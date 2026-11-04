"""LLM prompt/response logging.

Covers the properties the feature has to hold: ids present, ids absent (job_id fallback), a
hostile id that tries to escape the log root, a write failure that must not break the run, both
client roles distinguished, and — the one that protects model input — that the routing ids never
reach a prompt.
"""
import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.agents.prompt_log as prompt_log
from src.agents.llm_client import UnifiedLLMClient
from src.config import settings


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fake_response(content='hello', usage=None, status=200):
    """A response whose .text is a REAL string, so raw-body capture is exercised."""
    body = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}
    if usage:
        body['usage'] = usage
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status.return_value = None
    resp.text = json.dumps(body)
    resp.json.return_value = body
    return resp


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    """Point the log root at a tmp dir and reset the module's per-process state."""
    root = tmp_path / 'llm_logs'
    monkeypatch.setattr(settings, 'LLM_PROMPT_LOG_DIR', str(root))
    monkeypatch.setattr(settings, 'LLM_PROMPT_LOG_ENABLED', True)
    monkeypatch.setattr(settings, 'LLM_PROMPT_LOG_RETENTION_DAYS', 30)
    prompt_log._seq_counters.clear()
    prompt_log._warned.clear()
    return root


def _records(directory: Path):
    target = directory / prompt_log.CALLS_FILENAME
    if not target.exists():
        return []
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def _all_records(root: Path):
    out = []
    for path in root.rglob(prompt_log.CALLS_FILENAME):
        out.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
    return out


def _chat(client=None, **ctx):
    client = client or UnifiedLLMClient(provider='local')
    with prompt_log.log_context(**ctx):
        with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
            return client.chat([{'role': 'user', 'content': 'hi'}])


# ---------------------------------------------------------------------------
# 1. ids present
# ---------------------------------------------------------------------------

def test_ids_present_land_in_the_asked_for_tree(log_root):
    _chat(study_id='S1', analysis_id='A1', report_id='R1', job_id='job-9', step='step1')

    records = _records(log_root / 'S1' / 'A1' / 'R1')
    assert len(records) == 1
    rec = records[0]
    assert (rec['study_id'], rec['analysis_id'], rec['report_id']) == ('S1', 'A1', 'R1')
    assert rec['job_id'] == 'job-9'
    assert rec['step'] == 'step1'
    assert rec['seq'] == 1
    assert rec['http_status'] == 200
    assert rec['request']['messages'] == [{'role': 'user', 'content': 'hi'}]
    assert rec['response_text'] == 'hello'
    # The raw body, before any parsing — the thing step{N}_output.json does not already have.
    assert json.loads(rec['response_body'])['choices'][0]['message']['content'] == 'hello'
    assert rec['error'] is None
    assert isinstance(rec['duration_ms'], int)


def test_usage_and_sequence_are_recorded(log_root):
    client = UnifiedLLMClient(provider='local')
    usage = {'prompt_tokens': 11, 'completion_tokens': 3, 'total_tokens': 14}
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post',
                   return_value=_fake_response(usage=usage)):
            client.chat([{'role': 'user', 'content': 'a'}])
            client.chat([{'role': 'user', 'content': 'b'}])

    records = _records(log_root / 'S' / 'A' / 'R')
    assert [r['seq'] for r in records] == [1, 2]
    assert records[0]['usage'] == usage


# ---------------------------------------------------------------------------
# 2. ids absent -> job_id fallback
# ---------------------------------------------------------------------------

def test_no_ids_falls_back_to_job_id_and_is_never_dropped(log_root):
    _chat(job_id='job-42')

    expected = log_root / prompt_log.NO_STUDY / prompt_log.NO_ANALYSIS / 'job-42'
    assert len(_records(expected)) == 1


def test_partial_ids_still_bucket_at_full_depth(log_root):
    _chat(study_id='S1', job_id='job-7')

    expected = log_root / 'S1' / prompt_log.NO_ANALYSIS / 'job-7'
    assert len(_records(expected)) == 1


def test_no_ids_and_no_job_id_still_writes(log_root):
    _chat()
    assert len(_all_records(log_root)) == 1


# ---------------------------------------------------------------------------
# 3. hostile ids
# ---------------------------------------------------------------------------

HOSTILE = [
    '../../etc',
    '..',
    '../..',
    '/etc/passwd',
    'a/b',
    '....//....//etc',
    '',
    '   ',
    '.',
    'x' * 500,
]


@pytest.mark.parametrize('hostile', HOSTILE)
def test_hostile_ids_cannot_escape_the_log_root(log_root, hostile):
    _chat(study_id=hostile, analysis_id=hostile, report_id=hostile, job_id='job-1')

    root = log_root.resolve()
    written = list(root.rglob(prompt_log.CALLS_FILENAME))
    assert written, 'the record must still be written, not dropped'
    for path in written:
        # Raises ValueError if the file is not under the root.
        path.resolve().relative_to(root)
    # And nothing was created as a sibling of the root.
    siblings = [p for p in root.parent.iterdir() if p != root]
    assert siblings == []


@pytest.mark.parametrize('hostile', HOSTILE)
def test_resolve_dir_is_contained_for_hostile_ids(log_root, hostile):
    ctx = prompt_log.LogContext(
        study_id=hostile, analysis_id=hostile, report_id=hostile, job_id='j')
    directory = prompt_log.resolve_dir(ctx)
    assert directory is not None
    directory.relative_to(log_root.resolve())
    # Never a bare traversal component.
    assert '..' not in directory.parts


def test_non_string_ids_fall_back(log_root):
    ctx = prompt_log.LogContext(study_id={'a': 1}, analysis_id=['x'], report_id=7, job_id='j')
    directory = prompt_log.resolve_dir(ctx)
    assert directory == (log_root / prompt_log.NO_STUDY / prompt_log.NO_ANALYSIS / 'j').resolve()


# ---------------------------------------------------------------------------
# 4. a write failure must never break the run
# ---------------------------------------------------------------------------

def test_write_failure_does_not_break_the_call(log_root):
    # A regular file where the log root should be: every mkdir underneath it fails.
    log_root.parent.mkdir(parents=True, exist_ok=True)
    log_root.write_text('not a directory')

    assert _chat(study_id='S', analysis_id='A', report_id='R') == 'hello'


def test_write_failure_warns_only_once(log_root, caplog):
    log_root.parent.mkdir(parents=True, exist_ok=True)
    log_root.write_text('not a directory')

    client = UnifiedLLMClient(provider='local')
    with caplog.at_level('WARNING'):
        with prompt_log.log_context(study_id='S'):
            with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
                for _ in range(5):
                    client.chat([{'role': 'user', 'content': 'hi'}])

    warnings = [r for r in caplog.records if 'LLM prompt log' in r.getMessage()]
    assert len(warnings) == 1, 'a broken log root must not warn once per LLM call'


# ---------------------------------------------------------------------------
# 5. the two client roles are distinguished
# ---------------------------------------------------------------------------

def test_main_and_reviewer_roles_are_recorded_separately(log_root):
    main = UnifiedLLMClient(provider='local')
    reviewer = UnifiedLLMClient(provider='reviewer')
    assert main.role == 'main'
    assert reviewer.role == 'reviewer'

    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
            main.chat([{'role': 'user', 'content': 'x'}])
            reviewer.chat([{'role': 'user', 'content': 'y'}])

    records = _records(log_root / 'S' / 'A' / 'R')
    assert [r['role'] for r in records] == ['main', 'reviewer']
    assert records[0]['model'] == settings.LOCAL_LLM_MODEL
    assert records[1]['model'] == settings.REVIEWER_LLM_MODEL
    assert records[0]['endpoint'] != records[1]['endpoint']


def test_groq_client_alias_is_main_role(log_root):
    from src.agents.llm_client import GroqClient
    assert GroqClient().role == 'main'


# ---------------------------------------------------------------------------
# 6. chat_with_tools and complete() go through the same choke point
# ---------------------------------------------------------------------------

def test_chat_with_tools_is_logged_with_its_tools(log_root):
    client = UnifiedLLMClient(provider='local')
    tools = [{'type': 'function', 'function': {'name': 'search', 'parameters': {}}}]
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R', step='step4'):
        with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
            client.chat_with_tools([{'role': 'user', 'content': 'x'}], tools=tools)

    rec = _records(log_root / 'S' / 'A' / 'R')[0]
    assert rec['method'] == 'chat_with_tools'
    assert rec['request']['tools'] == tools
    assert rec['response_text'] == 'hello'


def test_complete_is_logged_via_chat(log_root):
    client = UnifiedLLMClient(provider='local')
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
            client.complete('a prompt', system_prompt='a system prompt')

    rec = _records(log_root / 'S' / 'A' / 'R')[0]
    assert rec['method'] == 'chat'
    roles = [m['role'] for m in rec['request']['messages']]
    assert roles == ['system', 'user']


def test_headers_are_never_logged(log_root):
    """The Groq path carries a bearer token; the record must not be able to leak it."""
    client = UnifiedLLMClient(provider='local')
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
            client.chat([{'role': 'user', 'content': 'x'}])

    raw = (log_root / 'S' / 'A' / 'R' / prompt_log.CALLS_FILENAME).read_text()
    assert 'Authorization' not in raw
    assert 'headers' not in json.loads(raw)


# ---------------------------------------------------------------------------
# 7. errors are recorded AND still raised
# ---------------------------------------------------------------------------

def test_request_error_is_recorded_and_still_raises(log_root):
    import requests

    client = UnifiedLLMClient(provider='local')
    boom = requests.exceptions.ConnectionError('upstream down')
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', side_effect=boom):
            with pytest.raises(Exception) as excinfo:
                client.chat([{'role': 'user', 'content': 'x'}])

    assert 'LOCAL API error' in str(excinfo.value)
    rec = _records(log_root / 'S' / 'A' / 'R')[0]
    assert rec['error']['type'] == 'ConnectionError'
    assert 'upstream down' in rec['error']['message']
    assert rec['response_text'] is None


def test_http_error_body_is_recorded(log_root):
    import requests

    client = UnifiedLLMClient(provider='local')
    err_resp = MagicMock()
    err_resp.status_code = 500
    err_resp.text = 'upstream exploded'
    boom = requests.exceptions.HTTPError('500 Server Error')
    boom.response = err_resp

    resp = _fake_response()
    resp.raise_for_status.side_effect = boom
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', return_value=resp):
            with pytest.raises(Exception):
                client.chat([{'role': 'user', 'content': 'x'}])

    rec = _records(log_root / 'S' / 'A' / 'R')[0]
    assert rec['error']['http_status'] == 500
    assert rec['error']['body'] == 'upstream exploded'


def test_unparseable_body_is_recorded_with_the_raw_text(log_root):
    """The case where the raw body matters most: it didn't parse."""
    client = UnifiedLLMClient(provider='local')
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status.return_value = None
    resp.text = 'this is not json'
    resp.json.side_effect = ValueError('no json')

    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', return_value=resp):
            with pytest.raises(ValueError):
                client.chat([{'role': 'user', 'content': 'x'}])

    rec = _records(log_root / 'S' / 'A' / 'R')[0]
    assert rec['response_body'] == 'this is not json'
    assert rec['error']['type'] == 'ValueError'


# ---------------------------------------------------------------------------
# 8. disabled
# ---------------------------------------------------------------------------

def test_disabled_writes_nothing(log_root, monkeypatch):
    monkeypatch.setattr(settings, 'LLM_PROMPT_LOG_ENABLED', False)
    assert _chat(study_id='S', analysis_id='A', report_id='R') == 'hello'
    assert not log_root.exists()


# ---------------------------------------------------------------------------
# 9. retention
# ---------------------------------------------------------------------------

def test_prune_removes_old_runs_and_keeps_recent_ones(log_root):
    old = log_root / 'S' / 'A' / 'old-run'
    new = log_root / 'S' / 'A' / 'new-run'
    for directory in (old, new):
        directory.mkdir(parents=True)
        (directory / prompt_log.CALLS_FILENAME).write_text('{}\n')

    ancient = time.time() - (40 * 86400)
    os.utime(old, (ancient, ancient))

    removed = prompt_log.prune(days=30)

    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_prune_tidies_empty_parents(log_root):
    old = log_root / 'S' / 'A' / 'old-run'
    old.mkdir(parents=True)
    ancient = time.time() - (40 * 86400)
    os.utime(old, (ancient, ancient))

    prompt_log.prune(days=30)

    assert not (log_root / 'S').exists()
    assert log_root.exists()


def test_prune_disabled_keeps_everything(log_root):
    old = log_root / 'S' / 'A' / 'old-run'
    old.mkdir(parents=True)
    ancient = time.time() - (400 * 86400)
    os.utime(old, (ancient, ancient))

    assert prompt_log.prune(days=0) == 0
    assert old.exists()


def test_prune_on_a_missing_root_is_a_noop(log_root):
    assert prompt_log.prune(days=30) == 0


# ---------------------------------------------------------------------------
# 10. context handling
# ---------------------------------------------------------------------------

def test_context_is_restored_after_the_block(log_root):
    before = prompt_log.current_context()
    with prompt_log.log_context(study_id='S', step='step1'):
        assert prompt_log.current_context().study_id == 'S'
    assert prompt_log.current_context() == before


def test_set_step_narrows_without_losing_the_ids(log_root):
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R', step='step1'):
        prompt_log.set_step('step4')
        ctx = prompt_log.current_context()
        assert ctx.step == 'step4'
        assert ctx.study_id == 'S'


def test_nested_context_does_not_clobber_with_none(log_root):
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with prompt_log.log_context(step='step2'):
            ctx = prompt_log.current_context()
            assert ctx.study_id == 'S'
            assert ctx.step == 'step2'


def test_context_reaches_a_thread_when_copied(log_root):
    """Step 4's reviewer prewarm runs in a thread; a fresh thread gets an EMPTY context."""
    import contextvars
    import threading

    seen = {}

    def worker():
        seen['ctx'] = prompt_log.current_context()

    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        ctx = contextvars.copy_context()
        thread = threading.Thread(target=lambda: ctx.run(worker))
        thread.start()
        thread.join()

    assert seen['ctx'].study_id == 'S'


# ---------------------------------------------------------------------------
# 11. the ids must never reach a prompt
# ---------------------------------------------------------------------------

def test_extract_ids_accepts_both_spellings():
    camel = prompt_log.extract_ids(
        {'studyId': 'S', 'analysisId': 'A', 'reportId': 'R', 'disease': 'x'})
    snake = prompt_log.extract_ids(
        {'study_id': 'S', 'analysis_id': 'A', 'report_id': 'R'})
    assert camel == snake == {'study_id': 'S', 'analysis_id': 'A', 'report_id': 'R'}


def test_extract_ids_ignores_blank_and_non_string():
    ids = prompt_log.extract_ids({'studyId': '   ', 'analysisId': 5, 'reportId': None})
    assert ids == {'study_id': None, 'analysis_id': None, 'report_id': None}


def test_strip_ids_removes_both_spellings_and_keeps_the_rest():
    stripped = prompt_log.strip_ids({
        'studyId': 'S', 'analysis_id': 'A', 'reportId': 'R',
        'disease': 'LUAD', 'organism': 'Homo sapiens',
    })
    assert stripped == {'disease': 'LUAD', 'organism': 'Homo sapiens'}


def test_orchestrator_keeps_the_ids_out_of_every_prompt(log_root, tmp_path):
    """The load-bearing guarantee.

    `metadata` is not inert: the orchestrator copies every non-organism key into `context`,
    which the steps render straight into their prompts, and Step 5 builds the report title from
    raw metadata keys. If the routing ids were left in, adding observability would have changed
    model input — so assert no id value appears in any logged request.
    """
    from src.pipeline.orchestrator import PipelineOrchestrator

    study, analysis, report = 'STUDY-ZZZ', 'ANALYSIS-YYY', 'REPORT-XXX'
    input_data = {
        'metadata': {
            'studyId': study, 'analysisId': analysis, 'reportId': report,
            'disease': 'LUAD', 'organism': 'Homo sapiens',
        },
        'genes': [],
        'pathways': [{'name': 'Glycolysis', 'source': 'KEGG', 'pathwayId': 'hsa00010',
                      'pValue': 0.001, 'pValueFDR': 0.01, 'genes': ['TP53', 'EGFR']}],
    }

    orchestrator = PipelineOrchestrator(
        output_dir=str(tmp_path / 'jobs'), enable_validation=False)

    with patch('src.agents.llm_client.requests.post', return_value=_fake_response(content='{}')):
        try:
            orchestrator.run_full_pipeline(input_data, job_id='job-leak')
        except Exception:
            # A stub LLM cannot satisfy every step; the logged prompts are what matter here.
            pass

    records = _all_records(log_root)
    assert records, 'the run must have logged at least one LLM call'
    for rec in records:
        blob = json.dumps(rec['request'])
        for identifier in (study, analysis, report):
            assert identifier not in blob, f'{identifier} leaked into a prompt'

    # ...but the ids DID route the logs, and the original input keeps them for provenance.
    assert (log_root / study / analysis / report).is_dir()
    saved = json.loads((tmp_path / 'jobs' / 'job-leak' / 'input_data.json').read_text())
    assert saved['metadata']['studyId'] == study


def test_orchestrator_writes_a_run_manifest(log_root, tmp_path):
    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator(
        output_dir=str(tmp_path / 'jobs'), enable_validation=False)
    input_data = {
        'metadata': {'studyId': 'S', 'analysisId': 'A', 'reportId': 'R'},
        'genes': [],
        'pathways': [{'name': 'Glycolysis', 'source': 'KEGG', 'pathwayId': 'hsa00010',
                      'pValue': 0.001, 'pValueFDR': 0.01, 'genes': ['TP53']}],
    }
    with patch('src.agents.llm_client.requests.post', return_value=_fake_response(content='{}')):
        try:
            orchestrator.run_full_pipeline(input_data, job_id='job-manifest')
        except Exception:
            pass

    manifest = json.loads(
        (log_root / 'S' / 'A' / 'R' / prompt_log.MANIFEST_FILENAME).read_text())
    assert manifest['job_id'] == 'job-manifest'
    assert manifest['study_id'] == 'S'
    assert 'started_at' in manifest and 'finished_at' in manifest


# ---------------------------------------------------------------------------
# 12. record placement and sequence numbering
# ---------------------------------------------------------------------------

def test_record_lands_in_its_own_run_not_the_ambient_one(log_root):
    """A record must follow the ids it was created with, even if the context moved on."""
    record = prompt_log.start_record(
        provider='local', role='main', model='m', endpoint='http://e',
        method='chat', stream=False, payload={'messages': []})
    assert record is not None

    # Simulate the context having moved on to a different run before the write lands.
    with prompt_log.log_context(study_id='OTHER', analysis_id='OTHER', report_id='OTHER'):
        record.update({'study_id': 'S1', 'analysis_id': 'A1', 'report_id': 'R1'})
        prompt_log.write(record)

    assert len(_records(log_root / 'S1' / 'A1' / 'R1')) == 1
    assert not (log_root / 'OTHER').exists()


def test_sequence_continues_across_a_process_restart(log_root):
    """A restart mid-run must not write a second call #1 into the same file."""
    with prompt_log.log_context(study_id='S', analysis_id='A', report_id='R'):
        with patch('src.agents.llm_client.requests.post', return_value=_fake_response()):
            client = UnifiedLLMClient(provider='local')
            client.chat([{'role': 'user', 'content': 'a'}])
            client.chat([{'role': 'user', 'content': 'b'}])

            # A fresh process has no in-memory counters, but the file is still there.
            prompt_log._seq_counters.clear()
            client.chat([{'role': 'user', 'content': 'c'}])

    assert [r['seq'] for r in _records(log_root / 'S' / 'A' / 'R')] == [1, 2, 3]


# ---------------------------------------------------------------------------
# 13. the container path and the code default must agree
# ---------------------------------------------------------------------------

def test_code_default_matches_the_compose_mount_point():
    """Defence in depth against the silent-wipe failure.

    docker-compose.prod.yml mounts the host log dir at /app/logs/llm_prompts and sets
    LLM_PROMPT_LOG_DIR to the same path. The code default is relative and must resolve to that
    identical path under the image's WORKDIR (/app), so the logs land on the persisted volume
    even if the compose environment entry were dropped. Assert the spelling rather than trust
    it: an underscore/hyphen slip here would silently send logs to the container's writable
    layer, which every redeploy wipes.
    """
    default = type(settings).model_fields['LLM_PROMPT_LOG_DIR'].default
    assert default == 'logs/llm_prompts'
    assert str(Path('/app') / default) == '/app/logs/llm_prompts'

    compose = (Path(__file__).resolve().parents[2] / 'docker-compose.prod.yml').read_text()
    assert 'LLM_PROMPT_LOG_DIR=/app/logs/llm_prompts' in compose
    assert ':/app/logs/llm_prompts' in compose
