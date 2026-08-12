"""
Step 3 batch mechanism interpretation must survive a malformed-JSON batch response.

Local models intermittently emit syntactically invalid JSON on long batch payloads (a
single missing comma fails json.loads for the whole batch). Rather than dropping all of a
batch's pathways, _interpret_pathway_batch parses tolerantly, then SALVAGES by splitting the
batch and retrying smaller sub-batches (shorter prompt -> shorter, valid response); a lone
pathway is retried once with a different seed before being skipped.
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms


# A batch response with a missing comma between two objects -> json.JSONDecodeError.
MALFORMED = '{"pathwayMechanisms": [{"pathway": "X"} {"pathway": "Y"}]}'


def _valid(*pathways):
    inner = ', '.join('{"pathway": "%s"}' % p for p in pathways)
    return '{"pathwayMechanisms": [%s]}' % inner


def _step(chat_returns=None, chat_side_effect=None):
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    step._build_system_prompt = lambda: 'sys'
    step._build_user_prompt = lambda batch, overlaps, themes: 'user'
    step.llm = MagicMock()
    if chat_side_effect is not None:
        step.llm.chat.side_effect = chat_side_effect
    else:
        step.llm.chat.return_value = chat_returns
    return step


def _batch(*names):
    return [{'pathway': n} for n in names]


def test_valid_batch_parses_without_retry():
    step = _step(chat_returns=_valid('P1', 'P2'))
    mech, failed = step._interpret_pathway_batch(_batch('P1', 'P2'), 'T', [], None)
    assert failed == []
    assert {m['pathway'] for m in mech} == {'P1', 'P2'}
    assert step.llm.chat.call_count == 1


def test_tolerates_json_code_fences():
    # A fenced response would break the old raw json.loads; the tolerant parser handles it.
    step = _step(chat_returns='```json\n' + _valid('P1') + '\n```')
    mech, failed = step._interpret_pathway_batch(_batch('P1'), 'T', [], None)
    assert failed == [] and mech[0]['pathway'] == 'P1'
    assert step.llm.chat.call_count == 1


def test_salvage_splits_and_recovers_all_pathways():
    # Full batch is malformed; each half then parses -> nothing is dropped.
    step = _step(chat_side_effect=[MALFORMED, _valid('P1'), _valid('P2')])
    mech, failed = step._interpret_pathway_batch(_batch('P1', 'P2'), 'T', [], None)
    assert failed == []
    assert {m['pathway'] for m in mech} == {'P1', 'P2'}
    assert step.llm.chat.call_count == 3          # full + 2 split halves


def test_salvage_recovers_the_good_half_and_skips_only_the_bad_pathway():
    # 2-pathway batch malformed; one half recovers, the other lone pathway stays malformed
    # (even on the seed-varied retry) -> only that one pathway is skipped.
    step = _step(chat_side_effect=[
        MALFORMED,        # full [P1, P2]
        _valid('P1'),     # split A [P1] -> ok
        MALFORMED,        # split B [P2] -> still bad
        MALFORMED,        # split B [P2] varied-seed retry -> still bad
    ])
    mech, failed = step._interpret_pathway_batch(_batch('P1', 'P2'), 'T', [], None)
    assert [m['pathway'] for m in mech] == ['P1']
    assert failed == ['P2']
    assert step.llm.chat.call_count == 4


def test_lone_pathway_recovered_on_seed_varied_retry():
    step = _step(chat_side_effect=[MALFORMED, _valid('P1')])
    mech, failed = step._interpret_pathway_batch(_batch('P1'), 'T', [], None)
    assert failed == [] and mech[0]['pathway'] == 'P1'
    assert step.llm.chat.call_count == 2          # initial + one varied-seed retry


def test_lone_pathway_retried_once_then_skipped():
    step = _step(chat_returns=MALFORMED)           # always malformed
    mech, failed = step._interpret_pathway_batch(_batch('P1'), 'T', [], None)
    assert mech == []
    assert failed == ['P1']
    assert step.llm.chat.call_count == 2          # exactly one retry, then give up


def test_seeds_differ_across_splits_and_retries():
    # The salvage must vary the seed so a temp=0 split/retry isn't a byte-for-byte repeat of
    # a call that already produced malformed JSON.
    step = _step(chat_side_effect=[MALFORMED, _valid('P1'), _valid('P2')])
    step._interpret_pathway_batch(_batch('P1', 'P2'), 'T', [], None)   # full + 2 split halves
    seeds = [c.kwargs.get('seed') for c in step.llm.chat.call_args_list]
    assert seeds == [42, 43, 44]                   # distinct, per split branch


def test_llm_exception_skips_batch_without_crashing():
    step = _step()
    step.llm.chat.side_effect = RuntimeError('500 from router')
    mech, failed = step._interpret_pathway_batch(_batch('P1', 'P2'), 'T', [], None)
    assert mech == []
    assert failed == ['P1', 'P2']
