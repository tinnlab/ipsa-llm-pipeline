"""
Item 10 (main L248): Step 4 queries every table pathway through tool calls; one reminder,
then the engine queries the rest; the tool results reach the regeneration paths.
Items 12/13 (main L252, L254): the rewrite-then-remove step, off by default.
Scripted fake LLM; no network.
"""

import json
from unittest.mock import MagicMock

import src.pipeline.steps.step04_hypothesis_generation as s4
from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration

from tests.test_bug4_tool_gating import _bare_step04, _structures_only_step3

FINAL = json.dumps({'hypotheses': [], 'centralMechanisticModel': '', 'keyPredictions': [],
                    'hypothesesSummary': ''})


def _mech():
    return {'pathway_mechanisms': [], 'pathway_overlaps': [],
            'pathway_structures': _structures_only_step3()['pathway_structures']}


def _run(step):
    return step.execute(genes=[], pathways=[], themes=None, hub_genes_result=None,
                        mechanisms_result=_mech(), analyses=None, context=None)


def _tool_call(name, args, cid='c1'):
    return {'content': None, 'finish_reason': 'tool_calls', 'tool_calls': [
        {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


def test_prompts_ask_for_every_table_pathway():
    step = _bare_step04()
    assert 'Take EVERY pathway in the summary table' in step._build_system_prompt()
    assert '3-5 pathways' not in step._build_system_prompt()


def test_one_reminder_then_model_queries_the_missing_pathway():
    step = _bare_step04()
    step.llm.chat.return_value = FINAL
    step.llm.chat_with_tools.side_effect = [
        {'content': FINAL, 'tool_calls': None, 'finish_reason': 'stop'},          # answers too early
        _tool_call('get_pathway_mechanism', {'pathway_name': 'Butanoate metabolism'}),
        {'content': FINAL, 'tool_calls': None, 'finish_reason': 'stop'},
    ]
    result = _run(step)
    log = result.metadata['tool_call_log']
    assert log['nudged'] is True
    assert log['queried_by_model'] == ['Butanoate metabolism']
    assert log['filled_by_engine'] == [] and log['missing_after_loop'] == []
    reminder = step.llm.chat_with_tools.call_args_list[1].kwargs['messages'][-1]['content']
    assert 'Butanoate metabolism' in reminder


def test_engine_fills_pathways_still_missing_after_the_reminder():
    step = _bare_step04()
    step.llm.chat.return_value = FINAL
    step.llm.chat_with_tools.return_value = {'content': FINAL, 'tool_calls': None, 'finish_reason': 'stop'}
    result = _run(step)
    log = result.metadata['tool_call_log']
    assert log['filled_by_engine'] == ['Butanoate metabolism']
    assert [c['by'] for c in log['calls']] == ['engine']
    final_msgs = next(c.kwargs['messages'] for c in step.llm.chat.call_args_list
                      if 'All table pathways have now been queried' in str(c))
    tool_msgs = [m for m in final_msgs if m.get('role') == 'tool']
    assert tool_msgs and 'Butanoate metabolism' in tool_msgs[0]['content']


def test_max_iterations_is_30():
    import inspect
    assert 'max_iterations = 30' in inspect.getsource(Step04HypothesisGeneration.execute)


def test_coverage_regeneration_keeps_tool_context():
    step = _bare_step04()
    step._tool_context = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                          {'role': 'tool', 'tool_call_id': 'x', 'name': 'get_pathway_mechanism',
                           'content': '{"curatedRelations": []}'}]
    step._validate_hypothesis_coverage = MagicMock(return_value=[
        {'direction': 'down', 'pathway': 'P', 'nes': -3.0, 'genes': [], 'metric': 'NES'}])
    step.llm.chat.return_value = '{"hypotheses": []}'
    step._enforce_hypothesis_coverage({'hypotheses': []}, [], [], 'ctx', 's', 'u', max_attempts=1)
    msgs = step.llm.chat.call_args.args[0]
    assert any(m.get('role') == 'tool' for m in msgs)


# ------------------------------------------------------------------ items 12/13 rewrite

GENES = [{'geneSymbol': 'SLC27A5', 'foldChange': -1.31}, {'geneSymbol': 'HRAS', 'foldChange': 1.22}]
BAD = {'hypothesis': 'Loss of FXR permits up-regulation of SLC27A5.', 'mechanisticModel': '',
       'mechanisticClaims': []}
GOOD = {'hypothesis': 'Loss of FXR permits down-regulation of SLC27A5.', 'mechanisticModel': '',
        'mechanisticClaims': []}


def test_rewrite_default_is_off():
    assert s4.STEP4_REWRITE_ENABLED is False


def test_rewrite_fixes_once():
    step = _bare_step04()
    step._collected_relations = []
    step.llm.chat.return_value = json.dumps(GOOD)
    log = []
    fixed = step._rewrite_or_remove(BAD, GENES, seed_offset=401, log=log, label='Hypothesis 1')
    assert fixed == GOOD
    prompt = step.llm.chat.call_args.args[0][0]['content']
    assert 'SLC27A5: -1.31' in prompt
    assert step.llm.chat.call_args.kwargs['temperature'] == 0.0
    assert log[0]['outcome'] == 'rewritten' and log[0]['before'] == BAD and log[0]['after'] == GOOD


def test_rewrite_that_still_fails_removes():
    step = _bare_step04()
    step._collected_relations = []
    step.llm.chat.return_value = json.dumps(BAD)
    log = []
    assert step._rewrite_or_remove(BAD, GENES, log=log) is None
    assert log[0]['outcome'] == 'removed'


def test_relation_error_triggers_rewrite_with_returned_relations_listed():
    step = _bare_step04()
    step._collected_relations = [{'source': 'E2F1', 'target': 'MCM2', 'type': 'expression'}]
    hyp = dict(GOOD, mechanisticClaims=[{'source': 'E2F1', 'target': 'CCNA2', 'relation': 'expression'}])
    step.llm.chat.return_value = json.dumps(dict(GOOD, mechanisticClaims=[
        {'source': 'E2F1', 'target': 'MCM2', 'relation': 'expression'}]))
    fixed = step._rewrite_or_remove(hyp, GENES)
    assert fixed['mechanisticClaims'][0]['target'] == 'MCM2'
    assert 'E2F1 --[expression]--> MCM2' in step.llm.chat.call_args.args[0][0]['content']


def _scripted(first, rest):
    """chat(): the first call (generation, no tools) returns `first`, later calls `rest`."""
    calls = {'n': 0}

    def chat(*args, **kwargs):
        calls['n'] += 1
        return first if calls['n'] == 1 else rest
    return chat


def test_gate_off_records_flags_without_rewriting(monkeypatch):
    step = _bare_step04()
    step.llm.chat.side_effect = _scripted(json.dumps(
        {'hypotheses': [BAD], 'centralMechanisticModel': '', 'keyPredictions': [], 'hypothesesSummary': ''}),
        FINAL)
    result = step.execute(genes=GENES, pathways=[], themes=None, hub_genes_result=None,
                          mechanisms_result=None, analyses=None, context=None)
    flags = result.metadata['hypothesis_check_flags']
    assert flags and flags[0]['direction_errors'][0]['gene'] == 'SLC27A5'
    assert result.metadata['rewrites'] == [] and result.metadata['rewrite_removed'] == 0
    assert len(result.hypotheses) == 1


def test_gate_on_removal_counts_toward_regeneration_and_summary(monkeypatch):
    monkeypatch.setattr(s4, 'STEP4_REWRITE_ENABLED', True)
    step = _bare_step04()
    # Generation returns BAD + GOOD; every later call (rewrite, regeneration) returns BAD.
    step.llm.chat.side_effect = _scripted(json.dumps(
        {'hypotheses': [BAD, GOOD], 'centralMechanisticModel': '', 'keyPredictions': [], 'hypothesesSummary': ''}),
        json.dumps(BAD))
    result = step.execute(genes=GENES, pathways=[], themes=None, hub_genes_result=None,
                          mechanisms_result=None, analyses=None, context=None)
    assert result.metadata['rewrite_removed'] == 1
    assert result.metadata['rewrites'][0]['outcome'] == 'removed'
    calls = [str(c) for c in step.llm.chat.call_args_list]
    assert any('replacement hypotheses' in c for c in calls), 'the "< 4" regeneration should run'
    assert any('survived validation' in c for c in calls), 'the summary rewrite should run'
