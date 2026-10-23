"""Meta step 9: the one remaining rendered LLM call (the per-dataset summary) is complete or
explicitly omitted, and the generators whose output is no longer rendered are not run."""
import logging
from dataclasses import asdict
from unittest.mock import MagicMock

from src.agents.llm_client import LLMEmptyContentError, LLMTruncatedError
from src.pipeline.meta_orchestrator import MetaPipelineOrchestrator
from src.pipeline.steps_meta.meta_step09_discovery_analysis import MetaStep09DiscoveryAnalysis


def _step9(reply='A paragraph.'):
    step = MetaStep09DiscoveryAnalysis.__new__(MetaStep09DiscoveryAnalysis)
    step.step_number, step.step_name = 9, 'Meta-Analysis Discovery Analysis'
    step.llm = MagicMock()
    if isinstance(reply, Exception):
        step.llm.chat.side_effect = reply
    else:
        step.llm.chat.return_value = reply
    step._warnings = []
    return step


def _summary(step):
    counts = {'OXPHOS': {'count': 10, 'best_pvalue': 1.4e-4},
              'Translation': {'count': 6, 'best_pvalue': 1.4e-4},
              'Lonely': {'count': 1, 'best_pvalue': 1e-3}}
    top = [{'name': n, **s} for n, s in counts.items()]
    return step._generate_individual_findings_summary(
        [f'd{i}' for i in range(14)], counts, top, 'Spaceflight')


def test_prompt_wording_and_budget():
    step = _step9()
    assert _summary(step) == 'A paragraph.'
    kwargs = step.llm.chat.call_args.kwargs
    prompt = kwargs['messages'][0]['content']
    assert 'Write ONE paragraph of at most 120 words' in prompt
    assert 'No headings, no bullet points. Do not restate the numbers above.' in prompt
    assert 'found in 10/14 datasets' in prompt
    assert 'Top 3 of 3 Pathways' in prompt
    assert 'studies' not in prompt.lower()  # "Study Context" is right: one study, many datasets
    assert '2-3 paragraphs' not in prompt
    assert 'BASELINE' not in prompt
    assert kwargs['require_complete'] is True
    assert kwargs['max_tokens'] == 4000


def test_truncated_summary_is_omitted_explicitly():
    for error in (LLMTruncatedError('cut'), LLMEmptyContentError('empty')):
        step = _step9(error)
        text = _summary(step)
        assert text == '*[Dataset summary could not be generated in full and is omitted.]*'
        assert step._warnings


def _synthetic(pathways):
    return {'steps': {'step1': {'themes': [], 'ungrouped_pathways': pathways},
                      'step2': {'network_hubs': []}, 'step3': {'pathway_mechanisms': []},
                      'step4': {'hypotheses': []}, 'step5': {'metadata': {'skipped': True}}},
            'input': {'genes': [], 'pathways': pathways, 'metadata': {}},
            'success': True, 'synthetic': True}


def _execute():
    pw = lambda name, p: {'name': name, 'pValue': p, 'pValueFDR': p, 'ES': -0.5}
    datasets = {'d1': _synthetic([pw('OXPHOS', 1e-4), pw('Translation', 1e-3)]),
                'd2': _synthetic([pw('OXPHOS', 2e-4)])}
    meta = {'steps': {'step1': {'themes': [{'name': 'Mito', 'pathways': [pw('OXPHOS', 1e-9),
                                                                         pw('Translation', 1e-8)]}],
                                'ungrouped': []},
                      'step4': {'hypotheses': []}}}
    step = _step9()
    result = step.execute(meta_results=meta, dataset_results=datasets,
                          dataset_names=['d1', 'd2'], metadata={'organism': 'Mus musculus'})
    return step, asdict(result)


def test_gated_generators_do_not_run():
    step, result = _execute()
    # Only the per-dataset summary reaches the LLM: no discovery-hypothesis fallback (#50),
    # no biological interpretation, no why-meta-unique explanations.
    assert step.llm.chat.call_count == 1
    assert 'Write ONE paragraph' in step.llm.chat.call_args.kwargs['messages'][0]['content']
    assert result['meta_generated_hypotheses'] == []
    assert result['biological_interpretation'] == ''
    assert set(result['summary']['gated_fields']) == set(MetaStep09DiscoveryAnalysis.GATED_FIELDS)
    assert result['summary']['meta_generated_hypotheses_count'] == 0


def test_validator_is_quiet_about_gated_fields(caplog):
    _, result = _execute()
    orch = MetaPipelineOrchestrator.__new__(MetaPipelineOrchestrator)
    with caplog.at_level(logging.INFO):
        orch._validate_step9_output(result)
    assert 'biological_interpretation is empty' not in caplog.text
    assert 'gated by design' in caplog.text


def test_validator_still_warns_on_a_real_empty_field(caplog):
    _, result = _execute()
    result['summary']['gated_fields'] = []
    orch = MetaPipelineOrchestrator.__new__(MetaPipelineOrchestrator)
    with caplog.at_level(logging.WARNING):
        orch._validate_step9_output(result)
    assert 'biological_interpretation is empty' in caplog.text
