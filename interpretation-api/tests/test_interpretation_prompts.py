"""
Tests for Group G interpretation improvements:
- strongest signals in BOTH directions reach the executive-summary prompt
- master regulators are described as network/topological hubs (not proven drivers)
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step06_report_generation import Step06ReportGeneration


def _bare_step06():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    return step




def test_top_enrichment_signals_tolerates_string_scores():
    """R2-1: a string-valued NES/ES on a raw pathway dict must not crash."""
    step = _bare_step06()
    input_data = {'pathways': [
        {'name': 'A', 'NES': '2.5'},        # numeric string
        {'name': 'B', 'NES': -3.0},
        {'name': 'C', 'NES': 'not-a-number'},  # unparseable -> skipped
        {'name': 'D'},                       # no score -> skipped
    ]}
    out = step._top_enrichment_signals(input_data)
    assert 'A' in out and 'B' in out
    assert 'C' not in out


def test_top_enrichment_signals_both_directions_sorted():
    step = _bare_step06()
    input_data = {'pathways': [
        {'name': 'Cell cycle', 'NES': 1.77},
        {'name': 'Tryptophan metabolism', 'NES': -3.37},
        {'name': 'Fatty acid degradation', 'NES': -3.40},
        {'name': 'DNA replication', 'NES': 1.78},
    ]}
    out = step._top_enrichment_signals(input_data)
    assert 'Up-regulated' in out and 'Down-regulated' in out
    # strongest down signal listed first among down
    down_line = [l for l in out.splitlines() if 'Down-regulated' in l][0]
    assert down_line.index('Fatty acid degradation') < down_line.index('Tryptophan metabolism')


def test_exec_summary_prompt_includes_down_signals_and_guidance():
    step = _bare_step06()
    step.llm.chat.return_value = 'summary'
    input_data = {'pathways': [
        {'name': 'Cell cycle', 'NES': 1.77},
        {'name': 'Fatty acid degradation', 'NES': -3.40},
    ]}
    step._generate_executive_summary(
        study_context='disease: HCC',
        input_data=input_data,
        step1={'themes': [{'name': 'Cell cycle control'}]},
        step2={'network_hubs': [{'gene': 'X'}], 'master_regulators': [{'gene': 'CDK1'}]},
        step3={}, step4={'hypotheses': [1]},
    )
    prompt = step.llm.chat.call_args.kwargs['messages'][0]['content']
    assert 'Fatty acid degradation' in prompt         # down-regulated signal surfaced
    assert 'BOTH directions' in prompt                 # guidance present
    assert 'strongest effect-size drivers' in prompt   # hub caveat


def test_hub_section_labels_topological_hubs():
    step = _bare_step06()
    section = step._generate_hub_genes_section(
        {'network_hubs': [{'gene': 'CDK1', 'degree': 40, 'hub_score': 0.8, 'fold_change': 2.4}],
         'master_regulators': [{'gene': 'CDK1', 'hub_score': 0.8, 'degree': 40,
                                'fold_change': 2.4, 'role': 'kinase', 'pathways': []}]},
        input_data={'genes': []},
    )
    assert 'topological hubs' in section.content
    assert 'not necessarily the largest effect-size drivers' in section.content
