"""
Fix 3 / AC2 — Key Findings must name the same dominant axes as the Executive
Summary, so a strongly enriched direction is never dropped from one while present
in the other. All helpers are deterministic; the LLM is mocked.
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step06_report_generation import Step06ReportGeneration


def _bare_step06():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    step.llm.chat.return_value = 'Narrative summary.'
    return step



INPUT_DATA = {'pathways': [
    {'name': 'Cell cycle', 'NES': 2.1},
    {'name': 'DNA replication', 'NES': 1.8},
    {'name': 'Fatty acid degradation', 'NES': -3.4},
    {'name': 'Valine, leucine and isoleucine degradation', 'NES': -2.6},
]}

STEP1 = {'themes': [
    {'name': 'Cell cycle regulation', 'pathways': [{'name': 'Cell cycle'}]},
    {'name': 'Fatty-acid catabolism', 'pathways': [{'name': 'Fatty acid degradation'}]},
]}


def test_dominant_axes_reports_both_directions():
    step = _bare_step06()
    axes = step._dominant_axes(INPUT_DATA)
    directions = {a['direction'] for a in axes}
    assert directions == {'up', 'down'}
    down = [a for a in axes if a['direction'] == 'down'][0]
    assert down['pathway'] == 'Fatty acid degradation'  # strongest by |NES|


def test_dominant_axes_labels_metric_honestly():
    """Metric label follows _enrichment_metric: classic ES stays 'ES'; |ES|>1 promotes to NES."""
    step = _bare_step06()
    es_axis = step._dominant_axes({'pathways': [{'name': 'P', 'ES': 0.5}]})
    assert es_axis[0]['metric'] == 'ES'
    nes_axis = step._dominant_axes({'pathways': [{'name': 'Q', 'ES': -3.2}]})
    assert nes_axis[0]['metric'] == 'NES'
    # and the rendered Key Findings / exec summary use that metric label
    section = step._generate_conclusions(
        study_context='x', input_data={'pathways': [{'name': 'P', 'ES': 0.5}]},
        step1=STEP1, step2={'master_regulators': []}, step3={},
        step4={'hypotheses': []})
    assert '(ES +0.50)' in section.content


def test_key_findings_names_the_down_axis():
    """AC2: the down-regulated axis must appear in Key Findings, not only up."""
    step = _bare_step06()
    section = step._generate_conclusions(
        study_context='disease: HCC',
        input_data=INPUT_DATA,
        step1=STEP1,
        step2={'master_regulators': [
            {'gene': 'CDK1', 'direction': 'up'},
            {'gene': 'ACADS', 'direction': 'down'},
        ]},
        step3={'upstream_regulators': [{'tf': 'HNF4A'}]},
        step4={'hypotheses': [1, 2]},
    )
    content = section.content
    assert 'Fatty acid degradation' in content          # down axis present
    assert 'Cell cycle' in content                        # up axis present
    # down-regulated master regulators surfaced separately (Fix 4 tie-in)
    assert 'ACADS' in content
    # TF candidate finding present (AC4 tie-in)
    assert 'HNF4A' in content


def test_key_findings_and_exec_summary_share_down_axis():
    """AC2: the same down axis name appears in BOTH sections."""
    step = _bare_step06()
    conclusions = step._generate_conclusions(
        study_context='disease: HCC', input_data=INPUT_DATA, step1=STEP1,
        step2={'master_regulators': [{'gene': 'CDK1', 'direction': 'up'}]},
        step3={}, step4={'hypotheses': [1]},
    )
    exec_summary = step._generate_executive_summary(
        study_context='disease: HCC', input_data=INPUT_DATA, step1=STEP1,
        step2={'network_hubs': [{'gene': 'CDK1'}],
               'master_regulators': [{'gene': 'CDK1', 'direction': 'up'}]},
        step3={}, step4={'hypotheses': [1]},
    )
    down = 'Fatty acid degradation'
    assert down in conclusions.content
    assert down in exec_summary.content   # deterministic "Dominant enrichment axes" metric


def test_key_findings_falls_back_without_enrichment_scores():
    step = _bare_step06()
    section = step._generate_conclusions(
        study_context='x', input_data={'pathways': []}, step1=STEP1,
        step2={'master_regulators': [{'gene': 'CDK1', 'direction': 'up'}]},
        step3={}, step4={'hypotheses': []},
    )
    # No NES data -> fall back to the leading theme name, still producing findings.
    assert 'Biological Theme' in section.content
