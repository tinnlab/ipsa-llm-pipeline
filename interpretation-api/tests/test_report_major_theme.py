"""
Item 1 — the major theme is the pathway theme with the most pathways (main L265),
not themes[0]. Ties: lower average FDR, then the alphabetical name of the theme's most
significant pathway (never the model-written theme name).
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step06_report_generation import Step06ReportGeneration, major_theme

SKIPPED = {'metadata': {'skipped': True}}


def _theme(name, pathways, avg=None):
    t = {'name': name, 'pathways': [{'name': n, 'p_value_fdr': f} for n, f in pathways]}
    if avg is not None:
        t['avg_p_value_fdr'] = avg
    return t


def test_most_pathways_wins_over_first_theme():
    themes = [
        _theme('Histone remodeling', [('Alcoholism', 0.01), ('Viral carcinogenesis', 0.02)]),
        _theme('DNA repair', [('Base excision repair', 0.03), ('Mismatch repair', 0.01),
                              ('Nucleotide excision repair', 0.02)]),
    ]
    assert major_theme(themes)['name'] == 'DNA repair'


def test_tie_breaks_on_average_fdr():
    themes = [
        _theme('A', [('P1', 0.04), ('P2', 0.04)], avg=0.04),
        _theme('B', [('P3', 0.01), ('P4', 0.02)], avg=0.015),
    ]
    assert major_theme(themes)['name'] == 'B'


def test_tie_breaks_on_top_pathway_name_not_theme_name():
    themes = [
        _theme('Zeta theme', [('Apoptosis', 0.01), ('Ferroptosis', 0.03)], avg=0.02),
        _theme('Alpha theme', [('Ribosome', 0.01), ('Spliceosome', 0.03)], avg=0.02),
    ]
    # Both have 2 pathways and avg 0.02; top pathways are Apoptosis vs Ribosome.
    assert major_theme(themes)['name'] == 'Zeta theme'


def test_fdr_zero_is_most_significant():
    themes = [
        _theme('A', [('Apoptosis', 0.0), ('Zinc transport', 0.05)], avg=0.025),
        _theme('B', [('B cell receptor', 0.01), ('Y', 0.04)], avg=0.025),
    ]
    # A's most significant pathway is Apoptosis (FDR 0.0, not treated as missing), which
    # sorts before B's "B cell receptor". Reading 0.0 as 1.0 would pick Zinc and lose.
    assert major_theme(themes)['name'] == 'A'


def test_empty_themes():
    assert major_theme([]) is None


def test_executive_summary_uses_major_theme():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    step.llm.chat.return_value = 'Narrative.'
    step1 = {'themes': [
        _theme('Small first theme', [('Alcoholism', 0.01)]),
        _theme('DNA repair', [('Base excision repair', 0.03), ('Mismatch repair', 0.01)]),
    ]}
    section = step._generate_executive_summary(
        study_context='x', input_data={'pathways': []}, step1=step1,
        step2={'network_hubs': [], 'master_regulators': []}, step3={},
        step4={'hypotheses': []}, step5=SKIPPED)
    assert '**Major biological theme:** DNA repair' in section.content
    prompt = step.llm.chat.call_args_list[0].kwargs['messages'][0]['content']
    assert 'Major biological theme: DNA repair' in prompt


def test_key_findings_fallback_uses_major_theme():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    step1 = {'themes': [
        _theme('Small first theme', [('Alcoholism', 0.01)]),
        _theme('DNA repair', [('Base excision repair', 0.03), ('Mismatch repair', 0.01)]),
    ]}
    section = step._generate_conclusions(
        study_context='x', input_data={'pathways': []}, step1=step1,
        step2={'master_regulators': []}, step3={}, step4={'hypotheses': []}, step5=SKIPPED)
    assert '**Biological Theme:** DNA repair' in section.content


def test_theme_overview_drops_major_wording():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    section = step._generate_pathway_themes_section(
        {'themes': [_theme('T', [('P', 0.01)])]}, {'genes': []})
    assert 'identified **1** biological theme(s)' in section.content
