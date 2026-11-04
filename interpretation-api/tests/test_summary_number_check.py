"""
Item 14 — wrong numbers in the executive summary are replaced (main L263).
The LLM is mocked; the checker is deterministic.
"""

from unittest.mock import MagicMock

from src.pipeline.summary_number_check import find_issues
from src.pipeline.steps.step06_report_generation import Step06ReportGeneration

SCORES = [
    (1.81, 'Ribosome'), (1.75, 'DNA replication'), (1.70, 'Cell cycle'),
    (-2.50, 'Histidine metabolism'), (-2.12, 'Primary bile acid biosynthesis'),
    (-1.73, 'Graft-versus-host disease'),
]

# The real Report 1 summary sentence (U+2011 minus signs, en-dash range, |NES| threshold).
REPORT1 = ('On the proliferative side, moderate‑strength enrichment (NES ≈ 1.7–1.8) of '
           'ribosome biogenesis, DNA replication. Concomitantly, strong down‑regulation '
           '(|NES| ≥ 2) of histidine metabolism (NES = ‑2.50) and primary bile‑acid '
           'biosynthesis (NES = ‑2.12) signals loss, while a moderate suppression of '
           'graft‑versus‑host disease signatures (NES = ‑1.73) reflects shifts. '
           'Network topology analysis identifies five high‑connectivity hub genes.')


def test_report1_summary_is_clean():
    text, rep = find_issues(REPORT1, SCORES, hub_counts=[5, 20], hypothesis_count=6)
    assert text == REPORT1
    assert rep['replaced'] == [] and rep['needs_rewrite'] == []
    assert rep['checked'] >= 5


def test_u2011_minus_is_read_as_minus():
    # Read as +2.50 this would be "unplaced"; read as -2.50 it is correct.
    _, rep = find_issues('histidine metabolism (NES = ‑2.50)', SCORES)
    assert rep['needs_rewrite'] == []


def test_u2212_minus_and_wrong_value_replaced_in_place():
    text, rep = find_issues('histidine metabolism (NES −2.31) is lost', SCORES)
    assert text == 'histidine metabolism (NES −2.50) is lost'
    assert rep['replaced'][0]['pathway'] == 'Histidine metabolism'


def test_replacement_uses_nearest_pathway_in_sentence():
    # Bile acid's value is wrong; histidine is named earlier in the same sentence and must
    # not capture the replacement.
    text, rep = find_issues('histidine metabolism (NES = -2.50) and primary bile-acid '
                            'biosynthesis (NES = -2.20) are lost', SCORES)
    assert '(NES = -2.12)' in text and '(NES = -2.50)' in text
    assert rep['replaced'][0]['pathway'] == 'Primary bile acid biosynthesis'


def test_ambiguous_nearest_word_is_not_replaced():
    scores = [(1.76, 'Mismatch repair'), (1.60, 'Base excision repair')]
    _, rep = find_issues('activation of DNA repair (NES 1.90)', scores)
    assert rep['replaced'] == []
    assert rep['needs_rewrite'][0]['kind'] == 'score_unplaced'


def test_threshold_skipped():
    _, rep = find_issues('signals with |NES| >= 2 and NES > 1.5 are strong', SCORES)
    assert rep['checked'] == 0


def test_range_outside_scores_needs_rewrite():
    _, rep = find_issues('enrichment (NES ≈ 2.6–2.9) of ribosome', SCORES)
    assert rep['needs_rewrite'][0]['kind'] == 'score_range'


def test_wrong_sign_needs_rewrite():
    _, rep = find_issues('histidine metabolism (NES = +2.50)', SCORES)
    assert rep['needs_rewrite'][0]['kind'] == 'score_sign'


def test_unsourced_fold_change_and_fdr_need_rewrite():
    _, rep = find_issues('CDK1 (log2FC = 3.2) and FDR = 0.001 are notable', SCORES)
    kinds = [i['kind'] for i in rep['needs_rewrite']]
    assert kinds.count('unsourced_stat') == 2


def test_fdr_threshold_is_not_a_claim():
    _, rep = find_issues('pathways at FDR <= 0.05 were used', SCORES)
    assert rep['needs_rewrite'] == []


def test_count_words_replaced():
    text, rep = find_issues('identifies Four hub genes and seven testable hypotheses',
                            SCORES, hub_counts=[5], hypothesis_count=6)
    assert text == 'identifies Five hub genes and six testable hypotheses'
    assert len(rep['replaced']) == 2


def test_ambiguous_hub_count_asks_for_rewrite_instead_of_guessing():
    text, rep = find_issues('identifies Four hub genes', SCORES, hub_counts=[5, 20])
    assert text == 'identifies Four hub genes'
    assert rep['needs_rewrite'][0]['kind'] == 'hub_count'


def test_counts_never_read_inside_words_or_from_other_phrases():
    # Reviewer cases: the 6 of IL6, the "one" of "None", "Hypothesis 2 and the other ...".
    for s in ('IL6 and CDK1 are hub genes.',
              'None of the hub genes is down-regulated.',
              'Hypothesis 2 and the other hypotheses agree.',
              'Two of the five hypotheses link CDK1 to E2F1.'):
        text, rep = find_issues(s, SCORES, hub_counts=[5], hypothesis_count=5)
        assert text == s, s
        assert rep['replaced'] == [], s


def test_zero_hypotheses_is_not_checked():
    text, rep = find_issues('two hypotheses', SCORES, hypothesis_count=0)
    assert text == 'two hypotheses' and rep['checked'] == 0


def test_step06_rewrites_once_then_records_unresolved():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    step.llm.chat.side_effect = ['histidine metabolism (NES = +2.50)',
                                 'histidine metabolism (NES = −2.50)']
    input_data = {'pathways': [{'name': 'Histidine metabolism', 'NES': -2.5}]}
    section = step._generate_executive_summary(
        study_context='x', input_data=input_data, step1={'themes': []},
        step2={'metadata': {'skipped': True}}, step3={}, step4={'hypotheses': []},
        step5={'metadata': {'skipped': True}})
    assert step.llm.chat.call_count == 2
    rewrite_msgs = step.llm.chat.call_args_list[1].kwargs['messages']
    assert 'Histidine metabolism: -2.50' in rewrite_msgs[-1]['content']
    check = section.source_data['summary_number_check']
    assert check['rewrite_requested'] is True and check['unresolved'] == []
    assert '(NES = −2.50)' in section.content


def test_step06_no_rewrite_when_clean():
    step = Step06ReportGeneration.__new__(Step06ReportGeneration)
    step.llm = MagicMock()
    step.llm.chat.return_value = 'histidine metabolism (NES = -2.50)'
    input_data = {'pathways': [{'name': 'Histidine metabolism', 'NES': -2.5}]}
    step._generate_executive_summary(
        study_context='x', input_data=input_data, step1={'themes': []},
        step2={'metadata': {'skipped': True}}, step3={}, step4={'hypotheses': []},
        step5={'metadata': {'skipped': True}})
    assert step.llm.chat.call_count == 1
