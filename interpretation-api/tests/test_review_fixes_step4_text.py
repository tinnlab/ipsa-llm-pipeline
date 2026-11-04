"""
Fixes from the independent review: the always-on Step 4 text corrections must never
change correct text.
- pathway NES/FDR: only a value attached to the name; never the next pathway's value;
  ES citations are not overwritten with NES; "Ribosome" is not read inside "Ribosome
  biogenesis ..." when only "Ribosome" has stats.
- gene p-value/FDR: the value belongs to the nearest known gene before it.
- direction detector: an active verb after a gene ("IFNG induced CXCL10") is not a
  direction claim about that gene.
- arrow fold changes are checked at a sentence end too.
"""

from src.pipeline.hypothesis_checks import (
    correct_arrow_fc,
    correct_pathway_stats,
    strict_direction_errors,
)
from tests.test_bug4_tool_gating import _bare_step04

STATS = {'Ribosome': {'nes': 2.1, 'fdr': 1.0e-3, 'metric': 'NES'},
         'p53 signaling pathway': {'nes': -1.2, 'fdr': 2.0e-2, 'metric': 'NES'}}


def test_next_pathways_value_is_not_overwritten():
    text = 'Ribosome has NES 2.1 and the p53 pathway has NES -1.2.'
    assert correct_pathway_stats(text, STATS) == (text, 0)


def test_wrong_value_attached_to_the_name_is_still_corrected():
    new, n = correct_pathway_stats('Ribosome (NES = 1.7, FDR = 3.0e-2) is up.', STATS)
    assert n == 2 and '(NES = 2.1, FDR = 1.00e-03)' in new


def test_pathway_enrichment_wording_still_corrected():
    # The real KIRC run_J case: "<name> pathway enrichment (FDR=...)".
    stats = {'Allograft rejection': {'nes': 2.0, 'fdr': 1.47e-3, 'metric': 'NES'}}
    new, n = correct_pathway_stats('The Allograft rejection pathway enrichment (FDR=1.79e-25) reflects',
                                   stats)
    assert n == 1 and '(FDR=1.47e-03)' in new


def test_es_citation_not_overwritten_with_nes():
    text = 'Ribosome, ES 0.5, is up.'
    assert correct_pathway_stats(text, STATS) == (text, 0)


def test_short_name_not_matched_inside_longer_pathway_name():
    text = 'Ribosome biogenesis in eukaryotes (NES 1.6) is up.'
    assert correct_pathway_stats(text, STATS) == (text, 0)


def _gene_stats_step():
    step = _bare_step04()
    step._pathways_for_stats = []
    return step


GENES = [{'geneSymbol': 'MYC', 'pValue': 3.0e-4, 'pValueFDR': 4.0e-3},
         {'geneSymbol': 'CDK1', 'pValue': 1.2e-5, 'pValueFDR': 2.0e-4}]


def test_gene_pvalue_goes_to_nearest_gene():
    result = {'hypotheses': [{'hypothesis': 'MYC activates CDK1 (p = 1.2e-05).'}]}
    n = _gene_stats_step()._autocorrect_gene_stats(result, GENES)
    assert n == 0
    assert result['hypotheses'][0]['hypothesis'] == 'MYC activates CDK1 (p = 1.2e-05).'


def test_gene_pvalue_after_a_capitalised_word_is_checked():
    result = {'hypotheses': [{'hypothesis': 'The CDK1 p=9.9e-02 signal.'}]}
    n = _gene_stats_step()._autocorrect_gene_stats(result, GENES)
    assert n == 1
    assert result['hypotheses'][0]['hypothesis'] == 'The CDK1 p=1.20e-05 signal.'


def test_gene_fdr_two_values_for_two_genes():
    result = {'hypotheses': [{'hypothesis': 'MYC (FDR=1.0e-01) and CDK1 (FDR=5.0e-01) rise.'}]}
    n = _gene_stats_step()._autocorrect_gene_stats(result, GENES)
    assert n == 2
    assert result['hypotheses'][0]['hypothesis'] == 'MYC (FDR=4.00e-03) and CDK1 (FDR=2.00e-04) rise.'


FC = {'IFNG': -1.4, 'TP53': 1.2, 'SOD2': 1.1, 'MYC': -0.9, 'CDK1': -2.0}


def test_active_verbs_are_not_direction_claims():
    for text in ('IFNG induced CXCL10 expression', 'TP53 suppressed tumour growth',
                 'SOD2 reduced ROS levels', 'MYC higher activity'):
        assert strict_direction_errors(text, FC) == [], text


def test_real_direction_claims_still_flagged():
    for text in ('CDK1 is up-regulated in tumours', 'CDK1 up-regulated in tumours',
                 'CDK1 up-regulated.', 'up-regulation of CDK1', 'CDK1 (↑2.0) is elevated'):
        errs = strict_direction_errors(text, FC)
        assert errs and errs[0]['gene'] == 'CDK1' and errs[0]['claimed'] == 'up', text


def test_arrow_at_sentence_end_is_checked():
    text, conflicts = correct_arrow_fc('CDK1 ↓2.50.', {'CDK1': -2.0})
    assert text == 'CDK1 ↓2.00.' and conflicts == set()
    _, conflicts = correct_arrow_fc('CDK1 ↑2.00.', {'CDK1': -2.0})
    assert conflicts == {'CDK1'}
