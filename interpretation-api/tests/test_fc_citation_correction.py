"""
Tests for report bug 4: the Central Mechanistic Model cited scrambled fold-changes
(e.g. "HEATR1 +1.15" when the DE value is +1.10). Fold-changes are now templated from
the DE table via a correction pass: same-sign magnitude errors are rewritten; sign
conflicts are detected (and, in drop mode, their number is removed) so the prose can
never contradict the true direction.
"""

from src.pipeline.fc_utils import (
    gene_fc_lookup, correct_fc_citations, drop_sentences_with_fc_conflicts,
)


DE = gene_fc_lookup([
    {'geneSymbol': 'HEATR1', 'foldChange': 1.10},
    {'geneSymbol': 'NOP56', 'foldChange': 1.24},
    {'geneSymbol': 'FBL', 'foldChange': 1.09},
    {'geneSymbol': 'NHP2', 'foldChange': 1.18},
    {'geneSymbol': 'ACADS', 'foldChange': -1.41},
])


def test_gene_fc_lookup_tolerates_field_names():
    lut = gene_fc_lookup([
        {'geneSymbol': 'A', 'foldChange': 1.0},
        {'gene_symbol': 'B', 'log2_fold_change': -2.0},
        {'gene': 'C', 'fold_change': 3.0},
    ])
    assert lut == {'A': 1.0, 'B': -2.0, 'C': 3.0}


def test_magnitude_only_errors_are_rewritten():
    # The exact scrambled sentence from the reported PDF.
    text = ('the coordinated up-regulation of ribosome biogenesis factors '
            '(HEATR1 +1.15-fold, NOP56 +1.20-fold, FBL +1.24-fold, NHP2 +1.12-fold) '
            'expands translational capacity')
    out, conflicts = correct_fc_citations(text, DE)
    assert conflicts == set()
    assert 'HEATR1 +1.10-fold' in out
    assert 'NOP56 +1.24-fold' in out
    assert 'FBL +1.09-fold' in out
    assert 'NHP2 +1.18-fold' in out
    # None of the wrong magnitudes survive.
    for wrong in ('+1.15', '+1.20', '+1.12'):
        assert wrong not in out


def test_correct_value_is_left_untouched():
    out, conflicts = correct_fc_citations('NOP56 +1.24-fold rises', DE)
    assert out == 'NOP56 +1.24-fold rises'
    assert conflicts == set()


def test_sign_conflict_detected_and_kept_by_default():
    out, conflicts = correct_fc_citations('increased ACADS +1.40 supports catabolism', DE)
    assert conflicts == {'ACADS'}
    assert out == 'increased ACADS +1.40 supports catabolism'  # unchanged (caller decides)


def test_pvalues_and_unknown_genes_are_untouched():
    out, conflicts = correct_fc_citations('ACADS at p=1.23e-10 and XYZ1 +9.99', DE)
    assert out == 'ACADS at p=1.23e-10 and XYZ1 +9.99'  # XYZ1 not in DE table; p-value safe
    assert conflicts == set()


def test_paren_and_fc_prefixed_forms():
    out, _ = correct_fc_citations('FBL (+1.24) and NHP2 (FC: +1.12) rise', DE)
    assert 'FBL (+1.09)' in out
    assert 'NHP2 (FC: +1.18)' in out


def test_gene_fc_lookup_preserves_zero_over_falsy_orchain():
    # Regression: an `or`-chain would treat 0.0 as falsy and mis-pick log2_fold_change.
    lut = gene_fc_lookup([{'geneSymbol': 'X', 'foldChange': 0.0, 'log2_fold_change': -2.0}])
    assert lut == {'X': 0.0}


def test_hyphenated_symbol_not_misparsed():
    # "IL-6" must NOT be read as gene "IL" with FC -6 (requires a real separator).
    lut = {'IL': 5.0}
    out, conflicts = correct_fc_citations('IL-6 signaling is active', lut)
    assert out == 'IL-6 signaling is active'
    assert conflicts == set()


def test_scientific_notation_not_corrupted():
    out, _ = correct_fc_citations('FOXP3 +1.2e-3 tiny effect', {'FOXP3': 2.5})
    assert out == 'FOXP3 +1.2e-3 tiny effect'  # not rewritten into "+2.50e-3"


def test_drop_sentences_removes_only_contradicting_sentence():
    # ACADS is down (-1.41); the middle sentence claims it is increased. The whole
    # sentence is dropped (not just the number) so no contradicting direction word remains.
    text = ('Cells grow fast. Increased ACADS +1.40 supports catabolism. Overall aggressive.')
    out = drop_sentences_with_fc_conflicts(text, DE)
    assert 'ACADS' not in out
    assert 'Cells grow fast.' in out
    assert 'Overall aggressive.' in out
