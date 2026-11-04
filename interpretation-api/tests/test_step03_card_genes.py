"""
Items 11/20 and 21: Step 3 card genes.

- item 20 (main L240): a gene that is neither DE nor in a retrieved relation is flagged
  on the card and removed while STEP3_GENE_CHECK_ENABLED is on (default on);
- item 21 (main L263, L269): the card's DE genes and fold changes are the engine's
  mapped DE genes of that pathway; the model's role text is kept; unmapped model genes
  (e.g. DE genes of another pathway) are dropped.
"""

from types import SimpleNamespace

from src.pipeline.steps import step03_pathway_mechanisms as s3
from src.pipeline.steps.step03_pathway_mechanisms import normalize_card_genes

GENES = [{'geneSymbol': s, 'foldChange': fc} for s, fc in
         (('RPL3', 1.2), ('RPS6', 0.9), ('HEATR1', 1.5), ('NOP56', 1.1))]


def _mapped(sym, fc):
    return SimpleNamespace(gene_symbol=sym, fold_change=fc, direction='up' if fc > 0 else 'down')


def _ribosome_ps(mapped=(('RPL3', 1.2), ('RPS6', 0.9)), rels=()):
    return {'pathway': 'Ribosome', 'source': 'kegg', 'confidence': 'high',
            'mapped_de_genes': [_mapped(s, fc) for s, fc in mapped],
            'de_relations': [SimpleNamespace(source=a, target=b, subtype='binding/association')
                             for a, b in rels]}


def _card(*genes):
    return {'pathway': 'Ribosome', 'deGeneInvolvement': [
        {'gene': g, 'foldChange': fc, 'roleInPathway': f'role of {g}'} for g, fc in genes]}


def test_ribosome_card_loses_genes_of_ribosome_biogenesis():
    # The brief's case: HEATR1 and NOP56 are DE but belong to hsa03008, not hsa03010.
    card = _card(('RPL3', 1.2), ('HEATR1', 1.5), ('NOP56', 1.1))
    counts = normalize_card_genes([card], [_ribosome_ps()], GENES)
    assert [g['gene'] for g in card['deGeneInvolvement']] == ['RPL3', 'RPS6']
    assert card['droppedGenes'] == ['HEATR1', 'NOP56']
    assert counts['dropped_unmapped'] == 2 and counts['added_mapped'] == 1


def test_fold_changes_come_from_the_engine_and_roles_are_kept():
    card = _card(('RPL3', 3.7))  # model wrote a wrong fold change
    normalize_card_genes([card], [_ribosome_ps()], GENES)
    rpl3 = card['deGeneInvolvement'][0]
    assert rpl3 == {'gene': 'RPL3', 'foldChange': 1.2, 'roleInPathway': 'role of RPL3'}
    assert card['deGeneInvolvement'][1]['roleInPathway'] == ''  # RPS6 added, no model text


def test_case_insensitive_match_keeps_engine_symbol():
    card = _card(('rpl3', 1.2))
    normalize_card_genes([card], [_ribosome_ps()], GENES)
    assert card['deGeneInvolvement'][0]['gene'] == 'RPL3'
    assert card['deGeneInvolvement'][0]['roleInPathway'] == 'role of rpl3'


def test_gene_check_flags_non_de_gene_not_in_retrieved_relation():
    card = _card(('RPL3', 1.2), ('FAKE1', 2.0), ('EEF2', 0.0))
    ps = _ribosome_ps(rels=[('RPL3', 'EEF2')])  # EEF2 is non-DE but in a retrieved relation
    normalize_card_genes([card], [ps], GENES, gene_check_enabled=False)
    assert card['geneCheckFlags'] == ['FAKE1']


def test_gene_check_default_is_on():
    assert s3.STEP3_GENE_CHECK_ENABLED is True


def test_gene_check_removes_when_enabled_without_item21_data():
    # No mapped DE genes (no DE data for the pathway), so only the gene check acts.
    card = _card(('RPL3', 1.2), ('FAKE1', 2.0))
    counts = normalize_card_genes([card], [_ribosome_ps(mapped=())], GENES, gene_check_enabled=True)
    assert [g['gene'] for g in card['deGeneInvolvement']] == ['RPL3']
    assert counts['removed_by_gene_check'] == 1


def test_card_without_structure_is_left_alone_except_flags():
    card = {'pathway': 'Unknown', 'deGeneInvolvement': [{'gene': 'RPL3', 'foldChange': 9.9}]}
    normalize_card_genes([card], [], GENES)
    assert card['deGeneInvolvement'] == [{'gene': 'RPL3', 'foldChange': 9.9}]
