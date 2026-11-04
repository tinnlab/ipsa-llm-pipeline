"""
Item 22 (main L238, suppl L256): crosstalk is computed through common DE genes. The
prompt lists, for each pathway, every partner with the shared DE gene names; crosstalk
text that names a gene outside the computed overlap is corrected.
"""

from types import SimpleNamespace

from src.pipeline.steps.step03_pathway_mechanisms import (
    PathwayOverlap, Step03PathwayMechanisms, check_crosstalk)


def _ps(name, *genes, direction='up'):
    return {'pathway': name, 'source': 'kegg', 'confidence': 'high',
            'structure': SimpleNamespace(id='x', genes=[]),
            'mapped_de_genes': [SimpleNamespace(gene_symbol=g, fold_change=1.0, p_value=0.01,
                                                direction=direction) for g in genes],
            'de_relations': []}


GENES = [{'geneSymbol': g} for g in ('PCNA', 'POLD1', 'MSH2', 'CDK1', 'CCNB1', 'SET', 'NFKB2')]


def _step():
    return Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)


def test_overlaps_use_de_genes_only():
    structs = [_ps('Mismatch repair', 'PCNA', 'MSH2'), _ps('DNA replication', 'PCNA', 'POLD1'),
               _ps('Chemokine signaling', 'PCNA', direction='unknown')]
    overlaps = _step()._calculate_pathway_overlaps(structs, [])
    assert [(o.pathway1, o.pathway2, o.shared_genes) for o in overlaps] == [
        ('Mismatch repair', 'DNA replication', ['PCNA'])]


def test_prompt_lists_every_partner_with_gene_names_per_batch_pathway():
    structs = [_ps('Mismatch repair', 'PCNA', 'MSH2'), _ps('Base excision repair', 'PCNA', 'POLD1'),
               _ps('Cell cycle', 'CDK1', 'CCNB1')]
    overlaps = _step()._calculate_pathway_overlaps(structs, [])
    prompt = _step()._build_user_prompt(structs[:1], overlaps, None)
    assert '- Mismatch repair:' in prompt
    assert '↔ Base excision repair: 1 shared DE gene(s): PCNA' in prompt
    assert 'Cell cycle ↔' not in prompt  # pairs not involving the batch's pathways


def _overlaps():
    return [PathwayOverlap('Mismatch repair', 'DNA replication', 1, ['PCNA'], []),
            PathwayOverlap('Cell cycle', 'DNA replication', 2, ['CDK1', 'CCNB1'], [])]


def _structs():
    return [_ps('Mismatch repair', 'PCNA', 'MSH2'), _ps('DNA replication', 'PCNA', 'CDK1', 'CCNB1'),
            _ps('Cell cycle', 'CDK1', 'CCNB1', 'NFKB2')]


def test_gene_outside_overlap_is_removed_from_a_list():
    mech = {'pathway': 'Mismatch repair',
            'crosstalk': ['Shares DE genes with DNA replication (PCNA, MSH2, POLD1), linking repair to replication.']}
    counts = check_crosstalk([mech], _overlaps(), _structs(), GENES)
    assert mech['crosstalk'] == ['Shares DE genes with DNA replication (PCNA), linking repair to replication.']
    assert counts['genes_removed'] == 2
    assert mech['crosstalkCorrections'][0]['removed_genes'] == ['MSH2', 'POLD1']


def test_clause_comma_is_not_a_list():
    mech = {'pathway': 'Mismatch repair',
            'crosstalk': ["Links to DNA replication; however, MSH2's role there is minor."]}
    check_crosstalk([mech], _overlaps(), _structs(), GENES)
    assert mech['crosstalk'] == []


def test_wrong_gene_inside_the_sentence_drops_the_line_instead_of_mangling_it():
    mech = {'pathway': 'Mismatch repair',
            'crosstalk': ['MSH2 is the sole shared gene with DNA replication.']}
    counts = check_crosstalk([mech], _overlaps(), _structs(), GENES)
    assert mech['crosstalk'] == []
    assert counts['lines_dropped'] == 1


def test_counts_in_the_text_are_left_as_written():
    line = 'Five shared DE genes (CDK1, CCNB1) connect to DNA replication.'
    mech = {'pathway': 'Cell cycle', 'crosstalk': [line]}
    check_crosstalk([mech], _overlaps(), _structs(), GENES)
    assert mech['crosstalk'] == [line]


def test_line_naming_partner_without_shared_gene_is_dropped():
    mech = {'pathway': 'Mismatch repair', 'crosstalk': ['Links to Cell cycle through CDK1.', 'Plain prose.']}
    check_crosstalk([mech], _overlaps(), _structs(), GENES)
    assert mech['crosstalk'] == ['Plain prose.']


def test_correct_line_is_untouched_and_english_words_are_not_genes():
    line = 'A set of shared genes (CDK1, CCNB1) couples Cell cycle and DNA replication.'
    mech = {'pathway': 'Cell cycle', 'crosstalk': [line]}
    check_crosstalk([mech], _overlaps(), _structs(), GENES)
    assert mech['crosstalk'] == [line]
    assert 'crosstalkCorrections' not in mech
