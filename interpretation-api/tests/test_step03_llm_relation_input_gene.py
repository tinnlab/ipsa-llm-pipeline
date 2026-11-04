"""
LLM-proposed relations (pathways with nothing retrieved from KEGG or Pathway Commons,
kept under decision D3) must have at least one input gene at either end (owner decision
2026-11-02). Case-insensitive. Retrieved-relation pathways are unaffected.
"""

from types import SimpleNamespace

from src.pipeline.steps.step03_pathway_mechanisms import (
    Step03PathwayMechanisms,
    _input_gene_set,
    filter_llm_relations,
)

GENES = [{'geneSymbol': 'Tfam', 'foldChange': 1.4}, {'geneSymbol': 'Ndufs1', 'foldChange': -0.8}]


def _inferred_ps(name='Mitochondrial central dogma', confidence='inferred'):
    return {'pathway': name, 'source': 'other', 'confidence': confidence,
            'structure': SimpleNamespace(id='x'), 'mapped_de_genes': [],
            'de_relations': [], 'pc_interactions': []}


def _mech(name, *pairs):
    return {'pathway': name, 'curatedRelations': [
        {'source': s, 'target': t, 'type': 'activation'} for s, t in pairs]}


def _run(mechs, structures, genes):
    step = Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)
    return step._validate_against_kegg({'pathwayMechanisms': mechs}, structures, genes)['pathwayMechanisms']


def _pairs(mech):
    return [(r['source'], r['target']) for r in mech['curatedRelations']]


def test_relation_with_no_input_gene_is_dropped():
    [out] = _run([_mech('Mitochondrial central dogma',
                        ('Tfam', 'Polrmt'), ('Polrmt', 'Ssbp1'), ('Mtpap', 'NDUFS1'))],
                 [_inferred_ps()], GENES)
    assert _pairs(out) == [('Tfam', 'Polrmt'), ('Mtpap', 'NDUFS1')]
    assert out['droppedRelations'] == [
        {'source': 'Polrmt', 'target': 'Ssbp1', 'type': 'activation', 'reference': 'input_genes'}]


def test_gene_set_llm_pathway_is_checked_too():
    [out] = _run([_mech('P', ('A1', 'B1'), ('A1', 'Tfam'))], [_inferred_ps('P', 'gene_set')], GENES)
    assert _pairs(out) == [('A1', 'Tfam')]


def test_no_input_genes_keeps_everything():
    [out] = _run([_mech('Mitochondrial central dogma', ('A1', 'B1'))], [_inferred_ps()], [])
    assert _pairs(out) == [('A1', 'B1')]
    assert 'droppedRelations' not in out


def test_retrieved_pathways_are_not_affected():
    # A Pathway Commons pathway uses the pair check only; the input-gene rule is for
    # LLM-proposed relations.
    ps = {'pathway': 'PC', 'source': 'other', 'confidence': 'pc_grounded',
          'structure': SimpleNamespace(id='x'), 'mapped_de_genes': [], 'de_relations': [],
          'pc_interactions': [SimpleNamespace(source='POLRMT', interaction_type='x', target='SSBP1')]}
    [out] = _run([_mech('PC', ('Polrmt', 'Ssbp1'))], [ps], GENES)
    assert _pairs(out) == [('Polrmt', 'Ssbp1')]


def test_filter_works_on_structure_relations_case_insensitive():
    rels = [SimpleNamespace(source='TFAM', target='POLRMT', subtype='inferred'),
            SimpleNamespace(source='POLRMT', target='SSBP1', subtype='inferred')]
    kept, dropped = filter_llm_relations(rels, _input_gene_set(GENES))
    assert [r.source for r in kept] == ['TFAM']
    assert [r.source for r in dropped] == ['POLRMT']


def test_input_gene_set_reads_both_keys():
    assert _input_gene_set([{'geneSymbol': 'Tfam'}, {'gene': 'ndufs1'}, {'gene': ''}]) == {'TFAM', 'NDUFS1'}
