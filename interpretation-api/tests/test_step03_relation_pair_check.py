"""
Items 16 and 19(d) (main L240): a Step 3 card keeps a relation only if its gene pair is
among the relations retrieved for that pathway (KGML DE relations, or Pathway Commons
interactions). A pathway with nothing retrieved keeps its relations (decision D3).
All comparisons are case-insensitive (mouse-cased model output vs uppercase sources).
"""

from types import SimpleNamespace

from src.pipeline.steps.step03_pathway_mechanisms import Step03PathwayMechanisms


def _step():
    return Step03PathwayMechanisms.__new__(Step03PathwayMechanisms)


def _kgml_rel(src, tgt, subtype):
    return SimpleNamespace(source=src, target=tgt, subtype=subtype, type='PPrel')


def _kegg_ps(name, rels, confidence='high'):
    structure = SimpleNamespace(id='hsa04110', genes=[SimpleNamespace(
        gene_symbol=s, name=s, names=[]) for r in rels for s in (r.source, r.target)],
        compounds=[])
    return {'pathway': name, 'source': 'kegg', 'confidence': confidence,
            'structure': structure, 'mapped_de_genes': [], 'de_relations': rels}


def _mech(name, *pairs):
    return {'pathway': name, 'curatedRelations': [
        {'source': s, 'target': t, 'type': 'activation'} for s, t in pairs]}


def _run(mechs, structures):
    out = _step()._validate_against_kegg({'pathwayMechanisms': mechs}, structures)
    return out['pathwayMechanisms']


def _pairs(mech):
    return [(r['source'], r['target']) for r in mech['curatedRelations']]


def test_relation_not_in_kgml_is_dropped():
    # The brief's case: E2F1 --[expression]--> CCNA2 is not a hsa04110 relation.
    ps = _kegg_ps('Cell cycle', [_kgml_rel('E2F1', 'MCM2', 'expression')])
    [mech] = _run([_mech('Cell cycle', ('E2F1', 'MCM2'), ('E2F1', 'CCNA2'))], [ps])
    assert _pairs(mech) == [('E2F1', 'MCM2')]
    assert mech['droppedRelations'][0]['target'] == 'CCNA2'


def test_directed_relation_reversed_is_dropped_undirected_kept():
    ps = _kegg_ps('Cell cycle', [_kgml_rel('CDK1', 'CDC25A', 'phosphorylation'),
                                 _kgml_rel('CDK1', 'CCNB1', 'binding/association')])
    [mech] = _run([_mech('Cell cycle', ('CDC25A', 'CDK1'), ('CCNB1', 'CDK1'))], [ps])
    assert _pairs(mech) == [('CCNB1', 'CDK1')]


def test_match_is_case_insensitive():
    ps = _kegg_ps('Cell cycle', [_kgml_rel('CDK1', 'CCNB1', 'activation')])
    [mech] = _run([_mech('Cell cycle', ('Cdk1', 'Ccnb1'))], [ps])
    assert _pairs(mech) == [('Cdk1', 'Ccnb1')]


def test_echoed_id_suffix_still_matches_pathway():
    ps = _kegg_ps('Cell cycle', [_kgml_rel('CDK1', 'CCNB1', 'activation')])
    [mech] = _run([_mech('Cell cycle (hsa04110)', ('CDK1', 'CCNB1'), ('CDK1', 'X1'))], [ps])
    assert _pairs(mech) == [('CDK1', 'CCNB1')]


def test_pair_check_is_per_pathway():
    a = _kegg_ps('Cell cycle', [_kgml_rel('CDK1', 'CCNB1', 'activation')])
    b = _kegg_ps('p53 signaling', [_kgml_rel('TP53', 'MDM2', 'expression')])
    [mech] = _run([_mech('Cell cycle', ('TP53', 'MDM2'))], [a, b])
    assert _pairs(mech) == []


def test_kgml_pathway_with_no_de_relation_keeps_nothing():
    ps = _kegg_ps('Ribosome', [], confidence='gene_set')
    [mech] = _run([_mech('Ribosome', ('RPL3', 'RPS6'))], [ps])
    assert _pairs(mech) == []


def test_pathway_commons_pair_check_either_direction_case_insensitive():
    ps = {'pathway': 'Mitochondrial central dogma', 'source': 'other',
          'confidence': 'pc_grounded', 'structure': SimpleNamespace(id='x'),
          'mapped_de_genes': [], 'de_relations': [],
          'pc_interactions': [SimpleNamespace(source='TFAM', interaction_type='controls-expression-of',
                                              target='POLRMT')]}
    [mech] = _run([_mech('Mitochondrial central dogma', ('Polrmt', 'Tfam'), ('Tfam', 'Ssbp1'))], [ps])
    assert _pairs(mech) == [('Polrmt', 'Tfam')]


def test_empty_retrieval_keeps_every_relation_d3():
    # Inferred pathway: no KGML, no PC interactions -> relations untouched (D3 = keep).
    ps = {'pathway': 'Mitochondrial central dogma', 'source': 'other', 'confidence': 'inferred',
          'structure': SimpleNamespace(id='x'), 'mapped_de_genes': [],
          'de_relations': [SimpleNamespace(source='Tfam', target='Polrmt', subtype='inferred')],
          'pc_interactions': []}
    mech = _mech('Mitochondrial central dogma', ('Tfam', 'Polrmt'), ('Tfam', 'Ssbp1'))
    [out] = _run([mech], [ps])
    assert _pairs(out) == [('Tfam', 'Polrmt'), ('Tfam', 'Ssbp1')]
    assert 'droppedRelations' not in out


def test_model_inferred_relations_are_not_a_reference_for_pc_pathways():
    # For a pc_grounded pathway, de_relations hold the MODEL's inferred relations; only
    # the PC interactions count as retrieved.
    ps = {'pathway': 'P', 'source': 'other', 'confidence': 'pc_grounded',
          'structure': SimpleNamespace(id='x'), 'mapped_de_genes': [],
          'de_relations': [SimpleNamespace(source='A1', target='B1', subtype='inferred')],
          'pc_interactions': [SimpleNamespace(source='C1', interaction_type='x', target='D1')]}
    [mech] = _run([_mech('P', ('A1', 'B1'))], [ps])
    assert _pairs(mech) == []
