"""
Tests for KEGG relation hygiene (kegg_service._extract_relations + symbol cleaning).

Regression coverage for report artifacts: "undefined" nodes, compound-ID / pathway-name
targets, self-loops, exploded duplicate relations, and truncated "..." gene identifiers.
"""

from src.pipeline.services.kegg_service import (
    KEGGService,
    PathwayEntry,
    _clean_symbol,
    _is_gene_symbol,
)


def _gene(eid, symbol):
    return PathwayEntry(id=eid, type='gene', name=symbol, names=[symbol], gene_symbol=symbol)


def _compound(eid, cid):
    return PathwayEntry(id=eid, type='compound', name=cid, names=[cid], gene_symbol=cid)


def _map(eid, mapname):
    return PathwayEntry(id=eid, type='map', name=mapname, names=[mapname], gene_symbol=mapname)


def _rel(e1, e2, subtype='activation', rtype='PPrel'):
    return {'@entry1': e1, '@entry2': e2, '@type': rtype,
            'subtype': {'@name': subtype, '@value': '-->'}}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_clean_symbol_strips_ellipsis():
    assert _clean_symbol('UGT2B11...') == 'UGT2B11'
    assert _clean_symbol('CNPY3-GNMT...') == 'CNPY3-GNMT'
    assert _clean_symbol('  TP53 ,') == 'TP53'


def test_clean_symbol_keeps_readthrough_names():
    assert _clean_symbol('NT5C1B-RDH14') == 'NT5C1B-RDH14'
    assert _clean_symbol('CYP3A7-CYP3A51P') == 'CYP3A7-CYP3A51P'


def test_is_gene_symbol_rejects_non_genes():
    assert _is_gene_symbol('TP53')
    assert not _is_gene_symbol('undefined')
    assert not _is_gene_symbol('C00338')          # compound id
    assert not _is_gene_symbol('Sulfur metabolism')  # map/pathway name (has space)
    assert not _is_gene_symbol('7157')            # unmapped Entrez id
    assert not _is_gene_symbol('')


# ---------------------------------------------------------------------------
# _extract_relations
# ---------------------------------------------------------------------------

def test_valid_gene_gene_relation_kept():
    svc = KEGGService()
    entries = [_gene('1', 'CDK1'), _gene('2', 'CDC20')]
    rels = svc._extract_relations([_rel('1', '2')], entries)
    assert len(rels) == 1
    assert (rels[0].source, rels[0].target) == ('CDK1', 'CDC20')


def test_self_loop_dropped():
    svc = KEGGService()
    entries = [_gene('1', 'CAD')]
    rels = svc._extract_relations([_rel('1', '1', subtype='compound', rtype='ECrel')], entries)
    assert rels == []


def test_compound_endpoint_dropped():
    svc = KEGGService()
    entries = [_compound('1', 'C00338'), _gene('2', 'CD14')]
    rels = svc._extract_relations([_rel('1', '2')], entries)
    assert rels == []


def test_map_endpoint_dropped():
    svc = KEGGService()
    entries = [_gene('1', 'AGXT'), _map('2', 'Sulfur metabolism')]
    rels = svc._extract_relations([_rel('1', '2', subtype='compound', rtype='maplink')], entries)
    assert rels == []


def test_duplicate_relations_deduped():
    svc = KEGGService()
    entries = [_gene('1', 'CAD'), _gene('2', 'DHODH')]
    dupes = [_rel('1', '2', subtype='compound', rtype='ECrel')] * 5
    rels = svc._extract_relations(dupes, entries)
    assert len(rels) == 1


def test_truncated_symbol_cleaned_in_relation():
    svc = KEGGService()
    entries = [_gene('1', 'AGXT'), _gene('2', 'UGT2B11...')]
    rels = svc._extract_relations([_rel('1', '2')], entries)
    assert len(rels) == 1
    assert rels[0].target == 'UGT2B11'


def test_extract_gene_symbol_strips_ellipsis():
    svc = KEGGService()
    assert svc._extract_gene_symbol('UGT2B11, UGT2B10...') == 'UGT2B11'


# ---------------------------------------------------------------------------
# Group (protein complex) expansion — F7
# ---------------------------------------------------------------------------

def _group(eid, component_ids):
    return PathwayEntry(id=eid, type='group', name='complex', names=[],
                        gene_symbol=None, components=list(component_ids))


def test_group_endpoint_expanded_to_member_genes():
    svc = KEGGService()
    # A relation from CDK1 to a complex {CCNB1, CCNB2} should yield two gene<->gene edges
    entries = [_gene('1', 'CDK1'), _gene('2', 'CCNB1'), _gene('3', 'CCNB2'), _group('4', ['2', '3'])]
    rels = svc._extract_relations([_rel('1', '4')], entries)
    pairs = {(r.source, r.target) for r in rels}
    assert pairs == {('CDK1', 'CCNB1'), ('CDK1', 'CCNB2')}


def test_group_expansion_drops_self_loops():
    svc = KEGGService()
    # Group containing CDK1 itself → the CDK1->group edge would self-loop on CDK1, dropped
    entries = [_gene('1', 'CDK1'), _gene('2', 'CCNB1'), _group('3', ['1', '2'])]
    rels = svc._extract_relations([_rel('1', '3')], entries)
    pairs = {(r.source, r.target) for r in rels}
    assert pairs == {('CDK1', 'CCNB1')}


# ---------------------------------------------------------------------------
# map_de_genes_to_pathway dedup — F/CYP1A1 x6
# ---------------------------------------------------------------------------

def test_group_with_dangling_or_nongene_components_skipped():
    svc = KEGGService()
    # component '9' is not in entry_map; '3' is a compound → both skipped, only CCNB1 resolves
    entries = [_gene('1', 'CDK1'), _gene('2', 'CCNB1'), _compound('3', 'C00001'),
               _group('4', ['2', '3', '9'])]
    rels = svc._extract_relations([_rel('1', '4')], entries)
    assert {(r.source, r.target) for r in rels} == {('CDK1', 'CCNB1')}


def test_group_with_no_resolvable_members_yields_nothing():
    svc = KEGGService()
    entries = [_gene('1', 'CDK1'), _compound('2', 'C00002'), _group('3', ['2'])]
    assert svc._extract_relations([_rel('1', '3')], entries) == []


def test_map_de_genes_dedups_repeated_symbol():
    from src.pipeline.services.kegg_service import PathwayStructure
    svc = KEGGService()
    # KEGG repeats CYP1A1 across several graphics nodes
    genes = [_gene(str(i), 'CYP1A1') for i in range(6)]
    structure = PathwayStructure(id='hsa00830', name='Retinol', organism='hsa', genes=genes)
    mapped = svc.map_de_genes_to_pathway(structure, [{'name': 'CYP1A1', 'foldChange': 1.48}])
    assert [m.gene_symbol for m in mapped] == ['CYP1A1']
