"""
Item 17 (main L236): keep every KGML relation with a DE gene at one end.

KGML facts these tests encode (checked on the live hsa04110 KGML, 2026-11-01):
- a gene box can hold several genes; its `name` lists their KEGG ids
  ("hsa:4171 hsa:4172 ..."), but its graphics name lists only the FIRST gene's symbol
  and that gene's aliases ("MCM2, BM28, CCNL1, CDCL1, ..."). CCNL1 is a different real
  gene, so graphics-name tokens must never be read as box members;
- the members come from the KEGG id -> symbol mapping (`list/<org>`).
"""

import pytest

from src.pipeline.services import kegg_service as ks
from src.pipeline.services.kegg_service import KEGGService, MappedDEGene

KGML = """<?xml version="1.0"?>
<pathway name="path:hsa04110" org="hsa" number="04110" title="Cell cycle">
  <entry id="6" name="hsa:4171 hsa:4172" type="gene">
    <graphics name="MCM2, BM28, CCNL1, CDCL1..." type="rectangle"/>
  </entry>
  <entry id="7" name="hsa:983" type="gene">
    <graphics name="CDK1, CDC2, CDC28A..." type="rectangle"/>
  </entry>
  <entry id="8" name="hsa:983" type="gene">
    <graphics name="CDK1, CDC2, CDC28A..." type="rectangle"/>
  </entry>
  <entry id="9" name="hsa:891" type="gene">
    <graphics name="CCNB1, CCNB..." type="rectangle"/>
  </entry>
  <entry id="10" name="hsa:1869" type="gene">
    <graphics name="E2F1, E2F-1, RBAP1..." type="rectangle"/>
  </entry>
  <entry id="11" name="hsa:7157" type="gene">
    <graphics name="TP53, BCC7, LFS1, P53..." type="rectangle"/>
  </entry>
  <entry id="20" name="undefined" type="group">
    <component id="9"/>
    <component id="11"/>
  </entry>
  <relation entry1="10" entry2="6" type="GErel">
    <subtype name="expression" value="--&gt;"/>
  </relation>
  <relation entry1="8" entry2="9" type="PPrel">
    <subtype name="binding/association" value="---"/>
  </relation>
  <relation entry1="20" entry2="7" type="PPrel">
    <subtype name="activation" value="--&gt;"/>
  </relation>
</pathway>
"""

SYMBOLS = {'4171': 'MCM2', '4172': 'MCM3', '983': 'CDK1', '891': 'CCNB1',
           '1869': 'E2F1', '7157': 'TP53'}


@pytest.fixture
def structure(monkeypatch):
    monkeypatch.setitem(ks._SYMBOL_MAP_CACHE, 'hsa', dict(SYMBOLS))
    return KEGGService()._parse_kgml(KGML, 'hsa04110')


def _de(*syms):
    return [{'geneSymbol': s, 'foldChange': 1.5, 'pValue': 1e-4} for s in syms]


def _pairs(rels):
    return {(r.source, r.target) for r in rels}


def test_parse_kegg_gene_list_both_formats():
    text = ('hsa:4171\tCDS\t3:complement(1..2)\tMCM2, BM28, CCNL1; DNA replication licensing\n'
            'hsa:4172\tMCM3, HCC5; DNA replication licensing factor MCM3\n'
            'hsa:999\tncRNA\t1:1..2\tno symbol here\n')
    assert ks.parse_kegg_gene_list(text) == {'4171': 'MCM2', '4172': 'MCM3'}


def test_multi_gene_box_members_come_from_ids_not_aliases(structure):
    box = structure.entry_map['6']
    assert box.gene_symbols == ['MCM2', 'MCM3']
    assert 'CCNL1' not in box.gene_symbols  # an alias token that is a real other gene


def test_de_gene_second_in_box_is_mapped(structure):
    mapped = KEGGService().map_de_genes_to_pathway(structure, _de('MCM3'))
    assert [g.gene_symbol for g in mapped] == ['MCM3']


def test_alias_token_does_not_map_a_de_gene(structure):
    # CCNL1 appears only as an alias in the MCM2 box's graphics name.
    assert KEGGService().map_de_genes_to_pathway(structure, _de('CCNL1')) == []


def test_relation_to_second_gene_in_box_is_kept(structure):
    svc = KEGGService()
    mapped = svc.map_de_genes_to_pathway(structure, _de('MCM3'))
    assert ('E2F1', 'MCM3') in _pairs(svc.find_de_gene_relations(structure, mapped))


def test_relation_on_duplicate_node_is_kept(structure):
    # CDK1 is in boxes 7 and 8; the CCNB1 binding is drawn on box 8 only. Mapping keeps
    # box 7 as CDK1's entry id, so the old id match lost this relation.
    svc = KEGGService()
    mapped = svc.map_de_genes_to_pathway(structure, _de('CDK1'))
    assert mapped[0].pathway_entry_id == '7'
    assert ('CDK1', 'CCNB1') in _pairs(svc.find_de_gene_relations(structure, mapped))


def test_relation_from_group_is_kept_for_de_member(structure):
    # Group 20 (CCNB1 + TP53) activates CDK1. A DE TP53 must keep TP53 -> CDK1, whose
    # entry1 is the group's id, not TP53's.
    svc = KEGGService()
    mapped = svc.map_de_genes_to_pathway(structure, _de('TP53'))
    assert ('TP53', 'CDK1') in _pairs(svc.find_de_gene_relations(structure, mapped))


def test_relations_without_de_gene_dropped(structure):
    svc = KEGGService()
    mapped = svc.map_de_genes_to_pathway(structure, _de('E2F1'))
    pairs = _pairs(svc.find_de_gene_relations(structure, mapped))
    assert pairs == {('E2F1', 'MCM2'), ('E2F1', 'MCM3')}


def test_match_is_case_insensitive():
    rel = ks.PathwayRelation(entry1='1', entry2='2', type='PPrel', subtype='activation',
                             source='Cdk1', target='Ccnb1')
    st = ks.PathwayStructure(id='x', name='x', organism='mmu', relations=[rel])
    mapped = [MappedDEGene('1', 'CDK1', [], 1.0, 0.01, 'up')]
    assert KEGGService().find_de_gene_relations(st, mapped) == [rel]


def test_without_symbol_map_falls_back_to_graphics_symbol(monkeypatch):
    monkeypatch.setattr(KEGGService, '_gene_symbol_map', lambda self, org: {})
    st = KEGGService()._parse_kgml(KGML, 'hsa04110')
    assert ks.entry_symbols(st.entry_map['6']) == ['MCM2']
