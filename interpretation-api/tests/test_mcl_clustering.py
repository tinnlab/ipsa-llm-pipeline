"""Tests for Markov Clustering (MCL) in the pathway-theme step.

Covers the core reason MCL replaces connected-components:

* Connected components on a thresholded Jaccard graph is single-linkage clustering — a
  single weak "bridge" pathway transitively merges two otherwise-distinct gene-overlap
  communities (the chaining / hairball problem).
* MCL (flow simulation) separates those communities.

Also guards the backward-compatible output shape, determinism, representative selection,
and the module-level MCL helpers on tiny known inputs.
"""

import pytest

from src.pipeline.services.pathway_clustering_service import (
    PathwayClusteringService,
    _build_weighted_adjacency,
    _mcl_partition,
    _select_representative,
    _jaccard,
)

# markov_clustering (and numpy) are required for the MCL path.
pytest.importorskip("markov_clustering")
pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Fixtures: two dense gene-overlap groups + one weak bridge pathway
# ---------------------------------------------------------------------------

# Group A pathways share {G1..G6}; group B pathways share {H1..H6}. The BRIDGE pathway
# overlaps 2 genes with each group — enough to create edges to both (so connected
# components chains A+B into one blob) but not enough to belong to either community.
_GA = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
_GB = ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']


def _two_groups_with_bridge():
    pathways = [
        {'name': 'A1', 'genes': _GA, 'p_value_fdr': 1e-4, 'NES': 2.1, 'database': 'KEGG'},
        {'name': 'A2', 'genes': _GA[:5] + ['G7'], 'p_value_fdr': 2e-4, 'NES': 1.9, 'database': 'KEGG'},
        {'name': 'A3', 'genes': _GA[:4] + ['G8', 'G9'], 'p_value_fdr': 3e-4, 'NES': 1.5, 'database': 'KEGG'},
        {'name': 'B1', 'genes': _GB, 'p_value_fdr': 1e-4, 'NES': 2.0, 'database': 'KEGG'},
        {'name': 'B2', 'genes': _GB[:5] + ['H7'], 'p_value_fdr': 2e-4, 'NES': 1.8, 'database': 'KEGG'},
        {'name': 'B3', 'genes': _GB[:4] + ['H8', 'H9'], 'p_value_fdr': 3e-4, 'NES': 1.4, 'database': 'KEGG'},
        {'name': 'BRIDGE', 'genes': ['G1', 'G2', 'H1', 'H2'], 'p_value_fdr': 0.01, 'NES': 1.0, 'database': 'KEGG'},
    ]
    all_symbols = set(_GA + _GB + ['G7', 'G8', 'G9', 'H7', 'H8', 'H9'])
    genes = [{'geneSymbol': g, 'foldChange': 1.5} for g in sorted(all_symbols)]
    return pathways, genes


def _cluster_of(result, pathway_name):
    """Return the set of pathway names in whichever cluster contains `pathway_name`."""
    for c in result['clusters']:
        if pathway_name in c['pathway_names']:
            return set(c['pathway_names'])
    return None


# ---------------------------------------------------------------------------
# Core: MCL splits the two dense groups; connected-components merges them
# ---------------------------------------------------------------------------

def test_connected_components_merges_via_bridge():
    pathways, genes = _two_groups_with_bridge()
    svc = PathwayClusteringService(method='connected_components', gene_source='de',
                                   jaccard_threshold=0.25, min_cluster_size=2)
    result = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    # The bridge chains everything into a single connected component.
    assert result['metadata']['cluster_count'] == 1
    merged = _cluster_of(result, 'A1')
    assert {'A1', 'B1'} <= merged  # A and B live in the SAME cluster


def test_mcl_splits_two_dense_groups():
    pathways, genes = _two_groups_with_bridge()
    # aPEAR defaults: full Jaccard matrix (floor 0.0) + inflation 2.5 + loop_value=0.
    svc = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5, min_cluster_size=2)
    result = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    a_cluster = _cluster_of(result, 'A1')
    b_cluster = _cluster_of(result, 'B1')
    # The two dense groups end up in DIFFERENT clusters (no chaining).
    assert a_cluster is not None and b_cluster is not None
    assert a_cluster != b_cluster
    assert {'A1', 'A2', 'A3'} <= a_cluster
    assert {'B1', 'B2', 'B3'} <= b_cluster
    # A and B are never co-clustered.
    assert 'B1' not in a_cluster and 'A1' not in b_cluster


# ---------------------------------------------------------------------------
# Backward-compatible output shape
# ---------------------------------------------------------------------------

_CLUSTER_KEYS = {
    'pathways', 'pathway_names', 'pathway_count', 'shared_genes', 'shared_gene_count',
    'avg_jaccard_overlap', 'avg_p_value_fdr', 'significance', 'key_genes_with_fc',
    'dominant_direction', 'max_abs_nes',
}
_PATHWAY_KEYS = {'name', 'p_value', 'p_value_fdr', 'database', 'gene_count', 'abs_nes'}
_SINGLETON_KEYS = {'name', 'p_value', 'p_value_fdr', 'database', 'gene_count'}


def test_mcl_output_shape_is_backward_compatible():
    pathways, genes = _two_groups_with_bridge()
    svc = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5)
    result = svc.cluster_pathways_by_gene_overlap(pathways, genes)

    assert set(result.keys()) >= {'clusters', 'singletons', 'metadata'}
    for c in result['clusters']:
        assert _CLUSTER_KEYS <= set(c.keys())
        assert 'representative' in c  # additive
        assert 0.0 <= c['avg_jaccard_overlap'] <= 1.0  # fraction, ×100'd downstream
        for p in c['pathways']:
            assert _PATHWAY_KEYS <= set(p.keys())
    for s in result['singletons']:
        assert _SINGLETON_KEYS <= set(s.keys())

    md = result['metadata']
    assert md['clustering_method'] == 'mcl'
    assert md['inflation'] == 2.5
    assert md['gene_source'] == 'de'
    assert md['similarity_floor'] == 0.0


def test_mcl_metadata_records_provenance():
    pathways, genes = _two_groups_with_bridge()
    svc = PathwayClusteringService(method='mcl', inflation=3.0, similarity_floor=0.2)
    md = svc.cluster_pathways_by_gene_overlap(pathways, genes)['metadata']
    assert md['clustering_method'] == 'mcl'
    assert md['inflation'] == 3.0
    assert md['similarity_floor'] == 0.2


# ---------------------------------------------------------------------------
# Determinism (MCL has no RNG)
# ---------------------------------------------------------------------------

def test_mcl_is_deterministic():
    pathways, genes = _two_groups_with_bridge()
    svc = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5)
    r1 = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    r2 = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    assign1 = sorted(sorted(c['pathway_names']) for c in r1['clusters'])
    assign2 = sorted(sorted(c['pathway_names']) for c in r2['clusters'])
    assert assign1 == assign2


# ---------------------------------------------------------------------------
# Representative
# ---------------------------------------------------------------------------

def test_representative_is_a_member_and_most_significant():
    pathways, genes = _two_groups_with_bridge()
    svc = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5)
    result = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    for c in result['clusters']:
        assert c['representative'] in c['pathway_names']


def test_select_representative_prefers_lowest_fdr_then_nes():
    pathways = [
        {'name': 'weak', 'p_value_fdr': 0.04, 'NES': 1.0},
        {'name': 'strong', 'p_value_fdr': 1e-5, 'NES': 1.2},
        {'name': 'mid', 'p_value_fdr': 0.001, 'NES': 3.0},
    ]
    assert _select_representative(pathways) == 'strong'
    # tie on FDR -> higher |NES| wins
    tie = [
        {'name': 'a', 'p_value_fdr': 0.001, 'NES': 1.1},
        {'name': 'b', 'p_value_fdr': 0.001, 'NES': 2.5},
    ]
    assert _select_representative(tie) == 'b'
    assert _select_representative([]) is None


# ---------------------------------------------------------------------------
# Module helpers on tiny known inputs / edge cases
# ---------------------------------------------------------------------------

def test_build_weighted_adjacency_floor_and_symmetry():
    a = {'X', 'Y', 'Z'}           # vs b: |∩|=2,|∪|=4 -> 0.5
    b = {'X', 'Y', 'W'}
    c = {'M', 'N'}                # disjoint from a/b -> 0.0
    adj = _build_weighted_adjacency([a, b, c], similarity_floor=0.1)
    # Diagonal carries 1.0 self-similarity (aPEAR full Jaccard matrix; supplies MCL self-loops
    # so run_mcl can use loop_value=0 / addLoops=FALSE without overwriting it).
    assert adj[0][0] == adj[1][1] == adj[2][2] == 1.0
    assert adj[0][1] == adj[1][0] == pytest.approx(0.5)
    assert adj[0][2] == 0.0 and adj[1][2] == 0.0  # below floor -> pruned (off-diagonal)

    # A high floor prunes the 0.5 edge too (diagonal self-similarity is unaffected).
    adj2 = _build_weighted_adjacency([a, b, c], similarity_floor=0.6)
    assert adj2[0][1] == 0.0
    assert adj2[0][0] == 1.0

    # aPEAR default floor 0.0 keeps every Jaccard>0 edge (full matrix); disjoint pairs stay 0.0.
    adj3 = _build_weighted_adjacency([a, b, c], similarity_floor=0.0)
    assert adj3[0][1] == pytest.approx(0.5)
    assert adj3[0][2] == 0.0 and adj3[1][2] == 0.0
    assert adj3[0][0] == adj3[1][1] == adj3[2][2] == 1.0


def test_mcl_partition_edge_cases():
    # aPEAR defaults: floor 0.0, inflation 2.5 (loop_value=0 inside _mcl_partition_from_adjacency).
    # empty
    assert _mcl_partition([], 0.0, 2.5) == []
    # single node
    assert _mcl_partition([{'A'}], 0.0, 2.5) == [[0]]
    # two disconnected nodes -> two singleton clusters (loop_value=0 + 1.0 diagonal: no crash)
    part = _mcl_partition([{'A'}, {'B'}], 0.0, 2.5)
    assert sorted(sorted(g) for g in part) == [[0], [1]]
    # a node with an empty gene set is isolated, never crashes, forms its own singleton
    part2 = _mcl_partition([{'A', 'B'}, {'A', 'B'}, set()], 0.0, 2.5)
    flat = sorted(i for g in part2 for i in g)
    assert flat == [0, 1, 2]           # every node accounted for exactly once
    assert [2] in [sorted(g) for g in part2]  # the empty-set node is its own cluster


def test_mcl_partition_is_a_strict_partition():
    # Every node appears in exactly one cluster (no overlap, none dropped).
    pathways, _ = _two_groups_with_bridge()
    gene_sets = [set(g.upper() for g in p['genes']) for p in pathways]
    part = _mcl_partition(gene_sets, 0.0, 2.5)
    flat = [i for g in part for i in g]
    assert sorted(flat) == list(range(len(gene_sets)))
    assert len(flat) == len(set(flat))  # no node in two clusters


def test_jaccard_basic():
    assert _jaccard({'A', 'B'}, {'A', 'B'}) == 1.0
    assert _jaccard({'A', 'B'}, {'C'}) == 0.0
    assert _jaccard(set(), set()) == 0.0
    assert _jaccard({'A', 'B'}, {'B', 'C'}) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# gene_source knob
# ---------------------------------------------------------------------------

def test_gene_source_full_uses_full_gene_sets_for_similarity():
    # Two pathways whose FULL sets overlap but whose DE-restricted sets do not.
    pathways = [
        {'name': 'P1', 'genes': ['G1', 'G2', 'DE1'], 'p_value_fdr': 1e-3, 'NES': 2.0},
        {'name': 'P2', 'genes': ['G1', 'G2', 'DE2'], 'p_value_fdr': 1e-3, 'NES': 2.0},
    ]
    genes = [{'geneSymbol': 'DE1', 'foldChange': 1.5},
             {'geneSymbol': 'DE2', 'foldChange': 1.5}]
    # gene_source='de': de sets are {DE1} and {DE2} -> Jaccard 0 -> no cluster (singletons).
    de_svc = PathwayClusteringService(method='mcl', gene_source='de',
                                      similarity_floor=0.0, inflation=2.5, min_cluster_size=2)
    de_res = de_svc.cluster_pathways_by_gene_overlap(pathways, genes)
    assert de_res['metadata']['cluster_count'] == 0
    # gene_source='full': full sets share G1,G2 -> Jaccard 0.5 -> one cluster of both.
    full_svc = PathwayClusteringService(method='mcl', gene_source='full',
                                        similarity_floor=0.0, inflation=2.5, min_cluster_size=2)
    full_res = full_svc.cluster_pathways_by_gene_overlap(pathways, genes)
    assert full_res['metadata']['cluster_count'] == 1
    assert full_res['metadata']['gene_source'] == 'full'


# ---------------------------------------------------------------------------
# aPEAR-default fidelity: full Jaccard matrix + inflation 2.5 + loop_value=0
# ---------------------------------------------------------------------------

def test_apear_defaults_separate_groups_without_fragmenting():
    """With aPEAR's defaults (floor 0.0 = full matrix, inflation 2.5, loop_value=0 via the
    1.0 self-similarity diagonal), MCL must still resolve the two dense communities as whole
    clusters — NOT shatter them into overlapping singletons (the failure mode when loop_value=0
    meets a 0.0 diagonal). This guards the diagonal-self-similarity fix."""
    pathways, genes = _two_groups_with_bridge()
    svc = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5, min_cluster_size=2)
    result = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    a_cluster = _cluster_of(result, 'A1')
    b_cluster = _cluster_of(result, 'B1')
    assert a_cluster is not None and b_cluster is not None
    assert a_cluster != b_cluster
    # Each dense group stays intact within a single cluster (no fragmentation).
    assert {'A1', 'A2', 'A3'} <= a_cluster
    assert {'B1', 'B2', 'B3'} <= b_cluster
    # Strict partition: every pathway appears in exactly one cluster or singleton.
    names_in_clusters = [n for c in result['clusters'] for n in c['pathway_names']]
    names_in_singletons = [s['name'] for s in result['singletons']]
    all_names = names_in_clusters + names_in_singletons
    assert sorted(all_names) == sorted(p['name'] for p in pathways)
    assert len(all_names) == len(set(all_names))  # no pathway in two places


def test_mcl_isolated_pathway_is_singleton_no_crash():
    """A pathway whose DE genes are disjoint from all others (an isolated node) must not crash
    MCL under loop_value=0 + full matrix, and must surface as a singleton — never dropped."""
    pathways = [
        {'name': 'A1', 'genes': _GA, 'p_value_fdr': 1e-4, 'NES': 2.1, 'database': 'KEGG'},
        {'name': 'A2', 'genes': _GA[:5] + ['G7'], 'p_value_fdr': 2e-4, 'NES': 1.9, 'database': 'KEGG'},
        {'name': 'LONER', 'genes': ['Z1', 'Z2', 'Z3'], 'p_value_fdr': 1e-3, 'NES': 1.5, 'database': 'KEGG'},
    ]
    all_symbols = set(_GA + ['G7', 'Z1', 'Z2', 'Z3'])
    genes = [{'geneSymbol': g, 'foldChange': 1.5} for g in sorted(all_symbols)]
    svc = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5, min_cluster_size=2)
    result = svc.cluster_pathways_by_gene_overlap(pathways, genes)
    # LONER is disjoint -> its own singleton; A1/A2 cluster together.
    assert 'LONER' in {s['name'] for s in result['singletons']}
    a_cluster = _cluster_of(result, 'A1')
    assert a_cluster is not None and {'A1', 'A2'} <= a_cluster and 'LONER' not in a_cluster


def test_apear_defaults_are_wired():
    """Guard the headline change: the aPEAR defaults must be the ACTUAL defaults, so a silent
    revert of the constructor defaults or the config values is caught even though every other
    test passes its params explicitly."""
    from src.config import settings

    # Config defaults mirror aPEAR (inflation 2.5, no threshold / full matrix).
    assert settings.CLUSTER_INFLATION == 2.5
    assert settings.CLUSTER_SIMILARITY_FLOOR == 0.0
    assert settings.CLUSTER_MIN_SIZE == 2
    assert settings.CLUSTER_GENE_SOURCE == 'de'

    # A bare MCL service carries the same aPEAR defaults.
    svc = PathwayClusteringService(method='mcl')
    assert svc.inflation == 2.5
    assert svc.similarity_floor == 0.0
    assert svc.min_cluster_size == 2
    assert svc.gene_source == 'de'
