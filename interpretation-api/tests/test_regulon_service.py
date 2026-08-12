"""
Tests for the TF->target regulon enrichment service (report bug 7).

Covers the pure statistics (hypergeometric survival, BH-FDR), sign-aware activity
inference, overlap/FDR gating that drops off-program TFs, ranking, and graceful
degradation when the regulon file is missing — which is the default, since the
CollecTRI network is fetched by the user rather than shipped with the repo.
"""

from math import comb

import pytest

from src.pipeline.services.regulon_service import (
    RegulonService,
    _DEFAULT_PATH,
    hypergeom_sf,
    bh_fdr,
    _activity_from_net,
)


# ---------------------------------------------------------------------------
# hypergeometric survival function (vs exact math.comb)
# ---------------------------------------------------------------------------

def _exact_sf(k, N, K, n):
    lo, hi = max(k, 0), min(K, n)
    return sum(comb(K, i) * comb(N - K, n - i) for i in range(lo, hi + 1)) / comb(N, n)


def test_hypergeom_matches_exact_bruteforce():
    for (k, N, K, n) in [(3, 8, 3, 3), (2, 20, 5, 6), (1, 50, 10, 7), (4, 100, 20, 15)]:
        assert hypergeom_sf(k, N, K, n) == pytest.approx(_exact_sf(k, N, K, n), rel=1e-9)


def test_hypergeom_edge_cases():
    assert hypergeom_sf(0, 20, 5, 6) == 1.0            # P(X >= 0) is certain
    assert hypergeom_sf(6, 20, 5, 6) == 0.0            # k > min(K, n) impossible
    assert hypergeom_sf(3, 0, 0, 0) == 0.0             # degenerate universe
    assert hypergeom_sf(1, 5, 3, 10) == 0.0            # n > N -> denom C(N,n) undefined -> 0


def test_hypergeom_monotonic_nonincreasing_in_k():
    vals = [hypergeom_sf(k, 100, 20, 15) for k in range(0, 8)]
    assert all(a >= b for a, b in zip(vals, vals[1:]))


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------

def test_bh_fdr_known_vector():
    q = bh_fdr([0.001, 0.01, 0.02, 0.5])
    # q_i = p_i * m / rank, enforced monotone from the top.
    assert q[0] == pytest.approx(0.004, rel=1e-9)
    assert q[1] == pytest.approx(0.02, rel=1e-9)
    assert q[2] == pytest.approx(0.02666667, rel=1e-6)
    assert q[3] == pytest.approx(0.5, rel=1e-9)
    assert all(0.0 <= x <= 1.0 for x in q)


def test_bh_fdr_empty():
    assert bh_fdr([]) == []


def test_bh_fdr_handles_ties_monotonically():
    q = bh_fdr([0.02, 0.02, 0.001, 0.04])
    # tied p-values collapse to the same (smallest) q via top-down cummin.
    assert q[0] == q[1]
    assert all(0.0 <= x <= 1.0 for x in q)
    assert q[2] < q[0]                        # a strictly smaller p gets a strictly smaller q
    assert q[2] == pytest.approx(0.004, rel=1e-9)


# ---------------------------------------------------------------------------
# sign-aware activity inference
# ---------------------------------------------------------------------------

def test_activity_from_net():
    assert _activity_from_net(-3, 3) == 'decreased'   # activators of down targets
    assert _activity_from_net(2, 2) == 'increased'     # repressors of down targets
    assert _activity_from_net(0, 4) == 'mixed'         # signed but balanced
    assert _activity_from_net(0, 0) == 'unknown'       # unsigned regulon


# ---------------------------------------------------------------------------
# RegulonService.rank_tfs
# ---------------------------------------------------------------------------

def _svc():
    # HNF4A activates 4 metabolic genes (all in the down set); FILLER genes pad the
    # universe; DECOY regulates only up-regulated genes.
    edges = [
        ('HNF4A', 'ACADS', 1), ('HNF4A', 'ALDH2', 1),
        ('HNF4A', 'CPT1A', 1), ('HNF4A', 'APOB', 1),
        ('DECOY', 'UPGENE1', 1), ('DECOY', 'UPGENE2', 1), ('DECOY', 'UPGENE3', 1),
        ('WEAK', 'ACADS', 1), ('WEAK', 'FILLER1', 1),   # only 1 down target -> below min_overlap
        ('HNF4A', 'FILLER2', 1), ('HNF4A', 'FILLER3', 1),
    ]
    return RegulonService.from_edges(edges)


DOWN = ['ACADS', 'ALDH2', 'CPT1A', 'APOB']

# Padding edges enlarge the background universe (none of these genes are in DOWN) so a
# genuine overlap is statistically significant rather than swamped by a tiny universe.
PAD = [('PAD', f'F{i}', 0) for i in range(40)]


def test_available_and_universe():
    svc = _svc()
    assert svc.available
    assert 'HNF4A' in svc.universe() and 'ACADS' in svc.universe()


def test_surfaces_master_tf_with_activity():
    hits = _svc().rank_tfs(DOWN, min_overlap=3, top_n=8)
    by_tf = {h['tf']: h for h in hits}
    assert 'HNF4A' in by_tf
    h = by_tf['HNF4A']
    assert h['overlap_count'] == 4
    assert set(h['targets']) == set(DOWN)
    assert h['inferred_tf_activity'] == 'decreased'   # activator of down targets
    assert h['enrichment_fdr'] <= 0.10


def test_off_program_tf_dropped_by_gate():
    # DECOY regulates only up genes -> zero overlap with the down set -> absent.
    # WEAK overlaps a single down gene -> below min_overlap -> absent.
    hits = _svc().rank_tfs(DOWN, min_overlap=3, top_n=8)
    tfs = {h['tf'] for h in hits}
    assert 'DECOY' not in tfs
    assert 'WEAK' not in tfs


def test_fdr_gate_excludes_marginal_tf_that_cleared_min_overlap():
    """The statistical gate (not just overlap/min_overlap) must be able to exclude a TF.
    MARGINAL clears min_overlap (3 down targets) but, over a large universe, its huge
    regulon makes the overlap unremarkable -> BH q-value fails the 0.10 gate. STRONG (a
    tight regulon = the down set) stays significant."""
    down = [f'D{i}' for i in range(5)]
    edges = []
    edges += [('STRONG', g, 1) for g in down]                      # tight, all-down regulon
    edges += [('MARGINAL', g, 1) for g in down[:3]]                # only 3 down targets...
    edges += [('MARGINAL', f'F{i}', 1) for i in range(197)]        # ...in a 200-gene regulon
    edges += [('PAD', f'G{i}', 0) for i in range(100)]             # enlarge the universe (~305)
    svc = RegulonService.from_edges(edges)

    strict = svc.rank_tfs(down, min_overlap=3, fdr_threshold=0.10, top_n=8)
    strict_tfs = {h['tf'] for h in strict}
    assert 'STRONG' in strict_tfs
    assert 'MARGINAL' not in strict_tfs               # dropped by the FDR gate, not overlap

    # Prove MARGINAL actually cleared min_overlap (was tested, then gated): with the gate
    # wide open it reappears with overlap >= 3.
    loose = {h['tf']: h for h in svc.rank_tfs(down, min_overlap=3, fdr_threshold=1.0, top_n=8)}
    assert 'MARGINAL' in loose and loose['MARGINAL']['overlap_count'] == 3
    assert loose['MARGINAL']['enrichment_fdr'] > 0.10


def test_repressor_of_down_targets_is_increased():
    edges = [('REP', g, -1) for g in DOWN] + [('REP', 'FILLER', -1)] + PAD
    hits = RegulonService.from_edges(edges).rank_tfs(DOWN, min_overlap=3)
    assert hits and hits[0]['tf'] == 'REP'
    assert hits[0]['inferred_tf_activity'] == 'increased'


def test_unsigned_regulon_activity_unknown():
    edges = [('T', g, 0) for g in DOWN] + PAD
    hits = RegulonService.from_edges(edges).rank_tfs(DOWN, min_overlap=3)
    assert hits and hits[0]['inferred_tf_activity'] == 'unknown'


def test_no_down_genes_returns_empty():
    assert _svc().rank_tfs([], min_overlap=3) == []


def test_rank_respects_top_n_and_is_deterministic():
    # Two equally-strong TFs over the same targets + padding; top_n truncates, order stable.
    edges = ([('TFA', g, 1) for g in DOWN] + [('TFB', g, 1) for g in DOWN] + PAD)
    svc = RegulonService.from_edges(edges)
    hits = svc.rank_tfs(DOWN, min_overlap=3, top_n=1)
    assert len(hits) == 1
    # deterministic tie-break by TF symbol (fdr and overlap identical) -> TFA first.
    assert hits[0]['tf'] == 'TFA'
    assert svc.rank_tfs(DOWN, min_overlap=3, top_n=1) == hits   # repeatable


def test_undersized_background_is_clamped_not_spuriously_significant():
    # An explicit background smaller than the regulon universe must be clamped up, so the
    # p-value isn't spuriously driven to 0.
    svc = _svc()
    big = svc.rank_tfs(DOWN, min_overlap=3, background_n=len(svc.universe()))
    clamped = svc.rank_tfs(DOWN, min_overlap=3, background_n=1)
    assert clamped and big
    assert clamped[0]['p_value'] == big[0]['p_value']


def test_larger_background_is_honored_and_sharpens_pvalue():
    # A background LARGER than the regulon universe is used as-is: the same overlap is rarer
    # by chance in a bigger universe, so the p-value gets smaller (more significant).
    svc = _svc()
    default = {h['tf']: h for h in svc.rank_tfs(DOWN, min_overlap=3)}
    large = {h['tf']: h for h in svc.rank_tfs(DOWN, min_overlap=3, background_n=100_000)}
    assert 'HNF4A' in default and 'HNF4A' in large
    assert large['HNF4A']['p_value'] < default['HNF4A']['p_value']


# ---------------------------------------------------------------------------
# graceful degradation
# ---------------------------------------------------------------------------

def test_missing_file_is_unavailable_not_error():
    svc = RegulonService(path='/nonexistent/regulon.tsv')
    assert svc.available is False
    assert svc.rank_tfs(DOWN) == []          # no raise


@pytest.mark.skipif(
    not _DEFAULT_PATH.exists(),
    reason="CollecTRI regulon not fetched; see the README for how to install it",
)
def test_fetched_regulon_loads_and_contains_master_tfs():
    """If the user has fetched the CollecTRI network, it must load and know the
    hepatocyte master TFs — i.e. this doubles as a check that the fetch produced a
    correctly-shaped file at the path the loader expects."""
    svc = RegulonService()
    assert svc.available
    uni = svc.universe()
    for tf in ('HNF4A', 'PPARA', 'CEBPA'):
        assert tf in uni


# ---------------------------------------------------------------------------
# file loader header handling
# ---------------------------------------------------------------------------

def _write(tmp_path, text):
    p = tmp_path / 'reg.tsv'
    p.write_text(text)
    return str(p)


def test_loader_skips_named_header(tmp_path):
    svc = RegulonService(path=_write(
        tmp_path, 'source\ttarget\tweight\nHNF4A\tACADS\t1\nHNF4A\tALDH2\t-1\n'))
    assert svc.available
    assert svc.universe() == {'HNF4A', 'ACADS', 'ALDH2'}   # header row not ingested


def test_loader_keeps_headerless_first_row_with_nonnumeric_weight(tmp_path):
    # First data row has weight 'NA' -> must be ingested (not mistaken for a header).
    svc = RegulonService(path=_write(
        tmp_path, 'HNF4A\tACADS\tNA\nHNF4A\tALDH2\t1\n'))
    assert svc.available
    assert 'ACADS' in svc.universe() and 'ALDH2' in svc.universe()


def test_loader_two_column_header_not_ingested_as_edge(tmp_path):
    svc = RegulonService(path=_write(tmp_path, 'source\ttarget\nHNF4A\tACADS\n'))
    assert svc.universe() == {'HNF4A', 'ACADS'}
