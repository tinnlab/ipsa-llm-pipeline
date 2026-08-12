"""
Fix 4 — directional master-regulator sets.

Master regulators must not collapse to a single (up-regulated proliferation)
direction: down-regulated / metabolic hubs sitting on a strong |NES| axis have to
be surfaced too. These tests exercise the pure selection/record helpers, which
require no LLM.
"""

from src.pipeline.steps.step02_hub_genes import Step02HubGenes
from src.pipeline.services.network_service import HubGene


def _bare_step02():
    return Step02HubGenes.__new__(Step02HubGenes)


def _hub(gene, fc, score, degree=30):
    return HubGene(
        gene=gene, fold_change=fc, p_value=1e-5,
        degree=degree, betweenness=0.1, closeness=0.5, hub_score=score,
        degree_rank=score, betweenness_rank=score, closeness_rank=score,
    )


# network_hubs arrive sorted by hub_score desc (network_service guarantees this).
def _mixed_hubs():
    return [
        _hub('CDK1', +2.4, 0.95),   # up
        _hub('CCNB1', +3.1, 0.90),  # up
        _hub('MKI67', +1.8, 0.88),  # up
        _hub('UBE2T', +1.7, 0.85),  # up
        _hub('ACADS', -1.4, 0.70),  # down (metabolic)
        _hub('ALDH2', -1.4, 0.65),  # down (metabolic)
    ]


def test_select_returns_both_directions():
    step = _bare_step02()
    selected = step._select_master_regulator_hubs(_mixed_hubs(), k_per_direction=3)
    genes = {h.gene for h in selected}
    assert any(h.fold_change > 0 for h in selected), 'no up-regulated hub selected'
    assert any(h.fold_change < 0 for h in selected), 'no down-regulated hub selected'
    # The down-regulated metabolic hubs must be present despite lower hub_score.
    assert 'ACADS' in genes and 'ALDH2' in genes


def test_identify_master_regulators_tags_direction():
    step = _bare_step02()
    selected = step._select_master_regulator_hubs(_mixed_hubs(), k_per_direction=3)
    regs = step._identify_master_regulators(selected, interpretation={})
    ups = [r for r in regs if r['direction'] == 'up']
    downs = [r for r in regs if r['direction'] == 'down']
    assert ups and downs
    for r in regs:
        assert r['direction'] in ('up', 'down')
        # direction must agree with the sign of the fold change
        assert (r['direction'] == 'up') == (r['fold_change'] > 0)


def test_single_direction_backfills_to_min_total():
    """All-up dataset: keep the previous coverage (>= min_total) via backfill."""
    step = _bare_step02()
    up_only = [_hub(f'G{i}', +2.0 - i * 0.1, 0.9 - i * 0.05) for i in range(6)]
    selected = step._select_master_regulator_hubs(up_only, k_per_direction=3, min_total=5)
    assert len(selected) == 5
    assert all(h.fold_change > 0 for h in selected)


def test_no_duplicate_genes_in_selection():
    step = _bare_step02()
    selected = step._select_master_regulator_hubs(_mixed_hubs(), k_per_direction=3)
    genes = [h.gene for h in selected]
    assert len(genes) == len(set(genes))


def test_zero_fold_change_hub_tagged_neutral_not_down():
    """A backfilled exact-zero-FC hub must be tagged 'neutral' (via fc_direction),
    never silently mislabeled 'down'."""
    step = _bare_step02()
    hubs = [_hub('CDK1', +2.4, 0.9), _hub('ZERO', 0.0, 0.8)]
    selected = step._select_master_regulator_hubs(hubs, k_per_direction=3, min_total=2)
    regs = step._identify_master_regulators(selected, interpretation={})
    zero = [r for r in regs if r['gene'] == 'ZERO']
    assert zero and zero[0]['direction'] == 'neutral'
    # a zero-FC hub is not counted as an up or down directional hub
    up = step._select_master_regulator_hubs(hubs, k_per_direction=3, min_total=1)
    assert all(h.gene != 'ZERO' for h in up[:1])  # top pick is the real up hub
