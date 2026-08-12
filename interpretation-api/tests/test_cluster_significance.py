"""
Tests for report bug 3: the cluster "Significance: HIGH/MEDIUM" label was miscalibrated.

Two distinct defects have surfaced here:
  1. (batch2) Step 1 displayed the LLM's free-text significance instead of the computed
     score — fixed by always using the computed score keyed to FDR + NES magnitude.
  2. (this batch) The computed score is correct at cluster time, but Phase-3 tissue
     filtering strips pathways from a theme and recomputes ``avg_p_value_fdr`` WITHOUT
     recomputing ``significance``. So a multi-pathway cluster labelled MEDIUM that is
     filtered down to just its single strongest pathway (e.g. Cell cycle, FDR 3.76e-04)
     keeps the stale MEDIUM label while its shown FDR says HIGH. The fix recomputes the
     significance from the surviving pathways.
"""

import json

from src.pipeline.services.pathway_clustering_service import (
    PathwayClusteringService,
    _pathway_abs_nes,
)
from src.pipeline.steps.step01_pathway_themes import Step01PathwayThemes


def _step_with_svc():
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    step.clustering_service = PathwayClusteringService()
    return step


def _svc():
    return PathwayClusteringService()


# ---------------------------------------------------------------------------
# computed significance formula (FDR base + NES promotion)
# ---------------------------------------------------------------------------

def test_headline_low_fdr_cluster_is_high():
    # Cell-cycle theme: FDR ~4e-4 → high, regardless of pathway count.
    assert _svc()._classify_cluster_significance(3.76e-4, 1, 1.77) == 'high'


def test_weak_borderline_cluster_is_not_high():
    # OXPHOS/Thermogenesis theme: avg FDR ~0.01, weak NES → medium (was wrongly HIGH).
    assert _svc()._classify_cluster_significance(0.0104, 2, 1.05) == 'medium'


def test_strong_nes_promotes_one_tier():
    # Marginal FDR but strong effect size (|NES| >= 2) → promoted.
    assert _svc()._classify_cluster_significance(0.02, 1, 2.6) == 'high'
    assert _svc()._classify_cluster_significance(0.06, 3, 2.4) == 'medium'


def test_nes_never_demotes_significant_cluster():
    # Very significant FDR with a weak NES stays high (NES only promotes).
    assert _svc()._classify_cluster_significance(3.76e-4, 1, 1.0) == 'high'


def test_no_nes_falls_back_to_fdr_only():
    assert _svc()._classify_cluster_significance(3.76e-4, 1, None) == 'high'


def test_pathway_abs_nes_prefers_nes_and_promotes_mislabeled_es():
    assert _pathway_abs_nes({'NES': -3.52}) == 3.52
    assert _pathway_abs_nes({'ES': -2.1}) == 2.1          # |ES| > 1 can only be an NES
    assert _pathway_abs_nes({'ES': 0.4}) is None          # classic ES, not comparable to NES scale


# ---------------------------------------------------------------------------
# Step 1 must not let the LLM override the computed significance
# ---------------------------------------------------------------------------

def _cluster(sig):
    return {
        'pathways': [{'name': 'Cell cycle'}],
        'pathway_names': ['Cell cycle'],
        'pathway_count': 1,
        'shared_genes': ['CDK1'],
        'shared_gene_count': 1,
        'avg_jaccard_overlap': 0.3,
        'avg_p_value_fdr': 3.76e-4,
        'significance': sig,          # computed
        'key_genes_with_fc': [],
        'dominant_direction': 'up',
    }


def test_merge_uses_computed_significance_not_llm():
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    clusters = [_cluster('high')]
    llm_themes = [{'cluster_number': 1, 'name': 'Cell cycle control',
                   'significance': 'medium'}]  # LLM disagrees
    themes = step._merge_clusters_with_names(clusters, llm_themes)
    assert themes[0]['significance'] == 'high'


# ---------------------------------------------------------------------------
# Significance must be recomputed after tissue-filtering removes pathways
# ---------------------------------------------------------------------------

def test_cluster_pathway_dicts_carry_abs_nes():
    # The per-pathway dict must preserve |NES| so significance can be honestly
    # recomputed on a filtered subset downstream.
    svc = PathwayClusteringService()
    cluster = svc._enrich_cluster_with_stats(
        [{'name': 'Cell cycle', 'de_genes': ['CDK1', 'CCNB1'],
          'p_value_fdr': 3.76e-4, 'NES': 1.77}], [])
    assert cluster.pathways[0]['abs_nes'] == 1.77


def test_recompute_theme_significance_from_surviving_pathways():
    step = _step_with_svc()
    theme = {
        'significance': 'medium',          # stale value from the pre-filter cluster
        'avg_p_value_fdr': 0.02,
        'pathways': [{'name': 'Cell cycle', 'p_value_fdr': 3.76e-4, 'abs_nes': 1.77}],
    }
    step._recompute_theme_significance(theme)
    assert theme['significance'] == 'high'
    assert theme['avg_p_value_fdr'] == 3.76e-4
    assert theme['max_abs_nes'] == 1.77


def test_recompute_averages_fdr_and_takes_max_nes_over_multiple_survivors():
    # Multiple survivors with distinct FDR/NES: avg FDR must be the true mean, max_abs_nes
    # the max, and a strong |NES|>=2 in ONE survivor promotes a medium base to high.
    step = _step_with_svc()
    theme = {
        'significance': 'low',
        'pathways': [
            {'name': 'A', 'p_value_fdr': 0.015, 'abs_nes': 1.0},
            {'name': 'B', 'p_value_fdr': 0.025, 'abs_nes': 2.5},
        ],
    }
    step._recompute_theme_significance(theme)
    assert theme['avg_p_value_fdr'] == 0.02        # true mean of 0.015 and 0.025
    assert theme['max_abs_nes'] == 2.5             # max, not last/first
    assert theme['significance'] == 'high'         # medium base (avg 0.02) promoted by |NES|>=2


def test_tissue_filter_recomputes_significance_after_dropping_pathways(monkeypatch):
    """The real report bug: a MEDIUM cluster filtered down to its strong Cell-cycle
    pathway (FDR 3.76e-04) must be relabelled HIGH, not keep the stale MEDIUM."""
    step = _step_with_svc()
    theme = {
        'name': 'Cell cycle progression',
        'significance': 'medium',          # cluster-level, dragged down by weak members
        'avg_p_value_fdr': 0.02,
        'pathway_count': 3,
        'pathways': [
            {'name': 'Cell cycle', 'p_value_fdr': 3.76e-4, 'gene_count': 42, 'abs_nes': 1.77},
            {'name': 'Oocyte meiosis', 'p_value_fdr': 0.03, 'gene_count': 5, 'abs_nes': 1.1},
            {'name': 'Progesterone-mediated oocyte maturation',
             'p_value_fdr': 0.04, 'gene_count': 4, 'abs_nes': 1.0},
        ],
    }

    def fake_chat_json(messages):
        # Keep the general Cell cycle pathway; filter the two off-tissue oocyte maps.
        return json.dumps({'filtered_pathways': [
            {'pathway_name': 'Cell cycle', 'decision': 'KEEP'},
            {'pathway_name': 'Oocyte meiosis', 'decision': 'FILTER'},
            {'pathway_name': 'Progesterone-mediated oocyte maturation', 'decision': 'FILTER'},
        ]})

    monkeypatch.setattr(step, '_chat_json', fake_chat_json)
    cleaned, filtered = step._filter_off_tissue_pathways(
        [theme], {'tissue': 'liver', 'disease': 'Hepatocellular carcinoma'})

    assert filtered == 2
    assert len(cleaned) == 1
    assert cleaned[0]['pathway_count'] == 1
    assert cleaned[0]['avg_p_value_fdr'] == 3.76e-4
    assert cleaned[0]['significance'] == 'high'   # was stale 'medium' before the fix


def test_semantic_clustering_uses_same_computed_classifier(monkeypatch):
    """The no-DE-genes semantic path must use the SAME classifier (and carry abs_nes), so a
    theme's label doesn't shift if it later passes through the tissue-filter recompute."""
    import src.agents.llm_client as llm_mod

    class _FakeLLM:
        def __init__(self, *a, **k):
            pass

        def complete(self, *a, **k):
            return json.dumps({'clusters': [
                {'theme': 'X', 'pathway_ids': [0, 1], 'rationale': 'r'}], 'singletons': []})

    monkeypatch.setattr(llm_mod, 'UnifiedLLMClient', _FakeLLM)
    svc = PathwayClusteringService()
    pathways = [{'name': 'P1', 'p_value_fdr': 0.006}, {'name': 'P2', 'p_value_fdr': 0.006}]
    cluster = svc.cluster_pathways_by_semantic_similarity(pathways)['clusters'][0]

    # avg FDR 0.006 with 2 pathways, no NES -> computed classifier gives 'medium'
    # (the OLD inline rule wrongly gave 'high' for anything < 0.01).
    assert cluster['significance'] == 'medium'
    assert 'abs_nes' in cluster['pathways'][0]


# ---------------------------------------------------------------------------
# 0.0-safe parsing (no `or`-chain dropping real zeros)
# ---------------------------------------------------------------------------

def test_zero_fdr_is_preserved_not_treated_as_one():
    # A real FDR of 0.0 must classify as high, not fall through an `or`-chain to 1.0→low.
    svc = _svc()
    from src.pipeline.services.pathway_clustering_service import PathwayCluster
    cluster = svc._enrich_cluster_with_stats(
        [{'name': 'P1', 'de_genes': ['A', 'B'], 'p_value_fdr': 0.0}], [])
    assert cluster.avg_p_value_fdr == 0.0
    assert cluster.significance == 'high'


def test_zero_fold_change_preserved_in_direction_calc():
    # A gene with foldChange 0.0 must not be mis-read as its log2 field via `or`-chaining.
    svc = _svc()
    genes = [{'geneSymbol': 'X', 'foldChange': 0.0, 'log2_fold_change': -2.0},
             {'geneSymbol': 'Y', 'foldChange': -1.5}]
    gwf, direction = svc._calculate_gene_fold_changes(['X', 'Y'], genes)
    x = next(g for g in gwf if g['gene'] == 'X')
    assert x['fold_change'] == 0.0
    assert x['direction'] == 'neutral'
    assert direction == 'down'  # only Y counts
