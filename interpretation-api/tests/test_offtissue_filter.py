"""
Tests for tissue-specificity filtering (Group E).

Off-tissue / off-disease pathways (e.g. Bladder cancer, Small cell lung cancer,
Salmonella infection in a liver study) must be filtered — including ungrouped
singletons, which previously bypassed the theme-level filter.
"""

import json
from unittest.mock import MagicMock

from src.pipeline.services.pathway_clustering_service import PathwayClusteringService
from src.pipeline.steps.step01_pathway_themes import Step01PathwayThemes


def _bare_step01():
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    step.llm = MagicMock()
    # Tissue-filtering recomputes theme significance via the clustering classifier.
    step.clustering_service = PathwayClusteringService()
    return step


def _theme(pathways):
    return {'name': 'T', 'pathways': pathways, 'pathway_count': len(pathways),
            'avg_p_value_fdr': 1e-3}


def test_off_tissue_pathway_filtered():
    step = _bare_step01()
    step.llm.chat.return_value = json.dumps({'filtered_pathways': [
        {'pathway_name': 'Bladder cancer', 'decision': 'FILTER', 'rationale': 'wrong organ'},
        {'pathway_name': 'Cell cycle', 'decision': 'KEEP', 'rationale': 'general'},
    ]})
    themes = [_theme([
        {'name': 'Bladder cancer', 'p_value_fdr': 0.01, 'gene_count': 5},
        {'name': 'Cell cycle', 'p_value_fdr': 1e-4, 'gene_count': 40},
    ])]
    cleaned, filtered = step._filter_off_tissue_pathways(
        themes=themes, context={'disease': 'Hepatocellular carcinoma', 'tissue': 'liver'})
    assert filtered == 1
    names = [p['name'] for p in cleaned[0]['pathways']]
    assert names == ['Cell cycle']


def test_no_context_skips_filtering():
    step = _bare_step01()
    themes = [_theme([{'name': 'Bladder cancer', 'p_value_fdr': 0.01, 'gene_count': 5}])]
    cleaned, filtered = step._filter_off_tissue_pathways(themes=themes, context=None)
    assert filtered == 0
    assert len(cleaned[0]['pathways']) == 1


def test_singletons_are_filtered_via_wrapper():
    """The singleton path wraps ungrouped pathways as a pseudo-theme and filters them."""
    step = _bare_step01()
    step.llm.chat.return_value = json.dumps({'filtered_pathways': [
        {'pathway_name': 'Salmonella infection', 'decision': 'FILTER', 'rationale': 'off-disease'},
        {'pathway_name': 'p53 signaling', 'decision': 'KEEP', 'rationale': 'general'},
    ]})
    singletons = [
        {'name': 'Salmonella infection', 'p_value_fdr': 0.02, 'gene_count': 7},
        {'name': 'p53 signaling', 'p_value_fdr': 1e-3, 'gene_count': 14},
    ]
    wrapped, filtered = step._filter_off_tissue_pathways(
        themes=[{'name': 'Ungrouped pathways', 'pathways': singletons,
                 'pathway_count': len(singletons), 'avg_p_value_fdr': 1.0}],
        context={'disease': 'Hepatocellular carcinoma', 'tissue': 'liver'})
    assert filtered == 1
    kept = [p['name'] for p in wrapped[0]['pathways']]
    assert kept == ['p53 signaling']
