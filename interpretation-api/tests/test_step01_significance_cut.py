"""
W5 leftover (engine side): Step 1 keeps pathways with FDR at most 0.05 (main L367,
suppl L542 "at most 0.05"), and an FDR of exactly 0.0 counts as significant (the old
`or`-chain read 0.0 as 1.0 and dropped the most significant pathways).
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step01_pathway_themes import Step01PathwayThemes


def _clustered_names(pathways):
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    step.llm = MagicMock()
    step.clustering_service = MagicMock()
    step.clustering_service.cluster_pathways_by_gene_overlap.return_value = {
        'clusters': [], 'singletons': [], 'metadata': {}}
    try:
        step.execute(pathways=pathways, genes=[])
    except Exception:
        pass  # only the input handed to clustering matters here
    call = step.clustering_service.cluster_pathways_by_gene_overlap.call_args
    return [p['name'] for p in call.kwargs['pathways']] if call else []


def test_fdr_exactly_threshold_is_kept():
    names = _clustered_names([{'name': 'Edge', 'p_value_fdr': 0.05},
                              {'name': 'Out', 'p_value_fdr': 0.0501}])
    assert names == ['Edge']


def test_fdr_zero_is_kept():
    names = _clustered_names([{'name': 'Strongest', 'p_value_fdr': 0.0},
                              {'name': 'Camel', 'pValueFDR': 0.0}])
    assert names == ['Strongest', 'Camel']


def test_missing_fdr_is_not_significant():
    assert _clustered_names([{'name': 'NoFDR'}]) == []
