#!/usr/bin/env python
"""A/B harness: connected-components vs Markov Clustering on the same pathway payload.

Runs BOTH clustering methods over an in-memory sample job and prints a side-by-side view of
the resulting cluster assignments, so the change is inspectable. It demonstrates the core
difference: a single weak "bridge" pathway chains two distinct gene-overlap communities into
one blob under connected components, while MCL keeps them apart.

Run from ``interpretation-api/``:

    python tests/mcl_ab_demo.py

No files are written; only stdlib + the clustering service are used.
"""

import os
import sys

# Make `src` importable when run directly (mirrors tests/conftest.py).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline.services.pathway_clustering_service import PathwayClusteringService  # noqa: E402


def sample_payload():
    """Two dense gene-overlap groups (cell-cycle-like A, OXPHOS-like B) + one weak bridge."""
    ga = ['CDK1', 'CCNB1', 'AURKB', 'BUB1', 'PLK1', 'CDC20']
    gb = ['NDUFA1', 'NDUFB2', 'SDHA', 'UQCRC1', 'COX5A', 'ATP5F1A']
    pathways = [
        {'name': 'Cell cycle', 'genes': ga, 'p_value_fdr': 1e-5, 'NES': 2.3, 'database': 'KEGG'},
        {'name': 'Mitotic spindle', 'genes': ga[:5] + ['KIF11'], 'p_value_fdr': 2e-4, 'NES': 2.0, 'database': 'Reactome'},
        {'name': 'G2/M checkpoint', 'genes': ga[:4] + ['WEE1', 'CHEK1'], 'p_value_fdr': 4e-4, 'NES': 1.7, 'database': 'MSigDB'},
        {'name': 'Oxidative phosphorylation', 'genes': gb, 'p_value_fdr': 3e-5, 'NES': 2.2, 'database': 'KEGG'},
        {'name': 'Respiratory electron transport', 'genes': gb[:5] + ['CYC1'], 'p_value_fdr': 1e-4, 'NES': 1.9, 'database': 'Reactome'},
        {'name': 'TCA / ETC coupling', 'genes': gb[:4] + ['FH', 'MDH2'], 'p_value_fdr': 5e-4, 'NES': 1.6, 'database': 'MSigDB'},
        {'name': 'Metabolic bridge (weak link)', 'genes': ['CDK1', 'CCNB1', 'NDUFA1', 'NDUFB2'],
         'p_value_fdr': 0.02, 'NES': 1.0, 'database': 'KEGG'},
    ]
    all_symbols = sorted({g for p in pathways for g in p['genes']})
    genes = [{'geneSymbol': g, 'foldChange': 1.5} for g in all_symbols]
    return pathways, genes


def _print_result(title, result):
    md = result['metadata']
    print(f"\n{title}")
    print('-' * len(title))
    print(f"  method={md.get('clustering_method')} "
          f"clusters={md.get('cluster_count')} singletons={md.get('singleton_count')} "
          f"edges={md.get('graph_edge_count')}")
    for i, c in enumerate(result['clusters'], 1):
        rep = c.get('representative')
        print(f"  Cluster {i} [rep: {rep}] "
              f"(overlap {c['avg_jaccard_overlap'] * 100:.0f}%, {c['significance']}): "
              f"{', '.join(c['pathway_names'])}")
    if result['singletons']:
        print(f"  Singletons: {', '.join(s['name'] for s in result['singletons'])}")


def main():
    pathways, genes = sample_payload()

    cc = PathwayClusteringService(method='connected_components', gene_source='de',
                                  jaccard_threshold=0.25, min_cluster_size=2)
    # MCL uses aPEAR's published defaults: full Jaccard matrix (floor 0.0), inflation 2.5,
    # loop_value=0 (addLoops=FALSE via the matrix's 1.0 self-similarity diagonal).
    mcl = PathwayClusteringService(method='mcl', gene_source='de',
                                   similarity_floor=0.0, inflation=2.5, min_cluster_size=2)

    _print_result('A) connected_components (legacy single-linkage)',
                   cc.cluster_pathways_by_gene_overlap(pathways, genes))
    _print_result('B) mcl (Markov Clustering)',
                   mcl.cluster_pathways_by_gene_overlap(pathways, genes))

    print("\nExpected: (A) chains both groups into ONE cluster via the weak bridge; "
          "(B) separates the cell-cycle and OXPHOS communities.")


if __name__ == '__main__':
    main()
