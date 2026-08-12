"""
Pathway Clustering Service
Implements gene overlap-based pathway clustering (similar to EnrichmentMap).

Pathways are grouped in two stages: (1) a computational graph-clustering step over a
pairwise Jaccard gene-overlap graph, then (2) LLM theme naming (in step01). The
computational step supports two algorithms:

* ``mcl`` (default) — Markov Clustering (Van Dongen, "Graph Clustering by Flow Simulation",
  PhD thesis, Univ. Utrecht, 2000). This is the default community-detection method in aPEAR
  (Kerševičius & Gordevičius, Bioinformatics 2023) and the algorithm EnrichmentMap
  recommends for its AutoAnnotate step. MCL simulates flow (expansion + inflation) on the
  similarity graph, which naturally separates communities linked only by a weak "bridge"
  pathway — the single-linkage chaining/hairball failure mode of the legacy method. MCL is
  fully deterministic (no RNG), so runs are reproducible without a seed. Our MCL parameters
  mirror aPEAR's published defaults (ievaKer/aPEAR, R/cluster.R + R/similarity.R):
  ``mcl(sim, addLoops=FALSE, max.iter=500, expansion=2, inflation=2.5)`` over the FULL pairwise
  Jaccard similarity matrix with NO similarity threshold — i.e. inflation 2.5, similarity_floor
  0.0 (keep every Jaccard>0 edge), expansion 2, and no added self-loops (the Jaccard matrix's
  own 1.0 diagonal self-similarity supplies the loops; see ``_build_weighted_adjacency``).
* ``connected_components`` — the legacy method: threshold the Jaccard graph at a hard cutoff
  and take connected components via BFS. This is single-linkage clustering, kept for A/B
  comparison and as a fallback.
"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import json

from src.pipeline.fc_utils import fc_direction, to_float, gene_fc_lookup, NES_STRONG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _pathway_fdr(pathway: Dict, default: float = 1.0) -> float:
    """Adjusted p-value (FDR) for a pathway, tolerant of field names, 0.0-safe.

    Uses to_float + None checks (not ``or``-chaining) so a legitimate FDR of exactly 0.0
    is preserved instead of falling through to ``default``.
    """
    for key in ('p_value_fdr', 'pValueFDR', 'fdr'):
        v = to_float(pathway.get(key))
        if v is not None:
            return v
    return default


def _pathway_abs_nes(pathway: Dict) -> Optional[float]:
    """Return |enrichment score| on the NES scale for a pathway, or None.

    Prefers NES; falls back to the legacy ``ES`` field, promoting it to the NES scale
    only when |ES| > 1 (classic fgsea ES is bounded to [-1, 1], so a larger magnitude
    can only be an NES that upstream forwarded under the ``ES`` name). Mirrors
    ``_enrichment_metric`` in step03 so significance uses the same honest scale.

    Named ``_pathway_abs_nes`` (not ``_abs_nes``) to avoid confusion with the unrelated
    local ``_abs_nes`` in step03 which reads ``enrichment_score`` off serialized structures.
    """
    nes = to_float(pathway.get('NES'))
    if nes is not None:
        return abs(nes)
    es = to_float(pathway.get('ES'))
    if es is not None and abs(es) > 1.0:
        return abs(es)
    return None


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity |A ∩ B| / |A ∪ B| (0.0 when the union is empty)."""
    union = a | b
    return (len(a & b) / len(union)) if union else 0.0


def _build_weighted_adjacency(
    gene_sets: List[Set[str]],
    similarity_floor: float,
) -> List[List[float]]:
    """Full pairwise Jaccard similarity matrix (aPEAR-style), optionally sparsified.

    Off-diagonal entry ``[i][j]`` is the pairwise Jaccard similarity when it is
    ``>= similarity_floor``, else 0.0. With the aPEAR default ``similarity_floor=0.0`` every
    Jaccard>0 edge is kept (no threshold — aPEAR feeds the full similarity matrix); the 0.0
    entries this leaves for disjoint pairs are harmless no-op edges.

    The **diagonal is 1.0 self-similarity** (a set is identical to itself: Jaccard=1). aPEAR
    runs MCL with ``addLoops=FALSE`` on exactly this matrix, so the diagonal's own 1.0 supplies
    the self-loops MCL needs — the caller passes ``loop_value=0`` so ``markov_clustering`` does
    NOT overwrite the diagonal. (With ``loop_value=0`` and a 0.0 diagonal MCL would have no
    self-loops and degenerate into over-fragmented, overlapping clusters.) Returned as a dense
    list-of-lists; the caller converts to a numpy array for ``markov_clustering``.
    """
    n = len(gene_sets)
    adj = [[0.0] * n for _ in range(n)]
    for i in range(n):
        adj[i][i] = 1.0  # self-similarity (aPEAR full Jaccard matrix; supplies MCL self-loops)
        for j in range(i + 1, n):
            sim = _jaccard(gene_sets[i], gene_sets[j])
            if sim >= similarity_floor:
                adj[i][j] = sim
                adj[j][i] = sim
    return adj


def _mcl_partition_from_adjacency(
    adjacency: List[List[float]],
    inflation: float,
) -> List[List[int]]:
    """Cluster a prebuilt weighted adjacency via MCL, returned as a strict partition.

    Runs MCL (Van Dongen 2000) through the ``markov_clustering`` package, then flattens its
    result into a strict partition — each node lands in exactly one cluster (the first MCL
    cluster that contains it, scanned in order; any node MCL never emits becomes its own
    singleton). A partition (single membership per pathway) preserves the contract the legacy
    connected-components method provided, so downstream theme handling is unchanged.

    numpy / markov_clustering are imported lazily (like networkx in ``network_service``) so
    the dependency is only touched when MCL is actually selected. MCL has no RNG, so the
    partition is deterministic across runs.

    ``run_mcl`` args mirror aPEAR's defaults: ``loop_value=0`` = aPEAR ``addLoops=FALSE`` (the
    adjacency's 1.0 diagonal already carries self-similarity, so MCL is NOT asked to overwrite
    it — see ``_build_weighted_adjacency``), ``iterations=500`` = aPEAR ``max.iter=500``, and
    the package-default ``expansion=2`` already equals aPEAR's expansion.
    """
    n = len(adjacency)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    import numpy as np
    import markov_clustering as mc

    result = mc.run_mcl(
        np.array(adjacency, dtype=float),
        inflation=inflation,
        loop_value=0,       # aPEAR addLoops=FALSE (diagonal self-similarity supplies the loops)
        iterations=500,     # aPEAR max.iter=500 (convergence is usually reached well before this)
    )
    raw_clusters = mc.get_clusters(result)  # list of index tuples (may overlap)

    # Flatten to a strict partition: first cluster to claim a node keeps it.
    assigned: Set[int] = set()
    partition: List[List[int]] = []
    for cluster in raw_clusters:
        members = [idx for idx in cluster if idx not in assigned]
        if members:
            assigned.update(members)
            partition.append(members)

    # Any node MCL never returned becomes its own singleton so no pathway is silently dropped.
    # With loop_value=0 an isolated node (or a fully-disjoint graph) can be emitted as an empty
    # cluster set or dropped entirely; this recovery keeps the partition total and correct.
    for idx in range(n):
        if idx not in assigned:
            partition.append([idx])
            assigned.add(idx)

    return partition


def _mcl_partition(
    gene_sets: List[Set[str]],
    similarity_floor: float,
    inflation: float,
) -> List[List[int]]:
    """Partition node indices into clusters via Markov Clustering over the Jaccard graph.

    Convenience wrapper: builds the sparsified Jaccard adjacency, then delegates to
    :func:`_mcl_partition_from_adjacency`.
    """
    if len(gene_sets) <= 1:
        # Avoid building an (at most 1x1) adjacency for the trivial cases.
        return [[0]] if gene_sets else []
    adjacency = _build_weighted_adjacency(gene_sets, similarity_floor)
    return _mcl_partition_from_adjacency(adjacency, inflation)


def _select_representative(pathways: List[Dict]) -> Optional[str]:
    """Pick a representative pathway name for a cluster (aPEAR-style seed for LLM naming).

    Deterministic and method-agnostic: the most statistically significant pathway (lowest
    FDR), tie-broken by strongest effect size (highest |NES|) then name. (The MCL-attractor
    node is a theoretically nicer choice but would couple this to MCL internals and not work
    for the connected-components method, so it is left as a future option.)
    """
    if not pathways:
        return None

    def _name_of(p: Dict) -> str:
        return p.get('name') or p.get('pathway_name') or p.get('pathwayName') or ''

    def _key(p: Dict) -> Tuple[float, float, str]:
        abs_nes = _pathway_abs_nes(p)
        # Lower FDR first; higher |NES| first (negate); then name for a stable tie-break.
        return (_pathway_fdr(p), -(abs_nes if abs_nes is not None else -1.0), _name_of(p))

    return _name_of(min(pathways, key=_key)) or None


@dataclass
class PathwayCluster:
    """Represents a cluster of pathways grouped by gene overlap"""
    pathways: List[Dict]
    pathway_names: List[str]
    pathway_count: int
    shared_genes: List[str]
    shared_gene_count: int
    avg_jaccard_overlap: float
    avg_p_value_fdr: float
    significance: str
    key_genes_with_fc: List[Dict] = None  # NEW: genes with fold changes
    dominant_direction: str = 'mixed'  # NEW: overall regulation direction
    max_abs_nes: Optional[float] = None  # NEW: strongest |NES| in the cluster (effect size)
    representative: Optional[str] = None  # NEW: representative pathway name (seeds LLM naming)


class PathwayClusteringService:
    """
    Service for clustering pathways based on gene overlap using Jaccard similarity.

    Methodology follows the EnrichmentMap approach (Merico et al., PLoS ONE 2010):
    1. Calculate pairwise Jaccard similarity between pathways
    2. Build a similarity graph over that matrix
    3. Partition the graph into clusters — via Markov Clustering (``method='mcl'``, the
       aPEAR/EnrichmentMap-recommended community-detection algorithm) or via connected
       components (``method='connected_components'``, the legacy single-linkage method)
    4. Calculate cluster statistics
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.25,
        min_cluster_size: int = 2,
        method: str = 'connected_components',
        inflation: float = 2.5,
        similarity_floor: float = 0.0,
        gene_source: str = 'de',
    ):
        """
        Initialize clustering service.

        The defaults reproduce the legacy behavior (connected components over the
        DE-gene-restricted Jaccard graph) so a bare ``PathwayClusteringService()`` is
        unchanged for existing callers/tests. Production wiring (step01) injects the new
        MCL defaults from ``settings``.

        Args:
            jaccard_threshold: Minimum Jaccard similarity to connect pathways for the legacy
                connected-components method (default: 0.25 — the EnrichmentMap-style cutoff;
                NOT an aPEAR value and unused by MCL).
            min_cluster_size: Minimum number of pathways per cluster (default: 2, = aPEAR
                minClusterSize).
            method: 'mcl' (Markov Clustering) or 'connected_components' (legacy).
            inflation: MCL inflation/granularity parameter (default: 2.5, aPEAR default).
            similarity_floor: Minimum Jaccard to keep a weighted edge for MCL (default: 0.0 —
                no threshold; aPEAR feeds the full pairwise Jaccard similarity matrix).
            gene_source: Gene set used for similarity — 'de' (DE-restricted, current) or
                'full' (the pathway's full gene set, aPEAR/EnrichmentMap style).
        """
        self.jaccard_threshold = jaccard_threshold
        self.min_cluster_size = min_cluster_size
        self.method = method
        self.inflation = inflation
        self.similarity_floor = similarity_floor
        self.gene_source = gene_source

    def cluster_pathways_by_gene_overlap(
        self,
        pathways: List[Dict],
        genes: List[Dict],
        jaccard_threshold: float = None
    ) -> Dict:
        """
        Cluster pathways based on gene overlap (Jaccard similarity) or semantic similarity

        Args:
            pathways: List of pathway dicts with 'genes' or 'gene_symbols' field
            genes: List of differentially expressed gene dicts (can be empty)
            jaccard_threshold: Override default Jaccard threshold

        Returns:
            Dictionary with 'clusters', 'singletons', and 'metadata'
        """
        threshold = jaccard_threshold if jaccard_threshold is not None else self.jaccard_threshold

        logger.info(f'Clustering {len(pathways)} pathways...')

        # Extract DE gene names
        de_gene_set = set()
        for gene in genes:
            # Support both camelCase and snake_case
            gene_name = (gene.get('geneSymbol') or gene.get('gene_symbol') or
                        gene.get('gene') or gene.get('name'))
            if gene_name:
                de_gene_set.add(gene_name.upper())  # Normalize to uppercase

        # If no genes provided, use semantic similarity clustering
        if not de_gene_set:
            logger.info('No DE genes provided - using semantic similarity clustering based on pathway names')
            return self.cluster_pathways_by_semantic_similarity(pathways)

        # Otherwise, use gene overlap clustering
        logger.info(
            f'Using gene overlap clustering (method={self.method}, gene_source={self.gene_source})')

        # Filter pathways to only include DE genes
        pathways_with_de_genes = []
        for pathway in pathways:
            # Extract pathway genes (support multiple field names)
            pathway_genes = pathway.get('genes') or pathway.get('gene_symbols') or pathway.get('geneSymbols') or []

            # Normalize and filter to DE genes
            de_genes = [g.upper() for g in pathway_genes if g.upper() in de_gene_set]

            if de_genes:
                pathway_copy = pathway.copy()
                pathway_copy['de_genes'] = de_genes
                # Gene set used for SIMILARITY (clustering only): 'de' reproduces today's
                # behavior; 'full' uses the whole pathway gene set (aPEAR/EnrichmentMap
                # style). All per-cluster STATS still derive from de_genes, so downstream
                # semantics are unchanged regardless of this knob.
                if self.gene_source == 'full':
                    pathway_copy['sim_genes'] = [g.upper() for g in pathway_genes]
                else:
                    pathway_copy['sim_genes'] = de_genes
                pathways_with_de_genes.append(pathway_copy)

        logger.info(f'Pathways with DE genes: {len(pathways_with_de_genes)}')

        if not pathways_with_de_genes:
            logger.warning('No pathways contain DE genes')
            return {'clusters': [], 'singletons': [], 'metadata': {}}

        # Partition the similarity graph into components. MCL (flow simulation) resolves the
        # single-linkage chaining that connected-components suffers from; connected_components
        # is kept for A/B comparison and as a fallback.
        sim_gene_sets = [set(p['sim_genes']) for p in pathways_with_de_genes]
        if self.method == 'mcl':
            # Build the weighted Jaccard adjacency ONCE, then reuse it for both the MCL
            # partition and the edge count (avoids a second O(n^2) Jaccard pass). At the aPEAR
            # default floor 0.0 this is the full similarity matrix (no sparsification).
            adjacency = _build_weighted_adjacency(sim_gene_sets, self.similarity_floor)
            index_components = _mcl_partition_from_adjacency(adjacency, self.inflation)
            components = [[pathways_with_de_genes[i] for i in grp] for grp in index_components]
            n_nodes = len(sim_gene_sets)
            edge_count = sum(
                1
                for i in range(n_nodes)
                for j in range(i + 1, n_nodes)
                if adjacency[i][j] > 0.0
            )
            edge_cutoff = self.similarity_floor
            logger.info(
                f'MCL similarity graph: {len(pathways_with_de_genes)} nodes, {edge_count} edges '
                f'(floor={self.similarity_floor}, inflation={self.inflation})')
        else:
            similarity_matrix = self._calculate_jaccard_matrix(pathways_with_de_genes)
            graph = self._build_similarity_graph(
                pathways_with_de_genes, similarity_matrix, threshold)
            edge_count = graph['edge_count']
            edge_cutoff = threshold
            logger.info(
                f'Similarity graph: {len(pathways_with_de_genes)} nodes, {edge_count} edges')
            components = self._find_connected_components(graph)

        # Enrich clusters with statistics
        clusters = []
        singletons = []

        for component_pathways in components:
            if len(component_pathways) >= self.min_cluster_size:
                cluster = self._enrich_cluster_with_stats(
                    component_pathways,
                    genes
                )
                clusters.append(cluster)
            else:
                # Singleton pathways
                for pathway in component_pathways:
                    singletons.append({
                        'name': pathway.get('name') or pathway.get('pathway_name'),
                        'p_value': pathway.get('p_value') or pathway.get('pValue'),
                        'p_value_fdr': pathway.get('p_value_fdr') or pathway.get('pValueFDR'),
                        'database': pathway.get('database', 'Unknown'),
                        'gene_count': len(pathway.get('de_genes', []))
                    })

        logger.info(f'Clusters formed: {len(clusters)}')
        logger.info(f'Singleton pathways: {len(singletons)}')

        if clusters:
            avg_size = sum(c.pathway_count for c in clusters) / len(clusters)
            avg_overlap = sum(c.avg_jaccard_overlap for c in clusters) / len(clusters)
            logger.info(f'Average cluster size: {avg_size:.1f} pathways')
            logger.info(f'Average intra-cluster overlap: {avg_overlap*100:.1f}%')

        return {
            'clusters': [self._cluster_to_dict(c) for c in clusters],
            'singletons': singletons,
            'metadata': {
                # Kept for back-compat: the edge cutoff actually used (threshold for
                # connected_components, similarity_floor for MCL).
                'jaccard_threshold': edge_cutoff,
                'total_pathways': len(pathways),
                'pathways_with_de_genes': len(pathways_with_de_genes),
                'cluster_count': len(clusters),
                'singleton_count': len(singletons),
                'graph_edge_count': edge_count,
                # Additive clustering provenance (safe — no consumer depends on these).
                'clustering_method': self.method,
                'inflation': self.inflation if self.method == 'mcl' else None,
                'similarity_floor': self.similarity_floor if self.method == 'mcl' else None,
                'gene_source': self.gene_source,
            }
        }

    def _calculate_jaccard_matrix(self, pathways: List[Dict]) -> List[List[float]]:
        """
        Calculate Jaccard similarity matrix for all pathway pairs

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        """
        n = len(pathways)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            matrix[i][i] = 1.0  # Self-similarity

            genes_i = set(pathways[i]['de_genes'])

            for j in range(i + 1, n):
                genes_j = set(pathways[j]['de_genes'])

                intersection = genes_i & genes_j
                union = genes_i | genes_j

                jaccard = len(intersection) / len(union) if union else 0.0

                matrix[i][j] = jaccard
                matrix[j][i] = jaccard  # Symmetric

        return matrix

    def _build_similarity_graph(
        self,
        pathways: List[Dict],
        similarity_matrix: List[List[float]],
        threshold: float
    ) -> Dict:
        """Build undirected graph where edges connect pathways with Jaccard ≥ threshold"""

        adjacency = defaultdict(list)
        edge_count = 0

        for i in range(len(pathways)):
            for j in range(i + 1, len(pathways)):
                if similarity_matrix[i][j] >= threshold:
                    adjacency[i].append(j)
                    adjacency[j].append(i)
                    edge_count += 1

        return {
            'nodes': pathways,
            'adjacency': adjacency,
            'edge_count': edge_count
        }

    def _find_connected_components(self, graph: Dict) -> List[List[Dict]]:
        """
        Find connected components using BFS
        Each component = one pathway cluster
        """
        visited = set()
        components = []
        nodes = graph['nodes']
        adjacency = graph['adjacency']

        for i in range(len(nodes)):
            if i in visited:
                continue

            # BFS to find all nodes in this component
            component = []
            queue = deque([i])
            visited.add(i)

            while queue:
                node_idx = queue.popleft()
                component.append(nodes[node_idx])

                for neighbor in adjacency.get(node_idx, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(component)

        return components

    def _enrich_cluster_with_stats(
        self,
        pathways: List[Dict],
        all_genes: List[Dict]
    ) -> PathwayCluster:
        """Enrich cluster with statistics and shared genes"""

        # Find shared genes across pathways in cluster
        shared_genes = self._find_shared_genes(pathways)

        # Calculate average Jaccard overlap within cluster
        total_jaccard = 0.0
        pair_count = 0

        for i in range(len(pathways)):
            genes_i = set(pathways[i]['de_genes'])
            for j in range(i + 1, len(pathways)):
                genes_j = set(pathways[j]['de_genes'])
                intersection = genes_i & genes_j
                union = genes_i | genes_j
                jaccard = len(intersection) / len(union) if union else 0.0
                total_jaccard += jaccard
                pair_count += 1

        avg_jaccard = total_jaccard / pair_count if pair_count > 0 else 0.0

        # Average pathway FDR (0.0-safe via the shared extractor, not `or`-chaining).
        p_values_fdr = [_pathway_fdr(p) for p in pathways]
        avg_p_value_fdr = sum(p_values_fdr) / len(p_values_fdr) if p_values_fdr else 1.0

        # Effect-size magnitude: strongest |NES| across the cluster's pathways (drives the
        # significance promotion; None when no enrichment scores were provided).
        abs_nes_values = [v for v in (_pathway_abs_nes(p) for p in pathways) if v is not None]
        max_abs_nes = max(abs_nes_values) if abs_nes_values else None

        # Classify significance from statistical strength (FDR) AND effect size (NES)
        significance = self._classify_cluster_significance(
            avg_p_value_fdr, len(pathways), max_abs_nes)

        # Extract pathway names
        pathway_names = [
            p.get('name') or p.get('pathway_name') or p.get('pathwayName') or f'Pathway {i+1}'
            for i, p in enumerate(pathways)
        ]

        # NEW: Calculate fold changes and direction for key genes
        key_genes_with_fc, dominant_direction = self._calculate_gene_fold_changes(
            shared_genes[:10],  # Top 10 shared genes
            all_genes
        )

        return PathwayCluster(
            pathways=[{
                'name': p.get('name') or p.get('pathway_name'),
                'p_value': p.get('p_value') or p.get('pValue'),
                'p_value_fdr': p.get('p_value_fdr') or p.get('pValueFDR'),
                'database': p.get('database', 'Unknown'),
                'gene_count': len(p.get('de_genes', [])),
                # Preserve |NES| per pathway so significance can be honestly recomputed
                # if a downstream step (e.g. tissue-filtering) drops pathways from the
                # theme — otherwise the label would reflect the pre-filter cluster.
                'abs_nes': _pathway_abs_nes(p),
            } for p in pathways],
            pathway_names=pathway_names,
            pathway_count=len(pathways),
            shared_genes=shared_genes,
            shared_gene_count=len(shared_genes),
            avg_jaccard_overlap=avg_jaccard,
            avg_p_value_fdr=avg_p_value_fdr,
            significance=significance,
            key_genes_with_fc=key_genes_with_fc,
            dominant_direction=dominant_direction,
            max_abs_nes=max_abs_nes,
            representative=_select_representative(pathways)
        )

    def _find_shared_genes(self, pathways: List[Dict]) -> List[str]:
        """
        Find genes shared by multiple pathways in cluster

        Returns genes appearing in at least 2 pathways (or 30% if cluster is large)
        """
        if not pathways:
            return []

        # Count how many pathways each gene appears in
        gene_counts = defaultdict(int)

        for pathway in pathways:
            genes = pathway.get('de_genes', [])
            for gene in genes:
                gene_counts[gene] += 1

        # Threshold: at least 2 pathways, or 30% of pathways
        threshold = max(2, int(len(pathways) * 0.3))

        # Return genes meeting threshold, sorted by frequency
        shared = [
            gene for gene, count in gene_counts.items()
            if count >= threshold
        ]

        # Sort by frequency (descending)
        shared.sort(key=lambda g: gene_counts[g], reverse=True)

        return shared

    def _classify_cluster_significance(
        self,
        avg_p_value_fdr: float,
        pathway_count: int,
        max_abs_nes: Optional[float] = None
    ) -> str:
        """Classify cluster significance from statistical strength AND effect size.

        Statistical strength (FDR) sets the base tier so a highly significant cluster is
        not under-ranked just because it has few pathways; cluster size is a secondary
        booster. Effect size (strongest |NES| in the cluster) can then PROMOTE the tier so
        a biologically dominant signal is not read as weaker than a marginal one. NES only
        promotes — it never demotes a statistically significant cluster, since FDR already
        provides the downside gating and NES-based demotion would risk masking real
        findings. The strong-|NES| threshold matches the codebase magnitude convention
        (|NES| >= 2 = strong; see ``_magnitude_label`` in step03).
        """
        # 1) Base tier from FDR (+ pathway count as a minor booster).
        if avg_p_value_fdr < 0.001:
            base = 'high'
        elif avg_p_value_fdr < 0.01:
            base = 'high' if pathway_count >= 3 else 'medium'
        elif avg_p_value_fdr < 0.05:
            base = 'medium'
        else:
            base = 'low'

        # 2) Effect-size promotion: a strong enrichment (|NES| >= NES_STRONG) bumps one
        #    tier. Skipped when no enrichment scores were provided.
        if max_abs_nes is not None and max_abs_nes >= NES_STRONG:
            order = ['low', 'medium', 'high']
            base = order[min(order.index(base) + 1, len(order) - 1)]
        return base

    def _calculate_gene_fold_changes(
        self,
        gene_symbols: List[str],
        all_genes: List[Dict]
    ) -> Tuple[List[Dict], str]:
        """
        Calculate fold changes for given genes and determine dominant regulation direction

        Args:
            gene_symbols: List of gene symbols to look up
            all_genes: Full list of DE genes with fold change information

        Returns:
            Tuple of (genes_with_fc, dominant_direction)
            - genes_with_fc: List of dicts with gene, fold_change, direction
            - dominant_direction: 'up', 'down', or 'mixed'
        """
        # Build UPPER symbol -> signed fold change. Uses the shared helper (to_float +
        # None checks), so a real fold change of exactly 0.0 is preserved instead of the
        # old `or`-chain treating it as falsy and mis-picking a later field — which would
        # skew this cluster's dominant_direction.
        gene_lookup = gene_fc_lookup(all_genes)

        # Get fold changes for requested genes
        genes_with_fc = []
        up_count = 0
        down_count = 0

        for symbol in gene_symbols:
            fc = gene_lookup.get(symbol.upper(), None)
            if fc is not None:
                direction = fc_direction(fc)
                genes_with_fc.append({
                    'gene': symbol,
                    'fold_change': fc,
                    'direction': direction
                })
                if direction == 'up':
                    up_count += 1
                elif direction == 'down':
                    down_count += 1

        # Determine dominant direction (require 80% consensus)
        total = up_count + down_count
        if total == 0:
            dominant_direction = 'mixed'
        elif up_count / total >= 0.8:
            dominant_direction = 'up'
        elif down_count / total >= 0.8:
            dominant_direction = 'down'
        else:
            dominant_direction = 'mixed'

        return genes_with_fc, dominant_direction

    def _cluster_to_dict(self, cluster: PathwayCluster) -> Dict:
        """Convert PathwayCluster to dictionary"""
        return {
            'pathways': cluster.pathways,
            'pathway_names': cluster.pathway_names,
            'pathway_count': cluster.pathway_count,
            'shared_genes': cluster.shared_genes,
            'shared_gene_count': cluster.shared_gene_count,
            'avg_jaccard_overlap': round(cluster.avg_jaccard_overlap, 3),
            'avg_p_value_fdr': cluster.avg_p_value_fdr,
            'significance': cluster.significance,
            'key_genes_with_fc': cluster.key_genes_with_fc or [],
            'dominant_direction': cluster.dominant_direction,
            'max_abs_nes': cluster.max_abs_nes,
            'representative': cluster.representative
        }

    def cluster_pathways_by_semantic_similarity(self, pathways: List[Dict]) -> Dict:
        """
        Cluster pathways based on semantic similarity using LLM

        Used when no DE genes are available. Groups pathways by biological function/theme
        based on pathway names and descriptions.

        Args:
            pathways: List of pathway dicts with 'name' field

        Returns:
            Dictionary with 'clusters', 'singletons', and 'metadata'
        """
        from src.agents.llm_client import UnifiedLLMClient

        logger.info(f'Clustering {len(pathways)} pathways by semantic similarity using LLM...')

        if not pathways:
            return {'clusters': [], 'singletons': [], 'metadata': {}}

        # Extract pathway names and basic info
        pathway_list = []
        for i, p in enumerate(pathways):
            pathway_list.append({
                'id': i,
                'name': p.get('name') or p.get('pathway_name') or f'Pathway {i+1}',
                'source': p.get('source') or p.get('database', 'Unknown'),
                'pValueFDR': _pathway_fdr(p)
            })

        # Build LLM prompt
        system_prompt = """You are an expert in molecular biology and pathway analysis.
Your task is to group pathways into biologically meaningful clusters based on their names.
Group pathways that share common biological themes, functions, or processes."""

        pathway_names_str = '\n'.join([f"{p['id']}. {p['name']} ({p['source']})" for p in pathway_list])

        user_prompt = f"""Given the following list of pathways, group them into biological themes/clusters.

Pathways:
{pathway_names_str}

Instructions:
1. Identify common biological themes (e.g., immune response, metabolism, cell cycle, etc.)
2. Group pathways that belong to the same theme
3. A pathway can only belong to ONE cluster
4. Only create clusters with at least 2 pathways
5. Pathways that don't fit any cluster should be marked as singletons

Return your response as a JSON object with this structure:
{{
  "clusters": [
    {{
      "theme": "Theme name",
      "pathway_ids": [list of pathway IDs in this cluster],
      "rationale": "Brief explanation of why these pathways are grouped together"
    }}
  ],
  "singletons": [list of pathway IDs that don't fit any cluster]
}}

Return ONLY the JSON object, no additional text."""

        try:
            # Call LLM
            llm = UnifiedLLMClient()
            response = llm.complete(user_prompt, system_prompt=system_prompt, temperature=0.0)

            # Parse LLM response
            # Clean response (remove markdown code blocks if present)
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()

            clustering_result = json.loads(response)

            # Build clusters from LLM response
            clusters = []
            for cluster_data in clustering_result.get('clusters', []):
                pathway_ids = cluster_data.get('pathway_ids', [])
                if len(pathway_ids) >= self.min_cluster_size:
                    cluster_pathways = [pathways[pid] for pid in pathway_ids if pid < len(pathways)]

                    # Calculate average FDR for cluster (0.0-safe via shared extractor)
                    fdr_values = [_pathway_fdr(p) for p in cluster_pathways]
                    avg_fdr = sum(fdr_values) / len(fdr_values) if fdr_values else 1.0
                    abs_nes_values = [v for v in (_pathway_abs_nes(p) for p in cluster_pathways)
                                      if v is not None]
                    max_abs_nes = max(abs_nes_values) if abs_nes_values else None

                    # Use the SAME computed classifier as the gene-overlap path (FDR base
                    # tier + |NES| promotion), so a theme's label is identical whether it was
                    # clustered by gene overlap or semantically, and does not silently change
                    # if it later passes through the tissue-filter significance recompute.
                    significance = self._classify_cluster_significance(
                        avg_fdr, len(cluster_pathways), max_abs_nes)

                    pathway_names = [p.get('name') or p.get('pathway_name') for p in cluster_pathways]

                    clusters.append({
                        'pathways': [{
                            'name': p.get('name') or p.get('pathway_name'),
                            'p_value': p.get('p_value') or p.get('pValue'),
                            'p_value_fdr': p.get('p_value_fdr') or p.get('pValueFDR'),
                            'database': p.get('database') or p.get('source', 'Unknown'),
                            'gene_count': len(p.get('genes') or p.get('gene_symbols') or []),
                            'abs_nes': _pathway_abs_nes(p),
                        } for p in cluster_pathways],
                        'pathway_names': pathway_names,
                        'pathway_count': len(cluster_pathways),
                        'shared_genes': [],  # Not applicable for semantic clustering
                        'shared_gene_count': 0,
                        'avg_jaccard_overlap': 0.0,  # Not applicable
                        'avg_p_value_fdr': avg_fdr,
                        'significance': significance,
                        'max_abs_nes': max_abs_nes,
                        'key_genes_with_fc': [],
                        'dominant_direction': 'mixed',
                        'representative': _select_representative(cluster_pathways),
                        'cluster_theme': cluster_data.get('theme', 'Unknown'),
                        'cluster_rationale': cluster_data.get('rationale', '')
                    })

            # Build singletons
            singleton_ids = set(clustering_result.get('singletons', []))
            singletons = []
            for sid in singleton_ids:
                if sid < len(pathways):
                    p = pathways[sid]
                    singletons.append({
                        'name': p.get('name') or p.get('pathway_name'),
                        'p_value': p.get('p_value') or p.get('pValue'),
                        'p_value_fdr': p.get('p_value_fdr') or p.get('pValueFDR'),
                        'database': p.get('database') or p.get('source', 'Unknown'),
                        'gene_count': len(p.get('genes') or p.get('gene_symbols') or [])
                    })

            logger.info(f'Semantic clustering complete: {len(clusters)} clusters, {len(singletons)} singletons')

            return {
                'clusters': clusters,
                'singletons': singletons,
                'metadata': {
                    'clustering_method': 'semantic_similarity',
                    'total_pathways': len(pathways),
                    'cluster_count': len(clusters),
                    'singleton_count': len(singletons),
                    'llm_used': True
                }
            }

        except Exception as e:
            logger.error(f'Error in semantic clustering: {str(e)}')
            logger.warning('Falling back to treating all pathways as singletons')

            # Fallback: treat all as singletons
            singletons = []
            for p in pathways:
                singletons.append({
                    'name': p.get('name') or p.get('pathway_name'),
                    'p_value': p.get('p_value') or p.get('pValue'),
                    'p_value_fdr': p.get('p_value_fdr') or p.get('pValueFDR'),
                    'database': p.get('database') or p.get('source', 'Unknown'),
                    'gene_count': len(p.get('genes') or p.get('gene_symbols') or [])
                })

            return {
                'clusters': [],
                'singletons': singletons,
                'metadata': {
                    'clustering_method': 'fallback',
                    'total_pathways': len(pathways),
                    'cluster_count': 0,
                    'singleton_count': len(singletons),
                    'error': str(e)
                }
            }
