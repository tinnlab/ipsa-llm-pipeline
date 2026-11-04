"""
Network Analysis Service
Queries STRING database for protein-protein interactions and calculates network centrality metrics
"""

import requests
import time
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkEdge:
    """Represents an edge in the protein-protein interaction network"""
    source: str
    target: str
    score: float  # STRING combined score (0-999)


@dataclass
class NetworkMetrics:
    """Network centrality metrics for a gene"""
    gene: str
    degree: int  # Number of direct connections
    betweenness: float  # Fraction of shortest paths through this node
    closeness: float  # Inverse average distance to all nodes
    degree_rank: float  # Normalized rank (0-1)
    betweenness_rank: float
    closeness_rank: float
    hub_score: float  # Composite score


@dataclass
class HubGene:
    """Hub gene with network metrics and expression data"""
    gene: str
    fold_change: float
    p_value: float
    degree: int
    betweenness: float
    closeness: float
    hub_score: float
    degree_rank: float
    betweenness_rank: float
    closeness_rank: float


class NetworkAnalysisService:
    """Service for protein-protein interaction network analysis using STRING database"""

    def __init__(self, cache_enabled: bool = True):
        self.string_api_url = 'https://string-db.org/api'
        self.cache_enabled = cache_enabled
        self.cache: Dict[str, Dict] = {}

    def calculate_network_hubs(
        self,
        genes: List[Dict],
        species: str = '9606',  # Human by default
        top_n: int = 10,
        min_score: int = 400,  # STRING confidence threshold (0-999)
        context: Optional[Dict] = None  # Disease/tissue context for scoring adjustments
    ) -> List[HubGene]:
        """
        Calculate network hub genes using STRING PPI network

        Args:
            genes: List of gene dicts with 'gene_symbol', 'log2_fold_change', 'p_value'
            species: NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat)
            top_n: Number of top hub genes to return
            min_score: Minimum STRING confidence score (0-999, default 400=medium confidence)

        Returns:
            List of HubGene objects ranked by hub score
        """
        logger.info(f'Calculating network hubs for {len(genes)} genes...')

        # Extract gene names (support both camelCase and snake_case)
        gene_names = [(g.get('geneSymbol') or g.get('gene_symbol') or
                      g.get('gene') or g.get('name')) for g in genes]
        gene_names = [g for g in gene_names if g]

        if not gene_names:
            logger.warning('No valid gene names provided')
            return []

        # Get PPI network from STRING
        network = self._get_string_network(gene_names, species, min_score)

        if not network or not network['edges']:
            logger.warning('No PPI network retrieved from STRING, falling back to expression ranking')
            return self._fallback_to_expression_ranking(genes, top_n)

        logger.info(f'Retrieved PPI network: {len(network["nodes"])} nodes, {len(network["edges"])} edges')

        # Calculate centrality metrics
        centrality_scores = self._calculate_centrality_metrics(network)

        # Combine network metrics with expression data
        hub_genes = []
        for gene_data in genes:
            # Support both camelCase and snake_case
            gene_name = (gene_data.get('geneSymbol') or gene_data.get('gene_symbol') or
                        gene_data.get('gene') or gene_data.get('name'))

            if not gene_name or gene_name not in centrality_scores:
                continue

            scores = centrality_scores[gene_name]

            # Composite hub score: weighted combination of metrics
            hub_score = (
                scores['degree_rank'] * 0.5 +      # Degree is most important
                scores['betweenness_rank'] * 0.3 + # Betweenness for bridges
                scores['closeness_rank'] * 0.2      # Closeness for global importance
            )

            # Support both camelCase and snake_case for fold change and p-value
            fold_change = (gene_data.get('foldChange') or gene_data.get('log2_fold_change') or
                          gene_data.get('log2FoldChange') or 0)
            p_value = gene_data.get('pValue') or gene_data.get('p_value') or 1.0

            # Apply tissue-marker penalty for likely differentiation markers (Fix 2)
            tissue_penalty = self._compute_tissue_marker_penalty(gene_name, fold_change, context)
            if tissue_penalty > 0:
                hub_score = max(0.0, hub_score - tissue_penalty)
                logger.info(f'  Tissue-marker penalty for {gene_name}: -{tissue_penalty:.2f} (FC={fold_change:+.2f})')

            # Peripheral-node demotion (Fix 6): genes with degree=1, betweenness=0,
            # and modest fold change are likely noise — demote significantly
            if (scores['degree'] <= 1 and scores['betweenness'] == 0
                    and abs(fold_change) < 1.6):
                hub_score *= 0.5
                logger.info(f'  Peripheral-node demotion for {gene_name}: score halved (degree={scores["degree"]}, FC={fold_change:+.2f})')

            hub_gene = HubGene(
                gene=gene_name,
                fold_change=fold_change,
                p_value=p_value,
                degree=scores['degree'],
                betweenness=scores['betweenness'],
                closeness=scores['closeness'],
                hub_score=hub_score,
                degree_rank=scores['degree_rank'],
                betweenness_rank=scores['betweenness_rank'],
                closeness_rank=scores['closeness_rank']
            )
            hub_genes.append(hub_gene)

        # Sort by hub score and return top N
        hub_genes.sort(key=lambda x: x.hub_score, reverse=True)
        top_hubs = hub_genes[:top_n]

        if top_hubs:
            logger.info(f'Identified {len(top_hubs)} network hub genes')
            logger.info(f'Top hub: {top_hubs[0].gene} (degree={top_hubs[0].degree}, score={top_hubs[0].hub_score:.3f})')

        return top_hubs

    def _get_string_network(
        self,
        gene_names: List[str],
        species: str,
        min_score: int
    ) -> Optional[Dict]:
        """Query STRING database for PPI network"""

        # Limit to 500 genes (STRING API limit)
        gene_names = gene_names[:500]

        # Check cache
        cache_key = f'{species}_{min_score}_{"_".join(sorted(gene_names))}'
        if self.cache_enabled and cache_key in self.cache:
            logger.info('Using cached STRING network')
            return self.cache[cache_key]

        try:
            # Build STRING API request
            identifiers = '%0d'.join(gene_names)
            url = f'{self.string_api_url}/json/network'
            params = {
                'identifiers': identifiers,
                'species': species,
                'required_score': min_score
            }

            logger.info(f'Querying STRING API for {len(gene_names)} genes (species={species}, min_score={min_score})...')
            response = requests.get(url, params=params, timeout=30)

            if response.status_code != 200:
                logger.error(f'STRING API error: {response.status_code} - {response.text}')
                return None

            edges_data = response.json()

            if not edges_data or not isinstance(edges_data, list):
                logger.warning('No interactions returned from STRING')
                return None

            # Build node set
            nodes = set()
            edges = []

            for edge in edges_data:
                source = edge.get('preferredName_A') or edge.get('stringId_A')
                target = edge.get('preferredName_B') or edge.get('stringId_B')
                score = edge.get('score', 0)

                if source and target:
                    nodes.add(source)
                    nodes.add(target)
                    edges.append(NetworkEdge(source, target, score))

            network = {
                'nodes': list(nodes),
                'edges': edges
            }

            # Cache result
            if self.cache_enabled:
                self.cache[cache_key] = network

            return network

        except requests.Timeout:
            logger.error('STRING API request timeout')
            return None
        except Exception as e:
            logger.error(f'Error querying STRING API: {str(e)}')
            return None

    def _calculate_centrality_metrics(self, network: Dict) -> Dict[str, Dict]:
        """Calculate degree, betweenness, and closeness centrality for all nodes"""

        nodes = network['nodes']
        edges = network['edges']

        # Use NetworkX for fast centrality calculation (optimized C implementation)
        try:
            import networkx as nx

            # Build NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(nodes)
            for edge in edges:
                G.add_edge(edge.source, edge.target)

            # Calculate centralities using NetworkX (much faster!)
            degree_dict = dict(G.degree())

            # For large networks, use approximate betweenness (sampling)
            if len(nodes) > 200:
                # Use k-sampling for betweenness (much faster for large networks)
                k = min(100, len(nodes) // 2)
                betweenness_dict = nx.betweenness_centrality(G, k=k, normalized=True)
                logger.info(f'Using approximate betweenness (k={k} samples)')
            else:
                betweenness_dict = nx.betweenness_centrality(G, normalized=True)

            closeness_dict = nx.closeness_centrality(G)

            # Convert to our format
            scores = {}
            for node in nodes:
                scores[node] = {
                    'degree': degree_dict.get(node, 0),
                    'betweenness': betweenness_dict.get(node, 0.0),
                    'closeness': closeness_dict.get(node, 0.0)
                }

        except ImportError:
            # Fallback to custom implementation if NetworkX not available
            logger.warning('NetworkX not available, using slower custom implementation')

            # Build adjacency list
            adjacency = defaultdict(list)
            for edge in edges:
                adjacency[edge.source].append(edge.target)
                adjacency[edge.target].append(edge.source)

            # Calculate raw metrics
            scores = {}
            for node in nodes:
                scores[node] = {
                    'degree': self._calculate_degree(node, adjacency),
                    'betweenness': self._calculate_betweenness(node, adjacency, nodes),
                    'closeness': self._calculate_closeness(node, adjacency, nodes)
                }

        # Normalize to 0-1 ranks
        self._normalize_scores(scores)

        return scores

    def _calculate_degree(self, node: str, adjacency: Dict) -> int:
        """Degree centrality: number of direct connections"""
        return len(adjacency.get(node, []))

    def _calculate_betweenness(
        self,
        node: str,
        adjacency: Dict,
        all_nodes: List[str]
    ) -> float:
        """
        Betweenness centrality: fraction of shortest paths passing through node
        Using simplified algorithm (exact for small networks, approximation for large)
        """
        if len(all_nodes) > 100:
            # For large networks, sample node pairs for efficiency
            import random
            sample_size = min(100, len(all_nodes))
            sampled_nodes = random.sample(all_nodes, sample_size)
        else:
            sampled_nodes = all_nodes

        betweenness = 0.0
        pair_count = 0

        for i, source in enumerate(sampled_nodes):
            for target in sampled_nodes[i+1:]:
                if source == node or target == node:
                    continue

                # Find shortest paths from source to target
                paths = self._find_all_shortest_paths(source, target, adjacency)

                if not paths:
                    continue

                # Count paths through node
                paths_through_node = sum(1 for path in paths if node in path)
                betweenness += paths_through_node / len(paths)
                pair_count += 1

        return betweenness / pair_count if pair_count > 0 else 0.0

    def _calculate_closeness(
        self,
        node: str,
        adjacency: Dict,
        all_nodes: List[str]
    ) -> float:
        """Closeness centrality: inverse of average distance to all other nodes"""
        total_distance = 0
        reachable_nodes = 0

        for target in all_nodes:
            if target == node:
                continue

            distance = self._shortest_path_length(node, target, adjacency)
            if distance != float('inf'):
                total_distance += distance
                reachable_nodes += 1

        if reachable_nodes == 0:
            return 0.0

        # Closeness = (n-1) / sum of distances
        return reachable_nodes / total_distance

    def _shortest_path_length(
        self,
        source: str,
        target: str,
        adjacency: Dict
    ) -> float:
        """BFS to find shortest path length"""
        if source == target:
            return 0

        queue = deque([(source, 0)])
        visited = {source}

        while queue:
            node, distance = queue.popleft()

            for neighbor in adjacency.get(node, []):
                if neighbor == target:
                    return distance + 1

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        return float('inf')  # Not reachable

    def _find_all_shortest_paths(
        self,
        source: str,
        target: str,
        adjacency: Dict
    ) -> List[List[str]]:
        """Find all shortest paths from source to target using BFS"""

        # BFS to find shortest distance
        distances = {source: 0}
        queue = deque([source])

        while queue:
            node = queue.popleft()
            current_dist = distances[node]

            for neighbor in adjacency.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        if target not in distances:
            return []  # Not reachable

        # Backtrack to find all shortest paths
        shortest_length = distances[target]
        paths = []

        def backtrack(current: str, path: List[str]):
            if current == source:
                paths.append(list(reversed(path)))
                return

            for neighbor in adjacency.get(current, []):
                if neighbor in distances and distances[neighbor] == distances[current] - 1:
                    backtrack(neighbor, path + [neighbor])

        backtrack(target, [target])
        return paths

    def _normalize_scores(self, scores: Dict[str, Dict]):
        """Normalize scores to 0-1 ranks"""
        nodes = list(scores.keys())

        if not nodes:
            return

        # Get all values for each metric
        degrees = [scores[n]['degree'] for n in nodes]
        betweennesses = [scores[n]['betweenness'] for n in nodes]
        closenesses = [scores[n]['closeness'] for n in nodes]

        max_degree = max(degrees) if degrees else 1
        max_betweenness = max(betweennesses) if betweennesses else 1
        max_closeness = max(closenesses) if closenesses else 1

        # Normalize
        for node in nodes:
            scores[node]['degree_rank'] = scores[node]['degree'] / max_degree if max_degree > 0 else 0
            scores[node]['betweenness_rank'] = scores[node]['betweenness'] / max_betweenness if max_betweenness > 0 else 0
            scores[node]['closeness_rank'] = scores[node]['closeness'] / max_closeness if max_closeness > 0 else 0

    def _fallback_to_expression_ranking(
        self,
        genes: List[Dict],
        top_n: int
    ) -> List[HubGene]:
        """Fallback: rank by fold change if network unavailable"""
        logger.info('Using expression-based ranking (no network data)')

        hub_genes = []
        for gene_data in genes:
            # Support both camelCase and snake_case
            gene_name = (gene_data.get('geneSymbol') or gene_data.get('gene_symbol') or
                        gene_data.get('gene') or gene_data.get('name'))
            if not gene_name:
                continue

            # Support both camelCase and snake_case for fold change and p-value
            fold_change = (gene_data.get('foldChange') or gene_data.get('log2_fold_change') or
                          gene_data.get('log2FoldChange') or 0)
            p_value = gene_data.get('pValue') or gene_data.get('p_value') or 1.0

            hub_gene = HubGene(
                gene=gene_name,
                fold_change=fold_change,
                p_value=p_value,
                degree=0,
                betweenness=0.0,
                closeness=0.0,
                hub_score=abs(fold_change),  # Use absolute fold change as proxy
                degree_rank=0.0,
                betweenness_rank=0.0,
                closeness_rank=0.0
            )
            hub_genes.append(hub_gene)

        hub_genes.sort(key=lambda x: x.hub_score, reverse=True)
        return hub_genes[:top_n]

    def _compute_tissue_marker_penalty(
        self,
        gene_name: str,
        fold_change: float,
        context: Optional[Dict]
    ) -> float:
        """
        Compute a penalty for genes that are likely tissue-specific differentiation
        markers rather than disease drivers.

        Heuristic: genes that are downregulated (lost during dedifferentiation) AND
        belong to tissue-specific functional categories (ion channels, tight junctions,
        transporters) are likely passenger genes, not drivers.

        Args:
            gene_name: Gene symbol
            fold_change: log2 fold change (negative = downregulated)
            context: Disease/tissue context dict

        Returns:
            Penalty value (0.0 = no penalty, 0.3 = likely tissue marker)
        """
        if not context or fold_change >= -1.0:
            return 0.0

        # Tissue-specific gene family prefixes/patterns associated with
        # normal tissue homeostasis rather than disease driving
        TISSUE_MARKER_GENES = {
            # Ion channels (kidney-specific)
            'KCNJ', 'KCNK', 'SCNN', 'CLCN', 'TRPV', 'TRPM',
            # Tight junctions
            'CLDN', 'TJP', 'OCLN', 'CGN',
            # Solute carriers / transporters
            'SLC', 'ATP6V', 'ATP12A', 'AQP',
            # Kinases involved in ion homeostasis
            'WNK',
            # Hormones with no tissue-relevant function
            'GNRH',
        }

        # Check if gene matches any tissue-marker pattern
        gene_upper = gene_name.upper()
        is_tissue_marker = any(gene_upper.startswith(prefix) for prefix in TISSUE_MARKER_GENES)

        if is_tissue_marker:
            return 0.3

        return 0.0


# Species mapping
SPECIES_MAP = {
    'Homo sapiens': '9606',
    'homo sapiens': '9606',
    'human': '9606',
    'Mus musculus': '10090',
    'mus musculus': '10090',
    'mouse': '10090',
    'Rattus norvegicus': '10116',
    'rattus norvegicus': '10116',
    'rat': '10116'
}


def get_species_id(organism: str) -> str:
    """Convert organism name to NCBI taxonomy ID"""
    return SPECIES_MAP.get(organism, '9606')  # Default to human
