"""
Service for querying pathway data from Step 3 via tool calls in Step 4
"""
from typing import Dict, List, Optional, Any
import re


class PathwayQueryService:
    """
    Service for querying Step 3 pathway data via tool calls.

    Provides tools for:
    - Getting detailed pathway mechanisms
    - Searching pathways by gene
    - Analyzing pathway crosstalk (shared genes)
    """

    def __init__(self, step3_output: Dict):
        """
        Initialize service with Step 3 output data

        Args:
            step3_output: Complete step3_output dict with pathway_mechanisms,
                         pathway_structures, and pathway_overlaps
        """
        self.mechanisms = step3_output.get('pathway_mechanisms', [])
        self.structures = step3_output.get('pathway_structures', [])
        self.overlaps = step3_output.get('pathway_overlaps', [])

        # Build indexes for fast lookup
        self._build_indexes()

    def _build_indexes(self):
        """Build lookup indexes for efficient querying"""
        # Mechanism lookup by pathway name (normalized)
        self.mechanism_index = {}
        for mech in self.mechanisms:
            pathway = mech.get('pathway', '')
            normalized = self._normalize_pathway_name(pathway)
            self.mechanism_index[normalized] = mech

        # Structure lookup by pathway name
        self.structure_index = {}
        for struct in self.structures:
            pathway = struct.get('pathway', '')
            normalized = self._normalize_pathway_name(pathway)
            self.structure_index[normalized] = struct

        # Gene-to-pathways index
        self.gene_to_pathways = {}
        for struct in self.structures:
            pathway = struct.get('pathway', '')
            de_genes = struct.get('mapped_de_genes', [])
            for gene in de_genes:
                gene_symbol = gene.get('gene_symbol', '')
                if gene_symbol:
                    if gene_symbol not in self.gene_to_pathways:
                        self.gene_to_pathways[gene_symbol] = []
                    self.gene_to_pathways[gene_symbol].append({
                        'pathway': pathway,
                        'fold_change': gene.get('fold_change'),
                        'p_value': gene.get('p_value'),
                        'direction': gene.get('direction')
                    })

        # Overlap lookup by pathway pair
        self.overlap_index = {}
        for overlap in self.overlaps:
            pw1 = self._normalize_pathway_name(overlap.get('pathway1', ''))
            pw2 = self._normalize_pathway_name(overlap.get('pathway2', ''))
            # Store both orderings for easy lookup
            self.overlap_index[(pw1, pw2)] = overlap
            self.overlap_index[(pw2, pw1)] = overlap

    def _normalize_pathway_name(self, pathway: str) -> str:
        """Normalize pathway name for matching"""
        return pathway.lower().strip()

    def _fuzzy_match_pathway(self, query: str) -> Optional[str]:
        """
        Find best matching pathway name from query string

        Args:
            query: User's pathway query (may be partial or fuzzy)

        Returns:
            Normalized pathway name if found, None otherwise
        """
        query_norm = self._normalize_pathway_name(query)

        # Exact match
        if query_norm in self.mechanism_index:
            return query_norm

        # Partial match - check if query is substring of any pathway
        for pathway_name in self.mechanism_index.keys():
            if query_norm in pathway_name or pathway_name in query_norm:
                return pathway_name

        # Check structure index as fallback
        for pathway_name in self.structure_index.keys():
            if query_norm in pathway_name or pathway_name in query_norm:
                return pathway_name

        return None

    def get_pathway_mechanism(
        self,
        pathway_name: str,
        include_relations: bool = True
    ) -> Dict[str, Any]:
        """
        Tool: Get detailed mechanism analysis for a specific pathway

        Args:
            pathway_name: Pathway name (exact or partial match)
            include_relations: Whether to include curated pathway relations

        Returns:
            Dict containing:
            - pathway: Full pathway name
            - biologicalFunction: Mechanism description
            - deGeneInvolvement: List of DE genes with fold changes
            - curatedRelations: Curated regulatory relations from KEGG/Reactome (if requested)
            - functionalConsequences: Predicted consequences
            - enrichment_score: Pathway enrichment score
            - p_value_fdr: FDR-adjusted p-value
        """
        # Find matching pathway
        matched_pathway = self._fuzzy_match_pathway(pathway_name)

        if not matched_pathway:
            return {
                'error': f'Pathway not found: {pathway_name}',
                'suggestion': 'Use list_available_pathways to see all available pathways',
                'available_pathways': list((self.mechanism_index or self.structure_index).keys())[:5]
            }

        # Get mechanism data
        mechanism = self.mechanism_index.get(matched_pathway, {})
        structure = self.structure_index.get(matched_pathway, {})

        # If the mechanism-interpretation layer was empty, derive the DE-gene list
        # straight from the retrieved structure (real DE genes only) so the tool still
        # returns gene involvement, symmetric with the curatedRelations fallback below.
        de_gene_involvement = mechanism.get('deGeneInvolvement', [])
        if not de_gene_involvement and structure:
            de_gene_involvement = [
                {
                    'gene': g.get('gene_symbol', ''),
                    'foldChange': g.get('fold_change'),
                    'roleInPathway': ''
                }
                for g in structure.get('mapped_de_genes', [])
                if g.get('gene_symbol') and g.get('direction') in ('up', 'down')
            ]

        # Build response
        result = {
            'pathway': mechanism.get('pathway', structure.get('pathway', pathway_name)),
            'pathway_id': structure.get('pathway_id'),
            'source': structure.get('source'),
            'biologicalFunction': mechanism.get('biologicalFunction', ''),
            'deGeneInvolvement': de_gene_involvement,
            'functionalConsequences': mechanism.get('functionalConsequences', ''),
            'enrichment_score': structure.get('enrichment_score'),
            'enrichment_direction': structure.get('enrichment_direction'),
            'p_value': structure.get('p_value'),
            'p_value_fdr': structure.get('p_value_fdr'),
            'de_genes_count': structure.get('de_genes_count', 0)
        }

        # Add curated relations if requested
        if include_relations:
            relations = mechanism.get('curatedRelations', [])
            if not relations and structure:
                # Mechanism-interpretation layer was empty (e.g. Step 3's LLM step
                # failed) — surface the curated/inferred relations straight from the
                # retrieved structure so the tool still returns mechanistic grounding.
                # Mirror the step03 synthesized shape (incl. isDE) for consistency.
                de_symbols = {g.get('gene') for g in de_gene_involvement}
                relations = [
                    {
                        'source': r.get('source', ''),
                        'target': r.get('target', ''),
                        'type': r.get('subtype') or r.get('type', ''),
                        'isDE': r.get('source') in de_symbols and r.get('target') in de_symbols,
                        'interpretation': ''
                    }
                    for r in structure.get('de_relations', [])
                    if r.get('source') and r.get('target')
                ]
            result['curatedRelations'] = relations

        return result

    def search_pathways_by_gene(self, gene_symbol: str) -> Dict[str, Any]:
        """
        Tool: Find all pathways containing a specific gene

        Args:
            gene_symbol: Gene symbol (e.g., 'TP53', 'EGFR')

        Returns:
            Dict containing:
            - gene_symbol: The queried gene
            - pathways_found: Number of pathways containing this gene
            - pathways: List of pathway info dicts with roles
        """
        gene_symbol = gene_symbol.upper().strip()

        if gene_symbol not in self.gene_to_pathways:
            return {
                'gene_symbol': gene_symbol,
                'pathways_found': 0,
                'pathways': [],
                'error': f'Gene {gene_symbol} not found in any enriched pathway'
            }

        pathway_list = self.gene_to_pathways[gene_symbol]

        # Enrich with mechanism information
        enriched_pathways = []
        for pw_info in pathway_list:
            pathway = pw_info['pathway']
            matched = self._fuzzy_match_pathway(pathway)

            mechanism = self.mechanism_index.get(matched, {})

            # Extract gene role from mechanism
            gene_role = ''
            for gene_inv in mechanism.get('deGeneInvolvement', []):
                if gene_inv.get('gene') == gene_symbol:
                    gene_role = gene_inv.get('roleInPathway', '')
                    break

            enriched_pathways.append({
                'pathway': pathway,
                'fold_change': pw_info['fold_change'],
                'p_value': pw_info['p_value'],
                'direction': pw_info['direction'],
                'role_in_pathway': gene_role,
                'biological_function': mechanism.get('biologicalFunction', '')[:200]  # Truncate
            })

        return {
            'gene_symbol': gene_symbol,
            'pathways_found': len(enriched_pathways),
            'pathways': enriched_pathways
        }

    def get_pathway_crosstalk(
        self,
        pathway1: str,
        pathway2: str
    ) -> Dict[str, Any]:
        """
        Tool: Get genes shared between two pathways (pathway crosstalk)

        Args:
            pathway1: First pathway name
            pathway2: Second pathway name

        Returns:
            Dict containing:
            - pathway1: Full name of pathway 1
            - pathway2: Full name of pathway 2
            - shared_genes: List of shared gene symbols
            - shared_hub_genes: List of shared hub genes
            - shared_count: Number of shared genes
            - jaccard_index: Similarity score (0-1)
        """
        # Find matching pathways
        matched1 = self._fuzzy_match_pathway(pathway1)
        matched2 = self._fuzzy_match_pathway(pathway2)

        if not matched1:
            return {
                'error': f'Pathway 1 not found: {pathway1}',
                'available_pathways': list((self.mechanism_index or self.structure_index).keys())[:5]
            }

        if not matched2:
            return {
                'error': f'Pathway 2 not found: {pathway2}',
                'available_pathways': list((self.mechanism_index or self.structure_index).keys())[:5]
            }

        # Look up overlap data
        overlap_key = (matched1, matched2)
        overlap_data = self.overlap_index.get(overlap_key)

        if not overlap_data:
            # Calculate on-the-fly if not precomputed
            struct1 = self.structure_index.get(matched1, {})
            struct2 = self.structure_index.get(matched2, {})

            genes1 = set(g['gene_symbol'] for g in struct1.get('mapped_de_genes', []))
            genes2 = set(g['gene_symbol'] for g in struct2.get('mapped_de_genes', []))

            shared = genes1 & genes2

            return {
                'pathway1': struct1.get('pathway', pathway1),
                'pathway2': struct2.get('pathway', pathway2),
                'shared_genes': sorted(list(shared)),
                'shared_hub_genes': [],
                'shared_count': len(shared),
                'jaccard_index': len(shared) / len(genes1 | genes2) if genes1 or genes2 else 0
            }

        # Use precomputed overlap data
        return {
            'pathway1': overlap_data.get('pathway1'),
            'pathway2': overlap_data.get('pathway2'),
            'shared_genes': overlap_data.get('shared_genes', []),
            'shared_hub_genes': overlap_data.get('shared_hub_genes', []),
            'shared_count': overlap_data.get('shared_count', 0),
            'jaccard_index': overlap_data.get('jaccard_index', 0)
        }

    def list_available_pathways(
        self,
        filter_by_direction: str = 'all'
    ) -> Dict[str, Any]:
        """
        Tool: List all pathways analyzed in Step 3 with summary stats

        Args:
            filter_by_direction: 'upregulated', 'downregulated', or 'all'

        Returns:
            Dict containing:
            - total_pathways: Total number of pathways
            - filter: Applied filter
            - pathways: List of pathway summaries
        """
        filter_by_direction = filter_by_direction.lower().strip()

        pathway_list = []
        for struct in self.structures:
            direction = struct.get('enrichment_direction', '').lower()

            # Apply filter
            if filter_by_direction != 'all':
                if filter_by_direction not in direction:
                    continue

            pathway_list.append({
                'pathway': struct.get('pathway'),
                'pathway_id': struct.get('pathway_id'),
                'enrichment_score': struct.get('enrichment_score'),
                'enrichment_direction': struct.get('enrichment_direction'),
                'de_genes_count': struct.get('de_genes_count'),
                'p_value_fdr': struct.get('p_value_fdr'),
                'source': struct.get('source')
            })

        # Sort by enrichment score (descending)
        pathway_list.sort(key=lambda x: x.get('enrichment_score', 0), reverse=True)

        return {
            'total_pathways': len(pathway_list),
            'filter': filter_by_direction,
            'pathways': pathway_list
        }

    @staticmethod
    def get_tool_definitions() -> List[Dict]:
        """
        Get OpenAI-compatible tool definitions for all available tools

        Returns:
            List of tool definition dicts in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_pathway_mechanism",
                    "description": "Retrieve detailed mechanism analysis for a specific pathway including DE genes, curated pathway relations, and functional consequences",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pathway_name": {
                                "type": "string",
                                "description": "Pathway name (exact or partial match, e.g., 'NF-kappa B signaling' or 'PI3K-Akt')"
                            },
                            "include_relations": {
                                "type": "boolean",
                                "description": "Whether to include curated pathway regulatory relations (default: true)",
                                "default": True
                            }
                        },
                        "required": ["pathway_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_pathways_by_gene",
                    "description": "Find all pathways containing a specific gene and its role in each pathway",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "gene_symbol": {
                                "type": "string",
                                "description": "Gene symbol in uppercase (e.g., 'TP53', 'EGFR', 'VEGFA')"
                            }
                        },
                        "required": ["gene_symbol"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pathway_crosstalk",
                    "description": "Get genes shared between two pathways to understand pathway crosstalk and interactions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pathway1": {
                                "type": "string",
                                "description": "First pathway name"
                            },
                            "pathway2": {
                                "type": "string",
                                "description": "Second pathway name"
                            }
                        },
                        "required": ["pathway1", "pathway2"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        """
        Execute a tool by name with given arguments

        Args:
            tool_name: Name of the tool function
            tool_args: Dict of arguments to pass to the tool

        Returns:
            Tool execution result
        """
        if tool_name == "get_pathway_mechanism":
            return self.get_pathway_mechanism(**tool_args)
        elif tool_name == "search_pathways_by_gene":
            return self.search_pathways_by_gene(**tool_args)
        elif tool_name == "get_pathway_crosstalk":
            return self.get_pathway_crosstalk(**tool_args)
        elif tool_name == "list_available_pathways":
            return self.list_available_pathways(**tool_args)
        else:
            return {
                'error': f'Unknown tool: {tool_name}',
                'available_tools': ['get_pathway_mechanism', 'search_pathways_by_gene', 'get_pathway_crosstalk', 'list_available_pathways']
            }
