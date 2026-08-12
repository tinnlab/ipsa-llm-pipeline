"""Pathway Commons API client for retrieving gene-gene interactions"""
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class PCInteraction:
    source: str
    interaction_type: str
    target: str


class PathwayCommonsService:
    BASE_URL = "https://www.pathwaycommons.org/pc2"

    def __init__(self):
        self.cache = {}

    def get_interactions_between(
        self,
        gene_symbols: List[str],
        limit: int = 50
    ) -> List[PCInteraction]:
        """
        Get curated interactions between a set of genes using pathsbetween query.

        Args:
            gene_symbols: List of gene symbols (e.g., ["NDUFS1", "UQCRC1"])
            limit: Max genes to query (API can be slow with too many)

        Returns:
            List of PCInteraction (source, type, target)
        """
        if not gene_symbols or len(gene_symbols) < 2:
            return []

        # Limit genes to avoid overloading API
        query_genes = gene_symbols[:limit]
        cache_key = ",".join(sorted(query_genes))
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            url = f"{self.BASE_URL}/graph"
            params = {
                "source": ",".join(query_genes),
                "kind": "pathsbetween",
                "format": "SIF"
            }
            response = requests.get(url, params=params, timeout=60)
            if not response.ok or not response.text.strip():
                return []

            interactions = self._parse_sif(response.text, query_genes)
            self.cache[cache_key] = interactions
            return interactions

        except Exception as e:
            print(f"  Warning: Pathway Commons API error: {e}")
            return []

    def _parse_sif(self, sif_text: str, gene_set: List[str]) -> List[PCInteraction]:
        """Parse SIF format, filter to mechanistic interactions involving query genes"""
        gene_set_upper = set(g.upper() for g in gene_set)

        # Prioritize mechanistic interaction types
        mechanistic_types = {
            'controls-expression-of',
            'controls-state-change-of',
            'controls-phosphorylation-of',
            'controls-production-of',
            'controls-transport-of',
            'consumption-controlled-by',
            'in-complex-with',
            'reacts-with',
            'used-to-produce',
        }

        interactions = []
        for line in sif_text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) != 3:
                continue
            source, itype, target = parts

            # Skip non-gene entries (chemicals start with CHEBI:)
            if source.startswith('CHEBI:') or target.startswith('CHEBI:'):
                continue

            # Only keep mechanistic types (skip generic 'interacts-with')
            if itype not in mechanistic_types:
                continue

            # At least one gene must be in our query set
            if source.upper() in gene_set_upper or target.upper() in gene_set_upper:
                interactions.append(PCInteraction(
                    source=source,
                    interaction_type=itype,
                    target=target
                ))

        return interactions
