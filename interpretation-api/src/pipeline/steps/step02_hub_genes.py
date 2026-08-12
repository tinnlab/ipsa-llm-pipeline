"""
Step 2: Hub Gene Identification
Identifies network hub genes using protein-protein interaction analysis and provides biological interpretation
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.network_service import NetworkAnalysisService, HubGene, get_species_id
from src.agents.groq_client import GroqClient
from src.pipeline.fc_utils import fc_direction

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class Step02HubGenes:
    """
    Step 2: Hub Gene Identification

    Performs network topology analysis to identify hub genes based on protein-protein
    interaction networks from STRING database, then uses LLM to interpret their
    biological significance.

    Hybrid Approach:
    1. Computational: Query STRING for PPI network
    2. Computational: Calculate degree, betweenness, closeness centrality
    3. Computational: Rank genes by composite hub score
    4. LLM: Interpret biological significance of network hubs
    """

    def __init__(self):
        """Initialize Step 2"""
        self.step_number = 2
        self.step_name = 'Hub Gene Identification'
        self.dependencies = [1]  # Depends on Step 1 (pathway themes)

        self.network_service = NetworkAnalysisService(cache_enabled=True)

        # Initialize Groq LLM (uses settings from config)
        self.llm = GroqClient()

    def execute(
        self,
        genes: List[Dict],
        pathway_themes: Optional[List[Dict]] = None,
        organism: str = 'Homo sapiens',
        top_n: int = 20,
        min_string_score: int = 400,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Execute Step 2: Hub Gene Identification

        Args:
            genes: List of DE genes with gene_symbol, log2_fold_change, p_value (can be empty)
            pathway_themes: Optional pathway themes from Step 1 for context
            organism: Organism name (default: Homo sapiens)
            top_n: Number of top hub genes to identify (default: 20)
            min_string_score: Minimum STRING confidence score 0-999 (default: 400)

        Returns:
            Dictionary with network hubs and biological interpretation
        """
        logger.info('='*80)
        logger.info('STEP 2: HUB GENE IDENTIFICATION')
        logger.info('='*80)

        # Check if genes are provided
        if not genes or len(genes) == 0:
            logger.info('\n[Step 2 Skipped]')
            logger.info('  Reason: No differentially expressed genes provided')
            logger.info('  Hub gene analysis requires gene expression data')
            return {
                'network_hubs': [],
                'hub_interpretation': {},
                'master_regulators': [],
                'summary': 'Hub gene analysis was not performed because no differentially expressed genes were provided. This step requires gene expression data to perform protein-protein interaction network analysis.',
                'metadata': {
                    'skipped': True,
                    'reason': 'No genes provided'
                }
            }

        # Phase 1: Quantitative network analysis
        logger.info(f'\n[Phase 1] Network Topology Analysis')
        logger.info(f'  Input: {len(genes)} differentially expressed genes')
        logger.info(f'  Organism: {organism}')
        logger.info(f'  STRING confidence threshold: {min_string_score}/999')

        species_id = get_species_id(organism)

        network_hubs = self.network_service.calculate_network_hubs(
            genes=genes,
            species=species_id,
            top_n=top_n,
            min_score=min_string_score,
            context=context
        )

        if not network_hubs:
            logger.warning('No hub genes identified')
            return {
                'network_hubs': [],
                'hub_interpretation': {},
                'master_regulators': [],
                'summary': 'No network hubs could be identified from the input genes.'
            }

        # Display network hub summary
        logger.info(f'\n  Network Hub Genes Identified:')
        for i, hub in enumerate(network_hubs[:5], 1):
            logger.info(f'    {i}. {hub.gene}:')
            logger.info(f'       - Hub Score: {hub.hub_score:.3f}')
            logger.info(f'       - Degree: {hub.degree} (rank: {hub.degree_rank:.2f})')
            logger.info(f'       - Betweenness: {hub.betweenness:.3f} (rank: {hub.betweenness_rank:.2f})')
            logger.info(f'       - Fold Change: {hub.fold_change:+.2f}')

        # Select the DIRECTIONAL master-regulator hubs (top hubs within each
        # regulation direction) BEFORE interpretation so the LLM interprets
        # exactly the genes that become master regulators — a down-regulated
        # hub promoted here must not be left without a biological role.
        master_regulator_hubs = self._select_master_regulator_hubs(network_hubs)

        # Phase 2: LLM biological interpretation
        logger.info(f'\n[Phase 2] Biological Interpretation (LLM)')

        hub_interpretation = self._interpret_network_hubs(
            network_hubs=network_hubs,
            pathway_themes=pathway_themes,
            organism=organism,
            master_regulator_genes=[h.gene for h in master_regulator_hubs]
        )

        # Build master regulator records from the pre-selected directional hubs
        master_regulators = self._identify_master_regulators(
            master_regulator_hubs=master_regulator_hubs,
            interpretation=hub_interpretation
        )

        logger.info(f'\n  Master Regulators Identified:')
        for i, master in enumerate(master_regulators, 1):
            logger.info(f'    {i}. {master["gene"]} - {master["role"]}')

        # Generate summary
        summary = self._generate_summary(
            network_hubs=network_hubs,
            master_regulators=master_regulators,
            pathway_themes=pathway_themes
        )

        logger.info(f'\n[Step 2 Complete]')
        logger.info(f'  Hub genes identified: {len(network_hubs)}')
        logger.info(f'  Master regulators: {len(master_regulators)}')

        network_hubs_dicts = [self._hub_to_dict(hub) for hub in network_hubs]

        # Merge biological_role from LLM interpretations into hub dicts
        hub_interp_map = {
            h['gene']: h for h in hub_interpretation.get('hub_interpretations', [])
        }
        for hub_dict in network_hubs_dicts:
            interp = hub_interp_map.get(hub_dict['gene'], {})
            hub_dict['biological_role'] = interp.get('biological_role', '')

        return {
            'network_hubs': network_hubs_dicts,
            'hub_interpretation': hub_interpretation,
            'master_regulators': master_regulators,
            'summary': summary,
            'metadata': {
                'organism': organism,
                'species_id': species_id,
                'top_n': top_n,
                'min_string_score': min_string_score,
                'hubs_found': len(network_hubs)
            }
        }

    def _interpret_network_hubs(
        self,
        network_hubs: List[HubGene],
        pathway_themes: Optional[List[Dict]],
        organism: str,
        master_regulator_genes: Optional[List[str]] = None
    ) -> Dict:
        """Use LLM to interpret biological significance of network hubs"""

        # Build context from pathway themes if available
        themes_context = ''
        if pathway_themes:
            themes_context = '\n**Pathway Themes (from Step 1):**\n'
            for theme in pathway_themes[:5]:
                themes_context += f'- {theme.get("name", "Unknown")}: {theme.get("description", "")}\n'

        # Build network hubs data
        hubs_data = '\n**Network Hub Genes (ranked by centrality):**\n'
        for hub in network_hubs:
            hubs_data += f'- **{hub.gene}**: '
            hubs_data += f'Degree={hub.degree}, '
            hubs_data += f'Betweenness={hub.betweenness:.2f}, '
            hubs_data += f'Closeness={hub.closeness:.2f}, '
            hubs_data += f'FC={hub.fold_change:+.2f}, '
            hubs_data += f'p={hub.p_value:.2e}\n'

        system_prompt = """You are a molecular biology expert analyzing gene expression data with network topology information.

CRITICAL INSTRUCTIONS:
1. You are provided with QUANTITATIVE network hub genes identified through protein-protein interaction analysis
2. Explain the BIOLOGICAL SIGNIFICANCE of these network hubs
3. Ground your interpretation in the network metrics provided (degree, betweenness, closeness)
4. Explain WHY high degree/betweenness makes a gene important biologically
5. Connect network position to biological function
6. DO NOT invent network connections - only interpret the provided metrics

Return VALID JSON only."""

        # Interpret exactly the genes that become master regulators. These are
        # selected as top hubs WITHIN each regulation direction (up & down), so
        # the list is not simply the first 5 by hub score.
        if master_regulator_genes:
            interpret_target = (
                'Provide biological interpretation for the following master-regulator genes '
                '(selected as the top hubs within each regulation direction — up- and '
                f'down-regulated): {", ".join(master_regulator_genes)}.'
            )
        else:
            interpret_target = (
                'Provide biological interpretation for the FIRST 5 genes in the list above '
                '(these are the top 5 by hub score, which become the master regulators):'
            )

        user_prompt = f"""Analyze these network hub genes and explain their biological roles in {organism}:

{hubs_data}
{themes_context}

IMPORTANT: The genes above are already ranked by hub score (a composite metric of degree, betweenness, and closeness centrality).
{interpret_target}

Return JSON:
{{
  "hub_interpretations": [
    {{
      "gene": "GENE_NAME",
      "network_metrics": {{
        "degree": <copy from above>,
        "betweenness": <copy from above>,
        "closeness": <copy from above>,
        "fold_change": <copy from above>
      }},
      "biological_role": "Why this gene is central in biological networks (2-3 sentences)",
      "pathway_involvement": ["Pathway 1", "Pathway 2"],
      "mechanistic_importance": "How its network position (high degree/betweenness) relates to its biological function",
      "predicted_impact": "Expected downstream effects of its expression change"
    }}
  ],
  "network_interpretation": "Overall interpretation explaining the collective significance of these hub genes (3-4 sentences)"
}}"""

        try:
            # Get LLM interpretation
            response = self.llm.chat([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ], temperature=0.0, seed=42)

            # Parse JSON response (response is already a string)
            interpretation = json.loads(response)

            logger.info(f'  LLM interpretation generated for {len(interpretation.get("hub_interpretations", []))} hub genes')

            return interpretation

        except json.JSONDecodeError as e:
            logger.error(f'  Failed to parse LLM response as JSON: {e}')
            logger.error(f'  Response: {response[:200]}...')
            return {'hub_interpretations': [], 'network_interpretation': 'Failed to generate interpretation'}
        except Exception as e:
            logger.error(f'  Error calling LLM: {e}')
            return {'hub_interpretations': [], 'network_interpretation': 'Error generating interpretation'}

    def _select_master_regulator_hubs(
        self,
        network_hubs: List[HubGene],
        k_per_direction: int = 3,
        min_total: int = 5,
    ) -> List[HubGene]:
        """
        Select master-regulator hubs as DIRECTIONAL sets (up- and down-regulated).

        Historically the top-N hubs by composite hub_score were returned. Because
        STRING PPI centrality structurally favours large, densely-interconnected
        modules — in practice heavily-studied proliferation / cell-cycle programs
        — this collapsed the master-regulator list to a single (up-regulated)
        direction. Down-regulated / metabolic hubs never surfaced, even when they
        sat on the strongest |NES| axis of the dataset.

        We now take the top ``k_per_direction`` hubs by hub_score WITHIN each
        regulation direction (up vs down by fold change), then backfill by
        hub_score up to ``min_total`` when one direction is sparse or absent — so
        genuinely one-directional datasets keep their previous coverage. The
        returned list is ordered best-first by hub_score; each hub carries its own
        direction, used downstream to render up- and down-regulated hub sets.
        """
        # network_hubs arrives sorted by hub_score desc (see network_service),
        # so these per-direction slices are already ranked. Use fc_direction (with its
        # epsilon dead-zone) rather than a bare sign test, matching the rest of the
        # codebase — a near-zero fold change is 'neutral', not a confident up/down.
        up_hubs = [h for h in network_hubs if fc_direction(h.fold_change) == 'up']
        down_hubs = [h for h in network_hubs if fc_direction(h.fold_change) == 'down']

        selected: List[HubGene] = []
        seen: set = set()

        def _add(hub: HubGene) -> None:
            if hub.gene not in seen:
                selected.append(hub)
                seen.add(hub.gene)

        for hub in up_hubs[:k_per_direction]:
            _add(hub)
        for hub in down_hubs[:k_per_direction]:
            _add(hub)

        # Backfill by hub_score when a direction is sparse/absent so single-
        # direction datasets keep the previous number of master regulators.
        if len(selected) < min_total:
            for hub in network_hubs:  # already hub_score-sorted
                if len(selected) >= min_total:
                    break
                _add(hub)

        selected.sort(key=lambda h: h.hub_score, reverse=True)
        return selected

    def _identify_master_regulators(
        self,
        master_regulator_hubs: List[HubGene],
        interpretation: Dict
    ) -> List[Dict]:
        """
        Build master-regulator records from the pre-selected directional hubs.

        The hubs are chosen by :meth:`_select_master_regulator_hubs` so both the
        up-regulated and down-regulated axes are represented. Each record is
        tagged with ``direction`` ('up' | 'down') so the report can present
        separate up- and down-regulated (e.g. metabolic) hub sets rather than
        collapsing to proliferation hubs.
        """
        master_regulators = []

        hub_interpretations = {
            h['gene']: h
            for h in interpretation.get('hub_interpretations', [])
        }

        for hub in master_regulator_hubs:
            interp = hub_interpretations.get(hub.gene, {})

            master_regulators.append({
                'gene': hub.gene,
                'hub_score': round(hub.hub_score, 3),
                'degree': hub.degree,
                'fold_change': round(hub.fold_change, 2),
                'direction': fc_direction(hub.fold_change),  # 'up' | 'down' | 'neutral'
                'role': interp.get('biological_role', 'Key network hub'),
                'pathways': interp.get('pathway_involvement', [])
            })

        return master_regulators

    def _generate_summary(
        self,
        network_hubs: List[HubGene],
        master_regulators: List[Dict],
        pathway_themes: Optional[List[Dict]]
    ) -> str:
        """Generate concise summary of hub gene analysis"""

        summary_parts = []

        # Hub gene overview
        if network_hubs:
            top_genes = ', '.join([h.gene for h in network_hubs[:3]])
            summary_parts.append(
                f'Network analysis identified {len(network_hubs)} hub genes, '
                f'with {top_genes} showing highest centrality.'
            )

        # Master regulators
        if master_regulators:
            master_genes = ', '.join([m['gene'] for m in master_regulators[:3]])
            summary_parts.append(
                f'{len(master_regulators)} master regulators were identified: {master_genes}.'
            )

        # Pathway context
        if pathway_themes:
            theme_names = ', '.join([t.get('name', 'Unknown') for t in pathway_themes[:2]])
            summary_parts.append(
                f'These hub genes are central to pathway themes including {theme_names}.'
            )

        return ' '.join(summary_parts) if summary_parts else 'No hub genes identified.'

    def _hub_to_dict(self, hub: HubGene) -> Dict:
        """Convert HubGene object to dictionary"""
        return {
            'gene': hub.gene,
            'fold_change': round(hub.fold_change, 3),
            'p_value': hub.p_value,
            'degree': hub.degree,
            'betweenness': round(hub.betweenness, 3),
            'closeness': round(hub.closeness, 3),
            'hub_score': round(hub.hub_score, 3),
            'degree_rank': round(hub.degree_rank, 3),
            'betweenness_rank': round(hub.betweenness_rank, 3),
            'closeness_rank': round(hub.closeness_rank, 3)
        }


def main():
    """Command-line interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Step 2: Hub Gene Identification')
    parser.add_argument('input_file', help='Input JSON file with differentially expressed genes')
    parser.add_argument('-o', '--output', help='Output JSON file (default: hub_results.json)')
    parser.add_argument('-n', '--top-n', type=int, default=10, help='Number of top hubs (default: 10)')
    parser.add_argument('--min-score', type=int, default=400, help='Minimum STRING score (default: 400)')

    args = parser.parse_args()

    # Load input
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    # Extract genes and organism
    genes = input_data.get('differentially_expressed_genes', [])
    metadata = input_data.get('metadata', {})
    organism = metadata.get('organism', 'Homo sapiens')

    # Run Step 2
    step2 = Step02HubGenes()
    results = step2.execute(
        genes=genes,
        organism=organism,
        top_n=args.top_n,
        min_string_score=args.min_score
    )

    # Save results
    output_file = args.output or 'hub_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f'\n{"="*80}')
    logger.info(f'Results saved to: {output_file}')
    logger.info(f'{"="*80}')


if __name__ == '__main__':
    main()
