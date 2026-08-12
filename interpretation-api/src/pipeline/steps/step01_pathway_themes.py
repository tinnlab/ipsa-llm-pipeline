"""
Step 1: Pathway Theme Clustering
Clusters pathways by gene overlap and uses LLM to assign biological theme names
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.pathway_clustering_service import PathwayClusteringService, _pathway_fdr
from src.agents.groq_client import GroqClient
from src.config import settings
from src.pipeline.json_utils import parse_llm_json
from src.pipeline.fc_utils import to_float

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Robust LLM-JSON parsing lives in the shared json_utils module (balanced-brace
# extraction, fence stripping, size cap). Re-exported here for the existing call sites.
_parse_llm_json = parse_llm_json


def _coerce_int(val) -> Optional[int]:
    """Coerce an LLM-supplied cluster_number to int (accepts 1, "1", 1.0, "1.0")."""
    if isinstance(val, bool):  # bool is an int subclass — reject explicitly
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val) if val.is_integer() else None
    try:
        s = str(val).strip()
        return int(s)
    except (TypeError, ValueError):
        try:
            f = float(s)
            return int(f) if f.is_integer() else None
        except (TypeError, ValueError):
            return None


def _fallback_theme_name(index: int, cluster: Dict) -> str:
    """Informative fallback theme name including top shared genes."""
    genes = ', '.join(cluster.get('shared_genes', [])[:5])
    base = f'Pathway Cluster {index + 1}'
    return f'{base} (shared genes: {genes})' if genes else base


class Step01PathwayThemes:
    """
    Step 1: Pathway Theme Clustering

    Hybrid approach combining computational clustering with LLM interpretation:
    1. Computational: Cluster pathways by gene overlap (Jaccard similarity)
    2. Computational: Find connected components in similarity graph
    3. Computational: Calculate cluster statistics
    4. LLM: Assign biological theme names to clusters
    5. LLM: Explain why pathways cluster together

    This ensures clustering is data-driven (reproducible) while theme naming
    provides biological context (interpretable).
    """

    def __init__(self):
        """Initialize Step 1"""
        # Clustering knobs come from config (env-overridable). Defaults use Markov Clustering
        # (aPEAR/EnrichmentMap-recommended) over the DE-restricted Jaccard graph.
        self.clustering_service = PathwayClusteringService(
            jaccard_threshold=settings.CLUSTER_JACCARD_THRESHOLD,  # connected_components cutoff
            min_cluster_size=settings.CLUSTER_MIN_SIZE,
            method=settings.CLUSTER_METHOD,
            inflation=settings.CLUSTER_INFLATION,
            similarity_floor=settings.CLUSTER_SIMILARITY_FLOOR,
            gene_source=settings.CLUSTER_GENE_SOURCE,
        )

        # Initialize Groq LLM (uses settings from config)
        self.llm = GroqClient()

    def execute(
        self,
        pathways: List[Dict],
        genes: List[Dict],
        organism: str = 'Homo sapiens',
        context: Optional[Dict] = None,
        significance_threshold: float = 0.05,
        jaccard_threshold: float = 0.25
    ) -> Dict:
        """
        Execute Step 1: Pathway Theme Clustering

        Args:
            pathways: List of enriched pathways with p-values and gene lists
            genes: List of differentially expressed genes
            organism: Organism name (default: Homo sapiens)
            context: Optional experimental context (tissue, disease, etc.)
            significance_threshold: FDR threshold for pathway significance (default: 0.05)
            jaccard_threshold: Minimum Jaccard similarity to connect pathways (default: 0.25)

        Returns:
            Dictionary with themes, ungrouped pathways, and summary
        """
        logger.info('='*80)
        logger.info('STEP 1: PATHWAY THEME CLUSTERING')
        logger.info('='*80)

        logger.info(f'\n[Input]')
        logger.info(f'  Pathways: {len(pathways)}')
        logger.info(f'  Genes: {len(genes)}')
        logger.info(f'  Organism: {organism}')
        logger.info(f'  Significance threshold: FDR < {significance_threshold}')

        # Filter to significant pathways only
        significant_pathways = [
            p for p in pathways
            if (p.get('p_value_fdr') or p.get('pValueFDR') or 1.0) < significance_threshold
        ]

        logger.info(f'\n[Filtering]')
        logger.info(f'  Significant pathways (FDR < {significance_threshold}): {len(significant_pathways)}')

        if not significant_pathways:
            logger.warning('  No significant pathways available for clustering')
            return {
                'themes': [],
                'ungrouped': [],
                'themes_summary': 'No statistically significant pathways available for clustering.',
                'metadata': {
                    'total_pathways': len(pathways),
                    'significant_pathways': 0,
                    'cluster_count': 0
                }
            }

        # Phase 1: Computational clustering
        logger.info(f'\n[Phase 1] Computational Clustering by Gene Overlap')
        logger.info(f'  Jaccard threshold: {jaccard_threshold}')

        clustering_result = self.clustering_service.cluster_pathways_by_gene_overlap(
            pathways=significant_pathways,
            genes=genes,
            jaccard_threshold=jaccard_threshold
        )

        clusters = clustering_result['clusters']
        singletons = clustering_result['singletons']
        metadata = clustering_result['metadata']

        if not clusters:
            logger.warning('  No clusters formed (all pathways are singletons)')
            # Still filter off-tissue singletons — this is the common small-input case.
            singletons, singleton_filtered = self._filter_singletons(singletons, context)
            return {
                'themes': [],
                'ungrouped': singletons,
                'themes_summary': 'Pathways do not show sufficient gene overlap for clustering.',
                'metadata': {**metadata, 'tissue_filtered_count': singleton_filtered}
            }

        # Display cluster summary
        logger.info(f'\n  Clustering Results:')
        for i, cluster in enumerate(clusters, 1):
            logger.info(f'    Cluster {i}:')
            logger.info(f'      - Pathways: {cluster["pathway_count"]}')
            logger.info(f'      - Shared genes: {cluster["shared_gene_count"]}')
            logger.info(f'      - Avg overlap: {cluster["avg_jaccard_overlap"]*100:.1f}%')
            logger.info(f'      - Significance: {cluster["significance"]}')

        # Phase 2: LLM naming and interpretation
        logger.info(f'\n[Phase 2] LLM Theme Naming and Interpretation')

        themes = self._name_clusters_with_llm(
            clusters=clusters,
            context=context,
            organism=organism
        )

        # Phase 3: Tissue-specificity filtering (remove off-tissue pathways from themes)
        logger.info(f'\n[Phase 3] Tissue-Specificity Filtering')

        cleaned_themes, filtered_count = self._filter_off_tissue_pathways(
            themes=themes,
            context=context
        )

        # Ungrouped singletons bypass theme-level filtering — filter them too.
        singletons, singleton_filtered = self._filter_singletons(singletons, context)
        filtered_count += singleton_filtered
        logger.info(f'  Filtered {filtered_count} off-tissue pathways ({singleton_filtered} from ungrouped)')
        if filtered_count > 0:
            logger.info(f'  Themes after filtering:')
            for theme in cleaned_themes:
                logger.info(f'    - {theme["name"]}: {theme["pathway_count"]} pathways')

        # Generate overall summary
        summary = self._generate_summary(cleaned_themes, singletons)

        logger.info(f'\n[Step 1 Complete]')
        logger.info(f'  Themes identified: {len(cleaned_themes)}')
        logger.info(f'  Ungrouped pathways: {len(singletons)}')

        return {
            'themes': cleaned_themes,
            'ungrouped': singletons,
            'themes_summary': summary,
            'metadata': {
                **metadata,
                'organism': organism,
                'context': context,
                'significance_threshold': significance_threshold,
                'jaccard_threshold': jaccard_threshold,
                'tissue_filtered_count': filtered_count
            }
        }

    def _filter_singletons(self, singletons: List[Dict], context: Optional[Dict]):
        """Apply tissue-specificity filtering to ungrouped singleton pathways.

        Singletons bypass theme-level filtering, so off-tissue/off-disease singletons
        (e.g. Bladder cancer, Small cell lung cancer, Salmonella infection in a liver
        study) would otherwise reach Step 3 unfiltered. Wrap them as a pseudo-theme and
        reuse the existing filter. Returns (kept_singletons, filtered_count).
        """
        if not singletons:
            return singletons, 0
        wrapped, filtered = self._filter_off_tissue_pathways(
            themes=[{'name': 'Ungrouped pathways', 'pathways': singletons,
                     'pathway_count': len(singletons), 'avg_p_value_fdr': 1.0}],
            context=context
        )
        return (wrapped[0]['pathways'] if wrapped else []), filtered

    def _chat_json(self, messages: List[Dict]) -> Optional[str]:
        """Chat expecting a JSON object.

        Requests guided JSON output (``response_format``) but degrades gracefully:
        if the provider/endpoint rejects that parameter (raising), retry once
        without it so we don't lose the whole call and regress to fallback names.
        """
        try:
            return self.llm.chat(
                messages, temperature=0.0, seed=42,
                response_format={'type': 'json_object'}
            )
        except Exception as e:
            logger.warning(f'  response_format not accepted ({e}); retrying without it')
            return self.llm.chat(messages, temperature=0.0, seed=42)

    def _name_clusters_with_llm(
        self,
        clusters: List[Dict],
        context: Optional[Dict],
        organism: str
    ) -> List[Dict]:
        """Use LLM to assign biological theme names to clusters"""

        # Build experimental context string from ALL context fields dynamically
        context_str = ''
        if context:
            parts = []
            # Skip internal/technical fields
            skip_fields = {'datasetId', 'dataset_id', 'organism'}
            for key, value in context.items():
                if key in skip_fields:
                    continue
                if value and str(value).strip():
                    # Format the key for display
                    display_key = key.replace('_', ' ').title()
                    parts.append(f"{display_key}: {value}")
            context_str = ', '.join(parts) if parts else ''

        # Build cluster descriptions for LLM
        cluster_descriptions = []
        for i, cluster in enumerate(clusters, 1):
            shared_genes_str = ', '.join(cluster['shared_genes'][:20])
            if len(cluster['shared_genes']) > 20:
                shared_genes_str += '...'

            # Regulation direction MUST be passed to the LLM: without it the model
            # infers biology with no sign and inverts blurbs (e.g. calling a
            # down-regulated beta-oxidation cluster a "shift toward oxidative fuel
            # utilization"). The direction and signed fold changes are already computed.
            direction = cluster.get('dominant_direction', 'mixed')
            direction_label = {
                'up': 'UP-regulated (these genes are INCREASED in disease vs control)',
                'down': 'DOWN-regulated (these genes are DECREASED in disease vs control)',
                'mixed': 'MIXED direction (genes go both up and down)',
            }.get(direction, 'MIXED direction (genes go both up and down)')

            fc_bits = []
            for g in (cluster.get('key_genes_with_fc') or [])[:6]:
                fc = to_float(g.get('fold_change'))
                if fc is not None:
                    fc_bits.append(f"{g.get('gene')} ({fc:+.2f})")
            fc_str = ', '.join(fc_bits) if fc_bits else 'n/a'

            # Representative pathway (most significant member) seeds the naming, aPEAR-style.
            rep = cluster.get('representative')
            rep_line = f"\n- **Representative pathway**: {rep}" if rep else ''

            cluster_desc = f"""**Cluster {i}:**
- **Pathways** ({cluster['pathway_count']} pathways): {', '.join(cluster['pathway_names'])}{rep_line}
- **Shared genes** ({cluster['shared_gene_count']} genes): {shared_genes_str}
- **Regulation direction**: {direction_label}
- **Key gene fold changes**: {fc_str}
- **Average gene overlap**: {cluster['avg_jaccard_overlap']*100:.1f}%
- **Average FDR**: {cluster['avg_p_value_fdr']:.2e}
- **Significance**: {cluster['significance']}"""

            cluster_descriptions.append(cluster_desc)

        system_prompt = """You are a bioinformatics expert naming pathway clusters based on gene overlap analysis.

CRITICAL: These clusters are formed by COMPUTATIONAL analysis of gene overlap (Jaccard similarity), NOT by you.

Your task is to:
1. Assign a descriptive biological theme name to each cluster
2. Explain WHY these pathways cluster together based on shared genes
3. Describe the biological role of the shared genes in context

IMPORTANT - Theme naming principles:
- Name themes by BIOLOGICAL PROCESS or MOLECULAR MECHANISM, not disease names
- Base names on SHARED GENES and their functions
- Example: If "Alzheimer disease", "Parkinson disease", "OXPHOS" cluster and share mitochondrial genes → name "Mitochondrial oxidative phosphorylation", NOT "Neurodegenerative Diseases"
- Use molecular mechanisms evident from shared genes
- Focus on what the genes DO (their biological function), not which disease pathways they appear in

CRITICAL - Respect the regulation direction:
- Each cluster states its **Regulation direction** (UP / DOWN / MIXED) and signed fold changes.
- Your description MUST match that direction. Do NOT describe a DOWN-regulated program as
  increased, enhanced, or "a shift toward" that activity — down-regulation means the process is
  SUPPRESSED/REDUCED. Likewise do not describe an UP-regulated program as reduced.
- Example: a DOWN-regulated fatty-acid beta-oxidation cluster indicates SUPPRESSED fatty-acid
  catabolism, not increased reliance on it.

Return VALID JSON only (no markdown code blocks, no explanatory text)."""

        user_prompt = f"""Name and describe these pathway clusters formed by gene overlap analysis:

{f"**Experimental Context:** {context_str}" if context_str else ""}

{chr(10).join(cluster_descriptions)}

For each cluster, provide:
{{
  "themes": [
    {{
      "cluster_number": 1,
      "name": "Biological process theme name (not disease name)",
      "description": "Why these pathways cluster together based on shared genes and biological function (2-3 sentences)",
      "key_genes": ["list 3-5 most important shared genes"],
      "biological_context": "How this theme relates to {context_str or 'the biological system'} (1-2 sentences)"
    }}
  ],
  "themes_summary": "Overall summary of pathway clustering patterns and their biological meaning (3-4 sentences)"
}}

IMPORTANT:
- Cluster numbers must match (1, 2, 3, etc.)
- Theme names must reflect MOLECULAR MECHANISMS evident from shared genes
- Use shared genes to justify why pathways belong together"""

        response = None
        try:
            # Get LLM interpretation (force JSON output to avoid fenced/prose wrappers)
            response = self._chat_json([
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ])

            # Parse JSON response (tolerant of fences / prose / partial output)
            result = _parse_llm_json(response)

            logger.info(f'  LLM naming complete: {len(result.get("themes", []))} themes named')

            # Merge computational clusters with LLM names
            themes = self._merge_clusters_with_names(clusters, result.get('themes', []))

            return themes

        except json.JSONDecodeError as e:
            logger.error(f'  Failed to parse LLM response as JSON: {e}')
            logger.error(f'  Response: {str(response)[:200]}...')

            # Fallback: use generic names
            return self._generate_fallback_themes(clusters)

        except Exception as e:
            logger.error(f'  Error calling LLM: {e}')
            return self._generate_fallback_themes(clusters)

    def _merge_clusters_with_names(
        self,
        clusters: List[Dict],
        llm_themes: List[Dict]
    ) -> List[Dict]:
        """Merge computational clusters with LLM-generated theme names"""

        themes = []

        # Tolerant matching: index LLM themes by coerced cluster_number so
        # string ("1") or float (1.0) numbers still match. Only fall back to
        # positional matching when NO theme carries a usable cluster_number and the
        # counts line up — mixing number- and position-matching could assign one
        # theme to two clusters and drop another.
        by_number = {}
        for t in llm_themes:
            num = _coerce_int(t.get('cluster_number'))
            if num is not None and num not in by_number:
                by_number[num] = t
        positional_ok = (len(llm_themes) == len(clusters)) and not by_number

        for i, cluster in enumerate(clusters):
            llm_theme = by_number.get(i + 1)
            if llm_theme is None and positional_ok:
                llm_theme = llm_themes[i]

            # Require a usable (non-empty) name; otherwise fall back
            if llm_theme and str(llm_theme.get('name') or '').strip():
                theme = {
                    'cluster_number': i + 1,
                    'name': llm_theme.get('name'),
                    'description': llm_theme.get('description', ''),
                    # Significance is a COMPUTED score (FDR + NES effect size), never the
                    # LLM's free-text judgment — the model would editorialize and invert
                    # the ranking (e.g. tag a weak OXPHOS cluster HIGH over the dominant
                    # cell-cycle cluster). Always use the computed cluster value.
                    'significance': cluster['significance'],
                    'key_genes': llm_theme.get('key_genes', cluster['shared_genes'][:5]),
                    'biological_context': llm_theme.get('biological_context', ''),
                    # Include computational data
                    'pathways': cluster['pathways'],
                    'pathway_count': cluster['pathway_count'],
                    'shared_genes': cluster['shared_genes'],
                    'shared_gene_count': cluster['shared_gene_count'],
                    'avg_jaccard_overlap': cluster['avg_jaccard_overlap'],
                    'avg_p_value_fdr': cluster['avg_p_value_fdr'],
                    # NEW: Add gene fold changes and direction
                    'key_genes_with_fc': cluster.get('key_genes_with_fc', []),
                    'dominant_direction': cluster.get('dominant_direction', 'mixed')
                }
            else:
                # Fallback if LLM didn't provide a usable theme for this cluster
                theme = {
                    'cluster_number': i + 1,
                    'name': _fallback_theme_name(i, cluster),
                    'description': f'Cluster of {cluster["pathway_count"]} pathways sharing {cluster["shared_gene_count"]} genes',
                    'significance': cluster['significance'],
                    'key_genes': cluster['shared_genes'][:5],
                    'biological_context': '',
                    'pathways': cluster['pathways'],
                    'pathway_count': cluster['pathway_count'],
                    'shared_genes': cluster['shared_genes'],
                    'shared_gene_count': cluster['shared_gene_count'],
                    'avg_jaccard_overlap': cluster['avg_jaccard_overlap'],
                    'avg_p_value_fdr': cluster['avg_p_value_fdr'],
                    # NEW: Add gene fold changes and direction
                    'key_genes_with_fc': cluster.get('key_genes_with_fc', []),
                    'dominant_direction': cluster.get('dominant_direction', 'mixed')
                }

            themes.append(theme)

        return themes

    def _generate_fallback_themes(self, clusters: List[Dict]) -> List[Dict]:
        """Generate generic theme names if LLM fails"""

        logger.warning('  Using fallback theme names (LLM failed)')

        themes = []
        for i, cluster in enumerate(clusters, 1):
            themes.append({
                'cluster_number': i,
                'name': _fallback_theme_name(i - 1, cluster),
                'description': f'Cluster of {cluster["pathway_count"]} pathways sharing {cluster["shared_gene_count"]} genes with {cluster["avg_jaccard_overlap"]*100:.1f}% average overlap',
                'significance': cluster['significance'],
                'key_genes': cluster['shared_genes'][:5],
                'biological_context': 'LLM interpretation unavailable',
                'pathways': cluster['pathways'],
                'pathway_count': cluster['pathway_count'],
                'shared_genes': cluster['shared_genes'],
                'shared_gene_count': cluster['shared_gene_count'],
                'avg_jaccard_overlap': cluster['avg_jaccard_overlap'],
                'avg_p_value_fdr': cluster['avg_p_value_fdr'],
                # Carry regulation direction on the failure path too, so downstream steps
                # (which key off dominant_direction) don't silently regress to 'mixed'
                # and reintroduce the direction-inversion bug when LLM naming fails.
                'key_genes_with_fc': cluster.get('key_genes_with_fc', []),
                'dominant_direction': cluster.get('dominant_direction', 'mixed')
            })

        return themes

    def _filter_off_tissue_pathways(
        self,
        themes: List[Dict],
        context: Optional[Dict]
    ) -> tuple[List[Dict], int]:
        """
        Use LLM to filter off-tissue pathways from themes after clustering.

        This intelligently removes pathways from irrelevant tissues while keeping
        pathways that are biologically relevant to the experimental context.

        Args:
            themes: List of themes with pathways
            context: Experimental context (tissue, disease, etc.)

        Returns:
            Tuple of (cleaned_themes, filtered_count)
        """

        # Build context string from ALL context fields dynamically
        context_str = ''
        if context:
            parts = []
            # Skip internal/technical fields
            skip_fields = {'datasetId', 'dataset_id', 'organism'}
            for key, value in context.items():
                if key in skip_fields:
                    continue
                if value and str(value).strip():
                    # Format the key for display
                    display_key = key.replace('_', ' ').title()
                    parts.append(f"{display_key}: {value}")
            context_str = ', '.join(parts) if parts else ''

        if not context_str:
            logger.info('  No experimental context provided - skipping tissue filtering')
            return themes, 0

        logger.info(f'  Experimental context: {context_str}')

        cleaned_themes = []
        total_filtered = 0

        for theme in themes:
            original_pathways = theme.get('pathways', [])
            theme_name = theme.get('name', 'Unknown')

            # Build pathway list for LLM (all pathways at once)
            pathway_list = []
            for i, pw in enumerate(original_pathways, 1):
                pw_name = pw.get('name', 'Unknown')
                pw_fdr = pw.get('p_value_fdr', 1.0)
                pw_genes = pw.get('gene_count', 0)
                pathway_list.append(f"{i}. {pw_name} (FDR={pw_fdr:.2e}, {pw_genes} genes)")

            pathways_text = '\n'.join(pathway_list)

            # Build LLM prompt
            system_prompt = """You are a bioinformatics expert reviewing pathway relevance for tissue-specific analysis.

Your task: Filter out pathways that are NOT relevant to the experimental tissue/disease context.

TISSUE-SPECIFICITY DEFINITION:
- Tissue-specific pathway = pathway name explicitly mentions an organ/tissue/cell type (e.g., "PANCREATIC", "HEPATOCELLULAR", "CARDIAC MUSCLE", "DOPAMINERGIC SYNAPSE")
- Disease-specific pathway = pathway name mentions a specific disease/disorder (e.g., "PARKINSON DISEASE", "ALZHEIMER DISEASE", "HUNTINGTON DISEASE")
- General pathway = pathway describes a biological mechanism or process without naming a specific organ or disease (e.g., "PI3K-AKT SIGNALING", "CELL CYCLE", "OXIDATIVE PHOSPHORYLATION")

FILTERING RULES (apply in order):

1. **FILTER organ/tissue-named pathways from different tissues**
   - If pathway name contains organ/tissue different from target → FILTER
   - Examples: PANCREATIC CANCER, HEPATOCELLULAR CARCINOMA, GLIOMA in lung studies
   - Examples: CARDIAC MUSCLE, DOPAMINERGIC SYNAPSE, GASTRIC ACID in lung studies
   - Rule: Be STRICT - organ-named pathways are almost always tissue-specific

2. **FILTER disease-named pathways that don't match target disease** (NEW - STRICT RULE)
   - If pathway name contains disease/disorder keywords (e.g., "disease", "syndrome", "disorder") → check match
   - If disease pathway does NOT match target disease/tissue → FILTER
   - Examples: "Parkinson disease", "Huntington disease", "Alzheimer disease" in lung adenocarcinoma study → FILTER
   - Examples: "Type II diabetes mellitus", "Rheumatoid arthritis" in lung adenocarcinoma study → FILTER
   - Exception: Disease EXACTLY matches target (e.g., "Lung adenocarcinoma" pathway in lung adenocarcinoma study) → KEEP
   - Rationale: Disease pathways represent specific pathological states in specific organs/systems. Even if they share molecular mechanisms (e.g., mitochondrial dysfunction), they describe different disease contexts and should be filtered.
   - IMPORTANT: Do NOT keep disease pathways just because they share genes or mechanisms. Shared genes do NOT justify keeping off-disease pathways.

3. **KEEP general biological mechanisms**
   - Pathways describing processes without organ names or disease names → KEEP
   - Examples: PI3K-AKT SIGNALING, CELL CYCLE, P53 SIGNALING, OXIDATIVE PHOSPHORYLATION
   - Examples: PATHWAYS IN CANCER (general), APOPTOSIS, AUTOPHAGY, MITOCHONDRIAL DYSFUNCTION
   - Examples: DNA REPLICATION, RIBOSOME, SPLICEOSOME, NUCLEOTIDE METABOLISM

4. **KEEP pathways for the target tissue/disease**
   - If pathway matches target tissue/disease → KEEP
   - Example: RENAL CELL CARCINOMA in kidney studies → KEEP
   - Example: NON-SMALL CELL LUNG CANCER in lung studies → KEEP

IMPORTANT:
- Sharing genes (e.g., KRAS, TP53, MAPK1, mitochondrial genes) does NOT make an organ-specific or disease-specific pathway relevant
- Gene overlap is expected across diseases but does NOT justify keeping off-target disease pathways
- Be STRICT with disease pathways - they should only be kept if they match the exact target disease

Return VALID JSON only (no markdown, no explanatory text)."""

            user_prompt = f"""Review these pathways from the theme "{theme_name}" for relevance to the experimental context.

**Experimental Context:** {context_str}

**Pathways to Review:**
{pathways_text}

For each pathway, decide if it should be KEPT or FILTERED based on tissue-specificity rules.

Return JSON:
{{
  "filtered_pathways": [
    {{
      "pathway_name": "exact pathway name",
      "decision": "KEEP" or "FILTER",
      "rationale": "brief explanation citing specific rule (1 sentence)"
    }}
  ]
}}

DECISION CRITERIA:
- FILTER: Pathway name contains organ/tissue different from "{context_str}" (e.g., PANCREATIC, HEPATOCELLULAR, GLIOMA, CARDIAC, DOPAMINERGIC)
- FILTER: Pathway name contains disease/disorder keywords AND does NOT match target disease in "{context_str}" (e.g., Parkinson disease, Huntington disease, Alzheimer disease, Type II diabetes)
- KEEP: Pathway describes general mechanism without organ/disease names (e.g., PI3K-AKT SIGNALING, CELL CYCLE, OXIDATIVE PHOSPHORYLATION, DNA REPLICATION)
- KEEP: Pathway matches target tissue/disease in "{context_str}"

STRICT RULES:
1. When pathway name explicitly mentions a different organ/tissue → ALWAYS FILTER (even if it shares genes)
2. When pathway name explicitly mentions a different disease/disorder → ALWAYS FILTER (even if it shares molecular mechanisms like mitochondrial dysfunction)
3. Gene overlap or mechanistic similarity does NOT justify keeping off-target disease pathways
4. Be STRICT - only keep disease pathways that exactly match the target disease context"""

            filtered_pathways = []

            try:
                # Get LLM decision for all pathways (force JSON output)
                response = self._chat_json([
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ])

                # Parse JSON response (tolerant of fences / prose / partial output)
                result = _parse_llm_json(response)

                # Build decision map
                decisions = {}
                for item in result.get('filtered_pathways', []):
                    pw_name = item.get('pathway_name', '')
                    decision = item.get('decision', 'KEEP')
                    rationale = item.get('rationale', '')
                    decisions[pw_name] = {'decision': decision, 'rationale': rationale}

                # Filter pathways based on LLM decisions
                for pw in original_pathways:
                    pw_name = pw.get('name', '')
                    decision_info = decisions.get(pw_name, {'decision': 'KEEP', 'rationale': 'No decision'})

                    if decision_info['decision'] == 'KEEP':
                        filtered_pathways.append(pw)
                    else:
                        logger.info(f'    Filtered: {pw_name} - {decision_info["rationale"]}')
                        total_filtered += 1

            except json.JSONDecodeError as e:
                logger.error(f'  Failed to parse LLM filtering response: {e}')
                # Fallback: keep all pathways if LLM fails
                filtered_pathways = original_pathways

            except Exception as e:
                logger.error(f'  Error filtering theme: {e}')
                # Fallback: keep all pathways if LLM fails
                filtered_pathways = original_pathways

            # Recalculate theme statistics
            if filtered_pathways:
                cleaned_theme = theme.copy()
                cleaned_theme['pathways'] = filtered_pathways
                cleaned_theme['pathway_count'] = len(filtered_pathways)

                # Recompute BOTH avg FDR and the significance label from the pathways that
                # actually survive filtering. Recomputing FDR alone (as before) left a stale
                # significance keyed to the pre-filter cluster — so a theme trimmed to just
                # its strongest pathway (e.g. Cell cycle, FDR 3.76e-04) kept a MEDIUM label
                # while its shown FDR said HIGH.
                self._recompute_theme_significance(cleaned_theme)

                cleaned_themes.append(cleaned_theme)
            else:
                logger.info(f'  ⚠️  Theme "{theme_name}" removed (no pathways after filtering)')

        return cleaned_themes, total_filtered

    def _recompute_theme_significance(self, theme: Dict) -> None:
        """Recompute a theme's ``avg_p_value_fdr`` and ``significance`` from its current
        pathways, in place.

        Called after tissue-filtering removes pathways so the significance label stays
        consistent with the pathways actually shown. Reuses the SAME computed classifier
        as clustering (FDR base tier + |NES| promotion) via ``_pathway_fdr`` (0.0-safe)
        and the per-pathway ``abs_nes`` preserved at cluster time — never the LLM's value.
        """
        pathways = theme.get('pathways') or []
        if not pathways:
            return
        fdrs = [_pathway_fdr(p) for p in pathways]
        avg_fdr = sum(fdrs) / len(fdrs)
        abs_nes_values = [p.get('abs_nes') for p in pathways if p.get('abs_nes') is not None]
        max_abs_nes = max(abs_nes_values) if abs_nes_values else None

        theme['avg_p_value_fdr'] = avg_fdr
        theme['max_abs_nes'] = max_abs_nes
        theme['significance'] = self.clustering_service._classify_cluster_significance(
            avg_fdr, len(pathways), max_abs_nes)

    def _generate_summary(self, themes: List[Dict], singletons: List[Dict]) -> str:
        """Generate overall summary of pathway clustering"""

        summary_parts = []

        if themes:
            theme_names = ', '.join([t['name'] for t in themes[:3]])
            if len(themes) > 3:
                theme_names += f', and {len(themes)-3} more'

            summary_parts.append(
                f'Pathway analysis identified {len(themes)} biological themes: {theme_names}.'
            )

            # Average cluster size
            avg_size = sum(t['pathway_count'] for t in themes) / len(themes)
            summary_parts.append(
                f'On average, each theme comprises {avg_size:.1f} functionally related pathways.'
            )

        if singletons:
            summary_parts.append(
                f'{len(singletons)} pathways did not cluster with others and represent unique biological processes.'
            )

        return ' '.join(summary_parts) if summary_parts else 'No pathway themes identified.'


def main():
    """Command-line interface for testing"""
    import argparse

    parser = argparse.ArgumentParser(description='Step 1: Pathway Theme Clustering')
    parser.add_argument('input_file', help='Input JSON file with pathways and genes')
    parser.add_argument('-o', '--output', help='Output JSON file (default: theme_results.json)')
    parser.add_argument('--threshold', type=float, default=0.05, help='FDR threshold (default: 0.05)')
    parser.add_argument('--jaccard', type=float, default=0.25, help='Jaccard threshold (default: 0.25)')

    args = parser.parse_args()

    # Load input
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)

    # Extract required fields
    pathways = input_data.get('pathways', [])
    genes = input_data.get('genes') or input_data.get('differentially_expressed_genes', [])

    # Extract metadata
    metadata = input_data.get('metadata', {})
    organism = metadata.get('organism', 'Homo sapiens')

    # Build context from input data - include ALL metadata fields dynamically
    context = input_data.get('context')
    if not context:
        # Construct context from ALL metadata fields (dynamic, not hardcoded)
        context = {}
        for key, value in metadata.items():
            if key != 'organism' and value:  # Skip organism (handled separately)
                context[key] = value

    if not pathways:
        print('Error: No pathways found in input file')
        print('Expected field: "pathways"')
        sys.exit(1)

    if not genes:
        print('Error: No genes found in input file')
        print('Expected field: "genes" or "differentially_expressed_genes"')
        sys.exit(1)

    # Run Step 1
    step1 = Step01PathwayThemes()
    results = step1.execute(
        pathways=pathways,
        genes=genes,
        organism=organism,
        context=context,
        significance_threshold=args.threshold,
        jaccard_threshold=args.jaccard
    )

    # Save results
    output_file = args.output or 'theme_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f'\n{"="*80}')
    logger.info(f'Results saved to: {output_file}')
    logger.info(f'{"="*80}')


if __name__ == '__main__':
    main()
