"""
Meta-Step 09: Meta-Analysis Discovery Analysis

Performs deep comparison between meta-analysis and individual study results to identify:
1. Meta-unique biological themes (not found in any individual study)
2. Meta-unique mechanistic hypotheses
3. Enhanced significance findings (more significant in meta)
4. Cross-study validated mechanisms
5. Meta-unique therapeutic targets

This is the key step that extracts the VALUE of meta-analysis.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.groq_client import GroqClient

logger = logging.getLogger(__name__)


@dataclass
class MetaUniqueTheme:
    """A biological theme found only in meta-analysis"""
    theme_name: str
    pathways: List[str]
    shared_genes: List[str]
    biological_context: str
    why_meta_unique: str  # LLM explanation of why this emerged in meta


@dataclass
class EnhancedSignificancePathway:
    """A pathway with significantly enhanced statistics in meta-analysis"""
    pathway_name: str
    meta_pvalue: float
    meta_fdr: float
    best_individual_pvalue: float
    best_individual_dataset: str
    significance_gain_fold: float  # How much more significant in meta
    found_in_n_datasets: int


@dataclass
class MetaUniqueHypothesis:
    """A mechanistic hypothesis found only in meta-analysis"""
    hypothesis_title: str
    hypothesis_text: str
    supporting_pathways: List[str]
    supporting_genes: List[str]
    confidence_score: float
    similar_individual_hypotheses: List[str]  # Related but different hypotheses from individuals


@dataclass
class CrossStudyMechanism:
    """A mechanism validated across multiple studies"""
    mechanism_name: str
    pathway_name: str
    found_in_datasets: List[str]
    consistency_score: float
    meta_enhancement: str  # How meta-analysis improved understanding


@dataclass
class MetaUniqueTarget:
    """A therapeutic target identified only through meta-analysis"""
    gene_symbol: str
    target_type: str
    drugs: List[str]
    why_meta_unique: str
    clinical_relevance: str


@dataclass
class IndividualFindingsSummary:
    """Aggregated findings from individual datasets"""
    total_datasets: int
    total_unique_pathways: int
    pathways_in_multiple_studies: int
    pathways_in_single_study: int
    top_pathways: List[Dict[str, Any]]  # [{name, count, datasets, best_pvalue}]
    pathway_counts: Dict[str, Dict[str, Any]]
    llm_summary: str  # Natural language summary


@dataclass
class MetaDiscoveryResult:
    """Complete meta-analysis discovery results"""
    # Individual dataset findings (baseline)
    individual_findings: Dict[str, Any]

    # Core discoveries (meta-unique)
    meta_unique_themes: List[Dict[str, Any]]
    enhanced_significance_pathways: List[Dict[str, Any]]
    meta_unique_hypotheses: List[Dict[str, Any]]
    cross_study_mechanisms: List[Dict[str, Any]]
    meta_unique_targets: List[Dict[str, Any]]

    # Organizational themes (themes that group known pathways, not genuinely unique)
    organizational_themes: List[Dict[str, Any]] = field(default_factory=list)

    # Meta-generated hypotheses (shown but not claimed as unique when synthetic)
    meta_generated_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # Whether individual results are synthetic (from raw input, not full pipeline)
    all_synthetic: bool = False

    # New discovery types (emergent/computational)
    emergent_theme_combinations: List[Dict[str, Any]] = field(default_factory=list)
    cross_dataset_convergence: List[Dict[str, Any]] = field(default_factory=list)
    weak_signal_amplification: List[Dict[str, Any]] = field(default_factory=list)
    novel_cooccurrence_pairs: List[Dict[str, Any]] = field(default_factory=list)

    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)

    # LLM-generated biological interpretation
    biological_interpretation: str = ''

    # Key insights for report
    key_discoveries: List[str] = field(default_factory=list)

    # Warnings collected during execution
    warnings: List[str] = field(default_factory=list)


class MetaStep09DiscoveryAnalysis:
    """
    Meta-Step 09: Meta-Analysis Discovery Analysis

    Identifies unique biological insights that only emerge from meta-analysis,
    not visible in individual study analyses.
    """

    def __init__(self):
        self.step_number = 9
        self.step_name = 'Meta-Analysis Discovery Analysis'
        self.llm = GroqClient()
        self._warnings: List[str] = []

    def execute(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        dataset_names: List[str],
        metadata: Dict[str, Any] = None
    ) -> MetaDiscoveryResult:
        """
        Execute deep meta-analysis discovery comparison

        Args:
            meta_results: Full pipeline results from meta-analysis
            dataset_results: Dict mapping dataset_name -> full pipeline results
            dataset_names: List of dataset names
            metadata: Study metadata for context

        Returns:
            MetaDiscoveryResult with all unique discoveries
        """
        # Reset warnings for this execution
        self._warnings = []

        print(f'\n[Meta-Step {self.step_number}] {self.step_name}')
        print('=' * 80)
        print(f'  Analyzing meta-analysis discoveries across {len(dataset_names)} datasets')

        # Build study context
        study_context = self._build_study_context(metadata or {})

        # Detect if individual results are synthetic (raw input, not full pipeline)
        all_synthetic = all(
            ds_result.get('synthetic', False)
            for ds_result in dataset_results.values()
        )
        if all_synthetic:
            print('  Note: Individual results are synthetic (raw input only)')
            print('  Adjusting comparisons to avoid inflated discovery claims')

        # Phase 0: Aggregate individual dataset findings (baseline)
        print('\n  [Phase 0] Aggregating individual dataset findings...')
        individual_findings = self._aggregate_individual_findings(
            dataset_results, dataset_names, study_context
        )
        print(f'    Total unique pathways across individuals: {individual_findings["total_unique_pathways"]}')
        print(f'    Pathways found in multiple studies: {individual_findings["pathways_in_multiple"]}')


        # 1. Find meta-unique themes (or analyze as organizational when synthetic)
        organizational_themes = []
        if all_synthetic:
            print('\n  [Phase 1] Analyzing biological themes vs individual pathways...')
            meta_unique_themes, organizational_themes = self._analyze_themes_vs_individual_pathways(
                meta_results, dataset_results, study_context
            )
            print(f'    Found {len(meta_unique_themes)} genuinely novel themes')
            print(f'    Found {len(organizational_themes)} organizational themes (grouping known pathways)')
        else:
            print('\n  [Phase 1] Identifying meta-unique biological themes...')
            meta_unique_themes = self._find_meta_unique_themes(
                meta_results, dataset_results, study_context
            )
            print(f'    Found {len(meta_unique_themes)} meta-unique themes')

        # 1A. Emergent theme combinations (replaces broken theme-uniqueness check)
        print('\n  [Phase 1A] Finding emergent theme combinations...')
        emergent_theme_combinations = self._find_emergent_theme_combinations(
            meta_results, dataset_results
        )
        print(f'    Found {len(emergent_theme_combinations)} emergent theme combinations')

        # 1B. Cross-dataset pathway convergence
        print('\n  [Phase 1B] Finding cross-dataset pathway convergence...')
        cross_dataset_convergence = self._find_cross_dataset_convergence(
            meta_results, dataset_results
        )
        print(f'    Found {len(cross_dataset_convergence)} convergent themes')

        # 2. Find enhanced significance pathways (genuine finding regardless of synthetic)
        print('\n  [Phase 2] Identifying enhanced significance pathways...')
        enhanced_pathways = self._find_enhanced_significance_pathways(
            meta_results, dataset_results
        )
        print(f'    Found {len(enhanced_pathways)} pathways with enhanced significance')

        # 2B. Weak signal amplification
        print('\n  [Phase 2B] Finding weak signal amplification...')
        weak_signal_amplification = self._find_weak_signal_amplification(
            meta_results, dataset_results
        )
        print(f'    Found {len(weak_signal_amplification)} amplified weak signals')

        # 3. Find meta-unique hypotheses (skip comparison when synthetic)
        if all_synthetic:
            print('\n  [Phase 3] Meta hypotheses (skipping uniqueness comparison — individual hypotheses are empty/synthetic)...')
            meta_hypotheses = self._extract_hypotheses(meta_results)
            meta_unique_hypotheses = []  # Don't claim uniqueness
            meta_generated_hypotheses = self._normalize_hypotheses(meta_hypotheses)
            print(f'    {len(meta_generated_hypotheses)} hypotheses generated by meta-analysis (not claimed as unique)')
        else:
            print('\n  [Phase 3] Identifying meta-unique mechanistic hypotheses...')
            meta_unique_hypotheses = self._find_meta_unique_hypotheses(
                meta_results, dataset_results, study_context
            )
            meta_generated_hypotheses = []  # Not needed when full pipeline results available
            print(f'    Found {len(meta_unique_hypotheses)} meta-unique hypotheses')

        # 4. Find cross-study validated mechanisms (skip when synthetic)
        if all_synthetic:
            print('\n  [Phase 4] Skipping cross-study mechanisms (individual mechanisms are empty/synthetic)')
            cross_study_mechanisms = []
        else:
            print('\n  [Phase 4] Identifying cross-study validated mechanisms...')
            cross_study_mechanisms = self._find_cross_study_mechanisms(
                meta_results, dataset_results
            )
            print(f'    Found {len(cross_study_mechanisms)} cross-study validated mechanisms')

        # 4A. Novel co-occurrence pairs
        print('\n  [Phase 4C] Finding novel pathway co-occurrence pairs...')
        novel_cooccurrence_pairs = self._find_novel_cooccurrence_pairs(
            meta_results, dataset_results
        )
        print(f'    Found {len(novel_cooccurrence_pairs)} novel co-occurrence pairs')

        # 4D. Compute significance hierarchy early (needed for Phase 3A hypothesis generation)
        significance_hierarchy = self._compute_significance_hierarchy(enhanced_pathways)
        if significance_hierarchy and significance_hierarchy.get('tiers'):
            print(f'\n  [Significance Hierarchy] Meta resolves pathway ranking:')
            print(f'    Individual p-value spread: {significance_hierarchy["individual_log10_spread"]} orders of magnitude')
            print(f'    Meta p-value spread: {significance_hierarchy["meta_log10_spread"]} orders of magnitude')
            print(f'    Resolution gain: {significance_hierarchy["resolution_gain"]}x more ranking resolution')
            print(f'    Priority tiers: {significance_hierarchy["n_tiers"]}')

        # Phase 3A: Generate hypotheses from discovery findings if step4 produced none
        if not meta_generated_hypotheses:
            print('\n  [Phase 3A] Generating hypotheses from discovery findings...')
            meta_generated_hypotheses = self._generate_discovery_hypotheses(
                emergent_theme_combinations=emergent_theme_combinations,
                cross_dataset_convergence=cross_dataset_convergence,
                weak_signal_amplification=weak_signal_amplification,
                novel_cooccurrence_pairs=novel_cooccurrence_pairs,
                enhanced_pathways=enhanced_pathways,
                significance_hierarchy=significance_hierarchy,
                study_context=study_context,
                n_datasets=len(dataset_names)
            )
            print(f'    Generated {len(meta_generated_hypotheses)} hypotheses from discoveries')

        # 5. Find meta-unique therapeutic targets (skip when synthetic)
        if all_synthetic:
            print('\n  [Phase 5] Skipping meta-unique targets (individual targets are empty/synthetic)')
            meta_unique_targets = []
        else:
            print('\n  [Phase 5] Identifying meta-unique therapeutic targets...')
            meta_unique_targets = self._find_meta_unique_targets(
                meta_results, dataset_results
            )
            print(f'    Found {len(meta_unique_targets)} meta-unique therapeutic targets')

        # 6. Generate biological interpretation
        print('\n  [Phase 6] Generating biological interpretation...')
        biological_interpretation = self._generate_biological_interpretation(
            meta_unique_themes,
            enhanced_pathways,
            meta_unique_hypotheses,
            cross_study_mechanisms,
            meta_unique_targets,
            study_context,
            all_synthetic=all_synthetic,
            organizational_themes=organizational_themes,
            emergent_theme_combinations=emergent_theme_combinations,
            cross_dataset_convergence=cross_dataset_convergence,
            weak_signal_amplification=weak_signal_amplification,
            novel_cooccurrence_pairs=novel_cooccurrence_pairs,
            n_datasets=len(dataset_names)
        )

        # 7. (Significance hierarchy already computed in Phase 4D above)

        # 8. Generate key discoveries summary
        key_discoveries = self._generate_key_discoveries(
            meta_unique_themes,
            enhanced_pathways,
            meta_unique_hypotheses,
            meta_unique_targets,
            all_synthetic=all_synthetic,
            organizational_themes=organizational_themes,
            significance_hierarchy=significance_hierarchy,
            emergent_theme_combinations=emergent_theme_combinations,
            cross_dataset_convergence=cross_dataset_convergence,
            weak_signal_amplification=weak_signal_amplification,
            novel_cooccurrence_pairs=novel_cooccurrence_pairs
        )

        # New discovery type counts (shared between synthetic and full)
        new_discovery_counts = {
            'emergent_theme_combinations_count': len(emergent_theme_combinations),
            'cross_dataset_convergence_count': len(cross_dataset_convergence),
            'weak_signal_amplification_count': len(weak_signal_amplification),
            'novel_cooccurrence_pairs_count': len(novel_cooccurrence_pairs),
        }

        # Summary statistics — adjust for synthetic
        if all_synthetic:
            total_discoveries = (
                len(enhanced_pathways) +
                len(emergent_theme_combinations) +
                len(weak_signal_amplification)
            )
            summary = {
                'n_datasets': len(dataset_names),
                'all_synthetic': True,
                'meta_unique_themes_count': len(meta_unique_themes),
                'organizational_themes_count': len(organizational_themes),
                'enhanced_significance_pathways_count': len(enhanced_pathways),
                'meta_unique_hypotheses_count': 0,  # Not claimed
                'meta_generated_hypotheses_count': len(meta_generated_hypotheses),
                'cross_study_mechanisms_count': 0,  # Skipped
                'meta_unique_targets_count': 0,  # Skipped
                'total_discoveries': total_discoveries,
                'significance_hierarchy': significance_hierarchy,
                **new_discovery_counts
            }
        else:
            total_discoveries = (
                len(meta_unique_themes) +
                len(meta_unique_hypotheses) +
                len(meta_unique_targets) +
                len(emergent_theme_combinations) +
                len(weak_signal_amplification)
            )
            summary = {
                'n_datasets': len(dataset_names),
                'all_synthetic': False,
                'meta_unique_themes_count': len(meta_unique_themes),
                'organizational_themes_count': len(organizational_themes),
                'enhanced_significance_pathways_count': len(enhanced_pathways),
                'meta_unique_hypotheses_count': len(meta_unique_hypotheses),
                'cross_study_mechanisms_count': len(cross_study_mechanisms),
                'meta_unique_targets_count': len(meta_unique_targets),
                'total_discoveries': total_discoveries,
                'significance_hierarchy': significance_hierarchy,
                **new_discovery_counts
            }

        print(f'\n  Summary:')
        if all_synthetic:
            print(f'    Enhanced significance pathways (genuine): {summary["enhanced_significance_pathways_count"]}')
            print(f'    Organizational themes: {summary["organizational_themes_count"]}')
            print(f'    Genuinely novel themes: {summary["meta_unique_themes_count"]}')
            print(f'    Meta-generated hypotheses: {summary["meta_generated_hypotheses_count"]} (not claimed unique)')
        else:
            print(f'    Total meta-unique discoveries: {summary["total_discoveries"]}')
            print(f'    - Unique themes: {summary["meta_unique_themes_count"]}')
            print(f'    - Unique hypotheses: {summary["meta_unique_hypotheses_count"]}')
            print(f'    - Unique targets: {summary["meta_unique_targets_count"]}')
            print(f'    Enhanced significance pathways: {summary["enhanced_significance_pathways_count"]}')

        # Print new discovery type counts
        print(f'    --- New emergent discovery types ---')
        print(f'    Emergent theme combinations: {len(emergent_theme_combinations)}')
        print(f'    Cross-dataset convergence: {len(cross_dataset_convergence)}')
        print(f'    Weak signal amplification: {len(weak_signal_amplification)}')
        print(f'    Novel co-occurrence pairs: {len(novel_cooccurrence_pairs)}')

        if self._warnings:
            print(f'    Warnings: {len(self._warnings)}')

        return MetaDiscoveryResult(
            individual_findings=individual_findings,
            meta_unique_themes=[asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in meta_unique_themes],
            enhanced_significance_pathways=[asdict(p) if hasattr(p, '__dataclass_fields__') else p for p in enhanced_pathways],
            meta_unique_hypotheses=[asdict(h) if hasattr(h, '__dataclass_fields__') else h for h in meta_unique_hypotheses],
            cross_study_mechanisms=[asdict(m) if hasattr(m, '__dataclass_fields__') else m for m in cross_study_mechanisms],
            meta_unique_targets=[asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in meta_unique_targets],
            organizational_themes=[asdict(t) if hasattr(t, '__dataclass_fields__') else t for t in organizational_themes],
            meta_generated_hypotheses=meta_generated_hypotheses,
            all_synthetic=all_synthetic,
            emergent_theme_combinations=emergent_theme_combinations,
            cross_dataset_convergence=cross_dataset_convergence,
            weak_signal_amplification=weak_signal_amplification,
            novel_cooccurrence_pairs=novel_cooccurrence_pairs,
            summary=summary,
            biological_interpretation=biological_interpretation,
            key_discoveries=key_discoveries,
            warnings=list(self._warnings)
        )

    def _build_study_context(self, metadata: Dict[str, Any]) -> str:
        """Build study context string from metadata"""
        if not metadata:
            return "Unknown study context"

        parts = []
        skip_fields = {'datasetId', 'dataset_id', 'organism'}

        for key, value in metadata.items():
            if key in skip_fields:
                continue
            if value and str(value).strip():
                display_key = key.replace('_', ' ').title()
                parts.append(f"{display_key}: {value}")

        return ', '.join(parts) if parts else "Unknown study context"

    def _extract_themes(self, results: Dict[str, Any]) -> List[Dict]:
        """Extract themes from pipeline results"""
        step1 = results.get('step1_pathway_themes', {})
        if not step1:
            step1 = results.get('steps', {}).get('step1', {})
        return step1.get('themes', [])

    def _extract_hypotheses(self, results: Dict[str, Any]) -> List[Dict]:
        """Extract hypotheses from pipeline results"""
        step4 = results.get('step4_hypotheses', {})
        if not step4:
            step4 = results.get('steps', {}).get('step4', {})
        return step4.get('hypotheses', [])

    def _normalize_hypotheses(self, raw_hypotheses: List[Dict]) -> List[Dict[str, Any]]:
        """Normalize hypothesis field names from step 4 LLM output.

        Step 4 may produce varying field names depending on the LLM. This method
        normalizes them to a consistent schema for the report.
        """
        confidence_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
        normalized = []

        for hyp in raw_hypotheses:
            # Title: use 'title' if present, otherwise derive from hypothesis text
            title = hyp.get('title', '')
            if not title:
                text = hyp.get('hypothesis', '')
                # Use first sentence as title (up to 120 chars)
                first_sentence = text.split('.')[0].strip() if text else 'Untitled hypothesis'
                title = first_sentence[:120]

            # Confidence: normalize string to float
            confidence_raw = hyp.get('confidence_score', hyp.get('confidence', 0.5))
            if isinstance(confidence_raw, str):
                confidence = confidence_map.get(confidence_raw.lower(), 0.5)
            else:
                confidence = float(confidence_raw) if confidence_raw else 0.5

            # Supporting pathways: extract from evidenceSupporting if not present
            supporting_pathways = hyp.get('supporting_pathways', hyp.get('supportingPathways', []))
            if not supporting_pathways:
                # Try to extract pathway names from evidenceSupporting entries
                for evidence in hyp.get('evidenceSupporting', []):
                    if isinstance(evidence, str) and 'pathway' in evidence.lower():
                        # Extract pathway name before "pathway enrichment" or similar
                        parts = evidence.split(' pathway ')
                        if parts:
                            supporting_pathways.append(parts[0].strip())

            # Key players
            key_players = hyp.get('keyPlayers', hyp.get('key_players', []))

            normalized.append({
                'title': title,
                'hypothesis_text': hyp.get('hypothesis', ''),
                'confidence_score': confidence,
                'confidence_label': hyp.get('confidence', ''),
                'key_players': key_players,
                'supporting_pathways': supporting_pathways,
                'evidence': hyp.get('evidenceSupporting', []),
                'mechanistic_model': hyp.get('mechanisticModel', ''),
                'testability': hyp.get('testability', {}),
                'directional_prediction': hyp.get('directionalPrediction', ''),
                'novelty': hyp.get('novelty', ''),
            })

        return normalized

    def _extract_mechanisms(self, results: Dict[str, Any]) -> List[Dict]:
        """Extract pathway mechanisms from pipeline results"""
        step3 = results.get('step3_mechanisms', {})
        if not step3:
            step3 = results.get('steps', {}).get('step3', {})
        mechanisms = step3.get('pathway_mechanisms', [])

        # Normalize field names (step3 uses 'pathway' not 'pathway_name')
        normalized = []
        for mech in mechanisms:
            pathway_id = mech.get('pathwayId', mech.get('pathway_id', ''))
            normalized.append({
                'pathway_name': mech.get('pathway', mech.get('pathway_name', '')),
                'pathway_id': pathway_id,
                'mechanism_name': mech.get('pathway', mech.get('name', '')),  # Use pathway as mechanism name
                'biological_function': mech.get('biologicalFunction', mech.get('biological_function', '')),
                'crosstalk': mech.get('crosstalk', []),
                'functional_consequences': mech.get('functionalConsequences', mech.get('functional_consequences', ''))
            })
        return normalized

    def _extract_therapeutics(self, results: Dict[str, Any]) -> Dict:
        """Extract therapeutic results from pipeline results"""
        step5 = results.get('step5_therapeutics', {})
        if not step5:
            step5 = results.get('steps', {}).get('step5', {})
        return step5

    def _extract_pathways_with_stats(self, results: Dict[str, Any]) -> Dict[str, Dict]:
        """Extract all pathways with their statistics"""
        pathways = {}
        step1 = results.get('step1_pathway_themes', {})
        if not step1:
            step1 = results.get('steps', {}).get('step1', {})

        # From themes
        for theme in step1.get('themes', []):
            for pw in theme.get('pathways', []):
                name = pw.get('name', pw.get('pathway', ''))
                if name:
                    pathways[name] = {
                        'pValue': pw.get('pValue') or pw.get('p_value'),
                        'pValueFDR': pw.get('pValueFDR') or pw.get('p_value_fdr'),
                        'ES': pw.get('ES') or pw.get('es'),
                        'theme': theme.get('name', '')
                    }

        # From ungrouped
        for pw in step1.get('ungrouped_pathways', []) or step1.get('ungrouped', []):
            name = pw.get('name', pw.get('pathway', ''))
            if name and name not in pathways:
                pathways[name] = {
                    'pValue': pw.get('pValue') or pw.get('p_value'),
                    'pValueFDR': pw.get('pValueFDR') or pw.get('p_value_fdr'),
                    'ES': pw.get('ES') or pw.get('es'),
                    'theme': 'Ungrouped'
                }

        return pathways

    def _get_theme_pathway_names(self, theme: Dict) -> Set[str]:
        """Extract normalized pathway names from a theme dict."""
        names = set()
        for pw in theme.get('pathways', []):
            name = pw.get('name', pw.get('pathway', ''))
            if name:
                names.add(name.lower().strip())
        return names

    def _find_meta_unique_themes(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        study_context: str
    ) -> List[MetaUniqueTheme]:
        """Find biological themes that appear only in meta-analysis.

        Uses pathway-set overlap comparison: a meta theme is unique only if
        no individual theme has >50% pathway overlap (Jaccard on smaller set).
        """
        meta_themes = self._extract_themes(meta_results)

        # Collect all individual themes with their pathway name sets
        individual_themes_with_pathways = []
        for ds_results in dataset_results.values():
            ds_themes = self._extract_themes(ds_results)
            for theme in ds_themes:
                pw_names = self._get_theme_pathway_names(theme)
                if pw_names:
                    individual_themes_with_pathways.append(pw_names)

        meta_unique_themes = []

        for theme in meta_themes:
            theme_name = theme.get('name', '')
            if not theme_name:
                continue

            meta_pw_names = self._get_theme_pathway_names(theme)

            # Check if this meta theme overlaps significantly with any individual theme
            is_unique = True
            if meta_pw_names:
                for ind_pw_names in individual_themes_with_pathways:
                    overlap = meta_pw_names & ind_pw_names
                    smaller_set_size = min(len(meta_pw_names), len(ind_pw_names))
                    if smaller_set_size > 0 and len(overlap) / smaller_set_size > 0.5:
                        is_unique = False
                        break
            else:
                # No pathways to compare - fall back to name matching
                individual_theme_names = set()
                for ds_results in dataset_results.values():
                    ds_themes = self._extract_themes(ds_results)
                    for ds_theme in ds_themes:
                        individual_theme_names.add(ds_theme.get('name', '').lower().strip())
                theme_name_lower = theme_name.lower().strip()
                if theme_name_lower in individual_theme_names:
                    is_unique = False

            if is_unique:
                # Get pathway names and genes
                pathways = [pw.get('name', '') for pw in theme.get('pathways', [])]
                shared_genes = theme.get('shared_genes', [])[:20]  # Limit for display
                biological_context = theme.get('biological_context', '')

                # Generate LLM explanation of why this theme is meta-unique
                why_unique = self._explain_meta_unique_theme(
                    theme_name, pathways, shared_genes, study_context
                )

                meta_unique_themes.append(MetaUniqueTheme(
                    theme_name=theme_name,
                    pathways=pathways,
                    shared_genes=shared_genes,
                    biological_context=biological_context,
                    why_meta_unique=why_unique
                ))

        return meta_unique_themes

    def _analyze_themes_vs_individual_pathways(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        study_context: str
    ) -> Tuple[List[MetaUniqueTheme], List[Dict[str, Any]]]:
        """Analyze meta themes against individual pathway lists when results are synthetic.

        Instead of comparing themes-vs-themes (which is meaningless when individuals
        have themes=[]), compare each meta theme's pathways against the actual pathway
        names present in individual datasets.

        Returns:
            Tuple of (genuinely_novel_themes, organizational_themes)
            - genuinely_novel_themes: themes with pathways not found in any individual dataset
            - organizational_themes: themes that group pathways already in individual datasets
        """
        meta_themes = self._extract_themes(meta_results)

        # Collect all individual pathway names (from ungrouped_pathways across all datasets)
        individual_pathway_names = set()
        for ds_results in dataset_results.values():
            ds_pathways = self._extract_pathways_with_stats(ds_results)
            for pw_name in ds_pathways:
                individual_pathway_names.add(pw_name.lower().strip())

        genuinely_novel = []
        organizational = []

        for theme in meta_themes:
            theme_name = theme.get('name', '')
            if not theme_name:
                continue

            # Get pathway names from this theme
            theme_pathways = [pw.get('name', pw.get('pathway', '')) for pw in theme.get('pathways', [])]
            if not theme_pathways:
                continue

            # Check what fraction of theme pathways exist in individual datasets
            in_individual = 0
            meta_only = 0
            meta_only_pathways = []
            for pw_name in theme_pathways:
                if pw_name.lower().strip() in individual_pathway_names:
                    in_individual += 1
                else:
                    meta_only += 1
                    meta_only_pathways.append(pw_name)

            total = len(theme_pathways)
            fraction_in_individual = in_individual / total if total > 0 else 0

            shared_genes = theme.get('shared_genes', [])[:20]
            biological_context = theme.get('biological_context', '')

            if meta_only == total:
                # All pathways meta-only — genuinely novel theme
                why_unique = self._explain_meta_unique_theme(
                    theme_name, theme_pathways, shared_genes, study_context
                )
                genuinely_novel.append(MetaUniqueTheme(
                    theme_name=theme_name,
                    pathways=theme_pathways,
                    shared_genes=shared_genes,
                    biological_context=biological_context,
                    why_meta_unique=why_unique
                ))
            elif meta_only > 0:
                # Some pathways meta-only — partially novel
                why_unique = self._explain_meta_unique_theme(
                    theme_name, meta_only_pathways, shared_genes, study_context
                )
                genuinely_novel.append(MetaUniqueTheme(
                    theme_name=theme_name,
                    pathways=theme_pathways,
                    shared_genes=shared_genes,
                    biological_context=biological_context,
                    why_meta_unique=f"Partially novel: {meta_only}/{total} pathways detected only in meta-analysis. {why_unique}"
                ))
            else:
                # All pathways in individuals — organizational insight
                organizational.append({
                    'theme_name': theme_name,
                    'pathways': theme_pathways,
                    'shared_genes': shared_genes,
                    'biological_context': biological_context,
                    'category': 'organizational',
                    'description': (
                        f"Groups {total} pathways already found in individual datasets "
                        f"into a coherent biological narrative. This is a useful organizational "
                        f"insight from meta-analysis, not a novel discovery."
                    ),
                    'pathways_in_individual': in_individual,
                    'pathways_meta_only': meta_only
                })

        return genuinely_novel, organizational

    def _explain_meta_unique_theme(
        self,
        theme_name: str,
        pathways: List[str],
        genes: List[str],
        study_context: str
    ) -> str:
        """Use LLM to explain why a theme emerged only in meta-analysis"""
        prompt = f"""A biological theme was identified ONLY in meta-analysis (not in any individual study).

**Study Context:** {study_context}
**Theme Name:** {theme_name}
**Pathways in Theme:** {', '.join(pathways[:5])}{'...' if len(pathways) > 5 else ''}
**Key Shared Genes:** {', '.join(genes[:10])}{'...' if len(genes) > 10 else ''}

Explain in 2-3 sentences why this theme might have emerged only when combining multiple studies (meta-analysis), not visible in individual datasets. Consider:
- Statistical power and sample size effects
- Heterogeneity across studies that averages out
- Subtle biological signals that accumulate"""

        try:
            response = self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=300,
                temperature=0,
                seed=42
            )
            return response.strip()
        except Exception as e:
            msg = f"LLM failed in _explain_meta_unique_theme for '{theme_name}': {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return f"This theme emerged from the combined statistical power of multiple studies."

    def _find_enhanced_significance_pathways(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> List[EnhancedSignificancePathway]:
        """Find pathways with significantly better p-values in meta-analysis"""
        meta_pathways = self._extract_pathways_with_stats(meta_results)

        # Pre-compute dataset pathway maps once (instead of per-pathway per-dataset)
        dataset_pathway_maps = {
            ds_name: self._extract_pathways_with_stats(ds_results)
            for ds_name, ds_results in dataset_results.items()
        }

        # Get best p-value for each pathway across individual datasets
        individual_best = {}
        for ds_name, ds_pathways in dataset_pathway_maps.items():
            for pw_name, pw_stats in ds_pathways.items():
                pval = pw_stats.get('pValue')
                if pval is not None:
                    if pw_name not in individual_best or pval < individual_best[pw_name]['pvalue']:
                        individual_best[pw_name] = {
                            'pvalue': pval,
                            'dataset': ds_name
                        }

        enhanced_pathways = []

        for pw_name, meta_stats in meta_pathways.items():
            meta_pval = meta_stats.get('pValue')
            meta_fdr = meta_stats.get('pValueFDR')

            if meta_pval is None:
                continue

            # Count how many datasets have this pathway (using pre-computed maps)
            found_in_count = sum(
                1 for ds_pathways in dataset_pathway_maps.values()
                if pw_name in ds_pathways
            )

            if pw_name in individual_best:
                best_ind = individual_best[pw_name]
                best_ind_pval = best_ind['pvalue']

                # Calculate significance gain (fold improvement in p-value)
                if best_ind_pval > 0 and meta_pval > 0:
                    # -log10 difference shows how many orders of magnitude better
                    meta_log = -math.log10(meta_pval) if meta_pval > 0 else 0
                    ind_log = -math.log10(best_ind_pval) if best_ind_pval > 0 else 0
                    significance_gain = meta_log - ind_log

                    # Only include if meta is at least 2-fold better (-log scale)
                    if significance_gain >= 1.0:  # 10x more significant
                        enhanced_pathways.append(EnhancedSignificancePathway(
                            pathway_name=pw_name,
                            meta_pvalue=meta_pval,
                            meta_fdr=meta_fdr or 0,
                            best_individual_pvalue=best_ind_pval,
                            best_individual_dataset=best_ind['dataset'],
                            significance_gain_fold=10 ** significance_gain,
                            found_in_n_datasets=found_in_count
                        ))
            else:
                # Pathway found in meta but not in any individual study
                enhanced_pathways.append(EnhancedSignificancePathway(
                    pathway_name=pw_name,
                    meta_pvalue=meta_pval,
                    meta_fdr=meta_fdr or 0,
                    best_individual_pvalue=1.0,
                    best_individual_dataset='None',
                    significance_gain_fold=float('inf'),
                    found_in_n_datasets=0
                ))

        # Sort by significance gain
        enhanced_pathways.sort(key=lambda x: -x.significance_gain_fold if x.significance_gain_fold != float('inf') else -1e10)

        return enhanced_pathways[:50]  # Top 50

    def _find_meta_unique_hypotheses(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        study_context: str
    ) -> List[MetaUniqueHypothesis]:
        """Find mechanistic hypotheses unique to meta-analysis"""
        meta_hypotheses = self._extract_hypotheses(meta_results)

        # Collect all individual hypotheses as full dicts for keyPlayers comparison
        individual_hypotheses = []
        for ds_results in dataset_results.values():
            ds_hypotheses = self._extract_hypotheses(ds_results)
            individual_hypotheses.extend(ds_hypotheses)

        meta_unique_hypotheses = []

        for hyp in meta_hypotheses:
            title = hyp.get('title', '')
            hypothesis_text = hyp.get('hypothesis', '')

            # Check if this hypothesis is truly unique using keyPlayers comparison
            is_unique, similar_hypotheses = self._check_hypothesis_uniqueness(
                hyp, individual_hypotheses
            )

            if is_unique:
                meta_unique_hypotheses.append(MetaUniqueHypothesis(
                    hypothesis_title=title,
                    hypothesis_text=hypothesis_text,
                    supporting_pathways=hyp.get('supporting_pathways', []),
                    supporting_genes=hyp.get('supporting_genes', []),
                    confidence_score=hyp.get('confidence_score', 0.5),
                    similar_individual_hypotheses=similar_hypotheses
                ))

        return meta_unique_hypotheses

    def _check_hypothesis_uniqueness(
        self,
        meta_hypothesis: Dict,
        individual_hypotheses: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """Check if a hypothesis is unique compared to individual study hypotheses.

        Uses keyPlayers comparison: if >50% of keyPlayers overlap, not unique.
        Falls back to exact title match when keyPlayers are empty.
        """
        if not individual_hypotheses:
            return True, []

        meta_title = meta_hypothesis.get('title', '')
        meta_key_players = set(
            p.lower().strip() for p in meta_hypothesis.get('keyPlayers', []) if p
        )

        similar = []
        for ind_hyp in individual_hypotheses:
            ind_title = ind_hyp.get('title', '')
            ind_key_players = set(
                p.lower().strip() for p in ind_hyp.get('keyPlayers', []) if p
            )

            # Compare using keyPlayers if available
            if meta_key_players and ind_key_players:
                overlap = meta_key_players & ind_key_players
                if len(overlap) / len(meta_key_players) > 0.5:
                    ind_text = f"{ind_title}: {ind_hyp.get('hypothesis', '')}"
                    similar.append(ind_text[:100] + '...' if len(ind_text) > 100 else ind_text)
            else:
                # Fallback to exact title match
                if meta_title and ind_title and meta_title.lower().strip() == ind_title.lower().strip():
                    ind_text = f"{ind_title}: {ind_hyp.get('hypothesis', '')}"
                    similar.append(ind_text[:100] + '...' if len(ind_text) > 100 else ind_text)

        # If no similar hypotheses found, it's unique
        is_unique = len(similar) == 0

        return is_unique, similar[:3]  # Return up to 3 similar

    def _find_cross_study_mechanisms(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> List[CrossStudyMechanism]:
        """Find mechanisms validated across multiple studies.

        Matches by pathway_id first, falls back to exact normalized name.
        """
        meta_mechanisms = self._extract_mechanisms(meta_results)

        cross_study_mechanisms = []

        for mechanism in meta_mechanisms:
            pathway_name = mechanism.get('pathway_name', '')
            pathway_id = mechanism.get('pathway_id', '')
            mechanism_name = mechanism.get('mechanism_name', mechanism.get('name', ''))
            pathway_name_normalized = pathway_name.lower().strip()

            # Check which datasets also found this pathway/mechanism
            found_in = []
            for ds_name, ds_results in dataset_results.items():
                ds_mechanisms = self._extract_mechanisms(ds_results)
                for ds_mech in ds_mechanisms:
                    ds_pathway_id = ds_mech.get('pathway_id', '')
                    ds_pathway_name = ds_mech.get('pathway_name', '')

                    # Match by pathway_id first (stable KEGG/Reactome IDs)
                    if pathway_id and ds_pathway_id and pathway_id == ds_pathway_id:
                        found_in.append(ds_name)
                        break
                    # Fall back to exact normalized name match
                    elif ds_pathway_name.lower().strip() == pathway_name_normalized:
                        found_in.append(ds_name)
                        break

            if len(found_in) >= 2:  # Found in at least 2 datasets
                consistency = len(found_in) / len(dataset_results)
                cross_study_mechanisms.append(CrossStudyMechanism(
                    mechanism_name=mechanism_name,
                    pathway_name=pathway_name,
                    found_in_datasets=found_in,
                    consistency_score=consistency,
                    meta_enhancement=f"Mechanism validated across {len(found_in)} studies with {consistency*100:.0f}% consistency"
                ))

        # Sort by consistency
        cross_study_mechanisms.sort(key=lambda x: -x.consistency_score)

        return cross_study_mechanisms

    def _find_meta_unique_targets(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> List[MetaUniqueTarget]:
        """Find therapeutic targets identified only in meta-analysis"""
        meta_therapeutics = self._extract_therapeutics(meta_results)

        # Collect all targets from individual datasets
        individual_targets = set()
        for ds_results in dataset_results.values():
            ds_therapeutics = self._extract_therapeutics(ds_results)
            for target in ds_therapeutics.get('therapeutic_targets', []):
                individual_targets.add(target.get('gene', '').upper())
            for drug in ds_therapeutics.get('drug_recommendations', []):
                individual_targets.add(drug.get('target_gene', '').upper())

        meta_unique_targets = []

        # Check meta targets
        for target in meta_therapeutics.get('therapeutic_targets', []):
            gene = target.get('gene', '')
            if gene.upper() not in individual_targets and gene:
                meta_unique_targets.append(MetaUniqueTarget(
                    gene_symbol=gene,
                    target_type=target.get('target_type', 'Unknown'),
                    drugs=target.get('drugs', [])[:5],
                    why_meta_unique="Target emerged from combined statistical power of meta-analysis",
                    clinical_relevance=target.get('clinical_relevance', '')
                ))

        # Check meta drug recommendations
        for drug in meta_therapeutics.get('drug_recommendations', []):
            gene = drug.get('target_gene', '')
            if gene.upper() not in individual_targets and gene:
                meta_unique_targets.append(MetaUniqueTarget(
                    gene_symbol=gene,
                    target_type='Drug target',
                    drugs=[drug.get('drug_name', '')],
                    why_meta_unique="Drug-gene interaction identified through meta-analysis power",
                    clinical_relevance=drug.get('clinical_relevance', drug.get('approval_status', ''))
                ))

        return meta_unique_targets

    # ------------------------------------------------------------------ #
    # New emergent discovery methods (purely computational, no LLM calls) #
    # ------------------------------------------------------------------ #

    def _find_emergent_theme_combinations(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        fdr_threshold: float = 0.05,
        high_coverage_threshold: float = 0.75,
        max_high_coverage_fraction: float = 0.25
    ) -> List[Dict[str, Any]]:
        """Type 1: Find emergent theme combinations.

        A meta theme groups pathways [A, B, C].  The combination is emergent
        if fewer than max_high_coverage_fraction (default 25%) of datasets have
        >high_coverage_threshold (default 75%) of the theme's pathways.

        This handles the reality that some datasets are very large (e.g., 46/54
        total pathways) and single-handedly cover every theme.  The combination
        is still emergent if *most* datasets cannot observe it.
        """
        meta_themes = self._extract_themes(meta_results)

        # Pre-compute per-dataset sets of significant pathway names
        ds_sig_sets: Dict[str, Set[str]] = {}
        for ds_name, ds_res in dataset_results.items():
            pw_stats = self._extract_pathways_with_stats(ds_res)
            sig = set()
            for pw_name, stats in pw_stats.items():
                fdr = stats.get('pValueFDR') or stats.get('pValue') or 1.0
                if fdr < fdr_threshold:
                    sig.add(pw_name.lower().strip())
            ds_sig_sets[ds_name] = sig

        n_datasets = len(ds_sig_sets)
        if n_datasets == 0:
            return []

        emergent = []
        for theme in meta_themes:
            theme_name = theme.get('name', '')
            if not theme_name:
                continue
            theme_pw_names = self._get_theme_pathway_names(theme)
            if len(theme_pw_names) < 2:
                continue

            # Compute per-dataset coverage of this theme
            coverages = {}
            for ds_name, sig_set in ds_sig_sets.items():
                overlap = theme_pw_names & sig_set
                coverages[ds_name] = len(overlap) / len(theme_pw_names)

            # Count how many datasets have high coverage (>threshold)
            n_high_coverage = sum(
                1 for cov in coverages.values()
                if cov > high_coverage_threshold
            )
            fraction_high = n_high_coverage / n_datasets

            # Median coverage across all datasets
            sorted_covs = sorted(coverages.values())
            median_coverage = sorted_covs[len(sorted_covs) // 2]

            # Compute best single-dataset coverage
            best_ds = max(coverages, key=coverages.get)
            best_cov = coverages[best_ds]

            # Emergent if few datasets see the full combination
            if fraction_high <= max_high_coverage_fraction:
                if best_cov >= 1.0:
                    continue  # Not emergent: fully visible in at least one dataset
                emergent.append({
                    'theme_name': theme_name,
                    'pathways': [pw.get('name', pw.get('pathway', ''))
                                 for pw in theme.get('pathways', [])],
                    'n_pathways': len(theme_pw_names),
                    'median_dataset_coverage': round(median_coverage, 3),
                    'n_datasets_high_coverage': n_high_coverage,
                    'fraction_datasets_high_coverage': round(fraction_high, 3),
                    'best_individual_coverage': round(best_cov, 3),
                    'best_individual_dataset': best_ds,
                    'description': (
                        f"Only {n_high_coverage}/{n_datasets} datasets ({fraction_high:.0%}) "
                        f"cover >{high_coverage_threshold:.0%} of this theme's "
                        f"{len(theme_pw_names)} pathways. Median coverage: {median_coverage:.0%}. "
                        f"The full combination is an emergent meta-analysis insight."
                    )
                })

        emergent.sort(key=lambda x: x['median_dataset_coverage'])
        return emergent

    def _find_cross_dataset_convergence(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        max_avg_jaccard: float = 0.5,
        omnibus_threshold: float = 0.80
    ) -> List[Dict[str, Any]]:
        """Type 2: Find cross-dataset pathway convergence.

        Pathways within a meta theme come from largely non-overlapping subsets of
        datasets — i.e. different datasets contribute different pathways, but
        meta-analysis groups them together.

        Omnibus datasets (those containing >omnibus_threshold of all known pathways)
        are excluded from the Jaccard computation because they contribute every
        pathway and mask true convergence patterns.
        """
        meta_themes = self._extract_themes(meta_results)

        # Pre-compute per-dataset pathway name sets
        ds_pw_sets: Dict[str, Set[str]] = {}
        for ds_name, ds_res in dataset_results.items():
            pw_stats = self._extract_pathways_with_stats(ds_res)
            ds_pw_sets[ds_name] = {n.lower().strip() for n in pw_stats}

        # Compute all known pathways and identify omnibus datasets
        all_known = set()
        for s in ds_pw_sets.values():
            all_known |= s
        n_all = len(all_known) if all_known else 1

        filtered_ds_pw_sets = {
            ds: s for ds, s in ds_pw_sets.items()
            if len(s) / n_all <= omnibus_threshold
        }

        # Need at least 2 non-omnibus datasets for meaningful comparison
        if len(filtered_ds_pw_sets) < 2:
            filtered_ds_pw_sets = ds_pw_sets  # Fall back to all

        convergent = []
        for theme in meta_themes:
            theme_name = theme.get('name', '')
            if not theme_name:
                continue
            theme_pw_names = self._get_theme_pathway_names(theme)
            if len(theme_pw_names) < 2:
                continue

            # For each pathway in the theme, find which (non-omnibus) datasets contain it
            pw_dataset_map: Dict[str, Set[str]] = {}
            for pw_norm in theme_pw_names:
                contributing_ds = set()
                for ds_name, ds_set in filtered_ds_pw_sets.items():
                    if pw_norm in ds_set:
                        contributing_ds.add(ds_name)
                pw_dataset_map[pw_norm] = contributing_ds

            # Compute pairwise Jaccard between contributing dataset sets
            pw_list = list(pw_dataset_map.keys())
            if len(pw_list) < 2:
                continue

            jaccard_values = []
            for i in range(len(pw_list)):
                for j in range(i + 1, len(pw_list)):
                    s1 = pw_dataset_map[pw_list[i]]
                    s2 = pw_dataset_map[pw_list[j]]
                    union_size = len(s1 | s2)
                    if union_size > 0:
                        jaccard_values.append(len(s1 & s2) / union_size)
                    else:
                        jaccard_values.append(0.0)

            avg_jaccard = sum(jaccard_values) / len(jaccard_values) if jaccard_values else 1.0

            if avg_jaccard < max_avg_jaccard:
                convergent.append({
                    'theme_name': theme_name,
                    'pathways': [pw.get('name', pw.get('pathway', ''))
                                 for pw in theme.get('pathways', [])],
                    'avg_dataset_overlap': round(avg_jaccard, 3),
                    'n_pathways': len(theme_pw_names),
                    'description': (
                        f"Pathways in this theme are contributed by largely independent "
                        f"datasets (avg Jaccard={avg_jaccard:.2f}). The meta-analysis reveals "
                        f"a connection between pathways that were studied in different experiments."
                    )
                })

        convergent.sort(key=lambda x: x['avg_dataset_overlap'])
        return convergent

    def _find_weak_signal_amplification(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        meta_p_threshold: float = 0.001,
        individual_strong_threshold: float = 0.001,
        individual_weak_lower: float = 0.01,
        individual_weak_upper: float = 0.05,
        min_weak_datasets: int = 2
    ) -> List[Dict[str, Any]]:
        """Type 5: Find weak signal amplification.

        Pathways only marginally significant in individuals (p 0.01-0.05) but
        strongly significant in meta (p < 0.001), where NO individual has p < 0.001
        and ≥2 have weak significance.
        """
        meta_pathways = self._extract_pathways_with_stats(meta_results)

        # Pre-compute individual dataset pathway stats
        ds_pw_stats: Dict[str, Dict[str, Dict]] = {}
        for ds_name, ds_res in dataset_results.items():
            ds_pw_stats[ds_name] = self._extract_pathways_with_stats(ds_res)

        amplified = []
        for pw_name, meta_stats in meta_pathways.items():
            meta_pval = meta_stats.get('pValue')
            if meta_pval is None or meta_pval >= meta_p_threshold:
                continue

            pw_norm = pw_name.lower().strip()

            # Check individual datasets
            has_strong = False
            n_weak = 0
            individual_pvals = {}
            for ds_name, ds_map in ds_pw_stats.items():
                # Match by normalized name
                ind_pval = None
                for ind_name, ind_stats in ds_map.items():
                    if ind_name.lower().strip() == pw_norm:
                        ind_pval = ind_stats.get('pValue')
                        break
                if ind_pval is not None:
                    individual_pvals[ds_name] = ind_pval
                    if ind_pval < individual_strong_threshold:
                        has_strong = True
                        break
                    if individual_weak_lower <= ind_pval <= individual_weak_upper:
                        n_weak += 1

            if not has_strong and n_weak >= min_weak_datasets:
                amplified.append({
                    'pathway': pw_name,
                    'meta_pvalue': meta_pval,
                    'best_individual_pvalue': min(individual_pvals.values()) if individual_pvals else None,
                    'n_weakly_significant': n_weak,
                    'n_weak_individual': n_weak,
                    'n_datasets_with_pathway': len(individual_pvals),
                    'individual_pvalues': {ds: round(p, 6) for ds, p in individual_pvals.items()},
                    'amplification_description': (
                        f"No individual dataset had p<{individual_strong_threshold}, "
                        f"but {n_weak} had weak signals ({individual_weak_lower}-{individual_weak_upper}). "
                        f"Meta-analysis amplified to p={meta_pval:.2e}."
                    )
                })

        amplified.sort(key=lambda x: x['meta_pvalue'])
        return amplified

    def _is_trivial_pathway_pair(self, name_a: str, name_b: str, similarity_threshold: float = 0.7) -> bool:
        """Check if two pathway names are trivially related (substring or high similarity)."""
        a, b = name_a.lower().strip(), name_b.lower().strip()
        # Substring containment
        if a in b or b in a:
            return True
        # High string similarity (catches near-duplicates like
        # "Protein import and sorting" vs "Protein import, sorting and homeostasis")
        from difflib import SequenceMatcher
        if SequenceMatcher(None, a, b).ratio() > similarity_threshold:
            return True
        return False

    def _find_novel_cooccurrence_pairs(
        self,
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        fdr_threshold: float = 0.05,
        omnibus_threshold: float = 0.80,
        max_results: int = 50
    ) -> List[Dict[str, Any]]:
        """Type 6: Find novel pathway co-occurrence pairs.

        Pathway pairs that never co-occur as significant in any individual dataset
        but are grouped in the same meta theme.

        Omnibus datasets (>omnibus_threshold of all known pathways) are excluded
        because they trivially co-occur with everything.
        """
        meta_themes = self._extract_themes(meta_results)

        # Pre-compute per-dataset significant pathway sets
        ds_sig: Dict[str, Set[str]] = {}
        for ds_name, ds_res in dataset_results.items():
            pw_stats = self._extract_pathways_with_stats(ds_res)
            sig = set()
            for pw_name, stats in pw_stats.items():
                fdr = stats.get('pValueFDR') or stats.get('pValue') or 1.0
                if fdr < fdr_threshold:
                    sig.add(pw_name.lower().strip())
            ds_sig[ds_name] = sig

        # Identify and exclude omnibus datasets
        all_known = set()
        for s in ds_sig.values():
            all_known |= s
        n_all = len(all_known) if all_known else 1

        filtered_ds_sig = {
            ds: s for ds, s in ds_sig.items()
            if len(s) / n_all <= omnibus_threshold
        }
        if len(filtered_ds_sig) < 2:
            filtered_ds_sig = ds_sig  # Fall back

        # Build co-occurrence matrix from non-omnibus datasets
        cooccurring_pairs: Set[Tuple[str, str]] = set()
        for sig_set in filtered_ds_sig.values():
            sig_list = sorted(sig_set)
            for i in range(len(sig_list)):
                for j in range(i + 1, len(sig_list)):
                    cooccurring_pairs.add((sig_list[i], sig_list[j]))

        novel_pairs = []
        for theme in meta_themes:
            theme_name = theme.get('name', '')
            if not theme_name:
                continue
            theme_pw_names = sorted(self._get_theme_pathway_names(theme))
            if len(theme_pw_names) < 2:
                continue

            # Find pairs within this theme that never co-occur in individuals
            for i in range(len(theme_pw_names)):
                for j in range(i + 1, len(theme_pw_names)):
                    pair = (theme_pw_names[i], theme_pw_names[j])
                    if pair not in cooccurring_pairs:
                        if self._is_trivial_pathway_pair(pair[0], pair[1]):
                            continue  # Skip trivially related pathways (hierarchy artifact)
                        # Map back to original (non-normalized) names
                        orig_names = []
                        for pw in theme.get('pathways', []):
                            n = pw.get('name', pw.get('pathway', ''))
                            if n.lower().strip() in pair:
                                orig_names.append(n)
                        novel_pairs.append({
                            'pathway_a': orig_names[0] if len(orig_names) > 0 else pair[0],
                            'pathway_b': orig_names[1] if len(orig_names) > 1 else pair[1],
                            'theme_name': theme_name,
                            'description': (
                                f"'{orig_names[0] if orig_names else pair[0]}' and "
                                f"'{orig_names[1] if len(orig_names) > 1 else pair[1]}' "
                                f"are grouped in theme '{theme_name}' but were never "
                                f"both significant in the same individual dataset "
                                f"(excluding omnibus datasets)."
                            )
                        })

        return novel_pairs[:max_results]

    def _generate_discovery_hypotheses(
        self,
        emergent_theme_combinations: List[Dict[str, Any]],
        cross_dataset_convergence: List[Dict[str, Any]],
        weak_signal_amplification: List[Dict[str, Any]],
        novel_cooccurrence_pairs: List[Dict[str, Any]],
        enhanced_pathways: List[EnhancedSignificancePathway],
        significance_hierarchy: Optional[Dict[str, Any]],
        study_context: str,
        n_datasets: int
    ) -> List[Dict[str, Any]]:
        """Generate structured hypotheses from discovery findings using LLM.

        When the meta-analysis has no gene-level data (empty genes list), step 4
        produces no hypotheses. This method fills that gap by generating
        pathway-level hypotheses grounded in the discovery findings that are
        unique to meta-analysis (convergence, amplification, co-occurrence, hierarchy).

        Returns list of hypothesis dicts matching the _normalize_hypotheses() schema.
        """
        import json as _json

        # Build discovery data sections for the prompt
        sections = []

        if cross_dataset_convergence:
            items = []
            for c in cross_dataset_convergence[:10]:
                items.append(f"  - Theme '{c.get('theme_name', '')}': pathways {c.get('pathways', [])}, "
                             f"found in {c.get('n_datasets', '?')} datasets")
            sections.append("CROSS-DATASET CONVERGENCE (pathways independently significant across datasets):\n" + "\n".join(items))

        if weak_signal_amplification:
            items = []
            for w in weak_signal_amplification[:10]:
                items.append(f"  - '{w.get('pathway_name', w.get('pathway', ''))}': individually sub-significant in "
                             f"{w.get('n_datasets_subsig', '?')} datasets, meta p={w.get('meta_pvalue', '?')}")
            sections.append("WEAK SIGNAL AMPLIFICATION (pathways invisible individually, significant in meta):\n" + "\n".join(items))

        if novel_cooccurrence_pairs:
            items = []
            for p in novel_cooccurrence_pairs[:10]:
                items.append(f"  - '{p.get('pathway_a', '')}' + '{p.get('pathway_b', '')}' "
                             f"(theme: {p.get('theme_name', '')})")
            sections.append("NOVEL CO-OCCURRENCE PAIRS (each pathway may be individually significant, but they were never BOTH significant in the same single dataset — their co-occurrence is only visible in the meta-analysis):\n" + "\n".join(items))

        if significance_hierarchy and significance_hierarchy.get('tiers'):
            tier_items = []
            for t in significance_hierarchy['tiers']:
                tier_items.append(f"  - Tier {t['tier_number']}: {t['pathways'][:5]} "
                                  f"(p-value range: {t['pvalue_lower']:.2e} to {t['pvalue_upper']:.2e})")
            sections.append(f"SIGNIFICANCE HIERARCHY (meta resolves {significance_hierarchy.get('resolution_gain', '?')}x "
                            f"more ranking resolution):\n" + "\n".join(tier_items))

        if emergent_theme_combinations:
            items = []
            for e in emergent_theme_combinations[:10]:
                items.append(f"  - Theme '{e.get('theme_name', '')}': {e.get('description', '')}")
            sections.append("EMERGENT THEME COMBINATIONS (themes only visible when combining datasets):\n" + "\n".join(items))

        if not sections:
            return []

        # Track which grounding types have actual data for post-validation
        valid_grounding_types = set()
        if cross_dataset_convergence:
            valid_grounding_types.add('convergence')
        if weak_signal_amplification:
            valid_grounding_types.add('amplified_signal')
        if novel_cooccurrence_pairs:
            valid_grounding_types.add('novel_pair')
        if significance_hierarchy and significance_hierarchy.get('tiers'):
            valid_grounding_types.add('hierarchy')
        if emergent_theme_combinations:
            valid_grounding_types.add('emergent_theme')
        # Fallback default: first available grounding type (prefer convergence)
        default_grounding = 'convergence' if 'convergence' in valid_grounding_types else (
            next(iter(valid_grounding_types)) if valid_grounding_types else 'convergence'
        )

        discovery_text = "\n\n".join(sections)

        prompt = f"""Given these meta-analysis discovery findings from {n_datasets} {study_context} datasets:

{discovery_text}

Generate 3-5 structured mechanistic hypotheses. Each hypothesis MUST:
- Reference ONLY pathway names listed above (do NOT invent gene names or pathway names)
- Be grounded in a specific discovery type (convergence, amplified_signal, novel_pair, hierarchy, or emergent_theme)
- Include a testable directional prediction
- Include 1-2 experimental approaches for testability

Return ONLY a JSON array (no markdown fences, no commentary) where each element has these fields:
- "title": concise mechanistic title (string)
- "hypothesis_text": 1-3 sentence mechanism description (string)
- "confidence": "high", "medium", or "low" (string)
- "key_players": list of pathway names involved (array of strings)
- "supporting_pathways": list of pathways from the discovery findings (array of strings)
- "evidence": list of statistical evidence strings (array of strings)
- "mechanistic_model": how the pathways connect mechanistically (string)
- "directional_prediction": a testable prediction (string)
- "testability": object with 1-2 keys describing experimental approaches (object)
- "novelty": why this is a meta-analysis insight not visible in single studies (string)
- "grounding_type": one of "convergence", "amplified_signal", "novel_pair", "hierarchy", "emergent_theme" (string)
"""

        try:
            response = self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=20000,
                temperature=0,
                seed=42
            )

            # Parse JSON from response (strip markdown fences if present)
            text = response.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                if text.endswith('```'):
                    text = text[:-3]
                text = text.strip()

            # Fix common LLM JSON issues: trailing commas before } or ]
            import re as _re
            text = _re.sub(r',\s*([}\]])', r'\1', text)

            try:
                hypotheses_raw = _json.loads(text)
            except _json.JSONDecodeError:
                # Fallback: try to extract JSON array via regex
                json_match = _re.search(r'\[.*\]', text, _re.DOTALL)
                if json_match:
                    cleaned = _re.sub(r',\s*([}\]])', r'\1', json_match.group())
                    hypotheses_raw = _json.loads(cleaned)
                else:
                    raise
            if not isinstance(hypotheses_raw, list):
                hypotheses_raw = [hypotheses_raw]

            # Normalize confidence labels to scores and ensure schema compliance
            confidence_map = {'high': 0.8, 'medium': 0.6, 'low': 0.4}
            normalized = []
            for hyp in hypotheses_raw:
                confidence_raw = hyp.get('confidence', 'medium')
                if isinstance(confidence_raw, str):
                    confidence_score = confidence_map.get(confidence_raw.lower(), 0.5)
                    confidence_label = confidence_raw.lower()
                else:
                    confidence_score = float(confidence_raw) if confidence_raw else 0.5
                    confidence_label = 'high' if confidence_score >= 0.7 else ('medium' if confidence_score >= 0.5 else 'low')

                # Validate grounding_type against discovery sections with actual data
                grounding = hyp.get('grounding_type', default_grounding)
                if grounding not in valid_grounding_types:
                    grounding = default_grounding

                normalized.append({
                    'title': hyp.get('title', 'Untitled hypothesis'),
                    'hypothesis_text': hyp.get('hypothesis_text', ''),
                    'confidence_score': confidence_score,
                    'confidence_label': confidence_label,
                    'key_players': hyp.get('key_players', []),
                    'supporting_pathways': hyp.get('supporting_pathways', []),
                    'evidence': hyp.get('evidence', []),
                    'mechanistic_model': hyp.get('mechanistic_model', ''),
                    'directional_prediction': hyp.get('directional_prediction', ''),
                    'testability': hyp.get('testability', {}),
                    'novelty': hyp.get('novelty', ''),
                    'grounding_type': grounding,
                })

            return normalized

        except _json.JSONDecodeError as e:
            msg = f"Failed to parse discovery hypotheses JSON: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return []
        except Exception as e:
            msg = f"LLM failed in _generate_discovery_hypotheses: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return []

    def _generate_biological_interpretation(
        self,
        themes: List[MetaUniqueTheme],
        enhanced_pathways: List[EnhancedSignificancePathway],
        hypotheses: List[MetaUniqueHypothesis],
        mechanisms: List[CrossStudyMechanism],
        targets: List[MetaUniqueTarget],
        study_context: str,
        all_synthetic: bool = False,
        organizational_themes: List[Dict[str, Any]] = None,
        emergent_theme_combinations: List[Dict[str, Any]] = None,
        cross_dataset_convergence: List[Dict[str, Any]] = None,
        weak_signal_amplification: List[Dict[str, Any]] = None,
        novel_cooccurrence_pairs: List[Dict[str, Any]] = None,
        n_datasets: int = 0
    ) -> str:
        """Generate comprehensive biological interpretation of meta-analysis discoveries"""
        organizational_themes = organizational_themes or []
        emergent_theme_combinations = emergent_theme_combinations or []
        cross_dataset_convergence = cross_dataset_convergence or []
        weak_signal_amplification = weak_signal_amplification or []
        novel_cooccurrence_pairs = novel_cooccurrence_pairs or []

        # Build summary of findings
        findings_summary = []

        if all_synthetic and organizational_themes:
            org_names = [t['theme_name'] for t in organizational_themes[:5]]
            findings_summary.append(
                f"Organizational biological themes (grouping known pathways): {', '.join(org_names)}"
            )

        if themes:
            theme_names = [t.theme_name for t in themes[:5]]
            findings_summary.append(f"Genuinely novel biological themes: {', '.join(theme_names)}")

        if enhanced_pathways:
            top_pathways = [p.pathway_name for p in enhanced_pathways[:5]]
            findings_summary.append(f"Most enhanced pathways: {', '.join(top_pathways)}")

        if hypotheses:
            hyp_titles = [h.hypothesis_title for h in hypotheses[:3]]
            findings_summary.append(f"Meta-unique hypotheses: {', '.join(hyp_titles)}")

        if targets:
            target_genes = [t.gene_symbol for t in targets[:5]]
            findings_summary.append(f"Meta-unique therapeutic targets: {', '.join(target_genes)}")

        # New emergent discovery types
        if emergent_theme_combinations:
            theme_lines = [f"  - {e['theme_name']}: {', '.join(e.get('pathways', [])[:5])}"
                           for e in emergent_theme_combinations[:3]]
            findings_summary.append(
                f"Emergent theme combinations ({len(emergent_theme_combinations)}):\n"
                + "\n".join(theme_lines)
            )

        if cross_dataset_convergence:
            names = [c['theme_name'] for c in cross_dataset_convergence[:3]]
            findings_summary.append(
                f"Cross-dataset convergence ({len(cross_dataset_convergence)} themes where pathways "
                f"come from largely independent datasets): {', '.join(names)}"
            )

        if weak_signal_amplification:
            pw_names = [w['pathway'] for w in weak_signal_amplification[:3]]
            findings_summary.append(
                f"Weak signal amplification ({len(weak_signal_amplification)} hidden signals "
                f"amplified by meta-analysis): {', '.join(pw_names)}"
            )

        if novel_cooccurrence_pairs:
            pair_lines = [f"  - '{p['pathway_a']}' + '{p['pathway_b']}' (theme: {p['theme_name']})"
                          for p in novel_cooccurrence_pairs[:5]]
            findings_summary.append(
                f"Novel co-occurrence pairs ({len(novel_cooccurrence_pairs)} total):\n"
                + "\n".join(pair_lines)
            )

        if not findings_summary:
            return "No significant meta-unique discoveries were identified in this analysis."

        if all_synthetic:
            context_note = (
                "\n\n**Important context:** Individual datasets were analyzed using raw pathway "
                "data only (no full pipeline run). Therefore, themes from meta-analysis represent "
                "organizational groupings of pathways that were already present in individual "
                "datasets, not genuinely unique discoveries. The primary value of this meta-analysis "
                "is emergent patterns visible only when combining datasets."
            )
        else:
            context_note = ""

        prompt = f"""Provide a comprehensive biological interpretation of the following meta-analysis discoveries.

**Study Context:** {study_context}{context_note}

**Key Findings:**
{chr(10).join('- ' + f for f in findings_summary)}

**Number of Datasets Combined:** {n_datasets} independent studies

Write a 3-4 paragraph interpretation that:
1. Explains the biological significance of the emergent discoveries (theme combinations, convergence, amplified signals)
2. Discusses how combining datasets revealed patterns invisible in individual studies
3. Suggests next steps for validation

Focus on genuine emergent patterns: pathway groupings no single study could reveal, weak signals amplified across studies, and novel co-occurrence relationships.

IMPORTANT: Only reference pathway names and relationships that appear in the findings above.
Do not invent specific gene names or pathway pairings not listed in the data."""

        try:
            response = self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1500,
                temperature=0,
                seed=42
            )
            return response.strip()
        except Exception as e:
            msg = f"LLM failed in _generate_biological_interpretation: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return f"Biological interpretation could not be generated: {str(e)}"

    def _generate_key_discoveries(
        self,
        themes: List[MetaUniqueTheme],
        enhanced_pathways: List[EnhancedSignificancePathway],
        hypotheses: List[MetaUniqueHypothesis],
        targets: List[MetaUniqueTarget],
        all_synthetic: bool = False,
        organizational_themes: List[Dict[str, Any]] = None,
        significance_hierarchy: Dict[str, Any] = None,
        emergent_theme_combinations: List[Dict[str, Any]] = None,
        cross_dataset_convergence: List[Dict[str, Any]] = None,
        weak_signal_amplification: List[Dict[str, Any]] = None,
        novel_cooccurrence_pairs: List[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate list of key discovery statements"""
        organizational_themes = organizational_themes or []
        significance_hierarchy = significance_hierarchy or {}
        emergent_theme_combinations = emergent_theme_combinations or []
        cross_dataset_convergence = cross_dataset_convergence or []
        weak_signal_amplification = weak_signal_amplification or []
        novel_cooccurrence_pairs = novel_cooccurrence_pairs or []
        discoveries = []

        if all_synthetic:
            # Honest framing for synthetic results
            if organizational_themes:
                discoveries.append(
                    f"Organized pathways into {len(organizational_themes)} biological theme(s) "
                    f"that group known pathways into coherent narratives: "
                    f"{', '.join(t['theme_name'] for t in organizational_themes[:3])}"
                )

            if themes:
                discoveries.append(
                    f"Identified {len(themes)} genuinely novel biological theme(s) with pathways "
                    f"not found in individual datasets: "
                    f"{', '.join(t.theme_name for t in themes[:3])}"
                )

            if enhanced_pathways:
                top = enhanced_pathways[0]
                discoveries.append(
                    f"Found {len(enhanced_pathways)} pathways with significantly enhanced statistical power "
                    f"(e.g., {top.pathway_name} - {top.significance_gain_fold:.1f}x more significant in meta)"
                )
        else:
            # Standard framing for full pipeline results
            if themes:
                discoveries.append(
                    f"Identified {len(themes)} biological theme(s) visible only through meta-analysis: "
                    f"{', '.join(t.theme_name for t in themes[:3])}"
                )

            if enhanced_pathways:
                top = enhanced_pathways[0]
                discoveries.append(
                    f"Found {len(enhanced_pathways)} pathways with significantly enhanced statistical power "
                    f"(e.g., {top.pathway_name} - {top.significance_gain_fold:.1f}x more significant in meta)"
                )

            if hypotheses:
                discoveries.append(
                    f"Generated {len(hypotheses)} novel mechanistic hypothesis(es) unique to meta-analysis"
                )

            if targets:
                discoveries.append(
                    f"Identified {len(targets)} therapeutic target(s) detectable only through combined analysis: "
                    f"{', '.join(t.gene_symbol for t in targets[:5])}"
                )

        # New emergent discovery types (apply to both synthetic and full)
        if emergent_theme_combinations:
            discoveries.append(
                f"Discovered {len(emergent_theme_combinations)} emergent theme combination(s) — "
                f"pathway groupings where no single dataset had all pathways significant together"
            )

        if cross_dataset_convergence:
            discoveries.append(
                f"Found {len(cross_dataset_convergence)} theme(s) with cross-dataset convergence — "
                f"pathways contributed by independent datasets, revealing hidden biological connections"
            )

        if weak_signal_amplification:
            top_amp = weak_signal_amplification[0]
            discoveries.append(
                f"Amplified {len(weak_signal_amplification)} weak signal(s) — individually marginal "
                f"pathways reaching strong meta-significance (e.g., {top_amp['pathway']} meta p={top_amp['meta_pvalue']:.2e})"
            )

        if novel_cooccurrence_pairs:
            discoveries.append(
                f"Revealed {len(novel_cooccurrence_pairs)} novel pathway co-occurrence pair(s) — "
                f"pathways grouped by meta-analysis that were never both significant in any single study"
            )

        # Add ranking insight if hierarchy was computed (applies to both synthetic and full)
        tiers = significance_hierarchy.get('tiers', [])
        resolution_gain = significance_hierarchy.get('resolution_gain', 0)
        if len(tiers) >= 2 and resolution_gain >= 2.0:
            top_tier = tiers[0]
            top_names = top_tier['pathways'][:3]
            discoveries.append(
                f"Meta-analysis revealed a pathway priority hierarchy not visible in individual studies "
                f"({resolution_gain:.0f}x more ranking resolution): "
                f"top-priority pathways include {', '.join(top_names)}"
            )

        return discoveries

    def _compute_significance_hierarchy(
        self,
        enhanced_pathways: List[EnhancedSignificancePathway]
    ) -> Dict[str, Any]:
        """Compute significance hierarchy from enhanced pathways.

        The key insight: individual studies often have p-values clustered in a narrow
        range (all pathways look equally important), while meta-analysis spreads them
        across many orders of magnitude, revealing a clear priority ranking.

        Returns dict with:
        - individual_log10_spread: orders of magnitude spanned by individual best p-values
        - meta_log10_spread: orders of magnitude spanned by meta p-values
        - resolution_gain: how much more ranking resolution meta provides
        - tiers: list of {tier_label, pvalue_lower, pvalue_upper, pathways: [...]}
        """
        if len(enhanced_pathways) < 2:
            return {}

        # Collect p-values (filter out inf/zero/None)
        ind_pvals = [
            p.best_individual_pvalue for p in enhanced_pathways
            if p.best_individual_pvalue and 0 < p.best_individual_pvalue < 1.0
        ]
        meta_pvals = [
            p.meta_pvalue for p in enhanced_pathways
            if p.meta_pvalue and p.meta_pvalue > 0
        ]

        if len(ind_pvals) < 2 or len(meta_pvals) < 2:
            return {}

        # Compute log10 spreads
        ind_log_max = -math.log10(min(ind_pvals))  # most significant
        ind_log_min = -math.log10(max(ind_pvals))  # least significant
        ind_spread = ind_log_max - ind_log_min

        meta_log_max = -math.log10(min(meta_pvals))  # most significant
        meta_log_min = -math.log10(max(meta_pvals))  # least significant
        meta_spread = meta_log_max - meta_log_min

        resolution_gain = meta_spread / ind_spread if ind_spread > 0 else float('inf')

        # Only report hierarchy if meta has meaningfully more spread (>2x)
        if resolution_gain < 2.0:
            return {
                'individual_log10_spread': round(ind_spread, 1),
                'meta_log10_spread': round(meta_spread, 1),
                'resolution_gain': round(resolution_gain, 1),
                'tiers': []
            }

        # Auto-compute tiers: divide meta log10 range into bins of ~3 orders of magnitude
        bin_size = 3.0
        n_tiers = max(2, min(5, int(math.ceil(meta_spread / bin_size))))
        actual_bin_size = meta_spread / n_tiers

        tiers = []
        for i in range(n_tiers):
            tier_log_upper = meta_log_max - i * actual_bin_size       # most significant end
            tier_log_lower = meta_log_max - (i + 1) * actual_bin_size  # least significant end

            # Convert back to p-values (lower log = higher p-value)
            pvalue_lower = 10 ** (-tier_log_upper)   # smallest p-value in tier (most significant)
            pvalue_upper = 10 ** (-tier_log_lower)   # largest p-value in tier (least significant)

            # Find pathways in this tier
            tier_pathways = []
            for p in enhanced_pathways:
                if p.meta_pvalue <= 0:
                    continue
                p_log = -math.log10(p.meta_pvalue)
                # Include in tier if within bounds (inclusive on lower bound for last tier)
                if i < n_tiers - 1:
                    if tier_log_lower < p_log <= tier_log_upper:
                        tier_pathways.append(p.pathway_name)
                else:
                    if tier_log_lower <= p_log <= tier_log_upper:
                        tier_pathways.append(p.pathway_name)

            if tier_pathways:
                tiers.append({
                    'tier_number': i + 1,
                    'pvalue_lower': pvalue_lower,
                    'pvalue_upper': pvalue_upper,
                    'n_pathways': len(tier_pathways),
                    'pathways': tier_pathways
                })

        return {
            'individual_log10_spread': round(ind_spread, 1),
            'meta_log10_spread': round(meta_spread, 1),
            'resolution_gain': round(resolution_gain, 1),
            'n_tiers': len(tiers),
            'tiers': tiers
        }

    def _extract_and_group_metadata(
        self,
        dataset_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict], List[str], Dict[str, Dict[str, List[str]]]]:
        """
        Extract and group metadata from all datasets generically.

        Returns:
            Tuple of:
            - pathway_details_by_dataset: {dataset_name: {metadata, pathways, hub_genes}}
            - metadata_fields: list of all unique metadata field names found
            - metadata_groupings: {field_name: {value: [dataset_names]}}
        """
        pathway_details_by_dataset = {}
        all_metadata_values = {}  # field_name -> {value -> [datasets]}

        for ds_name, results in dataset_results.items():
            # Extract metadata from step1 results
            step1 = results.get('step1_pathway_themes', {})
            if not step1:
                step1 = results.get('steps', {}).get('step1', {})

            metadata = {}
            # Try step1.metadata.context first (full pipeline results)
            context = {}
            if step1:
                context = step1.get('metadata', {}).get('context', {})
            # Fallback: synthetic results store metadata at results['input']['metadata']
            if not context:
                context = results.get('input', {}).get('metadata', {})
            if context:
                # Extract ALL metadata fields (no hardcoded field names)
                for key, value in context.items():
                    if key not in ('datasetId', 'dataset_id') and value:
                        metadata[key] = str(value)
                        # Track for grouping
                        if key not in all_metadata_values:
                            all_metadata_values[key] = {}
                        str_value = str(value)
                        if str_value not in all_metadata_values[key]:
                            all_metadata_values[key][str_value] = []
                        all_metadata_values[key][str_value].append(ds_name)

            # Extract pathways with direction (ES values)
            pathways_with_direction = []
            pathways = self._extract_pathways_with_stats(results)
            for pw_name, pw_stats in pathways.items():
                es = pw_stats.get('ES') or pw_stats.get('es')
                direction = 'up' if es and es > 0 else 'down' if es and es < 0 else 'mixed'
                pathways_with_direction.append({
                    'name': pw_name,
                    'direction': direction,
                    'pValue': pw_stats.get('pValue') or pw_stats.get('p_value'),
                    'ES': es
                })
            # Sort by p-value
            pathways_with_direction.sort(key=lambda x: x.get('pValue') or 1.0)

            # Extract hub genes from step2 results
            hub_genes = []
            step2 = results.get('step2_hub_genes', {})
            if not step2:
                step2 = results.get('steps', {}).get('step2', {})
            if step2:
                network_hubs = step2.get('network_hubs', []) or step2.get('hub_genes', [])
                for hub in network_hubs[:10]:  # Top 10 hub genes
                    if isinstance(hub, dict):
                        hub_genes.append(hub.get('gene', hub.get('symbol', '')))
                    elif isinstance(hub, str):
                        hub_genes.append(hub)

            pathway_details_by_dataset[ds_name] = {
                'metadata': metadata,
                'pathways': pathways_with_direction[:15],  # Top 15 pathways
                'hub_genes': hub_genes
            }

        # Identify metadata fields and create groupings
        # Only include fields that have multiple distinct values (useful for grouping)
        metadata_fields = list(all_metadata_values.keys())
        metadata_groupings = {}
        for field_name, value_map in all_metadata_values.items():
            if len(value_map) > 1:  # Only include if there are multiple values
                metadata_groupings[field_name] = value_map

        return pathway_details_by_dataset, metadata_fields, metadata_groupings

    def _aggregate_individual_findings(
        self,
        dataset_results: Dict[str, Dict[str, Any]],
        dataset_names: List[str],
        study_context: str
    ) -> Dict[str, Any]:
        """
        Aggregate findings across all individual datasets to establish baseline.

        Returns dict with:
        - total_unique_pathways: int
        - pathways_in_multiple: int
        - pathways_in_single: int
        - top_pathways: list of {name, count, datasets, best_pvalue}
        - llm_summary: natural language summary
        - pathway_details_by_dataset: per-dataset pathways with directions
        - metadata_fields: list of metadata fields found
        """
        # Extract per-dataset details using the generic helper
        pathway_details_by_dataset, metadata_fields, _metadata_groupings = \
            self._extract_and_group_metadata(dataset_results)

        # Count pathway frequency across datasets
        pathway_counts = {}

        for ds_name, results in dataset_results.items():
            pathways = self._extract_pathways_with_stats(results)
            for pw_name, stats in pathways.items():
                if pw_name not in pathway_counts:
                    pathway_counts[pw_name] = {
                        'count': 0,
                        'datasets': [],
                        'best_pvalue': 1.0,
                        'pvalues': []
                    }
                pathway_counts[pw_name]['count'] += 1
                pathway_counts[pw_name]['datasets'].append(ds_name)
                pval = stats.get('pValue') or stats.get('p_value') or 1.0
                if pval and pval < pathway_counts[pw_name]['best_pvalue']:
                    pathway_counts[pw_name]['best_pvalue'] = pval
                if pval:
                    pathway_counts[pw_name]['pvalues'].append(pval)

        # Calculate statistics
        total_unique = len(pathway_counts)
        in_multiple = sum(1 for p in pathway_counts.values() if p['count'] >= 2)
        in_single = sum(1 for p in pathway_counts.values() if p['count'] == 1)

        # Sort by frequency
        top_pathways = sorted(
            [{'name': name, **stats} for name, stats in pathway_counts.items()],
            key=lambda x: (-x['count'], x['best_pvalue'])
        )[:20]

        # Generate LLM summary of individual findings with enriched data
        llm_summary = self._generate_individual_findings_summary(
            dataset_names,
            pathway_counts,
            top_pathways,
            study_context,
            pathway_details_by_dataset
        )

        return {
            'total_datasets': len(dataset_names),
            'total_unique_pathways': total_unique,
            'pathways_in_multiple': in_multiple,
            'pathways_in_single': in_single,
            'top_pathways': top_pathways,
            'pathway_counts': pathway_counts,
            'llm_summary': llm_summary,
            'pathway_details_by_dataset': pathway_details_by_dataset,
            'metadata_fields': metadata_fields
        }

    def _generate_individual_findings_summary(
        self,
        dataset_names: List[str],
        pathway_counts: Dict[str, Dict],
        top_pathways: List[Dict],
        study_context: str,
        pathway_details_by_dataset: Dict[str, Dict] = None
    ) -> str:
        """Generate natural language summary of individual dataset findings."""

        # Build pathway frequency text
        pathway_lines = []
        for pw in top_pathways[:10]:
            pathway_lines.append(f"- {pw['name']}: found in {pw['count']}/{len(dataset_names)} studies (best p={pw['best_pvalue']:.2e})")
        pathway_text = '\n'.join(pathway_lines)

        total_unique = len(pathway_counts)
        in_multiple = sum(1 for p in pathway_counts.values() if p['count'] >= 2)
        in_3_plus = sum(1 for p in pathway_counts.values() if p['count'] >= 3)

        prompt = f"""Summarize the findings from {len(dataset_names)} individual studies BEFORE meta-analysis.

**Study Context:** {study_context}

**Top Pathways by Frequency Across Individual Studies:**
{pathway_text}

**Statistics:**
- Total unique pathways identified: {total_unique}
- Pathways found in >=2 studies: {in_multiple}
- Pathways found in >=3 studies: {in_3_plus}
- Pathways found in only 1 study: {total_unique - in_multiple}

Write 2-3 paragraphs describing:
1. What are the main biological pathways found across individual studies?
2. What do these pathways tell us about the biological response being studied?
3. What are the limitations - why might some pathways appear in few studies?

Focus on biological meaning. This establishes the BASELINE before showing what meta-analysis adds."""

        try:
            response = self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=800,
                temperature=0,
                seed=42
            )
            return response.strip()
        except Exception as e:
            msg = f"LLM failed in _generate_individual_findings_summary: {e}"
            logger.warning(msg)
            self._warnings.append(msg)
            return f"Individual findings summary could not be generated: {str(e)}"
