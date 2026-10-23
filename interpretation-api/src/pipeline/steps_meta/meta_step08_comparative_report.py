"""
Meta-Step 08: Comparative Report Generation

Generates the meta-analysis report. It mirrors the single-route report: the
combined-data run's step-6 report supplies the Executive Summary and analysis
sections, and "Findings Across Individual Datasets" is the one meta-only section.
"""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.groq_client import GroqClient


class MetaStep08ComparativeReport:
    """
    Meta-Step 08: Comparative Report Generation

    Generates markdown report comparing meta-analysis with individual datasets.
    No visualizations - just text, tables, and summary statistics.
    """

    def __init__(self):
        self.step_number = 8
        self.step_name = 'Comparative Report Generation'
        self.llm = GroqClient()

    def execute(
        self,
        reproducibility_results: Dict[str, Any],
        discovery_results: Dict[str, Any],
        meta_results: Dict[str, Any],
        dataset_results: Dict[str, Dict[str, Any]],
        dataset_names: List[str],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Execute comparative report generation

        Args:
            reproducibility_results: Output from Meta-Step 7
            discovery_results: Output from Meta-Step 9 (meta-unique discoveries)
            meta_results: Pipeline results from meta-analysis
            dataset_results: Dict mapping dataset_name -> pipeline results
            dataset_names: List of dataset names
            metadata: Full metadata dict with flexible fields

        Returns:
            Markdown report as string
        """
        print(f'\n[Meta-Step {self.step_number}] {self.step_name}')
        print('  Generating comparative report...')

        # Extract display info from flexible metadata
        metadata = metadata or {}
        discovery_results = discovery_results or {}
        study_type = (
            metadata.get('disease') or
            metadata.get('Dataset') or
            metadata.get('study_type') or
            metadata.get('description') or
            'Meta-analysis'
        )
        tissue = (
            metadata.get('tissue') or
            metadata.get('Tissue type') or
            metadata.get('tissue_type') or
            metadata.get('Tissue') or
            ''
        )

        meta_results = meta_results or {}
        individual = discovery_results.get('individual_findings') or {}
        n_selected = (meta_results.get('input_summary') or {}).get('num_pathways')

        # The meta report mirrors the single-route report: the combined-data run's own step-6
        # report supplies the Executive Summary and the analysis sections, and the per-dataset
        # findings are the one meta-only section. Removed sections are commented out below,
        # never deleted; each generator is intact, so re-enabling one is uncommenting a line.
        step6_chunks = self._split_markdown_sections(
            ((meta_results.get('steps') or {}).get('step5') or {}).get('markdown_content') or ''
        )
        step6_by_kind = {}
        for kind, heading, body in step6_chunks:
            step6_by_kind.setdefault(kind, []).append((kind, heading, body))

        # Ordered (number?, heading text, body). Numbered sections are renumbered 1..N
        # contiguously below; Executive Summary and Appendices stay unnumbered, as in step 6.
        ordered = []

        # Executive Summary (step 6) + one line of real meta facts
        meta_facts = self._meta_facts_line(individual, n_selected)
        if step6_by_kind.get('executive_summary'):
            for _, heading, body in step6_by_kind['executive_summary']:
                ordered.append((False, heading, body.rstrip() + '\n\n' + meta_facts))
        elif step6_chunks:
            ordered.append((False, 'Executive Summary', meta_facts))
        else:
            ordered.append((False, 'Executive Summary', '\n'.join([
                meta_facts,
                '',
                '> **The combined-data report was not produced**, so the theme, hub-gene, '
                'mechanism and hypothesis sections are missing from this report.',
                '> Error: ' + str(meta_results.get('error') or 'no step-6 report in the combined-data results'),
            ])))

        # Findings Across Individual Datasets: the only meta-only section
        if individual:
            heading, body = self._split_own_section(
                self._generate_individual_studies_summary(discovery_results, n_selected)
            )
            ordered.append((True, heading, body))

        # Step-6 analysis sections in step-6 order, body text verbatim. An unrecognised '##'
        # chunk is kept where it appeared, never dropped.
        for kind, heading, body in step6_chunks:
            if kind in ('executive_summary', 'appendices'):
                continue
            if kind == 'conclusions':
                body = body.rstrip() + '\n\n' + self._generate_dataset_conclusions(
                    individual, n_selected, meta_results
                )
            ordered.append((True, heading, body))

        for _, heading, body in step6_by_kind.get('appendices', []):
            ordered.append((False, heading, body))

        # --- Removed sections (meta-report restructure). Commented out, not deleted. ---
        # Key Meta-Analysis Findings / Discoveries
        # sections.append(self._generate_discovery_summary(discovery_results))
        # Old executive summary (reproducibility + power gain) — replaced by step 6's
        # sections.append(self._generate_executive_summary(reproducibility_results, discovery_results))
        # Meta-Analysis Insights: its themes and hypotheses duplicate step 1 and step 4; the
        # meta-unique hypotheses/targets subsections are permanently empty (all_synthetic)
        # sections.append(self._generate_meta_unique_discoveries_section(discovery_results))
        # Enhanced Significance: Power Gain, incl. Priority Hierarchy and full table (#11, #33)
        # sections.append(self._generate_enhanced_significance_section(discovery_results))
        # Cross-Study Validated Mechanisms: structurally empty when all_synthetic
        # sections.append(self._generate_cross_study_validation_section(discovery_results))
        # Emergent Theme Combinations
        # sections.append(self._generate_emergent_theme_combinations_section(discovery_results))
        # Cross-Dataset Pathway Convergence
        # sections.append(self._generate_cross_dataset_convergence_section(discovery_results))
        # Weak Signal Amplification: cannot fire. The finder wants per-dataset p in 0.01-0.05,
        # but per-dataset lists are pre-filtered at pValueFDR < 0.05 (cpa-website
        # aiWorkflow.js:344), so that band is empty by construction. Reviving it needs
        # unfiltered per-dataset pathway results.
        # sections.append(self._generate_weak_signal_amplification_section(discovery_results))
        # Novel Pathway Co-occurrence Pairs (#24)
        # sections.append(self._generate_novel_cooccurrence_pairs_section(discovery_results))
        # Reproducibility Overview (#25-#29: wrong word, union denominator)
        # sections.append(self._generate_reproducibility_overview(reproducibility_results))
        # Direction Concordance Analysis (#35-#37: the 99.0% is a sparsity artifact)
        # sections.append(self._generate_concordance_analysis(reproducibility_results))
        # Biological Interpretation of Meta-Analysis Findings (#45-#47)
        # sections.append(self._generate_biological_interpretation_section(discovery_results))
        # Old Conclusions: built almost entirely on the sections above; step 6's own
        # Conclusions plus _generate_dataset_conclusions replace it
        # sections.append(self._generate_conclusions(reproducibility_results, discovery_results))

        sections = [self._generate_header(study_type, tissue, len(dataset_names), metadata)]
        number = 0
        for numbered, heading, body in ordered:
            if numbered:
                number += 1
                heading = f'{number}. {heading}'
            sections.append(f'## {heading}\n\n{body.strip()}')

        report = '\n\n'.join(sections)

        print(f'  ✓ Report generated ({len(report)} characters)')

        return report

    # Step-6 section kinds, matched on the heading text after any "N. " prefix
    # (step 6 numbers its sections; step 3/4 report_section headings may differ).
    _STEP6_SECTION_KINDS = (
        ('Executive Summary', 'executive_summary'),
        ('Pathway Theme', 'themes'),
        ('Network Hub', 'hub_genes'),
        ('Pathway Mechanism', 'mechanisms'),
        ('Mechanistic Hypothes', 'hypotheses'),
        ('Therapeutic', 'therapeutics'),
        ('Summary and Conclusions', 'conclusions'),
        ('Appendices', 'appendices'),
    )

    @classmethod
    def _split_markdown_sections(cls, markdown: str) -> List[Tuple[str, str, str]]:
        """Split a step-6 report into ordered (kind, heading text, body) at its '## ' headings.

        The heading text has step 6's "N. " prefix removed so the meta report can renumber.
        Whatever precedes the first '## ' (step 6's title and metadata header) is dropped:
        the meta header already carries the metadata. Unknown headings get kind 'other'.
        """
        chunks = []
        heading, body = None, []
        for line in markdown.splitlines():
            if line.startswith('## '):
                if heading is not None:
                    chunks.append((heading, '\n'.join(body)))
                heading, body = line[3:].strip(), []
            elif heading is not None:
                body.append(line)
        if heading is not None:
            chunks.append((heading, '\n'.join(body)))

        result = []
        for heading, body in chunks:
            text = re.sub(r'^\d+\.\s*', '', heading)
            kind = next((k for prefix, k in cls._STEP6_SECTION_KINDS if text.startswith(prefix)), 'other')
            result.append((kind, text, body))
        return result

    @staticmethod
    def _split_own_section(section: str) -> Tuple[str, str]:
        """Split one generated '## Heading' section into (heading text, body)."""
        first, _, rest = section.lstrip().partition('\n')
        return first[3:].strip() if first.startswith('## ') else first.strip(), rest

    @staticmethod
    def _meta_facts_line(individual: Dict[str, Any], n_selected: Any) -> str:
        """The one meta-aware line added to step 6's dataset-blind Executive Summary."""
        n_datasets = individual.get('total_datasets')
        in_multiple = individual.get('pathways_in_multiple')
        total_unique = individual.get('total_unique_pathways')

        datasets_part = (f'{n_datasets} datasets pooled' if n_datasets is not None
                         else 'number of pooled datasets unavailable')
        if in_multiple is None:
            pathways_part = 'per-dataset pathway frequencies unavailable'
        elif n_selected is None:
            pathways_part = (f'{in_multiple} pathways are significant (FDR < 0.05) in two or more '
                             'datasets (the number of selected pathways is unavailable)')
        elif total_unique is not None and total_unique > n_selected:
            pathways_part = (f'{in_multiple} pathways are significant (FDR < 0.05) in two or more '
                             f'datasets; the per-dataset lists hold {total_unique} pathways, more than '
                             f'the {n_selected} selected, so they were not restricted to the selection')
        else:
            pathways_part = (f'{in_multiple} of {n_selected} selected pathways are significant '
                             '(FDR < 0.05) in two or more datasets')
        return f'**Meta-analysis scope:** {datasets_part}; {pathways_part}.'

    @staticmethod
    def _generate_dataset_conclusions(
        individual: Dict[str, Any], n_selected: Any, meta_results: Dict[str, Any]
    ) -> str:
        """Per-dataset frequency of the selected pathways, appended to step 6's Conclusions."""
        lines = ['### Across Individual Datasets', '']
        if individual:
            total_datasets = individual.get('total_datasets', 0)
            total_unique = individual.get('total_unique_pathways', 0)
            in_multiple = individual.get('pathways_in_multiple', 0)
            in_single = individual.get('pathways_in_single', 0)
            lines.append(f'Selected pathways by the number of the {total_datasets} individual datasets '
                         'in which they are significant (FDR < 0.05):')
            lines.append('')
            lines.append(f'- significant in two or more datasets: {in_multiple}')
            lines.append(f'- significant in exactly one dataset: {in_single}')
            if n_selected is not None and n_selected >= total_unique:
                lines.append(f'- significant in no individual dataset: {n_selected - total_unique} '
                             f'(of {n_selected} selected)')
            elif n_selected is not None:
                lines.append(f'- the per-dataset lists hold {total_unique} pathways, more than the '
                             f'{n_selected} selected; they were not restricted to the selection')
        else:
            lines.append('*Per-dataset pathway frequencies are unavailable for this run.*')

        step4 = (meta_results.get('steps') or {}).get('step4') or {}
        hypotheses = step4.get('hypotheses') or []
        lines.append('')
        if hypotheses:
            noun = 'hypothesis' if len(hypotheses) == 1 else 'hypotheses'
            lines.append(f'The combined-data analysis generated {len(hypotheses)} mechanistic '
                         f'{noun} (see Mechanistic Hypotheses above).')
        else:
            lines.append('No mechanistic hypotheses were generated from the combined data.')
        return '\n'.join(lines)

    def _generate_discovery_summary(self, discovery_results: Dict[str, Any]) -> str:
        """Generate prominent summary of meta-analysis discoveries"""
        summary = discovery_results.get('summary', {})
        key_discoveries = discovery_results.get('key_discoveries', [])
        all_synthetic = summary.get('all_synthetic', False)

        if all_synthetic:
            lines = [
                '## Key Meta-Analysis Findings',
                '',
                '> The primary value of this meta-analysis is **enhanced statistical significance**',
                '> from combining multiple datasets. Biological themes organize pathways already',
                '> found in individual studies into coherent narratives.',
                ''
            ]
        else:
            lines = [
                '## Key Meta-Analysis Discoveries',
                '',
                '> **These findings were identified ONLY through meta-analysis** - they were not',
                '> detectable in any individual study, demonstrating the unique value of combining',
                '> multiple datasets.',
                ''
            ]

        if key_discoveries:
            for discovery in key_discoveries:
                lines.append(f'- {discovery}')
            lines.append('')

        # Summary stats — different table for synthetic vs full
        if all_synthetic:
            enhanced_count = summary.get('enhanced_significance_pathways_count', 0)
            org_themes_count = summary.get('organizational_themes_count', 0)
            novel_themes_count = summary.get('meta_unique_themes_count', 0)
            generated_hyp_count = summary.get('meta_generated_hypotheses_count', 0)

            lines.extend([
                '### Finding Statistics',
                '',
                '| Finding Type | Count | Note |',
                '|--------------|-------|------|',
                f'| Enhanced significance pathways | {enhanced_count} | Genuine power gain |',
                f'| Organizational biological themes | {org_themes_count} | Group known pathways |',
                f'| Genuinely novel themes | {novel_themes_count} | Pathways not in individuals |',
                f'| Meta-generated hypotheses | {generated_hyp_count} | Not claimed as unique |',
                f'| Emergent theme combinations | {summary.get("emergent_theme_combinations_count", 0)} | Pathway groupings invisible in single studies |',
                f'| Cross-dataset convergence | {summary.get("cross_dataset_convergence_count", 0)} | Independent datasets contribute different pathways |',
                f'| Weak signal amplification | {summary.get("weak_signal_amplification_count", 0)} | Marginal signals amplified by meta |',
                f'| Novel co-occurrence pairs | {summary.get("novel_cooccurrence_pairs_count", 0)} | Never co-significant in any single study |',
            ])
        else:
            total_discoveries = summary.get('total_discoveries', 0)
            if total_discoveries > 0:
                lines.extend([
                    '### Discovery Statistics',
                    '',
                    '| Discovery Type | Count |',
                    '|----------------|-------|',
                    f'| Meta-unique biological themes | {summary.get("meta_unique_themes_count", 0)} |',
                    f'| Meta-unique hypotheses | {summary.get("meta_unique_hypotheses_count", 0)} |',
                    f'| Meta-unique therapeutic targets | {summary.get("meta_unique_targets_count", 0)} |',
                    f'| Enhanced significance pathways | {summary.get("enhanced_significance_pathways_count", 0)} |',
                    f'| Cross-study validated mechanisms | {summary.get("cross_study_mechanisms_count", 0)} |',
                    f'| Emergent theme combinations | {summary.get("emergent_theme_combinations_count", 0)} |',
                    f'| Cross-dataset convergence | {summary.get("cross_dataset_convergence_count", 0)} |',
                    f'| Weak signal amplification | {summary.get("weak_signal_amplification_count", 0)} |',
                    f'| Novel co-occurrence pairs | {summary.get("novel_cooccurrence_pairs_count", 0)} |',
                    f'| **Total unique discoveries** | **{total_discoveries}** |',
                ])
            else:
                lines.append('*No meta-unique discoveries were identified in this analysis.*')

        return '\n'.join(lines)

    def _generate_meta_unique_discoveries_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate detailed section on meta-unique discoveries"""
        summary = discovery_results.get('summary', {})
        all_synthetic = summary.get('all_synthetic', False)

        if all_synthetic:
            lines = [
                '## Meta-Analysis Insights',
                '',
                'This section describes biological insights from the meta-analysis, distinguishing',
                'between organizational groupings of known pathways and genuinely novel findings.',
                ''
            ]
        else:
            lines = [
                '## Meta-Unique Discoveries: The Value of Meta-Analysis',
                '',
                'These biological insights emerged ONLY from meta-analysis. They represent the',
                'true power gain from combining multiple independent studies.',
                ''
            ]

        # Organizational themes (synthetic only)
        organizational_themes = discovery_results.get('organizational_themes', [])
        if organizational_themes:
            lines.extend([
                '### Biological Themes from Meta-Analysis',
                '',
                'These themes organize pathways already found in individual studies into',
                'coherent biological narratives. While not novel discoveries, they provide',
                'useful high-level groupings of the data:',
                ''
            ])
            for theme in organizational_themes[:10]:
                theme_name = theme.get('theme_name', 'Unknown')
                pathways = theme.get('pathways', [])[:5]
                description = theme.get('description', '')

                lines.append(f'#### {theme_name}')
                lines.append('')
                if pathways:
                    lines.append(f'**Pathways:** {", ".join(pathways)}{"..." if len(theme.get("pathways", [])) > 5 else ""}')
                if description:
                    lines.append(f'\n*{description}*')
                lines.append('')

        # Meta-unique themes (genuinely novel)
        themes = discovery_results.get('meta_unique_themes', [])
        if themes:
            lines.extend([
                '### Genuinely Novel Biological Themes',
                '',
                'These biological themes contain pathways not found in any individual dataset:',
                ''
            ])
            for theme in themes[:10]:
                theme_name = theme.get('theme_name', 'Unknown')
                pathways = theme.get('pathways', [])[:5]
                why_unique = theme.get('why_meta_unique', '')

                lines.append(f'#### {theme_name}')
                lines.append('')
                if pathways:
                    lines.append(f'**Pathways:** {", ".join(pathways)}{"..." if len(theme.get("pathways", [])) > 5 else ""}')
                if why_unique:
                    lines.append(f'\n*Why meta-unique:* {why_unique}')
                lines.append('')

        # Meta-generated hypotheses (shown when synthetic, not claimed as unique)
        generated_hypotheses = discovery_results.get('meta_generated_hypotheses', [])
        if generated_hypotheses:
            lines.extend([
                '### Mechanistic Hypotheses from Meta-Analysis',
                '',
                'These mechanistic hypotheses were generated from the combined data.',
                'They represent plausible biological mechanisms suggested by the enriched',
                'pathways, but are not claimed as unique to meta-analysis since individual',
                'datasets were not analyzed with the full pipeline.',
                ''
            ])
            for hyp in generated_hypotheses[:10]:
                title = hyp.get('title', 'Untitled')
                text = hyp.get('hypothesis_text', '')
                confidence = hyp.get('confidence_score', 0)
                confidence_label = hyp.get('confidence_label', '')
                key_players = hyp.get('key_players', [])
                evidence = hyp.get('evidence', [])
                prediction = hyp.get('directional_prediction', '')
                testability = hyp.get('testability', {})

                lines.append(f'#### {title}')
                conf_str = f'{confidence_label} ({confidence:.0%})' if confidence_label else f'{confidence:.0%}'
                lines.append(f'*Confidence: {conf_str}*')
                lines.append('')
                if text:
                    lines.append(text)
                    lines.append('')
                if key_players:
                    lines.append(f'**Key Players:** {"; ".join(str(p) for p in key_players[:5])}')
                    lines.append('')
                if evidence:
                    lines.append('**Supporting Evidence:**')
                    for ev in evidence[:5]:
                        lines.append(f'- {ev}')
                    lines.append('')
                if prediction:
                    lines.append(f'**Prediction:** {prediction}')
                    lines.append('')
                if testability:
                    approaches = [v for k, v in testability.items() if k.startswith('approach')]
                    if approaches:
                        lines.append('**Testability:**')
                        for approach in approaches[:3]:
                            lines.append(f'- {approach}')
                        lines.append('')

        # Meta-unique hypotheses (shown when full pipeline results available)
        hypotheses = discovery_results.get('meta_unique_hypotheses', [])
        if hypotheses:
            lines.extend([
                '### Meta-Unique Mechanistic Hypotheses',
                '',
                'These mechanistic insights emerged only from combined analysis:',
                ''
            ])
            for hyp in hypotheses[:10]:
                title = hyp.get('hypothesis_title', 'Unknown')
                text = hyp.get('hypothesis_text', '')[:500]
                confidence = hyp.get('confidence_score', 0)

                lines.append(f'#### {title}')
                lines.append(f'*Confidence: {confidence:.0%}*')
                lines.append('')
                lines.append(text)
                lines.append('')

        # Meta-unique therapeutic targets
        targets = discovery_results.get('meta_unique_targets', [])
        if targets:
            lines.extend([
                '### Meta-Unique Therapeutic Targets',
                '',
                'These drug targets were identified only through meta-analysis:',
                '',
                '| Gene | Target Type | Drugs | Clinical Relevance |',
                '|------|-------------|-------|-------------------|'
            ])
            for target in targets[:20]:
                gene = target.get('gene_symbol', 'Unknown')
                target_type = target.get('target_type', '')
                drugs = ', '.join(target.get('drugs', [])[:3])
                relevance = target.get('clinical_relevance', '')[:50]
                lines.append(f'| {gene} | {target_type} | {drugs} | {relevance} |')
            lines.append('')

        if not themes and not hypotheses and not targets and not organizational_themes and not generated_hypotheses:
            lines.append('*No meta-unique discoveries were identified.*')

        return '\n'.join(lines)

    def _generate_enhanced_significance_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate section on enhanced significance pathways"""
        enhanced = discovery_results.get('enhanced_significance_pathways', [])
        summary = discovery_results.get('summary', {})
        hierarchy = summary.get('significance_hierarchy', {})

        lines = [
            '## Enhanced Significance: Power Gain from Meta-Analysis',
            '',
            'These pathways achieved significantly better statistical significance in meta-analysis',
            'compared to any individual study, demonstrating increased statistical power.',
            ''
        ]

        # Add hierarchy insight if available
        tiers = hierarchy.get('tiers', [])
        resolution_gain = hierarchy.get('resolution_gain', 0)
        if len(tiers) >= 2 and resolution_gain >= 2.0:
            ind_spread = hierarchy.get('individual_log10_spread', 0)
            meta_spread = hierarchy.get('meta_log10_spread', 0)

            lines.extend([
                '### Pathway Priority Hierarchy',
                '',
                'A key insight from meta-analysis is the ability to **rank pathways by biological importance**.',
                f'In individual studies, pathway p-values are clustered within a narrow range '
                f'(~{ind_spread:.1f} orders of magnitude), making all pathways appear equally significant.',
                f'Meta-analysis spreads the p-values across **{meta_spread:.1f} orders of magnitude** '
                f'({resolution_gain:.0f}x more resolution), revealing which processes are most '
                f'consistently and strongly affected:',
                ''
            ])

            for tier in tiers:
                tier_num = tier['tier_number']
                n_pw = tier['n_pathways']
                p_lower = tier['pvalue_lower']
                p_upper = tier['pvalue_upper']
                pw_names = tier['pathways']

                lines.append(f'**Tier {tier_num}** (p = {p_lower:.0e} to {p_upper:.0e}) — {n_pw} pathway(s)')
                # Show pathway names, truncate if many
                if len(pw_names) <= 5:
                    lines.append(f'  {", ".join(pw_names)}')
                else:
                    lines.append(f'  {", ".join(pw_names[:5])}... and {len(pw_names) - 5} more')
                lines.append('')

            lines.extend([
                '*Higher-tier pathways represent the most robust biological signals across all datasets,*',
                '*providing a prioritized list for downstream validation and intervention.*',
                ''
            ])

        # Full table
        lines.extend([
            '### Full Enhanced Significance Table',
            '',
            '| Pathway | Meta p-value | Best Individual p-value | Significance Gain | Found in N Studies |',
            '|---------|--------------|------------------------|-------------------|-------------------|'
        ])

        for pw in enhanced[:30]:
            name = pw.get('pathway_name', 'Unknown')[:50]
            meta_p = pw.get('meta_pvalue', 1.0)
            best_p = pw.get('best_individual_pvalue', 1.0)
            gain = pw.get('significance_gain_fold', 1.0)
            n_studies = pw.get('found_in_n_datasets', 0)

            gain_str = f'{gain:.1f}x' if gain != float('inf') else 'Meta-only'
            lines.append(f'| {name} | {meta_p:.2e} | {best_p:.2e} | {gain_str} | {n_studies} |')

        lines.append('')

        return '\n'.join(lines)

    def _generate_cross_study_validation_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate section on cross-study validated mechanisms"""
        mechanisms = discovery_results.get('cross_study_mechanisms', [])

        lines = [
            '## Cross-Study Validated Mechanisms',
            '',
            'These mechanisms were found consistently across multiple independent studies,',
            'providing high-confidence biological insights:',
            '',
            '| Mechanism | Pathway | Consistency | Studies |',
            '|-----------|---------|-------------|---------|'
        ]

        for mech in mechanisms[:20]:
            name = mech.get('mechanism_name', 'Unknown')[:40]
            pathway = mech.get('pathway_name', '')[:40]
            consistency = mech.get('consistency_score', 0)
            studies = ', '.join(mech.get('found_in_datasets', [])[:3])
            if len(mech.get('found_in_datasets', [])) > 3:
                studies += '...'

            lines.append(f'| {name} | {pathway} | {consistency:.0%} | {studies} |')

        lines.append('')

        return '\n'.join(lines)

    # ------------------------------------------------------------------ #
    # New emergent discovery report sections                              #
    # ------------------------------------------------------------------ #

    def _generate_emergent_theme_combinations_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate section on emergent theme combinations (Type 1)"""
        items = discovery_results.get('emergent_theme_combinations', [])
        lines = [
            '## Emergent Theme Combinations',
            '',
            'These biological themes group pathways where **most datasets lack the full combination**.',
            'The grouping itself is a genuinely emergent meta-analysis insight.',
            '',
            '| Theme | Pathways | Median Coverage | Datasets with >75% |',
            '|-------|----------|-----------------|---------------------|'
        ]
        for item in items[:20]:
            name = item.get('theme_name', 'Unknown')[:40]
            n_pw = item.get('n_pathways', 0)
            median_cov = item.get('median_dataset_coverage', 0)
            n_high = item.get('n_datasets_high_coverage', 0)
            frac_high = item.get('fraction_datasets_high_coverage', 0)
            lines.append(f'| {name} | {n_pw} | {median_cov:.0%} | {n_high} ({frac_high:.0%}) |')
        lines.append('')
        return '\n'.join(lines)

    def _generate_cross_dataset_convergence_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate section on cross-dataset pathway convergence (Type 2)"""
        items = discovery_results.get('cross_dataset_convergence', [])
        lines = [
            '## Cross-Dataset Pathway Convergence',
            '',
            'These themes contain pathways contributed by **largely independent datasets**.',
            'The meta-analysis reveals connections between pathways studied in different experiments.',
            '',
            '| Theme | Pathways | Avg Dataset Overlap (Jaccard) |',
            '|-------|----------|------------------------------|'
        ]
        for item in items[:20]:
            name = item.get('theme_name', 'Unknown')[:40]
            n_pw = item.get('n_pathways', 0)
            jaccard = item.get('avg_dataset_overlap', 0)
            lines.append(f'| {name} | {n_pw} | {jaccard:.2f} |')
        lines.append('')
        return '\n'.join(lines)

    def _generate_weak_signal_amplification_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate section on weak signal amplification (Type 5)"""
        items = discovery_results.get('weak_signal_amplification', [])
        lines = [
            '## Weak Signal Amplification',
            '',
            'These pathways were only marginally significant in individual datasets (p 0.01-0.05)',
            'but reached strong significance in meta-analysis (p < 0.001). No individual dataset',
            'had a strong signal alone — these are genuinely hidden signals amplified by combining studies.',
            '',
            '| Pathway | Meta p-value | Weak Individual Signals | Total Datasets |',
            '|---------|-------------|------------------------|----------------|'
        ]
        for item in items[:20]:
            name = item.get('pathway', 'Unknown')[:50]
            meta_p = item.get('meta_pvalue', 1.0)
            n_weak = item.get('n_weak_individual', 0)
            n_total = item.get('n_datasets_with_pathway', 0)
            lines.append(f'| {name} | {meta_p:.2e} | {n_weak} | {n_total} |')
        lines.append('')
        return '\n'.join(lines)

    def _generate_novel_cooccurrence_pairs_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate section on novel co-occurrence pairs (Type 6)"""
        items = discovery_results.get('novel_cooccurrence_pairs', [])
        lines = [
            '## Novel Pathway Co-occurrence Pairs',
            '',
            'These pathway pairs are grouped in the same meta-analysis theme but were **never both**',
            '**significant in any single dataset**. The meta-analysis reveals gene-level connections',
            'between pathways that never appeared together in one study.',
            '',
            '| Pathway A | Pathway B | Theme |',
            '|-----------|-----------|-------|'
        ]
        for item in items[:30]:
            pw_a = item.get('pathway_a', 'Unknown')[:40]
            pw_b = item.get('pathway_b', 'Unknown')[:40]
            theme = item.get('theme_name', '')[:30]
            lines.append(f'| {pw_a} | {pw_b} | {theme} |')
        lines.append('')
        return '\n'.join(lines)

    def _generate_biological_interpretation_section(self, discovery_results: Dict[str, Any]) -> str:
        """Generate biological interpretation section"""
        interpretation = discovery_results.get('biological_interpretation', '')

        lines = [
            '## Biological Interpretation of Meta-Analysis Findings',
            '',
            interpretation,
            ''
        ]

        return '\n'.join(lines)

    def _generate_individual_studies_summary(self, discovery_results: Dict[str, Any], n_selected: Any = None) -> str:
        """Generate the Findings Across Individual Datasets section (the one meta-only section)"""
        individual = discovery_results.get('individual_findings', {})

        if not individual:
            return ''

        total_datasets = individual.get('total_datasets', 0)
        total_pathways = individual.get('total_unique_pathways', 0)
        in_multiple = individual.get('pathways_in_multiple', 0)
        in_single = individual.get('pathways_in_single', 0)
        top_pathways = individual.get('top_pathways', [])
        llm_summary = individual.get('llm_summary', '')
        # top_pathways is capped upstream; the full frequency table is pathway_counts
        n_ranked = len(individual.get('pathway_counts') or {}) or total_pathways

        lines = [
            '## Findings Across Individual Datasets',
            '',
            f'This section summarizes findings from analyzing **{total_datasets} individual datasets**',
            'separately. For each dataset, a pathway counts when it is significant (FDR < 0.05)',
            'in that dataset.',
            ''
        ]

        # Add LLM-generated natural language summary
        if llm_summary:
            lines.extend([
                '### Summary of Dataset Findings',
                '',
                llm_summary,
                ''
            ])

        # Add statistics
        rows = [
            '### Dataset Statistics',
            '',
            '| Metric | Value |',
            '|--------|-------|',
            f'| Total datasets analyzed | {total_datasets} |',
        ]
        if n_selected is not None:
            rows.append(f'| Pathways selected for analysis | {n_selected} |')
        rows.extend([
            f'| Pathways significant in at least 1 dataset | {total_pathways} |',
            f'| Pathways found in ≥2 datasets | {in_multiple} |',
            f'| Pathways found in only 1 dataset | {in_single} |',
            ''
        ])
        lines.extend(rows)

        # Add top pathways table
        if top_pathways:
            shown = top_pathways[:15]
            lines.extend([
                '### Most Frequently Identified Pathways',
                '',
                f'*Showing top {len(shown)} of {n_ranked} pathways, ranked by the number of datasets.*',
                '',
                '| Pathway | Found in N Datasets | Best p-value |',
                '|---------|--------------------|--------------|'
            ])

            for pw in shown:
                name = pw.get('name', 'Unknown')  # full name; was cut at 50 chars
                count = pw.get('count', 0)
                pval = pw.get('best_pvalue', 1.0)
                lines.append(f'| {name} | {count} | {pval:.2e} |')

            lines.extend([
                '',
                '*Not shown here: which datasets each pathway was found in, and its per-dataset '
                'p-values and directions.*',
                ''
            ])

        # The former closing line ("The following sections show what meta-analysis
        # discovered BEYOND these individual findings.") is gone: the sections that follow
        # are the combined-data analysis, not meta-unique findings.

        return '\n'.join(lines)

    def _generate_header(self, study_type: str, tissue: str, n_datasets: int, metadata: Dict[str, Any] = None) -> str:
        """Generate report header with flexible metadata display"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata = metadata or {}

        # Build header with available metadata
        lines = ['# Meta-Analysis Comparative Report', '']

        # Display ALL metadata fields dynamically
        # Skip internal/technical fields
        skip_fields = {'datasetId', 'dataset_id'}

        for key, value in metadata.items():
            if key in skip_fields:
                continue
            if value and str(value).strip():
                # Format the key for display (capitalize, etc.)
                display_key = key.replace('_', ' ').title()
                lines.append(f'**{display_key}:** {value}')

        lines.append(f'**Number of Datasets:** {n_datasets}')
        lines.append(f'**Generated:** {timestamp}')
        lines.append('')
        lines.append('---')

        return '\n'.join(lines)

    def _generate_executive_summary(self, repro_results: Dict, discovery_results: Dict = None) -> str:
        """Generate executive summary with discovery highlights"""
        stats = repro_results['summary_stats']
        discovery_results = discovery_results or {}
        discovery_summary = discovery_results.get('summary', {})

        pathway_cats = stats['pathway_categories']
        gene_cats = stats['gene_categories']

        # Safe percentage calculation
        def safe_pct(count, total):
            return f"({count / total * 100:.1f}%)" if total > 0 else "(N/A)"

        total_pathways = stats['total_pathways_analyzed']
        total_genes = stats['total_genes_analyzed']

        # Build discovery highlights
        discovery_highlights = ""
        all_synthetic = discovery_summary.get('all_synthetic', False)
        total_discoveries = discovery_summary.get('total_discoveries', 0)

        if all_synthetic:
            enhanced_count = discovery_summary.get('enhanced_significance_pathways_count', 0)
            org_themes_count = discovery_summary.get('organizational_themes_count', 0)
            if enhanced_count > 0 or org_themes_count > 0:
                discovery_highlights = f"""
### Meta-Analysis Value

The meta-analysis provided:
- Pathways with enhanced significance: {enhanced_count}
- Organizational biological themes: {org_themes_count} (grouping known pathways)
- Genuinely novel themes: {discovery_summary.get('meta_unique_themes_count', 0)}
"""
        elif total_discoveries > 0:
            discovery_highlights = f"""
### Meta-Analysis Unique Value

The meta-analysis identified **{total_discoveries} unique discoveries** not visible in individual studies:
- Meta-unique biological themes: {discovery_summary.get('meta_unique_themes_count', 0)}
- Meta-unique hypotheses: {discovery_summary.get('meta_unique_hypotheses_count', 0)}
- Meta-unique therapeutic targets: {discovery_summary.get('meta_unique_targets_count', 0)}
- Pathways with enhanced significance: {discovery_summary.get('enhanced_significance_pathways_count', 0)}
"""

        return f"""## Executive Summary

This report compares meta-analysis results with {stats['n_datasets']} individual datasets to assess:
- **Reproducibility**: How consistently findings appear across studies
- **Power Gain**: Discoveries unique to meta-analysis (increased statistical power)
- **Novel Insights**: Biological mechanisms visible only through combined analysis
{discovery_highlights}
### Reproducibility Findings

**Pathways:**
- Total analyzed: {total_pathways}
- Highly reproducible: {pathway_cats['high']} {safe_pct(pathway_cats['high'], total_pathways)}
- Meta-unique (power gain): {pathway_cats['meta_unique']}

**Genes:**
- Total analyzed: {total_genes}
- Highly reproducible: {gene_cats['high']} {safe_pct(gene_cats['high'], total_genes)}
- Meta-unique (power gain): {gene_cats['meta_unique']}

**Direction Concordance:**
- Pathways: {stats['avg_pathway_direction_concordance']:.1%} agreement in up/down regulation
- Genes: {stats['avg_gene_direction_concordance']:.1%} agreement in expression direction"""

    def _generate_reproducibility_overview(self, repro_results: Dict) -> str:
        """Generate reproducibility overview section"""
        stats = repro_results['summary_stats']
        pathway_cats = stats['pathway_categories']
        gene_cats = stats['gene_categories']

        # Calculate percentages safely
        total_pathways = stats['total_pathways_analyzed']
        total_genes = stats['total_genes_analyzed']

        def safe_pct(count, total):
            return f"{count / total * 100:.1f}%" if total > 0 else "N/A"

        return f"""## Reproducibility Overview

### Reproducibility Categories

Findings are categorized based on how many datasets identified them:
- **High**: Found in ≥75% of datasets
- **Moderate**: Found in 50-74% of datasets
- **Low**: Found in <50% of datasets
- **Meta-unique**: Found only in meta-analysis (power gain)

### Pathway Reproducibility

| Category | Count | Percentage |
|----------|-------|------------|
| High | {pathway_cats['high']} | {safe_pct(pathway_cats['high'], total_pathways)} |
| Moderate | {pathway_cats['moderate']} | {safe_pct(pathway_cats['moderate'], total_pathways)} |
| Low | {pathway_cats['low']} | {safe_pct(pathway_cats['low'], total_pathways)} |
| Meta-unique | {pathway_cats['meta_unique']} | {safe_pct(pathway_cats['meta_unique'], total_pathways)} |

### Gene Reproducibility

| Category | Count | Percentage |
|----------|-------|------------|
| High | {gene_cats['high']} | {safe_pct(gene_cats['high'], total_genes)} |
| Moderate | {gene_cats['moderate']} | {safe_pct(gene_cats['moderate'], total_genes)} |
| Low | {gene_cats['low']} | {safe_pct(gene_cats['low'], total_genes)} |
| Meta-unique | {gene_cats['meta_unique']} | {safe_pct(gene_cats['meta_unique'], total_genes)} |"""

    def _generate_concordance_analysis(self, repro_results: Dict) -> str:
        """Generate direction concordance analysis"""
        stats = repro_results['summary_stats']

        return f"""## Direction Concordance Analysis

Direction concordance measures the agreement in up/down regulation (or enrichment direction)
across all studies. High concordance indicates consistent biological effects.

### Overall Concordance Metrics

- **Pathway Direction Concordance:** {stats['avg_pathway_direction_concordance']:.1%}
  - Measures agreement in pathway enrichment direction (upregulated vs downregulated)

- **Gene Direction Concordance:** {stats['avg_gene_direction_concordance']:.1%}
  - Measures agreement in gene expression direction (up vs down)

### Interpretation

- **>90% concordance**: Highly consistent effect direction across studies
- **75-90% concordance**: Good consistency with some variability
- **<75% concordance**: Substantial heterogeneity in effect direction"""

    def _generate_conclusions(self, repro_results: Dict, discovery_results: Dict = None) -> str:
        """Generate conclusions section with discovery insights"""
        stats = repro_results['summary_stats']
        discovery_results = discovery_results or {}
        discovery_summary = discovery_results.get('summary', {})

        pathway_cats = stats['pathway_categories']
        gene_cats = stats['gene_categories']

        # Calculate reproducibility rate
        pathway_repro_rate = (
            (pathway_cats['high'] + pathway_cats['moderate']) /
            stats['total_pathways_analyzed'] * 100
            if stats['total_pathways_analyzed'] > 0 else 0
        )

        gene_repro_rate = (
            (gene_cats['high'] + gene_cats['moderate']) /
            stats['total_genes_analyzed'] * 100
            if stats['total_genes_analyzed'] > 0 else 0
        )

        # Build discovery conclusions
        discovery_conclusions = ""
        all_synthetic = discovery_summary.get('all_synthetic', False)
        total_discoveries = discovery_summary.get('total_discoveries', 0)

        if all_synthetic:
            enhanced_count = discovery_summary.get('enhanced_significance_pathways_count', 0)
            org_themes_count = discovery_summary.get('organizational_themes_count', 0)
            if enhanced_count > 0 or org_themes_count > 0:
                discovery_conclusions = f"""
### Meta-Analysis Value

The primary value of this meta-analysis is enhanced statistical power:

- **{enhanced_count} pathways** achieved significantly enhanced statistical significance
- **{org_themes_count} biological themes** organize known pathways into coherent narratives
- **{discovery_summary.get('meta_unique_themes_count', 0)} genuinely novel themes** contain pathways not found in individual datasets

The enhanced significance findings represent the core contribution of combining
{discovery_summary.get('n_datasets', 'multiple')} datasets.
"""
        elif total_discoveries > 0:
            discovery_conclusions = f"""
### Meta-Analysis Discovery Value

This meta-analysis identified **{total_discoveries} unique biological insights** not detectable
in any individual study:

- **{discovery_summary.get('meta_unique_themes_count', 0)} biological themes** emerged only from combined data
- **{discovery_summary.get('meta_unique_hypotheses_count', 0)} mechanistic hypotheses** are unique to meta-analysis
- **{discovery_summary.get('meta_unique_targets_count', 0)} therapeutic targets** were identified through combined power
- **{discovery_summary.get('enhanced_significance_pathways_count', 0)} pathways** achieved enhanced statistical significance

These discoveries demonstrate the essential value of meta-analysis: revealing biological
patterns that remain hidden in underpowered individual studies.
"""

        return f"""## Conclusions

### Reproducibility Assessment

This meta-analysis demonstrates:

1. **Pathway Reproducibility:** {pathway_repro_rate:.1f}% of pathways show moderate-to-high
   reproducibility across studies, indicating consistent biological mechanisms.

2. **Gene Reproducibility:** {gene_repro_rate:.1f}% of genes show moderate-to-high
   reproducibility, suggesting reliable gene signatures.

3. **Power Gain:** Meta-analysis identified {discovery_summary.get('enhanced_significance_pathways_count', 0)} pathways with enhanced significance and
   {gene_cats['meta_unique']} unique genes not detected in individual datasets, demonstrating
   the value of increased statistical power.
{discovery_conclusions}
### Recommendations

- **High confidence findings:** Focus on highly reproducible pathways and genes for downstream
  validation and therapeutic targeting.

- **Meta-unique discoveries:** Prioritize investigation of meta-unique themes, hypotheses, and
  targets - these represent the unique value of meta-analysis and may reveal important
  biological mechanisms missed by individual studies.

- **Enhanced significance pathways:** Pathways with significantly better p-values in meta-analysis
  are strong candidates for follow-up validation.

- **Cross-study validated mechanisms:** Mechanisms found consistently across multiple studies
  provide the highest confidence for translational research.

- **Low reproducibility findings:** Exercise caution with low-reproducibility findings; they
  may represent study-specific effects or technical artifacts.

---

*End of Meta-Analysis Comparative Report*"""

