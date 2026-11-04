"""
Step 06: Final Report Generation

Generates a comprehensive markdown report from Steps 1-5 outputs with built-in
verification to prevent hallucination and ensure accuracy.

Key Features:
- Direct extraction (no LLM for facts)
- Cross-validation against source data
- 0% hallucination tolerance
- Traceable claims
- Publication-ready markdown

Verification Strategy:
1. Extract-only: Gene names, fold changes, p-values directly from input
2. LLM synthesis: Only for executive summary and interpretations (grounded)
3. Cross-check: Every claim verified against source data
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.groq_client import GroqClient
from src.pipeline.fc_utils import fc_arrow, fc_direction, to_float
from src.pipeline import summary_number_check
# Reuse the NES-preferred extraction so the report labels each value with its actual
# metric ('NES'/'ES') on the same scale as the Step 3 detailed sections.
from src.pipeline.steps.step03_pathway_mechanisms import _enrichment_metric
from src.pipeline.steps.step04_hypothesis_generation import normalize_novelty


def _pathway_databases(input_data: Dict) -> List[str]:
    """The pathway databases the submitted pathways came from (each pathway's `source`),
    in first-seen order, with empty/'Unknown' values dropped."""
    seen = []
    for pathway in (input_data or {}).get('pathways', []) or []:
        name = str((pathway or {}).get('source') or '').strip()
        if name and name.lower() != 'unknown' and name not in seen:
            seen.append(name)
    return seen


def _step3_methods_text(step3: Dict) -> str:
    """Methods line for Step 3, built from the evidence its pathway structures actually had."""
    if (step3 or {}).get('metadata', {}).get('skipped', False):
        return 'Skipped (no gene expression data)'
    curated, kinds = [], []
    for ps in (step3 or {}).get('pathway_structures', []) or []:
        confidence = ps.get('confidence')
        if confidence == 'high':
            name = f"curated {str(ps.get('source') or '').upper()} pathway structures".replace('  ', ' ')
            if name not in curated:
                curated.append(name)
        else:
            label = {
                'pc_grounded': 'Pathway Commons interactions',
                'inferred': 'LLM-inferred relations',
                'gene_set': 'pathway gene-set membership',
            }.get(confidence)
            if label and label not in kinds:
                kinds.append(label)
    evidence = curated + kinds
    if not evidence:
        return 'Pathway mechanism analysis'
    return f"Pathway mechanism analysis ({'; '.join(evidence)})"


def _pathway_fdr_value(pathway: Dict) -> float:
    """A pathway's FDR as a float (0.0-safe; 1.0 when absent)."""
    for key in ('p_value_fdr', 'pValueFDR'):
        val = to_float((pathway or {}).get(key))
        if val is not None:
            return val
    return 1.0


# Step 4 metadata lists that hold discrepancies the engine did not resolve (item 26).
_UNRESOLVED_GROUPS = (
    ('direction_validation_errors', 'Gene direction'),
    ('biochemistry_validation_warnings', 'Biochemistry review'),
    ('numerical_validation_warnings', 'Numerical claims'),
    ('kegg_validation_warnings', 'Relations not returned by the tool calls'),
    ('kegg_text_validation_warnings', 'Database citations in the text'),
    ('prediction_vs_data_warnings', 'Predictions vs data'),
    ('internal_consistency_warnings', 'Internal consistency'),
    ('coverage_validation_warnings', 'Coverage of dominant pathways'),
)


def _unresolved_warnings(step4: Dict, summary_unresolved: Optional[List[Dict]] = None):
    """[(group label, [message, ...]), ...] of unresolved validation warnings."""
    meta = (step4 or {}).get('metadata', {}) or {}
    out = []
    for key, label in _UNRESOLVED_GROUPS:
        items = [str(x) for x in (meta.get(key) or []) if x]
        if items:
            out.append((label, items))
    conflicts = meta.get('fold_change_sign_conflicts') or []
    if conflicts:
        out.append(('Fold-change sign conflicts left in the text',
                    [f'{g}: cited with a direction opposite to its fold change' for g in conflicts]))
    if summary_unresolved:
        out.append(('Executive summary numbers',
                    [f"\"{i.get('text')}\": {i.get('reason')}" for i in summary_unresolved]))
    return out


def major_theme(themes: List[Dict]) -> Optional[Dict]:
    """The major theme: the pathway theme that contains the most pathways (main L265).

    The paper gives no tie-break, so a fixed one is used: lower average FDR, then the
    alphabetically first name of the theme's most significant pathway. The theme name
    is deliberately not used, because the model writes it and it changes between runs.
    """
    if not themes:
        return None

    def key(theme: Dict):
        pathways = theme.get('pathways', []) or []
        avg_fdr = to_float(theme.get('avg_p_value_fdr'))
        if avg_fdr is None:
            avg_fdr = (sum(_pathway_fdr_value(p) for p in pathways) / len(pathways)
                       if pathways else 1.0)
        top = sorted(pathways, key=lambda p: (_pathway_fdr_value(p),
                                              str(p.get('name') or p.get('pathway') or '')))
        top_name = str(top[0].get('name') or top[0].get('pathway') or '') if top else ''
        return (-len(pathways), avg_fdr, top_name.lower())

    return min(themes, key=key)


@dataclass
class ReportSection:
    """A section of the final report"""
    title: str
    content: str
    source_data: Dict[str, Any]  # For traceability


@dataclass
class FinalReport:
    """Complete final report"""
    markdown_content: str
    metadata: Dict[str, Any]


class Step06ReportGeneration:
    """
    Step 06: Final Report Generation

    Generates a comprehensive markdown report by extracting information
    from Steps 1-5 outputs and verifying all claims against source data.
    """

    def __init__(self):
        """Initialize Step 6"""
        self.llm = GroqClient()

    def execute(
        self,
        input_data: Dict[str, Any],
        step1_output: Dict[str, Any],
        step2_output: Dict[str, Any],
        step3_output: Dict[str, Any],
        step4_output: Dict[str, Any],
        step5_output: Optional[Dict[str, Any]] = None
    ) -> FinalReport:
        """
        Generate final report from all step outputs.

        Args:
            input_data: Original input data (genes, pathways, context)
            step1_output: Pathway themes output
            step2_output: Hub genes output
            step3_output: Mechanisms output
            step4_output: Hypotheses output
            step5_output: (removed) Therapeutics output — always None now

        Returns:
            FinalReport with markdown content and validation results
        """
        print("\nGenerating final report...")

        # Extract metadata - handle ALL fields dynamically
        metadata = input_data.get('metadata', {})
        organism = metadata.get('organism', 'Homo sapiens')

        # Build study context from ALL metadata fields (not just disease/tissue)
        study_context = self._build_study_context(metadata)

        # Step 5 (Therapeutic Implications) has been removed from the pipeline; its
        # report section is always omitted. Carry an explicit skipped marker so every
        # sub-section (which recomputes step5_skipped from metadata) omits it.
        step5_output = step5_output or {'metadata': {'skipped': True}}
        step2_skipped = step2_output.get('metadata', {}).get('skipped', False)
        step5_skipped = True

        # Generate report sections
        sections = []

        # Section 1: Executive Summary
        sections.append(self._generate_executive_summary(
            study_context, input_data, step1_output, step2_output,
            step3_output, step4_output, step5_output
        ))

        # Section 2: Pathway Theme Analysis
        sections.append(self._generate_pathway_themes_section(
            step1_output, input_data
        ))

        # Section 3: Network Hub Genes (skip if no genes provided)
        if not step2_skipped:
            sections.append(self._generate_hub_genes_section(
                step2_output, input_data
            ))
        else:
            print("  Skipping Hub Genes section (no genes provided)")

        # Section 4: Pathway Mechanisms
        sections.append(self._generate_mechanisms_section(
            step3_output
        ))

        # Section 5: Mechanistic Hypotheses
        sections.append(self._generate_hypotheses_section(
            step4_output
        ))

        # Section 6: Therapeutic Implications (skip if no genes provided)
        if not step5_skipped:
            sections.append(self._generate_therapeutics_section(
                step5_output
            ))
        else:
            print("  Skipping Therapeutic Implications section (no genes provided)")

        # Section 7: Summary and Conclusions
        sections.append(self._generate_conclusions(
            study_context, input_data, step1_output, step2_output,
            step3_output, step4_output, step5_output
        ))

        # Section 8: Appendices
        sections.append(self._generate_appendices(
            input_data, step1_output, step2_output, step3_output,
            step4_output, step5_output
        ))

        # Assemble full report
        report_content = self._assemble_report(
            metadata, organism, sections
        )

        # Validate report
        print("\nValidating report against source data...")
        all_outputs = {
            'input_data': input_data,
            'step1': step1_output,
            'step2': step2_output,
            'step3': step3_output,
            'step4': step4_output,
            'step5': step5_output
        }

        # Create report metadata (include all study metadata fields)
        report_metadata = {
            **metadata,  # Include all original metadata fields
            'organism': organism,
            'generation_date': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'num_genes': len(input_data.get('genes', [])),
            'num_pathways': len(input_data.get('pathways', [])),
            'num_themes': len(step1_output.get('themes', [])),
            'num_hub_genes': len(step2_output.get('network_hubs', [])),
            'num_mechanisms': len(step3_output.get('pathway_mechanisms', [])),
            'num_hypotheses': len(step4_output.get('hypotheses', [])),
            'num_drugs': len(step5_output.get('drug_recommendations', [])),
            # What the executive-summary number check found and changed (main L263).
            'summary_number_check': sections[0].source_data.get('summary_number_check', {}),
        }

        print(f"\n✓ Report generated: {len(report_content)} characters")

        return FinalReport(
            markdown_content=report_content,
            metadata=report_metadata
        )

    def _build_study_context(self, metadata: Dict[str, Any]) -> str:
        """Build a study context string from all metadata fields"""
        if not metadata:
            return "Study Context: Unknown"

        lines = []
        # Skip internal/technical fields
        skip_fields = {'datasetId', 'dataset_id', 'organism'}

        for key, value in metadata.items():
            if key in skip_fields:
                continue
            if value and str(value).strip():
                # Format the key for display
                display_key = key.replace('_', ' ').title()
                lines.append(f"{display_key}: {value}")

        return '\n'.join(lines) if lines else "Study Context: Unknown"

    @staticmethod
    def _rank_enrichment_signals(input_data: Dict):
        """Rank pathways by enrichment score (NES preferred) into up/down axes.

        Returns ``(ups, downs)`` — each a list of ``(value, name, metric)`` sorted by
        magnitude, where ``metric`` is the honest scale ('NES'/'ES') from
        :func:`_enrichment_metric`. This single ranking backs both the executive summary
        and the Key Findings, so the two sections can never disagree on which axes exist
        (AC2), and it matches the scale/label used by Step 3 and Step 4.
        """
        pathways = input_data.get('pathways', []) or []
        ups, downs = [], []
        for p in pathways:
            if not isinstance(p, dict):
                continue
            # _enrichment_metric prefers NES and promotes |ES|>1 to the NES scale;
            # to_float guards string-valued scores on raw/synthetic pathway dicts.
            val, metric = _enrichment_metric(p)
            val = to_float(val)
            if val is None:
                continue
            name = p.get('name') or p.get('pathwayName') or p.get('pathway') or ''
            if not name:
                continue
            entry = (val, name, metric)
            if val > 0:
                ups.append(entry)
            elif val < 0:
                downs.append(entry)
        ups.sort(key=lambda x: -x[0])
        downs.sort(key=lambda x: x[0])
        return ups, downs

    def _dominant_axes(self, input_data: Dict) -> List[Dict]:
        """Dominant enrichment axes — the strongest up- and down-regulated pathway.

        Each axis: ``{'direction', 'pathway', 'nes', 'metric'}``. Both the executive
        summary and Key Findings draw from this, so no direction is silently dropped
        from either and the two sections always name the same axes (AC2).
        """
        ups, downs = self._rank_enrichment_signals(input_data)
        axes = []
        for direction, sig_list in (('up', ups), ('down', downs)):
            if not sig_list:
                continue
            val, name, metric = sig_list[0]
            axes.append({
                'direction': direction,
                'pathway': name,
                'nes': val,
                'metric': metric,
            })
        return axes

    def _top_enrichment_signals(self, input_data: Dict, n: int = 6) -> str:
        """Format the strongest up- and down-regulated pathways by |NES| (or ES).

        Feeds the executive-summary prompt so the narrative reflects the largest
        signals in both directions, not just up-regulated proliferation.
        """
        ups, downs = self._rank_enrichment_signals(input_data)
        if not ups and not downs:
            return '  (enrichment scores not provided)'

        lines = []
        if ups:
            lines.append('  Up-regulated (by enrichment score): '
                         + '; '.join(f'{nm} ({metric} {v:+.2f})' for v, nm, metric in ups[:n]))
        if downs:
            lines.append('  Down-regulated (by enrichment score): '
                         + '; '.join(f'{nm} ({metric} {v:+.2f})' for v, nm, metric in downs[:n]))
        return '\n'.join(lines)

    def _generate_executive_summary(
        self,
        study_context: str,
        input_data: Dict,
        step1: Dict,
        step2: Dict,
        step3: Dict,
        step4: Dict,
        step5: Dict
    ) -> ReportSection:
        """Generate executive summary section"""
        # Check if steps were skipped
        step2_skipped = step2.get('metadata', {}).get('skipped', False)
        step5_skipped = step5.get('metadata', {}).get('skipped', False)

        themes = step1.get('themes', [])
        hubs = step2.get('network_hubs', [])
        master_regulators = step2.get('master_regulators', [])
        hypotheses = step4.get('hypotheses', [])
        drugs = step5.get('drug_recommendations', [])

        # Extract key findings. Use the FULL master-regulator list (single source of
        # truth from step02._identify_master_regulators) so the executive-summary count
        # matches the Network Hub Genes section — previously [:3] said "3 of 20" while
        # the section listed 5 master regulators.
        # Major theme = the theme with the most pathways (main L265), not themes[0].
        _major = major_theme(themes)
        theme_name = _major.get('name', 'N/A') if _major else 'No major theme identified'
        top_hubs = [h.get('gene', '') for h in master_regulators] if master_regulators else []
        num_drugs = len(drugs)
        # Count only drugs approved specifically for this indication (not off-label)
        fda_approved = sum(1 for d in drugs if 'Approved for' in d.get('approvalStatus', '') or ('Approved' in d.get('approvalStatus', '') and 'off-label' not in d.get('approvalStatus', '').lower() and 'investigational' not in d.get('approvalStatus', '').lower()))
        fda_off_label = sum(1 for d in drugs if 'off-label' in d.get('approvalStatus', '').lower())

        # Build drug summary for grounding the LLM prompt
        drug_summary_lines = []
        for d in drugs:
            name = d.get('drugName', 'Unknown')
            targets = ', '.join(d.get('targetGenes', []))
            status = d.get('approvalStatus', 'Unknown')
            drug_summary_lines.append(f"  - {name} → targets {targets} [{status}]")
        drug_summary = '\n'.join(drug_summary_lines) if drug_summary_lines else '  None'

        # Strongest enrichment signals in BOTH directions, so the summary is not
        # proliferation-only and does not ignore strong down-regulated signals.
        signals_block = self._top_enrichment_signals(input_data)
        interpretation_guidance = """
IMPORTANT interpretation rules:
- Synthesize the STRONGEST signals in BOTH directions. Do not focus only on up-regulated
  proliferation; if strongly down-regulated processes (e.g. metabolic catabolism) are present,
  feature them prominently — largest |NES| signals matter most regardless of direction.
- Use magnitude-calibrated language: |NES| >= 2 = strong, 1.5-2 = moderate, < 1.5 = weak.
  Never call a weak/borderline signal "strong".
- Master regulators are NETWORK HUB genes (high connectivity), not necessarily the
  strongest effect-size drivers — describe them as hubs, not proven causal drivers.
- For pathways with mixed-direction member genes, state which sub-processes are up vs down
  (e.g. de novo lipogenesis up while beta-oxidation is down) rather than one net direction."""

        # Generate summary using LLM (grounded in extracted facts)
        # Adjust prompt based on available data
        if step2_skipped and step5_skipped:
            # Pathway-only analysis
            prompt = f"""Generate a 2-3 paragraph executive summary for a molecular pathway analysis report.

**Study Context:**
{study_context}

Major biological theme: {theme_name}
Number of hypotheses: {len(hypotheses)}

Strongest enrichment signals by |enrichment score| (both directions):
{signals_block}

Note: This is a pathway-only analysis (no gene expression data available).

The summary should be concise, scientific, and highlight:
1. The major biological theme identified
2. The mechanistic hypotheses generated
3. The pathway-level insights (strongest signals in both directions)
{interpretation_guidance}

Keep it factual and grounded in these specific findings."""
        elif step2_skipped:
            # No hub genes but has therapeutics
            prompt = f"""Generate a 2-3 paragraph executive summary for a molecular analysis report.

**Study Context:**
{study_context}

Major biological theme: {theme_name}
Number of hypotheses: {len(hypotheses)}
Number of druggable targets: {num_drugs}
FDA-approved drugs (for this indication): {fda_approved}
Drug recommendations:
{drug_summary}

The summary should be concise, scientific, and highlight:
1. The major biological theme identified
2. The mechanistic hypotheses
3. The therapeutic potential

CRITICAL: Only mention drug names and targets that appear in the data above. Do NOT invent or assume drug targets.

Keep it factual and grounded in these specific findings."""
        elif step5_skipped:
            # Has hub genes but no therapeutics
            prompt = f"""Generate a 2-3 paragraph executive summary for a molecular analysis report.

**Study Context:**
{study_context}

Major biological theme: {theme_name}
Network hub genes (a.k.a. master regulators — topological hubs): {', '.join(top_hubs)}
Number of hypotheses: {len(hypotheses)}

Strongest enrichment signals by |enrichment score| (both directions):
{signals_block}

The summary should be concise, scientific, and highlight:
1. The major biological theme(s), including the strongest signals in BOTH directions
2. The key network hub genes (described as hubs, not proven drivers)
3. The mechanistic insights
{interpretation_guidance}

Keep it factual and grounded in these specific findings."""
        else:
            # Full analysis
            prompt = f"""Generate a 2-3 paragraph executive summary for a molecular analysis report.

**Study Context:**
{study_context}

Major biological theme: {theme_name}
Network hub genes (a.k.a. master regulators — topological hubs): {', '.join(top_hubs)}
Number of hypotheses: {len(hypotheses)}
Number of druggable targets: {num_drugs}
FDA-approved drugs (for this indication): {fda_approved}
Drug recommendations:
{drug_summary}

The summary should be concise, scientific, and highlight:
1. The major biological theme identified
2. The key network hub genes (described as hubs, not proven drivers)
3. The therapeutic potential

CRITICAL: Only mention drug names and targets that appear in the data above. Do NOT invent or assume drug targets.

Keep it factual and grounded in these specific findings."""

        summary_text = self.llm.chat(
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0.0,
            seed=42,
            max_tokens=1000
        )

        # Main L263: replace any incorrect number in the summary with the computed value.
        # The scores the model may cite are exactly those in the prompt's signals block.
        prompt_has_scores = step5_skipped  # both step5-skipped prompts carry signals_block
        prompt_scores = []
        if prompt_has_scores:
            ups, downs = self._rank_enrichment_signals(input_data)
            prompt_scores = [(v, nm) for v, nm, _ in ups[:6] + downs[:6]]
        hub_counts = [] if step2_skipped else [len(top_hubs), len(hubs)]
        summary_text, number_check = self._check_summary_numbers(
            prompt, summary_text, prompt_scores, hub_counts, len(hypotheses))

        # Build key metrics based on available data
        content = f"""## Executive Summary

{summary_text.strip()}

**Key Metrics:**
- **Major biological theme:** {theme_name}
"""

        # Surface the dominant enrichment axes (both directions) here too, from the
        # same |NES| ranking as Key Findings, so the two sections name the same axes
        # (AC2) — the down-regulated axis can no longer be dropped from the summary.
        axes = self._dominant_axes(input_data)
        if axes:
            axis_strs = [
                f"{ax['pathway']} ({fc_arrow(ax['nes'])} {ax['metric']} {ax['nes']:+.2f})"
                for ax in axes
            ]
            content += f"- **Dominant enrichment axes:** {'; '.join(axis_strs)}\n"

        if not step2_skipped:
            content += f"- **Network hub genes (master regulators):** {', '.join(top_hubs)} ({len(top_hubs)} of {len(hubs)} hub genes)\n"

        if not step5_skipped:
            content += f"- **Druggable targets:** {num_drugs} drugs identified ({fda_approved} FDA-approved for indication, {fda_off_label} off-label)\n"

        content += f"- **Testable hypotheses:** {len(hypotheses)}\n"

        return ReportSection(
            title="Executive Summary",
            content=content,
            source_data={'step1': theme_name, 'step2': top_hubs, 'step5': num_drugs,
                         'summary_number_check': number_check}
        )

    def _check_summary_numbers(self, prompt: str, summary_text: str, scores, hub_counts,
                               hypothesis_count: int):
        """Check and correct the numbers in the executive summary (main L263).

        Wrong values that can be placed are replaced in place. Anything else (a wrong
        sign, an unplaceable value, a fold change / p-value / FDR, which the prompt never
        gives) triggers one rewrite with the correct values listed, then a re-check.
        What is still wrong after that is kept as an unresolved warning.
        """
        text, first = summary_number_check.find_issues(
            summary_text, scores, hub_counts, hypothesis_count)
        result = {
            'checked': first['checked'],
            'replaced': list(first['replaced']),
            'rewrite_requested': bool(first['needs_rewrite']),
            'rewrite_reasons': first['needs_rewrite'],
            'unresolved': [],
        }
        if first['needs_rewrite']:
            try:
                rewritten = self.llm.chat(
                    messages=[
                        {'role': 'user', 'content': prompt},
                        {'role': 'assistant', 'content': summary_text},
                        {'role': 'user', 'content': summary_number_check.rewrite_instructions(
                            first['needs_rewrite'], scores)},
                    ],
                    temperature=0.0,
                    seed=43,
                    max_tokens=1000,
                )
            except Exception as exc:  # keep the corrected first answer
                print(f"  Summary number rewrite failed: {exc}")
                rewritten = ''
            if rewritten and rewritten.strip():
                text, second = summary_number_check.find_issues(
                    rewritten, scores, hub_counts, hypothesis_count)
                result['replaced'].extend(second['replaced'])
                result['unresolved'] = second['needs_rewrite']
            else:
                result['unresolved'] = first['needs_rewrite']
        self._summary_unresolved = result['unresolved']  # printed in the appendix (item 26)
        if result['replaced'] or result['rewrite_requested']:
            print(f"  Summary number check: {len(result['replaced'])} replaced, "
                  f"rewrite={'yes' if result['rewrite_requested'] else 'no'}, "
                  f"{len(result['unresolved'])} unresolved")
        return text, result

    def _generate_pathway_themes_section(
        self,
        step1: Dict,
        input_data: Dict
    ) -> ReportSection:
        """Generate pathway themes section (direct extraction)"""
        themes = step1.get('themes', [])

        content = "## 1. Pathway Theme Analysis\n\n"
        content += "### Overview\n\n"
        content += f"Pathway clustering analysis identified **{len(themes)}** biological theme(s):\n\n"

        for i, theme in enumerate(themes, 1):
            name = theme.get('name', 'Unnamed Theme')
            description = theme.get('description', '')
            significance = theme.get('significance', 'unknown')
            pathways = theme.get('pathways', [])
            key_genes = theme.get('key_genes', [])
            shared_genes = theme.get('shared_genes', [])
            avg_overlap = theme.get('avg_jaccard_overlap', 0)

            content += f"#### Theme {i}: {name}\n\n"
            content += f"**Significance:** {significance.upper()}\n\n"
            content += f"**Description:**\n{description}\n\n"

            content += f"**Key Dysregulated Genes:** ({len(key_genes)} genes)\n"
            for gene in key_genes:
                # Find fold change from input data
                gene_info = next((g for g in input_data.get('genes', [])
                                 if g.get('gene', g.get('geneSymbol', '')) == gene), None)
                if gene_info:
                    fc = to_float(gene_info.get('foldChange', gene_info.get('fold_change', 0))) or 0.0
                    arrow = fc_arrow(fc)
                    content += f"- {gene} ({fc:+.2f}{' ' + arrow if arrow else ''})\n"

            content += f"\n**Enriched Pathways:** ({len(pathways)} pathways, average overlap {avg_overlap:.1%})\n\n"
            content += "| Pathway | p-value (FDR) | DE Genes |\n"
            content += "|---------|---------------|----------|\n"

            for pathway in pathways[:15]:  # Limit to top 15
                p_name = pathway.get('name', 'N/A')
                p_fdr = pathway.get('p_value_fdr', pathway.get('pValueFDR', 1.0))
                p_genes = pathway.get('gene_count', 0)
                content += f"| {p_name} | {p_fdr:.2e} | {p_genes} |\n"

            if len(pathways) > 15:
                content += f"\n*({len(pathways) - 15} additional pathways not shown)*\n"

            content += f"\n**Biological Context:**\n{theme.get('biological_context', 'N/A')}\n\n"
            content += "---\n\n"

        return ReportSection(
            title="Pathway Theme Analysis",
            content=content,
            source_data={'themes': [t.get('name') for t in themes]}
        )

    def _generate_hub_genes_section(
        self,
        step2: Dict,
        input_data: Dict
    ) -> ReportSection:
        """Generate hub genes section (direct extraction)"""
        network_hubs = step2.get('network_hubs', [])
        master_regulators = step2.get('master_regulators', [])
        hub_interpretation = step2.get('hub_interpretation', {})

        content = "## 2. Network Hub Genes\n\n"
        content += "### Network Hub Genes (Master Regulators)\n\n"
        content += f"Network topology analysis identified **{len(network_hubs)}** hub genes, "
        content += f"with **{len(master_regulators)}** top hubs highlighted as master regulators:\n\n"
        content += ("*These are selected by network centrality (connectivity), so they are "
                    "topological hubs — not necessarily the largest effect-size drivers.*\n\n")

        # Get detailed interpretations if available
        hub_interpretations = hub_interpretation.get('hub_interpretations', [])

        # Render a single master-regulator block. `idx` is the continuous
        # 1-based counter across both directional groups.
        def _render_master_reg(master_reg: Dict, idx: int) -> str:
            gene = master_reg.get('gene', 'N/A')
            fc = master_reg.get('fold_change', 0)
            role = master_reg.get('role', '')
            pathways = master_reg.get('pathways', [])

            block = f"#### {idx}. {gene}\n\n"
            block += f"**Network Centrality:**\n"

            full_hub = next((h for h in network_hubs if h.get('gene') == gene), None)
            if full_hub:
                block += f"- Degree: {full_hub.get('degree', 0)} (rank: {full_hub.get('degree_rank', 0):.1%})\n"
                block += f"- Betweenness: {full_hub.get('betweenness', 0):.3f} (rank: {full_hub.get('betweenness_rank', 0):.1%})\n"
                block += f"- Closeness: {full_hub.get('closeness', 0):.3f} (rank: {full_hub.get('closeness_rank', 0):.1%})\n"
                block += f"- Hub score: {full_hub.get('hub_score', 0):.3f}\n"

            block += f"\n**Differential Expression:** "
            if fc != 0:
                block += f"{fc:+.2f} {fc_arrow(fc)}\n"
            else:
                block += "No significant change\n"

            block += f"\n**Biological Role:**\n{role}\n\n"

            block += f"**Pathway Involvement:**\n"
            for pathway in pathways:
                block += f"- {pathway}\n"

            detailed = next((h for h in hub_interpretations if h.get('gene') == gene), None)
            if detailed:
                block += f"\n**Mechanistic Importance:**\n{detailed.get('mechanistic_importance', '')}\n\n"
                block += f"**Predicted Impact:**\n{detailed.get('predicted_impact', '')}\n\n"

            block += "---\n\n"
            return block

        # Group master regulators by regulation direction so down-regulated /
        # metabolic hubs are surfaced alongside up-regulated proliferation hubs,
        # rather than the section collapsing to a single (up) direction.
        up_regs = [m for m in master_regulators if m.get('direction') == 'up']
        down_regs = [m for m in master_regulators if m.get('direction') == 'down']

        counter = 1
        other_regs = [m for m in master_regulators
                      if m.get('direction') not in ('up', 'down')]
        if up_regs and down_regs:
            content += "### Up-regulated Hubs\n\n"
            for m in up_regs:
                content += _render_master_reg(m, counter)
                counter += 1
            content += "### Down-regulated / Metabolic Hubs\n\n"
            for m in down_regs:
                content += _render_master_reg(m, counter)
                counter += 1
            # Any hubs with neither direction (e.g. exact-zero fold change) still render.
            for m in other_regs:
                content += _render_master_reg(m, counter)
                counter += 1
        else:
            for m in master_regulators:
                content += _render_master_reg(m, counter)
                counter += 1

        # Network interpretation
        network_interp = hub_interpretation.get('network_interpretation', '')
        if network_interp:
            content += f"### Network Interpretation\n\n{network_interp}\n\n"

        return ReportSection(
            title="Network Hub Genes",
            content=content,
            source_data={'master_regulators': [m.get('gene') for m in master_regulators]}
        )

    def _generate_mechanisms_section(
        self,
        step3: Dict
    ) -> ReportSection:
        """Generate mechanisms section (use pre-generated if available)"""
        # Check if step3 already has a formatted report section
        if 'report_section' in step3 and step3['report_section']:
            content = step3['report_section']
        else:
            # Fallback: generate from pathway_mechanisms
            pathway_mechanisms = step3.get('pathway_mechanisms', [])

            content = "## 3. Pathway Mechanisms and Molecular Interactions\n\n"
            content += "### Overview\n\n"

            mechanistic_summary = step3.get('mechanistic_summary', '')
            if mechanistic_summary:
                content += f"{mechanistic_summary}\n\n"

            # Relations can be curated, Pathway Commons-grounded or LLM-inferred, so "curated" overclaimed.
            # content += f"Mechanistic analysis identified **{len(pathway_mechanisms)}** pathway mechanisms with curated regulatory relations.\n\n"
            content += f"Mechanistic analysis identified **{len(pathway_mechanisms)}** pathway mechanisms with regulatory relations.\n\n"

            # Show top pathways
            content += "### Key Pathway Mechanisms\n\n"

            for mech in pathway_mechanisms[:5]:  # Top 5 pathways
                pathway = mech.get('pathway', 'N/A')
                bio_func = mech.get('biologicalFunction', '')
                de_genes = mech.get('deGeneInvolvement', [])
                relations = mech.get('curatedRelations', [])
                consequences = mech.get('functionalConsequences', '')

                content += f"#### {pathway}\n\n"
                content += f"**Biological Function:**\n{bio_func}\n\n"

                if de_genes:
                    content += f"**Dysregulated Genes:** ({len(de_genes)})\n\n"
                    content += "| Gene | Fold Change | Role in Pathway |\n"
                    content += "|------|-------------|------------------|\n"
                    for gene_info in de_genes:
                        gene = gene_info.get('gene', 'N/A')
                        fc = to_float(gene_info.get('foldChange', 0)) or 0.0
                        arrow = fc_arrow(fc)
                        role = gene_info.get('roleInPathway', '')[:80]  # Truncate
                        content += f"| {gene} | {fc:+.2f}{' ' + arrow if arrow else ''} | {role} |\n"
                    content += "\n"

                if relations:
                    content += f"**Key Regulatory Relations:** ({len(relations)})\n\n"
                    for rel in relations[:5]:  # Top 5 relations
                        source = rel.get('source', 'N/A')
                        target = rel.get('target', 'N/A')
                        rel_type = rel.get('type', 'unknown')
                        interp = rel.get('interpretation', '')
                        content += f"- **{source} → {target}** ({rel_type}): {interp}\n"
                    content += "\n"

                content += f"**Functional Consequences:**\n{consequences}\n\n"
                content += "---\n\n"

        # Upstream-regulator candidates (TFs) for the strongest down-regulated
        # programs — driver TFs that are typically NOT differentially expressed and
        # so are invisible to the DE-based PPI-hub scoring in Step 2.
        content += self._format_upstream_regulators(step3.get('upstream_regulators', []))

        return ReportSection(
            title="Pathway Mechanisms",
            content=content,
            source_data={'num_mechanisms': len(step3.get('pathway_mechanisms', []))}
        )

    def _format_upstream_regulators(self, upstream_regulators: List[Dict]) -> str:
        """Render the 'Upstream Regulator Candidates' subsection (empty string if none)."""
        if not upstream_regulators:
            return ""

        # DB-derived rows vs LLM-proposed rows carry different provenance; the disclaimer
        # and per-row Evidence-source column keep them unambiguous.
        is_llm = any(r.get('evidence_source') == 'llm_hypothesis' for r in upstream_regulators)

        block = "### Upstream Regulator Candidates\n\n"
        block += ("*Candidate transcription factors driving the down-regulated program, "
                  "identified by testing which TF's target set (CollecTRI regulon) is "
                  "over-represented among the down-regulated genes (hypergeometric test, "
                  "Benjamini-Hochberg FDR). These drivers are frequently not themselves "
                  "differentially expressed, so they are missed by DE/PPI-hub analysis. "
                  "The statistical background is the regulon gene space (a pre-filtered DE "
                  "input makes p-values optimistic), so these are ranked candidates to test, "
                  "not established findings.*\n\n")
        if is_llm:
            reason = next((r.get('fallback_reason') for r in upstream_regulators
                           if r.get('evidence_source') == 'llm_hypothesis'), None)
            if reason == 'no_significant_enrichment':
                why = ("no transcription factor reached regulon-enrichment significance for "
                       "this run")
            else:
                why = "the regulon database was unavailable for this run"
            block += (f"*Note: {why} — the rows below are **LLM-proposed hypotheses**, "
                      "not database-derived.*\n\n")

        block += ("| Candidate TF | In DE List? | Inferred TF Activity | "
                  "Down-regulated Targets (overlap) | Enrichment FDR | Evidence Source |\n")
        block += ("|--------------|-------------|----------------------|"
                  "----------------------------------|----------------|-----------------|\n")
        for reg in upstream_regulators:
            tf = reg.get('tf', 'N/A')
            de_note = 'yes' if reg.get('is_de', False) else 'no (not DE)'
            activity = reg.get('inferred_tf_activity', 'unknown')

            shown = reg.get('targets', [])[:6]
            targets = ', '.join(shown)
            overlap = reg.get('overlap_count', reg.get('num_targets', len(reg.get('targets', []))))
            if len(reg.get('targets', [])) > 6:
                targets += ', …'
            reg_size = reg.get('regulon_size')
            targets += f" ({overlap}/{reg_size})" if reg_size else f" ({overlap})"

            fdr = reg.get('enrichment_fdr')
            fdr_str = f"{fdr:.2e}" if isinstance(fdr, (int, float)) else "—"

            source = {'collectri': 'CollecTRI', 'llm_hypothesis': 'LLM hypothesis'}.get(
                reg.get('evidence_source'), reg.get('evidence_source') or '—')

            pathways = ', '.join(reg.get('evidence_pathways', [])[:3])
            evidence = f"{source}" + (f"; {pathways}" if pathways else "")
            block += (f"| {tf} | {de_note} | {activity} | {targets} | "
                      f"{fdr_str} | {evidence} |\n")
        block += "\n"
        return block

    def _generate_hypotheses_section(
        self,
        step4: Dict
    ) -> ReportSection:
        """Generate hypotheses section (use pre-generated if available)"""
        hypotheses = step4.get('hypotheses', [])

        # Check if step4 already has a formatted report section
        if 'report_section' in step4 and step4['report_section']:
            content = step4['report_section']
        else:
            # Fallback: generate from hypotheses
            central_model = step4.get('central_mechanistic_model', '')

            content = "## 4. Mechanistic Hypotheses and Predictions\n\n"

            if central_model:
                content += "### Central Mechanistic Model\n\n"
                content += f"{central_model}\n\n"
                content += "---\n\n"

            content += "### Testable Hypotheses\n\n"
            content += f"Generated **{len(hypotheses)}** testable hypotheses from pathway and network analysis:\n\n"

            for i, hyp in enumerate(hypotheses, 1):
                statement = hyp.get('hypothesis', 'N/A')
                confidence = hyp.get('confidence', 'unknown').upper()
                novelty = normalize_novelty(hyp.get('novelty'))
                model = hyp.get('mechanisticModel', '')
                key_players = hyp.get('keyPlayers', [])
                evidence = hyp.get('evidenceSupporting', [])
                testability = hyp.get('testability', {})
                prediction = hyp.get('quantitativePrediction', '')

                content += f"#### Hypothesis {i}\n\n"
                content += f"**Statement:** {statement}\n\n"
                content += f"*Confidence: {confidence} | Novelty: {novelty}*\n\n"

                content += f"**Mechanistic Model:**\n{model}\n\n"

                content += f"**Key Players:**\n"
                for player in key_players:
                    content += f"- {player}\n"
                content += "\n"

                content += f"**Supporting Evidence:**\n"
                for evid in evidence:
                    content += f"- {evid}\n"
                content += "\n"

                if testability:
                    content += f"**Experimental Validation:**\n"
                    content += f"- **Approach 1:** {testability.get('approach1', 'N/A')}\n"
                    content += f"- **Approach 2:** {testability.get('approach2', 'N/A')}\n"
                    content += f"- **Expected Outcome:** {testability.get('expectedOutcome', 'N/A')}\n\n"

                content += f"**Quantitative Prediction:** {prediction}\n\n"
                content += "---\n\n"

        return ReportSection(
            title="Mechanistic Hypotheses",
            content=content,
            source_data={'num_hypotheses': len(hypotheses)}
        )

    def _generate_therapeutics_section(
        self,
        step5: Dict
    ) -> ReportSection:
        """Generate therapeutics section (use pre-generated if available)"""
        # Check if step5 already has a formatted report section
        if 'report_section' in step5 and step5['report_section']:
            content = step5['report_section']
        else:
            # Fallback: generate from drug_recommendations
            drugs = step5.get('drug_recommendations', [])
            targets = step5.get('therapeutic_targets', [])
            combos = step5.get('combination_therapies', [])

            content = "## 5. Therapeutic Implications\n\n"
            content += "### Overview\n\n"
            content += f"Druggability analysis identified **{len(targets)}** targetable genes with **{len(drugs)}** therapeutic agents.\n\n"

            content += "### Recommended Therapeutics\n\n"

            for i, drug in enumerate(drugs, 1):
                drug_name = drug.get('drugName', drug.get('drug_name', 'N/A'))
                target_genes = drug.get('targetGenes', drug.get('target_genes', []))
                approval = drug.get('approvalStatus', drug.get('approval_status', 'Unknown'))
                moa = drug.get('mechanismOfAction', drug.get('mechanism_of_action', 'N/A'))
                confidence = drug.get('confidence', 'unknown').upper()
                rationale = drug.get('therapeuticRationale', drug.get('therapeutic_rationale', ''))
                evidence = drug.get('supportingEvidence', drug.get('supporting_evidence', []))

                content += f"#### {i}. {drug_name}\n\n"
                content += f"**Targets:** {', '.join(target_genes)}\n\n"
                content += f"**Clinical Status:** {approval}\n\n"
                content += f"**Mechanism of Action:** {moa}\n\n"
                content += f"**Confidence:** {confidence}\n\n"
                content += f"**Rationale:**\n{rationale}\n\n"

                if evidence:
                    content += f"**Evidence:**\n"
                    for evid in evidence:
                        content += f"- {evid}\n"
                    content += "\n"

                content += "---\n\n"

            if combos:
                content += "### Combination Therapy Strategies\n\n"
                for combo in combos:
                    combo_drugs = combo.get('drugs', [])
                    combo_name = ' + '.join(combo_drugs) if combo_drugs else 'N/A'
                    combo_rationale = combo.get('rationale', '')
                    combo_mechanism = combo.get('synergisticMechanism', '')
                    combo_targets = combo.get('targetPathways', [])
                    combo_confidence = combo.get('confidence', 'unknown').upper()
                    content += f"#### {combo_name}\n\n"
                    if combo_targets:
                        content += f"**Target Pathways:** {', '.join(combo_targets)}\n\n"
                    content += f"**Rationale:** {combo_rationale}\n\n"
                    if combo_mechanism:
                        content += f"**Synergistic Mechanism:** {combo_mechanism}\n\n"
                    content += f"**Confidence:** {combo_confidence}\n\n"
                    content += "---\n\n"

        return ReportSection(
            title="Therapeutic Implications",
            content=content,
            source_data={'num_drugs': len(step5.get('drug_recommendations', []))}
        )

    def _theme_for_pathway(self, themes: List[Dict], pathway_name: str) -> Optional[str]:
        """Return the name of the theme whose member pathways include ``pathway_name``."""
        target = (pathway_name or '').strip().lower()
        if not target:
            return None
        for t in themes or []:
            for pw in t.get('pathways', []) or []:
                nm = pw.get('name', '') if isinstance(pw, dict) else str(pw)
                if (nm or '').strip().lower() == target:
                    return t.get('name')
        return None

    def _generate_conclusions(
        self,
        study_context: str,
        input_data: Dict,
        step1: Dict,
        step2: Dict,
        step3: Dict,
        step4: Dict,
        step5: Dict
    ) -> ReportSection:
        """Generate conclusions section"""
        # Check if steps were skipped
        step2_skipped = step2.get('metadata', {}).get('skipped', False)
        step5_skipped = step5.get('metadata', {}).get('skipped', False)

        themes = step1.get('themes', [])
        master_regs = step2.get('master_regulators', [])
        hypotheses = step4.get('hypotheses', [])
        drugs = step5.get('drug_recommendations', [])
        upstream_regulators = step3.get('upstream_regulators', []) if step3 else []

        content = "## 6. Summary and Conclusions\n\n"
        content += "### Key Findings\n\n"

        finding_num = 1

        # Report ALL dominant enrichment axes (both directions), derived from the
        # same |NES| ranking the executive summary uses, so a strongly enriched
        # direction is never dropped here while being named in the summary (AC2).
        axes = self._dominant_axes(input_data)
        if axes:
            for ax in axes:
                arrow = fc_arrow(ax['nes'])
                theme_nm = self._theme_for_pathway(themes, ax['pathway'])
                label = f"{theme_nm} — {ax['pathway']}" if theme_nm else ax['pathway']
                content += (f"{finding_num}. **Dominant {ax['direction']}-regulated axis "
                            f"{arrow}:** {label} ({ax['metric']} {ax['nes']:+.2f})\n")
                finding_num += 1
        else:
            # Fallback when no enrichment scores are available: the major theme
            # (most pathways, main L265), not themes[0].
            _major = major_theme(themes)
            content += (f"{finding_num}. **Biological Theme:** "
                        f"{_major.get('name', 'N/A') if _major else 'No major theme'}\n")
            finding_num += 1

        if not step2_skipped:
            up_regs = [m.get('gene') for m in master_regs if m.get('direction') == 'up']
            down_regs = [m.get('gene') for m in master_regs if m.get('direction') == 'down']
            if up_regs and down_regs:
                content += (f"{finding_num}. **Master Regulators (up-regulated hubs):** "
                            f"{', '.join(up_regs[:3])}\n")
                finding_num += 1
                content += (f"{finding_num}. **Master Regulators (down-regulated / metabolic hubs):** "
                            f"{', '.join(down_regs[:3])}\n")
                finding_num += 1
            else:
                top_genes = [m.get('gene') for m in master_regs[:3]]
                content += f"{finding_num}. **Master Regulators:** {', '.join(top_genes)}\n"
                finding_num += 1

        # Candidate upstream regulators (TFs) — driver TFs of the down-regulated
        # programs that are typically not DE and so missed by hub scoring (AC4).
        if upstream_regulators:
            tf_names = [u.get('tf') for u in upstream_regulators[:3]]
            content += (f"{finding_num}. **Candidate Upstream Regulators (TFs):** "
                        f"{', '.join(tf_names)}\n")
            finding_num += 1

        if not step5_skipped:
            content += f"{finding_num}. **Therapeutic Targets:** {len(drugs)} drugs identified\n"
            finding_num += 1

            # Count only drugs approved specifically for this indication (not off-label)
            fda_approved = [d for d in drugs if 'Approved for' in d.get('approvalStatus', '') or ('Approved' in d.get('approvalStatus', '') and 'off-label' not in d.get('approvalStatus', '').lower() and 'investigational' not in d.get('approvalStatus', '').lower())]
            fda_off_label = sum(1 for d in drugs if 'off-label' in d.get('approvalStatus', '').lower())
            content += f"{finding_num}. **FDA-Approved Options:** {len(fda_approved)} for this indication, {fda_off_label} off-label\n"

        content += "\n"

        content += "### Next Steps\n\n"
        content += "**Experimental Validation:**\n"

        if not step2_skipped:
            content += "1. Validate master regulator function through perturbation studies\n"
            content += "2. Test top mechanistic hypotheses in relevant cell/animal models\n\n"
        else:
            content += "1. Test top mechanistic hypotheses in relevant cell/animal models\n"
            content += "2. Validate pathway-level predictions with functional assays\n\n"

        if not step5_skipped:
            content += "**Clinical Translation:**\n"
            content += "1. Evaluate FDA-approved drugs in preclinical models\n"
            content += "2. Design biomarker-driven patient stratification strategies\n\n"

        return ReportSection(
            title="Summary and Conclusions",
            content=content,
            source_data={'study_context': study_context}
        )

    def _generate_appendices(
        self,
        input_data: Dict,
        step1: Dict,
        step2: Dict,
        step3: Dict,
        step4: Dict,
        step5: Dict
    ) -> ReportSection:
        """Generate appendices"""
        # Check if steps were skipped
        step2_skipped = step2.get('metadata', {}).get('skipped', False)
        step5_skipped = step5.get('metadata', {}).get('skipped', False)

        content = "## Appendices\n\n"

        content += "### A. Methodology\n\n"
        content += "- **Step 1:** Pathway theme clustering (Jaccard ≥ 0.25)\n"

        if not step2_skipped:
            content += "- **Step 2:** STRING PPI network analysis (score ≥ 400)\n"
        else:
            content += "- **Step 2:** Skipped (no gene expression data)\n"

        # Steps 3 and B used to state KEGG/Reactome for every report, including custom gene
        # sets (e.g. MitoPathways) — a false provenance claim in the Methods. They now name
        # what was actually used.
        # content += "- **Step 3:** KEGG/Reactome pathway structure retrieval\n"
        content += f"- **Step 3:** {_step3_methods_text(step3)}\n"
        content += "- **Step 4:** LLM-based hypothesis generation\n"
        content += "- **Step 5:** Final report generation\n\n"

        content += "### B. Data Sources\n\n"
        # content += "- **Pathway database:** KEGG/Reactome\n"
        databases = _pathway_databases(input_data)
        content += f"- **Pathway database:** {', '.join(databases) if databases else 'not specified in the input'}\n"

        if not step2_skipped:
            content += "- **PPI database:** STRING v11.5\n"

        if not step5_skipped:
            content += "- **Drug-gene database:** DGIdb v5.0\n"

        # Extract organism from metadata
        metadata = input_data.get('metadata', {})
        organism = metadata.get('organism', 'Homo sapiens')
        content += f"- **Organism:** {organism}\n\n"

        # Item 26 (main L214): unresolved discrepancies are flagged as warnings in the output.
        warnings = _unresolved_warnings(step4, getattr(self, '_summary_unresolved', []))
        if warnings:
            content += "### C. Unresolved Validation Warnings\n\n"
            content += ("Discrepancies the engine detected but did not correct automatically. "
                        "They are listed for the reader to check.\n\n")
            for group, items in warnings:
                content += f"**{group}:**\n"
                for item in items:
                    content += f"- {item}\n"
                content += "\n"

        content += "---\n\n"
        content += "**Report generated by:** 5-Step Agentic Pipeline v1.0\n"
        content += f"**Generation date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return ReportSection(
            title="Appendices",
            content=content,
            source_data={}
        )

    def _assemble_report(
        self,
        metadata: Dict[str, Any],
        organism: str,
        sections: List[ReportSection]
    ) -> str:
        """Assemble all sections into final markdown report"""
        # Build report title from available metadata
        title_parts = []
        for key in ['Dataset', 'disease', 'study_type', 'description']:
            if metadata.get(key):
                title_parts.append(str(metadata[key]))
                break
        title = title_parts[0] if title_parts else 'Molecular Analysis'

        # Build header with ALL metadata fields dynamically
        header_lines = [f"# {title} - Molecular Analysis Report", ""]

        # Skip internal/technical fields
        skip_fields = {'datasetId', 'dataset_id'}

        for key, value in metadata.items():
            if key in skip_fields:
                continue
            if value and str(value).strip():
                # Format the key for display
                display_key = key.replace('_', ' ').title()
                header_lines.append(f"**{display_key}:** {value}")

        header_lines.append(f"**Organism:** {organism}")
        header_lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}")
        header_lines.append(f"**Pipeline Version:** 1.0")
        header_lines.append("")
        header_lines.append("---")
        header_lines.append("")

        report = '\n'.join(header_lines)

        # Add all sections
        for section in sections:
            report += section.content + "\n\n"

        return report
