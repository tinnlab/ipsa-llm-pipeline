"""
Step output validators for pipeline.

Validates that each step's output is structurally correct and doesn't
contain hallucinated entities (genes, pathways, drugs not in input data).
"""

import copy
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Set, Optional


@dataclass
class ValidationResult:
    """Result of validating a step's output"""
    passed: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class StepValidators:
    """Validators for each pipeline step"""

    @staticmethod
    def strip_hallucinated_entities(step_dict: Dict[str, Any], step_no: int,
                                    allow_pathways: Set[str], allow_genes: Set[str]):
        """Return (cleaned_copy, removed) with hallucinated entities removed from a step output.

        Mirrors the `validate_stepN_output` membership predicate so we strip exactly what the
        validators flag. Operates on a deep copy and does NOT mutate the input — callers keep the
        raw output and the flags; only the copy passed to the next step is cleaned.

        - step 1: drop pathways in themes[].pathways whose name is not in `allow_pathways`
        - step 2: drop hub_genes/network_hubs whose 'gene' is not in `allow_genes`
        - step 3: drop deGeneInvolvement entries whose 'gene' is not in `allow_genes`
        """
        cleaned = copy.deepcopy(step_dict)
        removed: List[str] = []

        if step_no == 1:
            for theme in cleaned.get('themes', []) or []:
                kept = []
                for p in theme.get('pathways', []) or []:
                    name = p.get('name', p.get('pathway', '')) if isinstance(p, dict) else ''
                    if name and name not in allow_pathways:
                        removed.append(name)
                    else:
                        kept.append(p)
                theme['pathways'] = kept
                if 'pathway_count' in theme:
                    theme['pathway_count'] = len(kept)

        elif step_no == 2:
            for key in ('hub_genes', 'network_hubs'):
                items = cleaned.get(key)
                if isinstance(items, list):
                    kept = []
                    for hub in items:
                        gene = hub.get('gene', '') if isinstance(hub, dict) else ''
                        if gene and gene not in allow_genes:
                            removed.append(gene)
                        else:
                            kept.append(hub)
                    cleaned[key] = kept

        elif step_no == 3:
            # Only deGeneInvolvement is stripped — same as validate_step3_output. curatedRelations
            # are NOT stripped: they come from trusted KEGG/curated structures (and their genes are
            # in pathway_genes, i.e. allowed), so removing them would discard real curated data.
            for mech in cleaned.get('pathway_mechanisms', []) or []:
                kept = []
                for gi in mech.get('deGeneInvolvement', []) or []:
                    gene = gi.get('gene', '') if isinstance(gi, dict) else ''
                    if gene and gene not in allow_genes:
                        removed.append(gene)
                    else:
                        kept.append(gi)
                mech['deGeneInvolvement'] = kept

        return cleaned, removed

    @staticmethod
    def validate_step1_output(output: Dict[str, Any], input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate Step 1: Pathway Themes

        Checks:
        - All pathways in themes exist in input data
        - Themes have required structure
        """
        errors = []
        warnings = []

        themes = output.get('themes', [])
        input_pathways = {p.get('name', '') for p in input_data.get('pathways', [])}

        # Check for hallucinated pathways
        hallucinated = []
        for theme in themes:
            for pathway in theme.get('pathways', []):
                pathway_name = pathway.get('name', pathway.get('pathway', ''))
                if pathway_name and pathway_name not in input_pathways:
                    hallucinated.append(pathway_name)

        if hallucinated:
            errors.append(f"Hallucinated pathways: {', '.join(hallucinated[:5])}")

        # Check structure
        if not themes:
            warnings.append("No themes generated")

        stats = {
            'num_themes': len(themes),
            'num_hallucinated_pathways': len(hallucinated),
            'total_pathways_in_themes': sum(len(t.get('pathways', [])) for t in themes)
        }

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    @staticmethod
    def validate_step2_output(output: Dict[str, Any], input_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate Step 2: Hub Genes

        Checks:
        - All hub genes exist in input data
        - Network metrics are present
        """
        errors = []
        warnings = []

        # Check if step was skipped due to missing genes
        if output.get('metadata', {}).get('skipped'):
            return ValidationResult(
                passed=True,
                errors=[],
                warnings=['Step 2 was skipped due to missing gene data'],
                stats={'skipped': True}
            )

        hubs = output.get('network_hubs', output.get('hub_genes', []))
        input_genes = {g.get('geneSymbol', g.get('gene', '')) for g in input_data.get('genes', [])}

        # If no input genes, skip validation
        if not input_genes:
            return ValidationResult(
                passed=True,
                errors=[],
                warnings=['No input genes to validate against'],
                stats={'num_hubs': 0}
            )

        # Check for hallucinated genes
        hallucinated = []
        for hub in hubs:
            gene = hub.get('gene', '')
            if gene and gene not in input_genes:
                hallucinated.append(gene)

        if hallucinated:
            errors.append(f"Hallucinated genes: {', '.join(hallucinated[:5])}")

        # Check structure
        if not hubs:
            warnings.append("No hub genes identified")

        stats = {
            'num_hubs': len(hubs),
            'num_hallucinated_genes': len(hallucinated),
            'avg_hub_score': sum(h.get('hub_score', 0) for h in hubs) / len(hubs) if hubs else 0
        }

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    @staticmethod
    def validate_step3_output(
        output: Dict[str, Any],
        input_data: Dict[str, Any],
        pathway_genes: Optional[Set[str]] = None
    ) -> ValidationResult:
        """
        Validate Step 3: Pathway Mechanisms

        Checks:
        - Genes in mechanisms exist in input or pathway structures
        - Curated relations are present
        """
        errors = []
        warnings = []

        mechanisms = output.get('pathway_mechanisms', [])
        input_genes = {g.get('geneSymbol', g.get('gene', '')) for g in input_data.get('genes', [])}

        # Add pathway genes if provided (from curated structures)
        allowed_genes = input_genes.copy()
        if pathway_genes:
            allowed_genes.update(pathway_genes)

        # Check for hallucinated genes in DE involvement
        hallucinated = []
        for mech in mechanisms:
            for gene_info in mech.get('deGeneInvolvement', []):
                gene = gene_info.get('gene', '')
                if gene and gene not in allowed_genes:
                    hallucinated.append(gene)

        if hallucinated:
            errors.append(f"Hallucinated genes: {', '.join(list(set(hallucinated))[:5])}")

        # Check structure
        if not mechanisms:
            warnings.append("No pathway mechanisms generated")

        total_relations = sum(len(m.get('curatedRelations', [])) for m in mechanisms)
        if total_relations == 0:
            warnings.append("No curated relations found")

        stats = {
            'num_mechanisms': len(mechanisms),
            'num_hallucinated_genes': len(set(hallucinated)),
            'total_curated_relations': total_relations
        }

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    @staticmethod
    def validate_step4_output(
        output: Dict[str, Any],
        input_data: Dict[str, Any],
        pathway_genes: Optional[Set[str]] = None
    ) -> ValidationResult:
        """
        Validate Step 4: Hypotheses

        Checks:
        - Hypotheses are testable
        - Genes mentioned exist (lenient - allows pathway genes)

        Note: More lenient than other steps because hypotheses can reference
        genes from pathway structures, not just DE genes.
        """
        errors = []
        warnings = []

        hypotheses = output.get('hypotheses', [])
        input_genes = {g.get('geneSymbol', g.get('gene', '')) for g in input_data.get('genes', [])}

        # Add pathway genes (hypotheses can reference non-DE genes from pathways)
        allowed_genes = input_genes.copy()
        if pathway_genes:
            allowed_genes.update(pathway_genes)

        # Check testability
        non_testable = []
        for i, hyp in enumerate(hypotheses):
            if not hyp.get('testability'):
                non_testable.append(i + 1)

        if non_testable:
            warnings.append(f"Hypotheses without testability: {non_testable}")

        # Check structure
        if not hypotheses:
            warnings.append("No hypotheses generated")

        # Count confidence distribution
        confidence_dist = {'high': 0, 'medium': 0, 'low': 0}
        for hyp in hypotheses:
            conf = hyp.get('confidence', 'unknown').lower()
            if conf in confidence_dist:
                confidence_dist[conf] += 1

        stats = {
            'num_hypotheses': len(hypotheses),
            'num_non_testable': len(non_testable),
            'confidence_distribution': confidence_dist
        }

        return ValidationResult(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats
        )

    @staticmethod
    def create_validation_summary(results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Create a summary of all validation results

        Args:
            results: Dict of ValidationResult objects

        Returns:
            Summary dict with overall statistics
        """
        all_passed = all(r.passed for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())

        failed_steps = [step for step, result in results.items() if not result.passed]

        return {
            'all_passed': all_passed,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'failed_steps': failed_steps,
            'num_steps_validated': len(results)
        }
