"""
Pipeline orchestrator for running the 5-step biological analysis workflow.

Coordinates execution of all pipeline steps and manages data flow between them.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from .steps.step01_pathway_themes import Step01PathwayThemes
from .steps.step02_hub_genes import Step02HubGenes
from .steps.step03_pathway_mechanisms import Step03PathwayMechanisms
from .steps.step04_hypothesis_generation import Step04HypothesisGeneration
from .steps.step06_report_generation import Step06ReportGeneration
from .services.step_validators import StepValidators
from .services.baseline_comparison_service import BaselineComparisonService
from ..api.job_manager import job_manager
from ..config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the full 5-step biological analysis pipeline.

    Manages data flow between steps, saves outputs, and handles errors.
    Optionally validates step outputs to catch LLM hallucinations.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        enable_validation: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize pipeline orchestrator.

        Args:
            output_dir: Directory to save step outputs (default: output/pipeline_jobs)
            enable_validation: Enable step output validation (default: True)
            strict_mode: Fail pipeline if validation errors found (default: False)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output/pipeline_jobs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_validation = enable_validation
        self.strict_mode = strict_mode

        # Initialize step instances
        self.step1 = Step01PathwayThemes()
        self.step2 = Step02HubGenes()
        self.step3 = Step03PathwayMechanisms()
        self.step4 = Step04HypothesisGeneration()
        self.step6 = Step06ReportGeneration()

    def _get_filtered_pathways_from_step1(
        self,
        step1_output: Dict[str, Any],
        original_pathways: list
    ) -> list:
        """
        Extract ALL pathways from Step 1 output (themes + ungrouped).

        Step 1 filters out off-tissue and off-disease pathways using LLM-based
        tissue-specificity rules. This function retrieves only the pathways that
        passed Step 1's filtering, maintaining the biological context.

        Pathways are prioritized:
        1. Themed pathways (sorted by theme significance, then FDR within theme)
        2. Ungrouped singleton pathways (sorted by FDR)

        Args:
            step1_output: Step 1 output dictionary with themes and ungrouped pathways
            original_pathways: Original pathway list from input (contains full metadata)

        Returns:
            List of filtered pathway objects, sorted by priority
        """
        filtered_pathway_names = []

        # 1. Extract pathway names from themes (prioritize by significance)
        themes = step1_output.get('themes', [])

        # Sort themes by significance (high > medium > low)
        significance_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_themes = sorted(
            themes,
            key=lambda t: significance_order.get(t.get('significance', 'low'), 2)
        )

        for theme in sorted_themes:
            theme_pathways = theme.get('pathways', [])
            # Sort pathways within theme by FDR
            sorted_theme_pathways = sorted(
                theme_pathways,
                key=lambda p: p.get('p_value_fdr', 1.0)
            )
            for pathway in sorted_theme_pathways:
                pathway_name = pathway.get('name', pathway.get('pathway', ''))
                if pathway_name and pathway_name not in filtered_pathway_names:
                    filtered_pathway_names.append(pathway_name)

        # 2. Extract pathway names from ungrouped (sorted by FDR)
        ungrouped = step1_output.get('ungrouped', [])
        sorted_ungrouped = sorted(
            ungrouped,
            key=lambda p: p.get('p_value_fdr', 1.0)
        )
        for pathway in sorted_ungrouped:
            pathway_name = pathway.get('name', pathway.get('pathway', ''))
            if pathway_name and pathway_name not in filtered_pathway_names:
                filtered_pathway_names.append(pathway_name)

        # 3. Match pathway names back to original pathway objects
        # (need full metadata including pathway ID for KEGG/Reactome API calls)
        pathway_lookup = {
            p.get('name', p.get('pathwayName', '')): p
            for p in original_pathways
        }

        filtered_pathways = []
        for name in filtered_pathway_names:
            if name in pathway_lookup:
                filtered_pathways.append(pathway_lookup[name])
            else:
                logger.warning(f"  ⚠️  Pathway '{name}' from Step 1 not found in original pathway list")

        return filtered_pathways

    def run_full_pipeline(
        self,
        input_data: Dict[str, Any],
        job_id: str,
        tracking_job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the full 5-step pipeline.

        Args:
            input_data: Pipeline input containing genes, pathways, disease info
            job_id: Unique job identifier for output organization
            tracking_job_id: Optional separate job ID for progress tracking
                (used by meta-pipeline so nested pipeline updates the parent job)

        Returns:
            Dictionary containing all step results and execution metadata
        """
        start_time = time.time()
        progress_id = tracking_job_id or job_id
        job_output_dir = self.output_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting pipeline execution for job {job_id}")
        logger.info(f"Output directory: {job_output_dir}")

        # Save input data to job directory for reproducibility
        input_file = job_output_dir / 'input_data.json'
        with open(input_file, 'w') as f:
            json.dump(input_data, f, indent=2)
        logger.info(f"Input data saved to: {input_file}")

        # Extract input components
        genes = input_data['genes']
        pathways = input_data['pathways']

        # Ground-truth allow-lists used to strip hallucinated entities before they are passed to
        # the next step. These mirror the membership predicates in StepValidators so we remove
        # exactly what the validators flag (the flags themselves are preserved in results['validation']).
        allow_pathways = {p.get('name', '') for p in pathways}
        allow_genes = {g.get('geneSymbol', g.get('gene', '')) for g in genes}

        # Extract metadata - include ALL metadata fields for flexible analysis
        metadata = input_data.get('metadata', {})
        organism = metadata.get('organism', 'Homo sapiens')

        # Build context from ALL metadata fields (not just disease/tissue)
        # This allows the pipeline to work with any kind of study metadata
        context = {}
        for key, value in metadata.items():
            if key != 'organism' and value:  # Skip organism (handled separately) and empty values
                context[key] = value

        # Check if genes are provided
        has_genes = genes and len(genes) > 0
        if not has_genes:
            logger.info("\n⚠️  Note: No DE genes provided in input")
            logger.info("   Steps 2 (Hub Genes) and 3 (Pathway Mechanisms) will be skipped")
            logger.info("   Pipeline will run with pathway-only analysis\n")

        # Build input summary with all metadata fields (dynamic, not hardcoded)
        input_summary = {
            'num_genes': len(genes),
            'num_pathways': len(pathways),
            'has_genes': has_genes,
            'metadata': metadata  # Store full metadata for flexible access
        }

        results = {
            'job_id': job_id,
            'input_summary': input_summary,
            'steps': {},
            'execution_times': {},
            'validation': {} if self.enable_validation else None
        }

        # Collect pathway genes for validation (genes from curated pathway structures)
        pathway_genes = set()

        try:
            # Step 1: Pathway Themes
            step_start = time.time()
            job_manager.set_current_step(progress_id, 1)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 1: Pathway Theme Identification")
            logger.info("=" * 80)

            step1_output = self.step1.execute(
                pathways=pathways,
                genes=genes,
                organism=organism,
                context=context
            )

            step1_dict = self._to_dict(step1_output)
            self._save_step_output(job_output_dir, 1, step1_dict)
            results['steps']['step1'] = step1_dict
            step1_time = time.time() - step_start
            results['execution_times']['step1'] = step1_time
            job_manager.set_step_result(progress_id, 1, step1_dict, step1_time)

            # Validate Step 1 output
            themes_source = step1_dict  # forwarded copy; cleaned below if validation strips hallucinations
            if self.enable_validation:
                validation = StepValidators.validate_step1_output(step1_dict, input_data)
                results['validation']['step1'] = validation.to_dict()
                logger.info(f"Step 1 Validation: {'PASSED' if validation.passed else 'FAILED'}")
                if validation.errors:
                    logger.warning(f"  Errors: {validation.errors}")
                if validation.warnings:
                    logger.info(f"  Warnings: {validation.warnings}")
                if not validation.passed and self.strict_mode:
                    raise ValueError(f"Step 1 validation failed: {validation.errors}")
                # Strip hallucinated pathways before downstream use (raw output + flags preserved)
                themes_source, removed = StepValidators.strip_hallucinated_entities(step1_dict, 1, allow_pathways, allow_genes)
                if removed:
                    results['validation']['step1'].setdefault('stats', {})['removed_entities'] = removed
                    logger.info(f"  Step 1: stripped {len(removed)} hallucinated pathway(s) before downstream use: {removed[:5]}")

            # Extract themes from Step 1 (needed for Step 2, 3, 4, 5) — from the cleaned copy
            themes = themes_source.get('themes', [])

            # Step 2: Hub Genes (skip if no genes provided)
            step_start = time.time()
            job_manager.set_current_step(progress_id, 2)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 2: Hub Gene Detection")
            logger.info("=" * 80)

            if has_genes:
                step2_output = self.step2.execute(
                    genes=genes[:100],  # Limit for STRING API
                    pathway_themes=themes,
                    organism=organism,
                    context=context
                )

                step2_dict = self._to_dict(step2_output)
                self._save_step_output(job_output_dir, 2, step2_dict)
                results['steps']['step2'] = step2_dict
                step2_time = time.time() - step_start
                results['execution_times']['step2'] = step2_time
                job_manager.set_step_result(progress_id, 2, step2_dict, step2_time)

                # Validate Step 2 output
                if self.enable_validation:
                    validation = StepValidators.validate_step2_output(step2_dict, input_data)
                    results['validation']['step2'] = validation.to_dict()
                    logger.info(f"Step 2 Validation: {'PASSED' if validation.passed else 'FAILED'}")
                    if validation.errors:
                        logger.warning(f"  Errors: {validation.errors}")
                    if validation.warnings:
                        logger.info(f"  Warnings: {validation.warnings}")
                    if not validation.passed and self.strict_mode:
                        raise ValueError(f"Step 2 validation failed: {validation.errors}")
                    # Strip hallucinated hub genes before downstream use (raw output + flags preserved)
                    step2_clean, removed = StepValidators.strip_hallucinated_entities(step2_dict, 2, allow_pathways, allow_genes)
                    if removed:
                        results['validation']['step2'].setdefault('stats', {})['removed_entities'] = removed
                        logger.info(f"  Step 2: stripped {len(removed)} hallucinated hub gene(s) before downstream use: {removed[:5]}")
                    step2_dict = step2_clean  # forward cleaned copy; results['steps']['step2'] keeps raw
            else:
                # Create skipped output for Step 2
                step2_dict = {
                    'network_hubs': [],
                    'hub_interpretation': {},
                    'master_regulators': [],
                    'summary': 'Hub gene analysis was not performed because no DE genes were provided.',
                    'metadata': {'skipped': True, 'reason': 'No genes provided'}
                }
                self._save_step_output(job_output_dir, 2, step2_dict)
                results['steps']['step2'] = step2_dict
                step2_time = time.time() - step_start
                results['execution_times']['step2'] = step2_time
                job_manager.set_step_result(progress_id, 2, step2_dict, step2_time)
                logger.info("  Step 2 skipped (no genes provided)")

            # Step 3: Pathway Mechanisms (skip if no genes provided)
            step_start = time.time()
            job_manager.set_current_step(progress_id, 3)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 3: Pathway Mechanisms")
            logger.info("=" * 80)

            if has_genes:
                # Get ALL filtered pathways from Step 1 (themes + ungrouped)
                # This ensures Step 3 only analyzes pathways that passed Step 1's tissue-specificity filtering
                filtered_pathways = self._get_filtered_pathways_from_step1(step1_dict, pathways)

                # Determine pathway sources for dynamic message
                pathway_sources = set()
                for p in filtered_pathways:
                    source = p.get('source', '').upper()
                    if 'KEGG' in source:
                        pathway_sources.add('KEGG')
                    elif 'REACTOME' in source:
                        pathway_sources.add('Reactome')

                # Set dynamic message based on pathway sources
                if pathway_sources == {'KEGG'}:
                    step3_message = "Fetching molecular interactions from KEGG pathway database..."
                elif pathway_sources == {'Reactome'}:
                    step3_message = "Fetching molecular interactions from Reactome pathway database..."
                elif 'KEGG' in pathway_sources and 'Reactome' in pathway_sources:
                    step3_message = "Fetching molecular interactions from KEGG and Reactome databases..."
                else:
                    step3_message = "Fetching molecular interactions from pathway databases..."

                job_manager.set_current_step(progress_id, 3, step3_message)

                # Count themed vs ungrouped pathways for logging
                themed_pathway_count = sum(t.get('pathway_count', 0) for t in themes)
                ungrouped_count = len(step1_dict.get('ungrouped', []))

                logger.info(f"  Analyzing {len(filtered_pathways)} filtered pathways from Step 1:")
                logger.info(f"    - Themed pathways: {themed_pathway_count} (across {len(themes)} themes)")
                logger.info(f"    - Ungrouped pathways: {ungrouped_count}")

                hub_genes = step2_dict.get('hub_genes', [])
                step3_output = self.step3.execute(
                    pathways=filtered_pathways,  # ALL filtered pathways from Step 1 (not limited to 10)
                    genes=genes,
                    analyses=[{'organismId': organism}],
                    themes=themes,
                    hub_genes=hub_genes
                )

                step3_dict = self._to_dict(step3_output)
                self._save_step_output(job_output_dir, 3, step3_dict)
                results['steps']['step3'] = step3_dict
                step3_time = time.time() - step_start
                results['execution_times']['step3'] = step3_time
                job_manager.set_step_result(progress_id, 3, step3_dict, step3_time)

                # Collect pathway genes from curated relations for downstream validation
                for mech in step3_dict.get('pathway_mechanisms', []):
                    for rel in mech.get('curatedRelations', []):
                        pathway_genes.add(rel.get('source', ''))
                        pathway_genes.add(rel.get('target', ''))

                # Validate Step 3 output
                if self.enable_validation:
                    validation = StepValidators.validate_step3_output(step3_dict, input_data, pathway_genes)
                    results['validation']['step3'] = validation.to_dict()
                    logger.info(f"Step 3 Validation: {'PASSED' if validation.passed else 'FAILED'}")
                    if validation.errors:
                        logger.warning(f"  Errors: {validation.errors}")
                    if validation.warnings:
                        logger.info(f"  Warnings: {validation.warnings}")
                    if not validation.passed and self.strict_mode:
                        raise ValueError(f"Step 3 validation failed: {validation.errors}")
                    # Strip hallucinated DE genes before Step 4 (raw output + flags preserved)
                    step3_clean, removed = StepValidators.strip_hallucinated_entities(
                        step3_dict, 3, allow_pathways, allow_genes | set(pathway_genes))
                    if removed:
                        results['validation']['step3'].setdefault('stats', {})['removed_entities'] = removed
                        logger.info(f"  Step 3: stripped {len(removed)} hallucinated gene(s) before downstream use: {removed[:5]}")
                    step3_dict = step3_clean  # forward cleaned copy; results['steps']['step3'] keeps raw
            else:
                # Create skipped output for Step 3
                step3_dict = {
                    'pathway_mechanisms': [],
                    'metadata': {'skipped': True, 'reason': 'No genes provided'}
                }
                self._save_step_output(job_output_dir, 3, step3_dict)
                results['steps']['step3'] = step3_dict
                step3_time = time.time() - step_start
                results['execution_times']['step3'] = step3_time
                job_manager.set_step_result(progress_id, 3, step3_dict, step3_time)
                logger.info("  Step 3 skipped (no genes provided)")

            # Step 4: Hypothesis Generation
            step_start = time.time()
            job_manager.set_current_step(progress_id, 4)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 4: Hypothesis Generation")
            logger.info("=" * 80)

            step4_output = self.step4.execute(
                genes=genes,
                pathways=pathways,
                themes=themes,
                hub_genes_result=step2_dict,
                mechanisms_result=step3_dict,
                context=context
            )
            step4_dict = self._to_dict(step4_output)

            # Retry once with a different seed if LLM returned 0 hypotheses
            if not step4_dict.get('hypotheses'):
                logger.warning("Step 4 produced 0 hypotheses, retrying with different seed...")
                from .steps.step04_hypothesis_generation import Step04HypothesisGeneration
                step4_retry = Step04HypothesisGeneration()
                step4_output = step4_retry.execute(
                    genes=genes,
                    pathways=pathways,
                    themes=themes,
                    hub_genes_result=step2_dict,
                    mechanisms_result=step3_dict,
                    context=context,
                    seed=1042
                )
                step4_dict = self._to_dict(step4_output)
                logger.info(f"Retry produced {len(step4_dict.get('hypotheses', []))} hypotheses")

            self._save_step_output(job_output_dir, 4, step4_dict)
            results['steps']['step4'] = step4_dict
            step4_time = time.time() - step_start
            results['execution_times']['step4'] = step4_time
            job_manager.set_step_result(progress_id, 4, step4_dict, step4_time)

            # Validate Step 4 output
            if self.enable_validation:
                validation = StepValidators.validate_step4_output(step4_dict, input_data, pathway_genes)
                results['validation']['step4'] = validation.to_dict()
                logger.info(f"Step 4 Validation: {'PASSED' if validation.passed else 'FAILED'}")
                if validation.errors:
                    logger.warning(f"  Errors: {validation.errors}")
                if validation.warnings:
                    logger.info(f"  Warnings: {validation.warnings}")
                if not validation.passed and self.strict_mode:
                    raise ValueError(f"Step 4 validation failed: {validation.errors}")

            # Baseline Comparison (optional — runs external LLMs with same input)
            if settings.ENABLE_BASELINE_COMPARISON:
                ctx_parts = []
                if context:
                    for k, v in context.items():
                        if k not in ('datasetId', 'dataset_id', 'organism') and v:
                            ctx_parts.append(f"{k}: {v}")
                experiment_context = ', '.join(ctx_parts) if ctx_parts else 'unknown'

                results['baseline_comparisons'] = {}
                for provider in ['openai', 'anthropic']:
                    try:
                        baseline_service = BaselineComparisonService(provider=provider)
                        baseline_result = baseline_service.run_baseline(
                            genes=genes,
                            pathways=pathways,
                            context=context,
                            experiment_context=experiment_context,
                            output_dir=job_output_dir
                        )
                        if baseline_result:
                            results['baseline_comparisons'][provider] = baseline_result
                    except Exception as e:
                        logger.warning(f"Baseline comparison ({provider}) failed: {e}")

            # Step 5: Final Report Generation (the last step)
            step_start = time.time()
            job_manager.set_current_step(progress_id, 5)
            logger.info("\n" + "=" * 80)
            logger.info("STEP 5: Final Report Generation")
            logger.info("=" * 80)

            step6_output = self.step6.execute(
                input_data=input_data,
                step1_output=step1_dict,
                step2_output=step2_dict,
                step3_output=step3_dict,
                step4_output=step4_dict
            )

            step6_dict = self._to_dict(step6_output)
            # `step5` is the report. Report generation was originally step 6; it moved into
            # the step-5 slot when the pipeline shrank. The key is deliberately kept as
            # `step5` because API consumers read `results.steps.step5.markdown_content`.
            self._save_step_output(job_output_dir, 5, step6_dict)
            results['steps']['step5'] = step6_dict
            step6_time = time.time() - step_start
            results['execution_times']['step5'] = step6_time
            job_manager.set_step_result(progress_id, 5, step6_dict, step6_time)

            # Save markdown report to separate file
            markdown_content = step6_dict.get('markdown_content', '')
            if markdown_content:
                report_file = job_output_dir / 'final_report.md'
                with open(report_file, 'w') as f:
                    f.write(markdown_content)
                logger.info(f"Saved markdown report to: {report_file}")
                results['report_path'] = str(report_file)

            # Calculate total execution time
            total_time = time.time() - start_time
            results['total_execution_time'] = total_time
            results['status'] = 'completed'
            results['output_path'] = str(job_output_dir)

            # Add validation summary
            if self.enable_validation:
                # Convert validation dicts back to ValidationResult objects for summary
                from .services.step_validators import ValidationResult
                validation_objects = {}
                for step, val_dict in results['validation'].items():
                    if val_dict:
                        validation_objects[step] = ValidationResult(
                            passed=val_dict['passed'],
                            errors=val_dict['errors'],
                            warnings=val_dict['warnings'],
                            stats=val_dict['stats']
                        )
                validation_summary = StepValidators.create_validation_summary(validation_objects)
                results['validation_summary'] = validation_summary
                logger.info("\n" + "=" * 80)
                logger.info("VALIDATION SUMMARY")
                logger.info("=" * 80)
                logger.info(f"All Passed: {validation_summary['all_passed']}")
                logger.info(f"Total Errors: {validation_summary['total_errors']}")
                logger.info(f"Total Warnings: {validation_summary['total_warnings']}")
                if validation_summary['failed_steps']:
                    logger.warning(f"Failed Steps: {validation_summary['failed_steps']}")

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Total execution time: {total_time:.2f}s")
            logger.info(f"Results saved to: {job_output_dir}")

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results['status'] = 'failed'
            results['error'] = str(e)
            results['total_execution_time'] = time.time() - start_time
            raise

    def run_single_step(
        self,
        step_number: int,
        input_data: Dict[str, Any],
        previous_results: Optional[Dict[str, Any]] = None,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a single pipeline step.

        Args:
            step_number: Step number to run (1-5)
            input_data: Pipeline input data
            previous_results: Results from previous steps (required for steps 2-5)
            job_id: Optional job ID for output organization

        Returns:
            Dictionary containing step results and execution metadata
        """
        if step_number < 1 or step_number > 5:
            raise ValueError(f"Invalid step number: {step_number}. Must be 1-5.")

        if step_number > 1 and not previous_results:
            raise ValueError(f"Step {step_number} requires previous_results from prior steps")

        start_time = time.time()

        # Extract input components
        genes = input_data['genes']
        pathways = input_data['pathways']

        # Allow-lists to strip hallucinated entities from prior-step inputs before this step
        # uses them — mirrors run_full_pipeline so step-by-step runs are protected too.
        allow_pathways = {p.get('name', '') for p in pathways}
        allow_genes = {g.get('geneSymbol', g.get('gene', '')) for g in genes}

        def _clean_prior(step_no, prev_key, extra_genes=None):
            """Return a cleaned copy of a prior step's output (does not mutate previous_results)."""
            ag = (allow_genes | set(extra_genes)) if extra_genes else allow_genes
            cleaned, _ = StepValidators.strip_hallucinated_entities(
                previous_results['steps'][prev_key], step_no, allow_pathways, ag)
            return cleaned

        def _pathway_genes_from_step3():
            pg = set()
            for mech in previous_results['steps'].get('step3', {}).get('pathway_mechanisms', []):
                for rel in mech.get('curatedRelations', []):
                    pg.add(rel.get('source', ''))
                    pg.add(rel.get('target', ''))
            return pg

        # Extract metadata - include ALL metadata fields for flexible analysis
        metadata = input_data.get('metadata', {})
        organism = metadata.get('organism', 'Homo sapiens')

        # Build context from ALL metadata fields
        context = {}
        for key, value in metadata.items():
            if key != 'organism' and value:
                context[key] = value

        logger.info(f"Running Step {step_number}")

        try:
            if step_number == 1:
                output = self.step1.execute(
                    pathways=pathways,
                    genes=genes,
                    organism=organism,
                    context=context
                )
            elif step_number == 2:
                themes = _clean_prior(1, 'step1').get('themes', [])
                output = self.step2.execute(
                    genes=genes[:100],
                    pathway_themes=themes,
                    organism=organism,
                    context=context
                )
            elif step_number == 3:
                themes = _clean_prior(1, 'step1').get('themes', [])
                hub_genes = _clean_prior(2, 'step2').get('hub_genes', [])

                # Get filtered pathways from Step 1 (themes + ungrouped)
                step1_dict = previous_results['steps']['step1']
                filtered_pathways = self._get_filtered_pathways_from_step1(step1_dict, pathways)
                logger.info(f"  Using {len(filtered_pathways)} filtered pathways from Step 1")

                output = self.step3.execute(
                    pathways=filtered_pathways,
                    genes=genes,
                    analyses=[{'organismId': organism}],
                    themes=themes,
                    hub_genes=hub_genes
                )
            elif step_number == 4:
                themes = _clean_prior(1, 'step1').get('themes', [])
                step2_dict = _clean_prior(2, 'step2')
                step3_dict = _clean_prior(3, 'step3', extra_genes=_pathway_genes_from_step3())
                output = self.step4.execute(
                    genes=genes,
                    pathways=pathways,
                    themes=themes,
                    hub_genes_result=step2_dict,
                    mechanisms_result=step3_dict
                )
            else:  # step_number == 5 (Final Report Generation; previously step 6)
                step1_dict = previous_results['steps']['step1']
                step2_dict = previous_results['steps']['step2']
                step3_dict = previous_results['steps']['step3']
                step4_dict = previous_results['steps']['step4']
                output = self.step6.execute(
                    input_data=input_data,
                    step1_output=step1_dict,
                    step2_output=step2_dict,
                    step3_output=step3_dict,
                    step4_output=step4_dict
                )

            output_dict = self._to_dict(output)
            execution_time = time.time() - start_time

            # Save output if job_id provided
            if job_id:
                job_output_dir = self.output_dir / job_id
                job_output_dir.mkdir(parents=True, exist_ok=True)
                self._save_step_output(job_output_dir, step_number, output_dict)

            return {
                'step_number': step_number,
                'results': output_dict,
                'execution_time': execution_time,
                'status': 'completed'
            }

        except Exception as e:
            logger.error(f"Step {step_number} failed: {str(e)}", exc_info=True)
            raise

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass or dict to dict."""
        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        return obj

    def _save_step_output(self, output_dir: Path, step_number: int, output: Dict[str, Any]) -> None:
        """Save step output to JSON file."""
        output_file = output_dir / f"step{step_number}_output.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved Step {step_number} output to: {output_file}")
