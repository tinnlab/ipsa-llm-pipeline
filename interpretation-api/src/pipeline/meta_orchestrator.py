"""
Meta-Pipeline Orchestrator

Runs the standard 5-step pipeline on the combined/meta-analysis data,
then uses raw input from individual datasets (without re-running the pipeline)
to perform reproducibility analysis and comparative reporting.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.orchestrator import PipelineOrchestrator
from src.pipeline.steps_meta.meta_step07_reproducibility import MetaStep07Reproducibility
from src.pipeline.steps_meta.meta_step08_comparative_report import MetaStep08ComparativeReport
from src.pipeline.steps_meta.meta_step09_discovery_analysis import MetaStep09DiscoveryAnalysis
from src.api.job_manager import job_manager

logger = logging.getLogger(__name__)


@dataclass
class MetaPipelineResult:
    """Result from meta-pipeline execution"""
    meta_results: Dict[str, Any]
    dataset_results: Dict[str, Dict[str, Any]]
    reproducibility_results: Dict[str, Any]
    discovery_results: Dict[str, Any]  # New: meta-unique discoveries
    comparative_report: str
    execution_time: float
    success: bool
    error: Optional[str] = None


class MetaPipelineOrchestrator:
    """
    Meta-Pipeline Orchestrator

    Runs the standard 5-step pipeline on the combined/meta data only,
    builds synthetic results from raw individual dataset inputs,
    then performs reproducibility analysis and generates comparative report.
    """

    def __init__(self):
        self.step7 = MetaStep07Reproducibility()
        self.step8 = MetaStep08ComparativeReport()
        self.step9 = MetaStep09DiscoveryAnalysis()

    def execute(
        self,
        meta_input: Dict[str, Any],
        dataset_inputs: List[Dict[str, Any]],
        max_workers: Optional[int] = None,
        job_id: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> MetaPipelineResult:
        """
        Execute meta-pipeline

        Args:
            meta_input: Input data for meta-analysis (combined data)
                {
                    "disease": str,
                    "tissue": str,
                    "description": str,
                    "genes": [...],
                    "pathways": [...]
                }
            dataset_inputs: List of input data for individual datasets
                [
                    {
                        "dataset_name": str,
                        "disease": str,
                        "tissue": str,
                        "description": str,
                        "genes": [...],
                        "pathways": [...]
                    },
                    ...
                ]
            max_workers: Max parallel workers (default: min(len(datasets)+1, 4))
            job_id: Optional job ID for status tracking
            output_dir: Optional directory to save all outputs in organized structure

        Returns:
            MetaPipelineResult with all results
        """
        start_time = time.time()

        try:
            # Validate inputs
            if not dataset_inputs or len(dataset_inputs) == 0:
                raise ValueError(
                    'No individual datasets provided in request. '
                    'Meta-analysis pipeline requires at least one individual dataset for reproducibility analysis. '
                    'Please include a "datasets" array in your request with at least one dataset.'
                )

            print('\n' + '=' * 80)
            print('META-ANALYSIS PIPELINE')
            print('=' * 80)
            print(f'\nAnalyzing:')
            # Extract metadata - use flexible field names
            metadata = meta_input.get('metadata', {})

            # Try to find a descriptive label from various possible field names
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

            # For display
            if tissue:
                print(f'  - Meta-analysis: {study_type} ({tissue})')
            else:
                print(f'  - Meta-analysis: {study_type}')
            print(f'  - Individual datasets: {len(dataset_inputs)}')

            # Create output directory if specified
            output_base_dir = None
            if output_dir:
                output_base_dir = Path(output_dir)
                output_base_dir.mkdir(parents=True, exist_ok=True)
                print(f'\nOutput directory: {output_base_dir}')

            print('\n' + '-' * 80)
            print('PHASE 1: Run pipeline on combined data; use raw input for individual datasets')
            print('-' * 80)

            # Build synthetic results from raw individual data (no pipeline runs needed)
            dataset_results = {}
            dataset_names = []
            for ds_input in dataset_inputs:
                ds_name = ds_input.get('dataset_name', f'Dataset_{len(dataset_names)}')
                dataset_results[ds_name] = self._build_synthetic_results(ds_input)
                dataset_names.append(ds_name)

            print(f'  Loaded {len(dataset_results)} individual datasets (from raw input)')

            # Run pipeline ONLY on the combined/meta data
            if job_id:
                job_manager.set_current_step(
                    job_id, 1,
                    "Running pipeline on combined/meta data"
                )

            print(f'  Running full pipeline on meta/combined data...')
            meta_output_dir = output_base_dir / 'meta_analysis' if output_base_dir else None
            meta_results = self._run_single_pipeline('meta', meta_input, meta_output_dir, tracking_job_id=job_id)

            if not meta_results.get('success', True):
                raise Exception('Meta-analysis pipeline failed')

            # Meta Step 6: Reproducibility Analysis
            if job_id:
                job_manager.set_current_step(job_id, 6, "Analyzing reproducibility across datasets")
            print('\n' + '-' * 80)
            print('PHASE 2: Reproducibility Analysis (Step 6)')
            print('-' * 80)

            step7_start = time.time()
            reproducibility_results = self.step7.execute(
                meta_results=meta_results,
                dataset_results=dataset_results,
                dataset_names=dataset_names
            )
            self._validate_step7_output(asdict(reproducibility_results))
            if job_id:
                step7_time = time.time() - step7_start
                job_manager.set_step_result(job_id, 6, asdict(reproducibility_results), step7_time)

            # Meta Step 8: Discovery Analysis — the main value of running a meta-analysis
            if job_id:
                job_manager.set_current_step(job_id, 8, "Identifying meta-unique discoveries")
            print('\n' + '-' * 80)
            print('PHASE 3: Meta-Analysis Discovery Analysis (Step 8)')
            print('-' * 80)

            step9_start = time.time()
            discovery_results = self.step9.execute(
                meta_results=meta_results,
                dataset_results=dataset_results,
                dataset_names=dataset_names,
                metadata=metadata
            )
            self._validate_step9_output(asdict(discovery_results))
            if job_id:
                step9_time = time.time() - step9_start
                job_manager.set_step_result(job_id, 8, asdict(discovery_results), step9_time)

            # Meta Step 7: Comparative Report
            if job_id:
                job_manager.set_current_step(job_id, 7, "Generating comparative report")
            print('\n' + '-' * 80)
            print('PHASE 4: Comparative Report Generation (Step 7)')
            print('-' * 80)

            step8_start = time.time()
            comparative_report = self.step8.execute(
                reproducibility_results=asdict(reproducibility_results),
                discovery_results=asdict(discovery_results),
                meta_results=meta_results,
                dataset_results=dataset_results,
                dataset_names=dataset_names,
                metadata=metadata  # Pass full metadata for flexible display
            )
            self._validate_step8_output(comparative_report)
            if job_id:
                step8_time = time.time() - step8_start
                job_manager.set_step_result(job_id, 7, {'report_length': len(comparative_report)}, step8_time)

            execution_time = time.time() - start_time

            print('\n' + '=' * 80)
            print(f'META-ANALYSIS PIPELINE COMPLETED ({execution_time:.1f}s)')
            print('=' * 80)

            return MetaPipelineResult(
                meta_results=meta_results,
                dataset_results=dataset_results,
                reproducibility_results=asdict(reproducibility_results),
                discovery_results=asdict(discovery_results),
                comparative_report=comparative_report,
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            print(f'\n❌ Meta-pipeline failed: {e}')

            return MetaPipelineResult(
                meta_results={},
                dataset_results={},
                reproducibility_results={},
                discovery_results={},
                comparative_report='',
                execution_time=execution_time,
                success=False,
                error=str(e)
            )

    def _build_synthetic_results(self, dataset_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert raw dataset input into a synthetic pipeline results dict.

        This allows meta steps 7/8/9 to extract pathway and gene data from
        individual datasets WITHOUT running the full 5-step pipeline on them.

        The extractors in meta steps read from:
          - step1.themes[].pathways -> names, pValues, ES
          - step1.ungrouped_pathways -> same
          - input.genes -> geneSymbol, foldChange, pValue
        """
        raw_pathways = dataset_input.get('pathways', [])
        raw_genes = dataset_input.get('genes', [])

        return {
            'steps': {
                'step1': {
                    'themes': [],  # No AI-generated themes for raw data
                    'ungrouped_pathways': raw_pathways  # All pathways available for matching
                },
                'step2': {
                    'network_hubs': []
                },
                'step3': {'pathway_mechanisms': []},
                'step4': {'hypotheses': []},
                # Step 5 is report generation.
                'step5': {
                    'metadata': {'skipped': True, 'reason': 'Synthetic results from raw input'}
                }
            },
            'input': {
                'genes': raw_genes,
                'pathways': raw_pathways,
                'metadata': dataset_input.get('metadata', {})
            },
            'success': True,
            'synthetic': True  # Flag indicating these are not from a full pipeline run
        }

    def _run_single_pipeline(
        self,
        name: str,
        input_data: Dict[str, Any],
        output_dir: Optional[Path] = None,
        tracking_job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run standard 5-step pipeline on a single dataset

        Args:
            name: Dataset name (or 'meta')
            input_data: Pipeline input data
            output_dir: Optional output directory for this pipeline's results
            tracking_job_id: Optional job ID for progress tracking (so nested
                pipeline updates the parent meta-pipeline job)

        Returns:
            Pipeline results dict
        """
        print(f'\n[{name}] Starting pipeline...')

        try:
            # Prepare input (remove dataset_name field if present)
            pipeline_input = {k: v for k, v in input_data.items() if k != 'dataset_name'}

            # Determine output directory and job_id
            if output_dir:
                # Use parent of output_dir as base, and folder name as job_id
                # This way files go directly into the output_dir
                base_dir = output_dir.parent
                job_id = output_dir.name
                orchestrator = PipelineOrchestrator(output_dir=str(base_dir))
            else:
                job_id = f"{name}_sub"
                orchestrator = PipelineOrchestrator()

            # Run pipeline
            result = orchestrator.run_full_pipeline(pipeline_input, job_id=job_id, tracking_job_id=tracking_job_id)

            # Add success flag and input for reference
            result['success'] = True
            result['input'] = pipeline_input
            result['output_dir'] = str(output_dir) if output_dir else None

            return result

        except Exception as e:
            print(f'\n[{name}] Pipeline error: {e}')
            return {
                'success': False,
                'error': str(e),
                'input': input_data
            }

    def _validate_step7_output(self, output: Dict[str, Any]) -> None:
        """Validate Step 7 (Reproducibility) output. Logs warnings, non-blocking."""
        required_fields = [
            'pathway_reproducibility', 'gene_reproducibility',
            'summary_stats', 'meta_unique_pathways', 'meta_unique_genes'
        ]
        for field in required_fields:
            if field not in output:
                logger.warning(f"Step 7 output missing field: {field}")

        stats = output.get('summary_stats', {})
        for count_field in ['total_pathways_analyzed', 'total_genes_analyzed']:
            val = stats.get(count_field)
            if val is not None and val < 0:
                logger.warning(f"Step 7 summary_stats.{count_field} is negative: {val}")

    def _validate_step9_output(self, output: Dict[str, Any]) -> None:
        """Validate Step 9 (Discovery) output. Logs warnings, non-blocking."""
        required_fields = [
            'individual_findings', 'meta_unique_themes',
            'enhanced_significance_pathways', 'meta_unique_hypotheses',
            'cross_study_mechanisms', 'meta_unique_targets',
            'summary', 'biological_interpretation', 'key_discoveries',
            'emergent_theme_combinations', 'cross_dataset_convergence',
            'weak_signal_amplification', 'novel_cooccurrence_pairs'
        ]
        for field in required_fields:
            if field not in output:
                logger.warning(f"Step 9 output missing field: {field}")

        interpretation = output.get('biological_interpretation', '')
        if not interpretation or not interpretation.strip():
            logger.warning("Step 9 biological_interpretation is empty")

        warnings = output.get('warnings', [])
        if warnings:
            logger.warning(f"Step 9 reported {len(warnings)} warning(s): {warnings}")

    def _validate_step8_output(self, report: str) -> None:
        """Validate Step 8 (Comparative Report) output. Logs warnings, non-blocking."""
        if not report or not report.strip():
            logger.warning("Step 8 comparative report is empty")
        elif len(report) < 200:
            logger.warning(
                f"Step 8 comparative report is suspiciously short ({len(report)} chars)"
            )


def main():
    """Example usage"""
    # Example meta-analysis input
    meta_input = {
        'metadata': {
            'disease': 'Lung Adenocarcinoma',
            'tissue': 'Lung',
            'description': 'Meta-analysis of 3 LUAD studies',
        },
        'genes': [
            {'geneSymbol': 'TP53', 'foldChange': -2.5, 'pValue': 1e-10, 'pValueFDR': 1e-8},
            {'geneSymbol': 'KRAS', 'foldChange': 1.8, 'pValue': 1e-8, 'pValueFDR': 1e-6},
        ],
        'pathways': [
            {
                'name': 'p53 signaling pathway',
                'source': 'KEGG',
                'pathwayId': 'hsa04115',
                'pValue': 1e-6,
                'pValueFDR': 1e-4,
                'ES': -0.6,
                'genes': ['TP53', 'MDM2', 'CDKN1A']
            }
        ]
    }

    # Example dataset inputs
    dataset_inputs = [
        {
            'dataset_name': 'GSE12345',
            'metadata': {
                'disease': 'Lung Adenocarcinoma',
                'tissue': 'Lung',
                'description': 'Study 1',
            },
            'genes': [
                {'geneSymbol': 'TP53', 'foldChange': -2.1, 'pValue': 1e-5, 'pValueFDR': 1e-3}
            ],
            'pathways': [
                {
                    'name': 'p53 signaling pathway',
                    'source': 'KEGG',
                    'pathwayId': 'hsa04115',
                    'pValue': 1e-3,
                    'pValueFDR': 0.01,
                    'ES': -0.5,
                    'genes': ['TP53', 'MDM2']
                }
            ]
        },
        {
            'dataset_name': 'GSE67890',
            'metadata': {
                'disease': 'Lung Adenocarcinoma',
                'tissue': 'Lung',
                'description': 'Study 2',
            },
            'genes': [
                {'geneSymbol': 'KRAS', 'foldChange': 1.9, 'pValue': 1e-6, 'pValueFDR': 1e-4}
            ],
            'pathways': []
        }
    ]

    # Run meta-pipeline
    orchestrator = MetaPipelineOrchestrator()
    result = orchestrator.execute(meta_input, dataset_inputs)

    if result.success:
        print('\n' + '=' * 80)
        print('RESULTS SUMMARY')
        print('=' * 80)
        print(f'\nMeta-unique pathways: {len(result.reproducibility_results["meta_unique_pathways"])}')
        print(f'Meta-unique genes: {len(result.reproducibility_results["meta_unique_genes"])}')
        print(f'\nComparative report length: {len(result.comparative_report)} characters')
    else:
        print(f'\nMeta-pipeline failed: {result.error}')


if __name__ == '__main__':
    main()
