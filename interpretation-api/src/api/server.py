from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import uvicorn
import time
import traceback
from ..config import settings

# Pipeline imports
from ..pipeline.orchestrator import PipelineOrchestrator
from .job_manager import job_manager, JobStatus
from .models.pipeline_models import (
    PipelineInput,
    DatasetInput,
    MetaPipelineInput,
    StepRequest,
    JobResponse,
    JobStatusResponse,
    JobListResponse,
    StepResponse,
    PipelineResponse,
    MetaPipelineResponse,
    ErrorDetail,
    ErrorResponse
)

app = FastAPI(
    title="IPSA Interpretation Pipeline API",
    description="Multi-step LLM interpretation of pathway-enrichment and differential-expression results",
    version="1.0.0"
)

# CORS. The default is permissive so the service works out of the box behind any
# front-end; restrict allow_origins to your own origins before exposing it publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
pipeline_orchestrator = None


# Request/Response models
class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrators on startup"""
    global pipeline_orchestrator
    pipeline_orchestrator = PipelineOrchestrator(
        output_dir=settings.PIPELINE_OUTPUT_DIR,
        enable_validation=settings.PIPELINE_ENABLE_VALIDATION,
        strict_mode=settings.PIPELINE_STRICT_MODE
    )
    print("Pipeline Service initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Pipeline Service shutdown")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return {
        "status": "running",
        "service": "IPSA Interpretation Pipeline API",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "IPSA Interpretation Pipeline API",
        "version": "1.0.0"
    }


# ============================================================================
# PIPELINE ENDPOINTS
# ============================================================================

def run_pipeline_job(job_id: str, input_data: Dict[str, Any], enable_validation: Optional[bool] = None, strict_mode: Optional[bool] = None):
    """
    Background task to run pipeline job.

    Args:
        job_id: Job identifier
        input_data: Pipeline input data
        enable_validation: Optional override for validation setting
        strict_mode: Optional override for strict mode setting
    """
    try:
        job_manager.update_status(job_id, JobStatus.RUNNING)

        # Create orchestrator with optional overrides
        orchestrator = PipelineOrchestrator(
            output_dir=settings.PIPELINE_OUTPUT_DIR,
            enable_validation=enable_validation if enable_validation is not None else settings.PIPELINE_ENABLE_VALIDATION,
            strict_mode=strict_mode if strict_mode is not None else settings.PIPELINE_STRICT_MODE
        )

        # Run pipeline
        results = orchestrator.run_full_pipeline(input_data, job_id)

        # Update job with results
        job_manager.set_results(job_id, results)
        job_manager.update_status(job_id, JobStatus.COMPLETED)

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()[:500]}"
        job_manager.set_error(job_id, error_msg)
        job_manager.update_status(job_id, JobStatus.FAILED)
        print(f"Pipeline job {job_id} failed: {error_msg}")


def run_meta_pipeline_job(job_id: str, meta_input: Dict[str, Any], dataset_inputs: List[Dict[str, Any]], max_workers: Optional[int] = None):
    """
    Background task to run meta-analysis pipeline job.

    Args:
        job_id: Job identifier
        meta_input: Meta-analysis input data
        dataset_inputs: List of individual dataset inputs
        max_workers: Max parallel workers
    """
    try:
        from src.pipeline.meta_orchestrator import MetaPipelineOrchestrator
        import json

        job_manager.update_status(job_id, JobStatus.RUNNING)

        # Create output directory with job_id
        output_dir = Path(settings.PIPELINE_OUTPUT_DIR) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create meta-orchestrator
        meta_orchestrator = MetaPipelineOrchestrator()

        # Run meta-pipeline with organized output directory
        result = meta_orchestrator.execute(
            meta_input=meta_input,
            dataset_inputs=dataset_inputs,
            max_workers=max_workers,
            job_id=job_id,
            output_dir=str(output_dir)
        )

        if result.success:
            # Save comparative report to root of job folder
            report_path = output_dir / 'comparative_report.md'
            report_path.write_text(result.comparative_report)

            # Save reproducibility results
            repro_path = output_dir / 'reproducibility_results.json'
            repro_path.write_text(json.dumps(result.reproducibility_results, indent=2))

            # Save discovery results
            discovery_path = output_dir / 'discovery_results.json'
            discovery_path.write_text(json.dumps(result.discovery_results, indent=2, default=str))

            # Save summary JSON with execution info
            summary_path = output_dir / 'meta_pipeline_summary.json'
            summary_data = {
                'job_id': job_id,
                'success': True,
                'execution_time': result.execution_time,
                'meta_analysis_dir': 'meta_analysis',
                'datasets_dir': 'datasets',
                'dataset_names': list(result.dataset_results.keys()),
                'files': {
                    'comparative_report': 'comparative_report.md',
                    'reproducibility_results': 'reproducibility_results.json',
                    'discovery_results': 'discovery_results.json'
                }
            }
            summary_path.write_text(json.dumps(summary_data, indent=2))

            # Update job with results
            job_manager.set_results(job_id, {
                'meta_results': result.meta_results,
                'dataset_results': result.dataset_results,
                'reproducibility_results': result.reproducibility_results,
                'discovery_results': result.discovery_results,
                'comparative_report': result.comparative_report,
                'execution_time': result.execution_time,
                'output_path': str(output_dir)
            })
            job_manager.update_status(job_id, JobStatus.COMPLETED)
        else:
            raise Exception(result.error)

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()[:500]}"
        job_manager.set_error(job_id, error_msg)
        job_manager.update_status(job_id, JobStatus.FAILED)
        print(f"Meta-pipeline job {job_id} failed: {error_msg}")


@app.post("/api/pipeline", response_model=JobResponse)
async def submit_pipeline_job(
    request: PipelineInput,
    background_tasks: BackgroundTasks
):
    """
    Submit a full pipeline job for async processing.

    Runs the pipeline steps:
    1. Pathway Theme Identification
    2. Hub Gene Detection
    3. Pathway Mechanisms
    4. Hypothesis Generation
    5. Final Report Generation (Markdown)

    Steps 2 and 3 are skipped when the input carries no differentially expressed genes.

    Returns a job ID that can be used to check status and retrieve results.
    """
    try:
        # Convert Pydantic model to dict
        input_data = request.model_dump()

        # Extract validation options (they shouldn't be part of input_data)
        enable_validation = input_data.pop('enable_validation', None)
        strict_mode = input_data.pop('strict_mode', None)

        # Check if input has DE genes to determine which steps to include
        genes = input_data.get('genes', [])
        has_genes = bool(genes and len(genes) > 0)

        # Build steps_info conditionally — skip gene-dependent steps when no genes
        steps_info = [
            {"step": 1, "name": "Pathway Theme Identification", "message": "Clustering pathways by shared biological functions and gene overlap..."},
        ]
        if has_genes:
            steps_info.append({"step": 2, "name": "Hub Gene Detection", "message": "Analyzing protein-protein interaction network via STRING database..."})
            steps_info.append({"step": 3, "name": "Pathway Mechanisms", "message": "Fetching molecular interactions from pathway databases..."})
        steps_info.append({"step": 4, "name": "Hypothesis Generation", "message": "Generating testable research hypotheses from pathway analysis..."})
        steps_info.append({"step": 5, "name": "Report Generation", "message": "Compiling analysis results into final report..."})

        # Create job
        job_id = job_manager.create_job(input_data, total_steps=len(steps_info), steps_info=steps_info)

        # Submit background task with validation options
        background_tasks.add_task(run_pipeline_job, job_id, input_data, enable_validation, strict_mode)

        return JobResponse(
            success=True,
            job_id=job_id,
            status="pending",
            message=f"Pipeline job submitted successfully. Use GET /api/pipeline/job/{job_id} to check status."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="PIPELINE_SUBMIT_ERROR",
                    message=str(e),
                    details=traceback.format_exc()[:500]
                )
            ).model_dump()
        )


@app.post("/api/meta-pipeline", response_model=JobResponse)
async def submit_meta_pipeline_job(
    request: MetaPipelineInput,
    background_tasks: BackgroundTasks
):
    """
    Submit a meta-analysis pipeline job for async processing.

    Runs the standard 5-step pipeline in parallel on:
    - Meta-analysis data (combined from all studies)
    - Individual dataset data (each study separately)

    Then performs:
    - Meta-Step 7: Reproducibility Analysis
    - Meta-Step 9: Discovery Analysis
    - Meta-Step 8: Comparative Report Generation

    Returns a job ID that can be used to check status and retrieve results.
    """
    try:
        # Convert Pydantic model to dict
        request_data = request.model_dump()

        # Extract components
        meta_input = request_data['meta']
        dataset_inputs = request_data['datasets']
        max_workers = request_data.get('max_workers')

        # Extract validation options from meta input
        enable_validation = request_data.get('enable_validation')
        strict_mode = request_data.get('strict_mode')

        # Apply validation settings to meta and all dataset inputs
        if enable_validation is not None:
            meta_input['enable_validation'] = enable_validation
            for ds in dataset_inputs:
                ds['enable_validation'] = enable_validation

        if strict_mode is not None:
            meta_input['strict_mode'] = strict_mode
            for ds in dataset_inputs:
                ds['strict_mode'] = strict_mode

        # Check if meta input has DE genes to determine which steps to include
        meta_genes = meta_input.get('genes', [])
        has_genes = bool(meta_genes and len(meta_genes) > 0)

        # Build steps_info conditionally — skip gene-dependent steps when no genes
        meta_steps_info = [
            {"step": 1, "name": "Pathway Themes", "message": "Identifying pathway themes from combined data"},
        ]
        if has_genes:
            meta_steps_info.append({"step": 2, "name": "Hub Genes", "message": "Detecting hub genes from combined data"})
            meta_steps_info.append({"step": 3, "name": "Pathway Mechanisms", "message": "Analyzing pathway mechanisms from combined data"})
        meta_steps_info.append({"step": 4, "name": "Hypothesis Generation", "message": "Generating mechanistic hypotheses from combined data"})
        # Step 5 = the per-analysis report; the meta-only steps follow it.
        meta_steps_info.extend([
            {"step": 5, "name": "Analysis Report", "message": "Generating analysis report from combined data"},
            {"step": 6, "name": "Reproducibility Analysis", "message": "Comparing results across datasets"},
            {"step": 8, "name": "Discovery Analysis", "message": "Identifying meta-unique findings"},
            {"step": 7, "name": "Comparative Report", "message": "Generating final comparative report"},
        ])

        # Create job with meta input as reference
        job_id = job_manager.create_job(
            input_data={
                'type': 'meta_pipeline',
                'meta': meta_input,
                'n_datasets': len(dataset_inputs),
                'dataset_names': [ds.get('dataset_name', f'Dataset {i}') for i, ds in enumerate(dataset_inputs)]
            },
            total_steps=len(meta_steps_info),
            steps_info=meta_steps_info
        )

        # Submit background task
        background_tasks.add_task(
            run_meta_pipeline_job,
            job_id,
            meta_input,
            dataset_inputs,
            max_workers
        )

        return JobResponse(
            success=True,
            job_id=job_id,
            status="pending",
            message=f"Meta-pipeline job submitted successfully. Use GET /api/pipeline/job/{job_id} to check status."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="META_PIPELINE_SUBMIT_ERROR",
                    message=str(e),
                    details=traceback.format_exc()[:500]
                )
            ).model_dump()
        )


@app.get("/api/pipeline/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get status and results of a pipeline job.

    Returns:
    - Job status (pending/running/completed/failed)
    - Results (if completed)
    - Error message (if failed)
    - Timestamps
    """
    try:
        job = job_manager.get_job(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )

        return JobStatusResponse(
            success=True,
            job_id=job.job_id,
            status=job.status.value,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            current_step=job.current_step,
            current_step_message=job.current_step_message,
            total_steps=job.total_steps,
            steps_info=job.steps_info,
            step_results=job.step_results,
            step_execution_times=job.step_execution_times,
            results=job.results,
            error=job.error
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="JOB_STATUS_ERROR",
                    message=str(e),
                    details=traceback.format_exc()[:500]
                )
            ).model_dump()
        )


@app.get("/api/pipeline/jobs", response_model=JobListResponse)
async def list_jobs(limit: int = 100):
    """
    List all pipeline jobs (most recent first).

    Args:
        limit: Maximum number of jobs to return (default: 100)
    """
    try:
        jobs = job_manager.list_jobs(limit=limit)

        return JobListResponse(
            success=True,
            jobs=jobs,
            count=len(jobs)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="JOB_LIST_ERROR",
                    message=str(e),
                    details=traceback.format_exc()[:500]
                )
            ).model_dump()
        )


@app.post("/api/pipeline/step/{step_number}", response_model=StepResponse)
async def run_single_step(
    step_number: int,
    request: StepRequest
):
    """
    Run a single pipeline step.

    Args:
        step_number: Step number to run (1-5)
        request: Step request with input data and previous results

    Steps:
    1. Pathway Theme Identification
    2. Hub Gene Detection (requires step 1 results)
    3. Pathway Mechanisms (requires steps 1-2 results)
    4. Hypothesis Generation (requires steps 1-3 results)
    5. Final Report Generation (requires steps 1-4 results)
    """
    try:
        if step_number < 1 or step_number > 5:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid step number: {step_number}. Must be 1-5."
            )

        # Convert Pydantic model to dict
        input_data = request.input_data.model_dump()
        previous_results = request.previous_results

        # Run step
        start_time = time.time()
        result = pipeline_orchestrator.run_single_step(
            step_number=step_number,
            input_data=input_data,
            previous_results=previous_results
        )

        execution_time = time.time() - start_time

        return StepResponse(
            success=True,
            step_number=step_number,
            results=result['results'],
            execution_time=execution_time
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="INVALID_STEP_REQUEST",
                    message=str(e),
                    details=None
                )
            ).model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="STEP_EXECUTION_ERROR",
                    message=str(e),
                    details=traceback.format_exc()[:500]
                )
            ).model_dump()
        )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
