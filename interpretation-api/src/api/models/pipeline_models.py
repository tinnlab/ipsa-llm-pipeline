"""
Pydantic models for pipeline API.

Defines request/response schemas matching the experimental pipeline structure.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Input models
class GeneInput(BaseModel):
    """Gene expression data."""
    geneSymbol: str = Field(..., description="Gene symbol")
    foldChange: float = Field(..., description="Fold change in expression")
    pValue: float = Field(..., description="Statistical p-value")
    pValueFDR: float = Field(..., description="FDR-adjusted p-value")


class PathwayInput(BaseModel):
    """Pathway enrichment data."""
    name: str = Field(..., description="Pathway name")
    source: str = Field(..., description="Pathway source (e.g., KEGG, Reactome, GO, MSigDB, WikiPathways, etc.)")
    pathwayId: str = Field(..., description="Pathway identifier")
    pValue: float = Field(..., description="Enrichment p-value")
    pValueFDR: float = Field(..., description="FDR-adjusted p-value")
    ES: Optional[float] = Field(None, description="fgsea enrichment score (positive=upregulated, negative=downregulated). Note: some producers forward the NES value here; a |value|>1 is treated as NES. Prefer sending NES explicitly.")
    NES: Optional[float] = Field(None, description="Normalized enrichment score from fgsea (unbounded, typically +/-1 to +/-4). Preferred over ES for magnitude/labelling.")
    genes: List[str] = Field(..., description="All genes in pathway")


class PipelineInput(BaseModel):
    """
    Input for the biological analysis pipeline.

    Matches the structure of luad_lung_adenocarcinoma_input.json.
    """
    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description=(
            "Optional flexible dataset metadata. Recommended fields: "
            "'disease', 'tissue', 'description', 'organism'. "
            "You can include any additional custom fields as needed. "
            "If not provided, defaults to empty dict."
        )
    )
    genes: List[GeneInput] = Field(default=[], description="Differentially expressed genes (optional - if not provided, gene-centric steps will be skipped)")
    pathways: List[PathwayInput] = Field(..., description="Enriched pathways")
    enable_validation: Optional[bool] = Field(None, description="Enable step output validation (overrides config default)")
    strict_mode: Optional[bool] = Field(None, description="Fail pipeline on validation errors (overrides config default)")


class DatasetInput(BaseModel):
    """
    Input for a single dataset in meta-analysis.

    Same structure as PipelineInput but with an additional dataset_name field.
    """
    dataset_name: str = Field(..., description="Unique identifier for this dataset (e.g., GSE12345)")
    metadata: Optional[Dict[str, Any]] = Field(
        default={},
        description=(
            "Optional flexible dataset metadata. Recommended fields: "
            "'disease', 'tissue', 'description', 'organism'. "
            "You can include any additional custom fields as needed. "
            "If not provided, defaults to empty dict."
        )
    )
    genes: List[GeneInput] = Field(default=[], description="Differentially expressed genes (optional - if not provided, gene-centric steps will be skipped)")
    pathways: List[PathwayInput] = Field(..., description="Enriched pathways")


class MetaPipelineInput(BaseModel):
    """
    Input for meta-analysis pipeline.

    Includes meta-analysis data (combined across all studies) plus individual dataset data.
    """
    meta: PipelineInput = Field(..., description="Meta-analysis data (combined from all datasets)")
    datasets: List[DatasetInput] = Field(
        ...,
        min_length=1,
        description="Individual dataset data (at least 1 required for reproducibility analysis)"
    )
    max_workers: Optional[int] = Field(None, description="Max parallel workers (default: min(n_datasets+1, 4))")
    enable_validation: Optional[bool] = Field(None, description="Enable step output validation (overrides config default)")
    strict_mode: Optional[bool] = Field(None, description="Fail pipeline on validation errors (overrides config default)")


class StepRequest(BaseModel):
    """Request to run an individual pipeline step."""
    step_number: int = Field(..., ge=1, le=5, description="Step number (1-5)")
    input_data: PipelineInput = Field(..., description="Pipeline input data")
    previous_results: Optional[Dict[str, Any]] = Field(
        None,
        description="Results from previous steps (required for steps 2-5)"
    )


# Response models
class JobResponse(BaseModel):
    """Response when creating a new job."""
    success: bool = Field(..., description="Whether job was created successfully")
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Initial job status")
    message: str = Field(..., description="Human-readable message")


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    success: bool = Field(..., description="Whether query was successful")
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    created_at: str = Field(..., description="Job creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    started_at: Optional[str] = Field(None, description="Job start timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    # Step progress tracking
    current_step: Optional[int] = Field(None, description="Currently executing step number (1-based)")
    current_step_message: Optional[str] = Field(None, description="Current step progress message (dynamic based on context)")
    total_steps: int = Field(default=5, description="Total number of steps in the pipeline")
    steps_info: List[Dict[str, Any]] = Field(default_factory=list, description="Step definitions with name and message for each step")
    step_results: Dict[str, Any] = Field(default_factory=dict, description="Results for each completed step")
    step_execution_times: Dict[str, float] = Field(default_factory=dict, description="Execution time for each completed step in seconds")
    # Final results
    results: Optional[Dict[str, Any]] = Field(None, description="Job results (if completed)")
    error: Optional[str] = Field(None, description="Error message (if failed)")


class JobListResponse(BaseModel):
    """Response for listing jobs."""
    success: bool = Field(..., description="Whether query was successful")
    jobs: List[Dict[str, Any]] = Field(..., description="List of jobs")
    count: int = Field(..., description="Number of jobs returned")


class StepResponse(BaseModel):
    """Response for individual step execution."""
    success: bool = Field(..., description="Whether step completed successfully")
    step_number: int = Field(..., description="Step number that was executed")
    results: Dict[str, Any] = Field(..., description="Step results")
    execution_time: float = Field(..., description="Execution time in seconds")


class PipelineResponse(BaseModel):
    """Response for full pipeline execution."""
    success: bool = Field(..., description="Whether pipeline completed successfully")
    job_id: str = Field(..., description="Job identifier")
    results: Dict[str, Any] = Field(..., description="All pipeline results")
    steps_completed: int = Field(..., description="Number of steps completed")
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    output_path: Optional[str] = Field(None, description="Path where results were saved")


class MetaPipelineResponse(BaseModel):
    """Response for meta-analysis pipeline execution."""
    success: bool = Field(..., description="Whether meta-pipeline completed successfully")
    job_id: str = Field(..., description="Job identifier")
    meta_results: Dict[str, Any] = Field(..., description="Meta-analysis pipeline results")
    dataset_results: Dict[str, Dict[str, Any]] = Field(..., description="Individual dataset pipeline results")
    reproducibility_results: Dict[str, Any] = Field(..., description="Reproducibility analysis results (Meta-Step 7)")
    discovery_results: Dict[str, Any] = Field(..., description="Discovery analysis results (Meta-Step 9)")
    comparative_report: str = Field(..., description="Markdown comparative report (Meta-Step 8)")
    total_execution_time: float = Field(..., description="Total execution time in seconds")
    output_path: Optional[str] = Field(None, description="Path where results were saved")


# Error models
class ErrorDetail(BaseModel):
    """Error detail information."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(default=False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")
