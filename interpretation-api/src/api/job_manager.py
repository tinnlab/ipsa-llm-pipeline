"""
Job manager for async pipeline execution.

Provides in-memory job tracking with thread-safe operations.
"""

import uuid
import threading
from datetime import datetime
from typing import Dict, Optional, Any, List
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Step definitions for the standard pipeline (5 steps)
PIPELINE_STEPS = [
    {
        "step": 1,
        "name": "Pathway Theme Identification",
        "message": "Clustering pathways by shared biological functions and gene overlap..."
    },
    {
        "step": 2,
        "name": "Hub Gene Detection",
        "message": "Analyzing protein-protein interaction network via STRING database..."
    },
    {
        "step": 3,
        "name": "Pathway Mechanisms",
        "message": "Fetching molecular interactions from pathway databases..."
    },
    {
        "step": 4,
        "name": "Hypothesis Generation",
        "message": "Generating testable research hypotheses from pathway analysis..."
    },
    {
        "step": 5,
        "name": "Report Generation",
        "message": "Compiling analysis results into final report..."
    }
]

# Additional steps that only run in the meta-analysis pipeline
META_PIPELINE_STEPS = PIPELINE_STEPS + [
    {
        "step": 6,
        "name": "Reproducibility Analysis",
        "message": "Comparing findings across individual datasets for reproducibility..."
    },
    {
        "step": 7,
        "name": "Comparative Report",
        "message": "Generating comparative analysis report..."
    }
]


class Job:
    """Represents a pipeline job."""

    def __init__(
        self,
        job_id: str,
        input_data: Dict[str, Any],
        total_steps: int = 5,
        steps_info: Optional[List[Dict[str, Any]]] = None
    ):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.input_data = input_data
        self.results: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        # Step progress tracking
        self.current_step: Optional[int] = None
        self.current_step_message: Optional[str] = None
        self.total_steps: int = total_steps
        self.step_results: Dict[str, Dict[str, Any]] = {}
        self.step_execution_times: Dict[str, float] = {}
        # Step definitions - use custom steps_info if provided, else default based on pipeline type
        if steps_info is not None:
            self.steps_info: List[Dict[str, Any]] = steps_info
        else:
            self.steps_info: List[Dict[str, Any]] = (
                META_PIPELINE_STEPS if total_steps == len(META_PIPELINE_STEPS) else PIPELINE_STEPS
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "input_data": self.input_data,
            "results": self.results,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step": self.current_step,
            "current_step_message": self.current_step_message,
            "total_steps": self.total_steps,
            "step_results": self.step_results,
            "step_execution_times": self.step_execution_times,
            "steps_info": self.steps_info
        }


class JobManager:
    """
    Thread-safe job manager for tracking pipeline jobs.

    Stores jobs in memory with options to persist to disk.
    """

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        input_data: Dict[str, Any],
        total_steps: int = 5,
        steps_info: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Create a new job.

        Args:
            input_data: Input data for the pipeline
            total_steps: Total number of steps in the pipeline (5 for the standard
                pipeline, 7 for meta-analysis)
            steps_info: Optional custom step definitions (overrides default based on total_steps)

        Returns:
            job_id: Unique job identifier
        """
        job_id = str(uuid.uuid4())
        job = Job(job_id, input_data, total_steps, steps_info)

        with self._lock:
            self.jobs[job_id] = job

        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        with self._lock:
            return self.jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update job status.

        Args:
            job_id: Job identifier
            status: New status
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.status = status
                job.updated_at = datetime.utcnow()

                if status == JobStatus.RUNNING and job.started_at is None:
                    job.started_at = datetime.utcnow()
                elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    job.completed_at = datetime.utcnow()

    def set_results(self, job_id: str, results: Dict[str, Any]) -> None:
        """
        Set job results.

        Args:
            job_id: Job identifier
            results: Job results
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.results = results
                job.updated_at = datetime.utcnow()

    def set_error(self, job_id: str, error: str) -> None:
        """
        Set job error.

        Args:
            job_id: Job identifier
            error: Error message
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.error = error
                job.updated_at = datetime.utcnow()

    def set_current_step(self, job_id: str, step_number: int, message: Optional[str] = None) -> None:
        """
        Set the currently executing step.

        Args:
            job_id: Job identifier
            step_number: Step number currently being executed (1-based)
            message: Optional custom message for this step (overrides default from steps_info)
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.current_step = step_number
                # Use custom message or default from steps_info. Look up by the "step"
                # field, not list position: a gene-free run omits steps 2 and 3, so the
                # numbers present are 1,4,5 and position would not match.
                if message:
                    job.current_step_message = message
                else:
                    entry = next((s for s in job.steps_info if s.get("step") == step_number), None)
                    if entry:
                        job.current_step_message = entry["message"]
                job.updated_at = datetime.utcnow()

    def set_step_result(self, job_id: str, step_number: int, result: Dict[str, Any], execution_time: float) -> None:
        """
        Store the result of a completed step.

        Args:
            job_id: Job identifier
            step_number: Step number that completed (1-based)
            result: Step output/result
            execution_time: Time taken to execute the step in seconds
        """
        with self._lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                step_key = f"step{step_number}"
                job.step_results[step_key] = result
                job.step_execution_times[step_key] = execution_time
                job.updated_at = datetime.utcnow()

    def list_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all jobs (most recent first).

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of job dictionaries
        """
        with self._lock:
            jobs = sorted(
                self.jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )
            return [job.to_dict() for job in jobs[:limit]]

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if job_id in self.jobs:
                del self.jobs[job_id]
                return True
            return False


# Global job manager instance
job_manager = JobManager()
