"""Meta progress steps: the declared steps_info and the step ids the orchestrator fires
must agree and run contiguously in execution order. Reproducibility (Meta-Step 7) is
disabled because nothing in the report reads its output."""
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import src.api.server as server
import src.pipeline.meta_orchestrator as mo
from src.api.job_manager import job_manager

PATHWAY = {"name": "Cell cycle", "source": "KEGG", "pathwayId": "hsa04110",
           "pValue": 0.01, "pValueFDR": 0.02, "genes": ["CDK1"]}
GENE = {"geneSymbol": "CDK1", "foldChange": 2.0, "pValue": 1e-4, "pValueFDR": 1e-3}


def _submit(monkeypatch, with_genes):
    monkeypatch.setattr(server, "run_meta_pipeline_job", lambda *a, **k: None)
    meta = {"pathways": [PATHWAY], "genes": [GENE] if with_genes else []}
    body = {"meta": meta, "datasets": [{"dataset_name": "d1", **meta}]}
    r = TestClient(server.app).post("/api/meta-pipeline", json=body)
    assert r.status_code == 200
    return job_manager.jobs[r.json()["job_id"]]


@pytest.mark.parametrize("with_genes", [True, False])
def test_meta_steps_info_contiguous_without_reproducibility(monkeypatch, with_genes):
    job = _submit(monkeypatch, with_genes)
    steps = [s["step"] for s in job.steps_info]
    assert steps == ([1, 2, 3, 4, 5, 6, 7] if with_genes else [1, 4, 5, 6, 7])
    names = [s["name"] for s in job.steps_info]
    assert names[-3:] == ["Analysis Report", "Cross-Dataset Findings", "Final Report"]
    assert "Reproducibility Analysis" not in names


def test_fallback_meta_steps_match_declared(monkeypatch):
    from src.api.job_manager import META_PIPELINE_STEPS
    job = _submit(monkeypatch, with_genes=True)
    assert [(s["step"], s["name"]) for s in META_PIPELINE_STEPS[-2:]] == \
           [(s["step"], s["name"]) for s in job.steps_info[-2:]]


@dataclass
class _Discovery:
    summary: dict = field(default_factory=dict)


def test_orchestrator_fires_declared_ids_in_order_and_skips_reproducibility(monkeypatch):
    jm = MagicMock()
    monkeypatch.setattr(mo, "job_manager", jm)
    orch = mo.MetaPipelineOrchestrator.__new__(mo.MetaPipelineOrchestrator)
    orch.step7, orch.step8, orch.step9 = MagicMock(), MagicMock(), MagicMock()
    orch.step9.execute.return_value = _Discovery()
    orch.step8.execute.return_value = "report"
    orch._run_single_pipeline = MagicMock(return_value={"success": True})
    orch._build_synthetic_results = MagicMock(return_value={})
    orch._validate_step9_output = MagicMock()
    orch._validate_step8_output = MagicMock()

    result = orch.execute({"metadata": {}}, [{"dataset_name": "d1"}], job_id="job")

    assert result.success, result.error
    orch.step7.execute.assert_not_called()
    assert [c.args[1] for c in jm.set_current_step.call_args_list] == [1, 6, 7]
    assert [c.args[1] for c in jm.set_step_result.call_args_list] == [6, 7]
    assert orch.step8.execute.call_args.kwargs["reproducibility_results"] == {}
    assert result.reproducibility_results == {}  # payload key kept for cpa-website
