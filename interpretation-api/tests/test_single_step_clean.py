"""Fix 2, run_single_step path: when a step is run in isolation, the hallucinated entities in the
prior-step inputs (passed via previous_results) are stripped before the step consumes them, without
mutating previous_results."""
from unittest.mock import MagicMock

import src.pipeline.orchestrator as orch_mod
from src.pipeline.orchestrator import PipelineOrchestrator


def _orch(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "job_manager", MagicMock())
    return PipelineOrchestrator(output_dir=str(tmp_path), enable_validation=True)


def test_single_step4_receives_cleaned_prior_outputs(tmp_path, input_data, monkeypatch):
    orch = _orch(tmp_path, monkeypatch)
    orch.step4 = MagicMock()
    orch.step4.execute.return_value = {"hypotheses": [{"hypothesis": "h"}], "metadata": {}}

    previous_results = {"steps": {
        "step1": {"themes": [{"pathways": [{"name": "Glycolysis"}, {"name": "FAKE_PATHWAY"}]}],
                  "ungrouped": []},
        "step2": {"network_hubs": [{"gene": "TP53"}, {"gene": "FAKEGENE"}]},
        "step3": {"pathway_mechanisms": [
            {"deGeneInvolvement": [{"gene": "BRCA1"}, {"gene": "FAKEDE"}], "curatedRelations": []}]},
    }}

    orch.run_single_step(4, input_data, previous_results=previous_results)

    _, kwargs = orch.step4.execute.call_args
    theme_pathways = [p["name"] for t in kwargs["themes"] for p in t["pathways"]]
    assert "FAKE_PATHWAY" not in theme_pathways and "Glycolysis" in theme_pathways
    assert "FAKEGENE" not in [h["gene"] for h in kwargs["hub_genes_result"]["network_hubs"]]
    de = [g["gene"] for m in kwargs["mechanisms_result"]["pathway_mechanisms"]
          for g in m["deGeneInvolvement"]]
    assert "FAKEDE" not in de and "BRCA1" in de

    # previous_results must NOT be mutated (the raw inputs are preserved)
    assert "FAKE_PATHWAY" in [p["name"] for t in previous_results["steps"]["step1"]["themes"]
                              for p in t["pathways"]]
    assert "FAKEGENE" in [h["gene"] for h in previous_results["steps"]["step2"]["network_hubs"]]
