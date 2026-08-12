"""Fix 2 wiring + preservation: the orchestrator strips hallucinated entities from each step's
output before passing it to the next step, while keeping the RAW outputs and the flags."""
from unittest.mock import MagicMock

import src.pipeline.orchestrator as orch_mod
from src.pipeline.orchestrator import PipelineOrchestrator


def test_clean_at_boundaries_and_preserve(tmp_path, input_data, monkeypatch):
    # Neutralize progress tracking (no real job registered).
    monkeypatch.setattr(orch_mod, "job_manager", MagicMock())

    orch = PipelineOrchestrator(output_dir=str(tmp_path), enable_validation=True, strict_mode=False)

    # Canned step outputs that each contain one hallucinated entity (not in input_data).
    step1_out = {
        "themes": [{
            "theme": "Energy", "significance": "high", "pathway_count": 2,
            "pathways": [
                {"name": "Glycolysis", "p_value_fdr": 0.01},
                {"name": "FAKE_PATHWAY", "p_value_fdr": 0.02},
            ],
        }],
        "ungrouped": [],
    }
    # Production Step 2 emits `network_hubs` (not `hub_genes`); use the real key here so the
    # wiring test exercises the same path Step 4 consumes.
    step2_out = {
        "network_hubs": [{"gene": "TP53", "hub_score": 1.0}, {"gene": "FAKEGENE", "hub_score": 0.5}],
        "summary": "", "metadata": {},
    }
    step3_out = {
        "pathway_mechanisms": [{
            "pathway": "Glycolysis",
            "deGeneInvolvement": [{"gene": "BRCA1"}, {"gene": "FAKEDE"}],
            "curatedRelations": [],
        }],
        "metadata": {},
    }

    orch.step1 = MagicMock(); orch.step1.execute.return_value = step1_out
    orch.step2 = MagicMock(); orch.step2.execute.return_value = step2_out
    orch.step3 = MagicMock(); orch.step3.execute.return_value = step3_out
    orch.step4 = MagicMock(); orch.step4.execute.return_value = {
        "hypotheses": [{"hypothesis": "h1"}], "report_section": "r",
        "centralMechanisticModel": "", "metadata": {},
    }
    orch.step6 = MagicMock(); orch.step6.execute.return_value = {"markdown_content": "# report"}

    results = orch.run_full_pipeline(input_data, job_id="testjob")

    # --- Step 4 received CLEANED upstream data (hallucinations removed) ---
    _, kwargs = orch.step4.execute.call_args
    theme_pathways = [p["name"] for t in kwargs["themes"] for p in t["pathways"]]
    assert "FAKE_PATHWAY" not in theme_pathways
    assert "Glycolysis" in theme_pathways

    hub_genes = [h["gene"] for h in kwargs["hub_genes_result"]["network_hubs"]]
    assert "FAKEGENE" not in hub_genes
    assert "TP53" in hub_genes

    de_genes = [g["gene"] for m in kwargs["mechanisms_result"]["pathway_mechanisms"]
                for g in m["deGeneInvolvement"]]
    assert "FAKEDE" not in de_genes
    assert "BRCA1" in de_genes

    # --- RAW step outputs preserved in results['steps'] ---
    raw_theme_pathways = [p["name"] for t in results["steps"]["step1"]["themes"]
                          for p in t["pathways"]]
    assert "FAKE_PATHWAY" in raw_theme_pathways
    assert "FAKEGENE" in [h["gene"] for h in results["steps"]["step2"]["network_hubs"]]

    # --- flags preserved, and what was removed is recorded inside the validation block ---
    assert "FAKE_PATHWAY" in results["validation"]["step1"]["stats"]["removed_entities"]
    assert "FAKEGENE" in results["validation"]["step2"]["stats"]["removed_entities"]
    assert "FAKEDE" in results["validation"]["step3"]["stats"]["removed_entities"]


def test_step3_receives_organism_from_metadata(tmp_path, input_data, monkeypatch):
    """The orchestrator must thread the real organism into Step 3 (via analyses), so a
    non-human dataset doesn't silently get queried against human KEGG."""
    monkeypatch.setattr(orch_mod, "job_manager", MagicMock())
    orch = PipelineOrchestrator(output_dir=str(tmp_path), enable_validation=True, strict_mode=False)

    data = dict(input_data)
    data["metadata"] = dict(input_data["metadata"], organism="Mus musculus")

    orch.step1 = MagicMock(); orch.step1.execute.return_value = {
        "themes": [{"theme": "Energy", "significance": "high", "pathway_count": 1,
                    "pathways": [{"name": "Glycolysis", "p_value_fdr": 0.01}]}],
        "ungrouped": [],
    }
    orch.step2 = MagicMock(); orch.step2.execute.return_value = {
        "network_hubs": [{"gene": "TP53", "hub_score": 1.0}], "summary": "", "metadata": {},
    }
    orch.step3 = MagicMock(); orch.step3.execute.return_value = {
        "pathway_mechanisms": [{"pathway": "Glycolysis", "deGeneInvolvement": [{"gene": "BRCA1"}],
                                "curatedRelations": []}],
        "metadata": {},
    }
    orch.step4 = MagicMock(); orch.step4.execute.return_value = {
        "hypotheses": [{"hypothesis": "h1"}], "report_section": "r",
        "centralMechanisticModel": "", "metadata": {},
    }
    orch.step6 = MagicMock(); orch.step6.execute.return_value = {"markdown_content": "# report"}

    orch.run_full_pipeline(data, job_id="orgjob")

    _, kwargs = orch.step3.execute.call_args
    assert kwargs["analyses"] == [{"organismId": "Mus musculus"}]


def test_single_step3_receives_organism_from_metadata(tmp_path, input_data, monkeypatch):
    """The run_single_step Step-3 path (separate API) must also thread the real organism."""
    monkeypatch.setattr(orch_mod, "job_manager", MagicMock())
    orch = PipelineOrchestrator(output_dir=str(tmp_path), enable_validation=True, strict_mode=False)
    orch.step3 = MagicMock()
    orch.step3.execute.return_value = {"pathway_mechanisms": [], "metadata": {}}

    data = dict(input_data)
    data["metadata"] = dict(input_data["metadata"], organism="Mus musculus")
    previous_results = {"steps": {
        "step1": {"themes": [{"pathways": [{"name": "Glycolysis"}]}], "ungrouped": []},
        "step2": {"network_hubs": [{"gene": "TP53"}]},
    }}

    orch.run_single_step(3, data, previous_results=previous_results)

    _, kwargs = orch.step3.execute.call_args
    assert kwargs["analyses"] == [{"organismId": "Mus musculus"}]
