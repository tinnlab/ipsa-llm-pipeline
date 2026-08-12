"""Fix 2 core: StepValidators.strip_hallucinated_entities removes exactly the flagged entities,
without mutating the input. Light import (no heavy deps)."""
from src.pipeline.services.step_validators import StepValidators as SV


def test_step1_removes_hallucinated_pathway(allow_pathways, allow_genes):
    step1 = {"themes": [
        {"theme": "Energy", "pathway_count": 2,
         "pathways": [{"name": "Glycolysis"}, {"name": "FAKE_PATHWAY"}]},
    ]}
    cleaned, removed = SV.strip_hallucinated_entities(step1, 1, allow_pathways, allow_genes)
    assert [p["name"] for p in cleaned["themes"][0]["pathways"]] == ["Glycolysis"]
    assert removed == ["FAKE_PATHWAY"]
    # pathway_count is kept consistent with the filtered list
    assert cleaned["themes"][0]["pathway_count"] == 1
    # original is NOT mutated (raw output is preserved by the caller)
    assert len(step1["themes"][0]["pathways"]) == 2


def test_step1_supports_pathway_key_alias(allow_pathways, allow_genes):
    step1 = {"themes": [{"pathways": [{"pathway": "TCA cycle"}, {"pathway": "BOGUS"}]}]}
    cleaned, removed = SV.strip_hallucinated_entities(step1, 1, allow_pathways, allow_genes)
    assert removed == ["BOGUS"]
    assert len(cleaned["themes"][0]["pathways"]) == 1


def test_step2_removes_hallucinated_hub_genes(allow_pathways, allow_genes):
    step2 = {
        "hub_genes": [{"gene": "TP53"}, {"gene": "NOTAGENE"}],
        "network_hubs": [{"gene": "EGFR"}, {"gene": "ZZZ9"}],
    }
    cleaned, removed = SV.strip_hallucinated_entities(step2, 2, allow_pathways, allow_genes)
    assert [h["gene"] for h in cleaned["hub_genes"]] == ["TP53"]
    assert [h["gene"] for h in cleaned["network_hubs"]] == ["EGFR"]
    assert set(removed) == {"NOTAGENE", "ZZZ9"}


def test_step3_removes_hallucinated_de_gene(allow_pathways, allow_genes):
    step3 = {"pathway_mechanisms": [
        {"deGeneInvolvement": [{"gene": "BRCA1"}, {"gene": "HALLUC1"}]},
    ]}
    cleaned, removed = SV.strip_hallucinated_entities(step3, 3, allow_pathways, allow_genes)
    assert [g["gene"] for g in cleaned["pathway_mechanisms"][0]["deGeneInvolvement"]] == ["BRCA1"]
    assert removed == ["HALLUC1"]


def test_step3_allow_list_includes_pathway_genes(allow_pathways, allow_genes):
    # a gene not in input genes but present in pathway (curated) genes should be KEPT
    step3 = {"pathway_mechanisms": [
        {"deGeneInvolvement": [{"gene": "PATHWAYGENE"}, {"gene": "HALLUC1"}]},
    ]}
    allow = allow_genes | {"PATHWAYGENE"}
    cleaned, removed = SV.strip_hallucinated_entities(step3, 3, allow_pathways, allow)
    assert [g["gene"] for g in cleaned["pathway_mechanisms"][0]["deGeneInvolvement"]] == ["PATHWAYGENE"]
    assert removed == ["HALLUC1"]


def test_all_valid_removes_nothing(allow_pathways, allow_genes):
    step2 = {"hub_genes": [{"gene": "TP53"}, {"gene": "EGFR"}]}
    cleaned, removed = SV.strip_hallucinated_entities(step2, 2, allow_pathways, allow_genes)
    assert removed == []
    assert len(cleaned["hub_genes"]) == 2


def test_all_invalid_removes_everything(allow_pathways, allow_genes):
    step2 = {"network_hubs": [{"gene": "BOGUS1"}, {"gene": "BOGUS2"}]}
    cleaned, removed = SV.strip_hallucinated_entities(step2, 2, allow_pathways, allow_genes)
    assert cleaned["network_hubs"] == []
    assert set(removed) == {"BOGUS1", "BOGUS2"}


def test_empty_inputs_safe(allow_pathways, allow_genes):
    cases = [(1, {"themes": []}), (2, {"hub_genes": []}), (3, {"pathway_mechanisms": []}), (1, {})]
    for step_no, d in cases:
        cleaned, removed = SV.strip_hallucinated_entities(d, step_no, allow_pathways, allow_genes)
        assert removed == []


def test_predicate_is_case_sensitive_like_validator(allow_pathways, allow_genes):
    # The validators use exact membership, so 'tp53' != 'TP53' and is treated as hallucinated.
    step2 = {"hub_genes": [{"gene": "tp53"}]}
    cleaned, removed = SV.strip_hallucinated_entities(step2, 2, allow_pathways, allow_genes)
    assert removed == ["tp53"]
    assert cleaned["hub_genes"] == []
