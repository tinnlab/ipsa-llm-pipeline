"""Regression: validate_step3_output previously did `set(hallucinated)[:5]` which raises
TypeError ('set' object is not subscriptable) whenever a Step-3 hallucination exists."""
from src.pipeline.services.step_validators import StepValidators as SV


def test_step3_validation_with_hallucinations_does_not_crash():
    input_data = {"genes": [{"geneSymbol": "TP53"}]}
    output = {"pathway_mechanisms": [{
        "deGeneInvolvement": [{"gene": "FAKE1"}, {"gene": "FAKE2"}, {"gene": "FAKE1"}],
        "curatedRelations": [],
    }]}
    # Must not raise; FAKE1/FAKE2 are not in input genes -> flagged.
    res = SV.validate_step3_output(output, input_data, pathway_genes=set())
    assert res.passed is False
    assert any("Hallucinated genes" in e for e in res.errors)
    assert res.stats["num_hallucinated_genes"] == 2


def test_step3_validation_clean_passes():
    input_data = {"genes": [{"geneSymbol": "TP53"}]}
    output = {"pathway_mechanisms": [{"deGeneInvolvement": [{"gene": "TP53"}], "curatedRelations": []}]}
    res = SV.validate_step3_output(output, input_data, pathway_genes=set())
    assert res.passed is True
    assert res.stats["num_hallucinated_genes"] == 0
