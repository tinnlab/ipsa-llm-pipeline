"""Fix 1 end-state: the biochemistry cross-check actually invokes the reviewer (bio) model,
confirms/downgrades flags via it, and fails safe when the reviewer call errors (e.g. timeout)."""
from unittest.mock import MagicMock

from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration

# A flag whose claim contains a real gene symbol so _build_independent_question yields a question.
GENERATOR_FLAG = (
    '{"issues": [{"hypothesis_index": 1, "category": "metabolic_direction", '
    '"claim": "IDH1 runs the reaction backward, depleting its product", '
    '"correction": "", "severity": "error"}]}'
)
RESULT = {
    "hypotheses": [{"hypothesis": "IDH1 drives the phenotype", "mechanisticModel": "m"}],
    "centralMechanisticModel": "",
}
GENES = [{"geneSymbol": "IDH1", "foldChange": 2.0}]


def _bare_step():
    return Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)


def test_reviewer_confirms_flag_as_error_and_is_actually_called():
    step = _bare_step()
    step.llm = MagicMock()
    step.llm.chat.return_value = GENERATOR_FLAG          # generator raises the flag
    step.reviewer_llm = MagicMock()
    # reviewer: independent-knowledge answer, then CONTRADICTS verdict
    step.reviewer_llm.chat.side_effect = ["IDH1 converts isocitrate to alpha-ketoglutarate.", "CONTRADICTS"]

    warnings = step._validate_biochemistry_with_llm(RESULT, GENES)

    assert step.reviewer_llm.chat.called  # <-- the bio model genuinely runs the cross-check
    assert any("Biochemistry review (error)" in w for w in warnings)


def test_reviewer_timeout_fails_safe_to_warning():
    step = _bare_step()
    step.llm = MagicMock()
    step.llm.chat.return_value = GENERATOR_FLAG
    step.reviewer_llm = MagicMock()
    step.reviewer_llm.chat.side_effect = Exception("read timeout")  # simulate cold-start timeout

    warnings = step._validate_biochemistry_with_llm(RESULT, GENES)

    assert step.reviewer_llm.chat.called
    assert any("(warning)" in w for w in warnings)        # downgraded, not confirmed
    assert not any("(error)" in w for w in warnings)      # nothing removed on a failed verification


def test_no_generator_flags_skips_reviewer():
    step = _bare_step()
    step.llm = MagicMock()
    step.llm.chat.return_value = '{"issues": []}'
    step.reviewer_llm = MagicMock()

    warnings = step._validate_biochemistry_with_llm(RESULT, GENES)

    assert warnings == []
    assert not step.reviewer_llm.chat.called              # nothing to verify -> no bio call
