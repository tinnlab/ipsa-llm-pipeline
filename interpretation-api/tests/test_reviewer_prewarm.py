"""Fix 1b: Step 4 pre-warms the reviewer (OpenBio) non-blockingly so the biochem cross-check
isn't a cold-start hang; warm-up failures must never break the step."""
from unittest.mock import MagicMock

from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration


def _bare_step():
    # Skip the heavy __init__ (which builds real LLM clients); we set llm/reviewer_llm ourselves.
    return Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)


def test_prewarm_fires_reviewer_when_separate():
    step = _bare_step()
    step.llm = MagicMock(name="generator")
    step.reviewer_llm = MagicMock(name="reviewer")
    t = step._prewarm_reviewer()
    assert t is not None
    t.join(timeout=5)
    assert step.reviewer_llm.chat.called
    # generator must NOT be used for warm-up
    assert not step.llm.chat.called


def test_prewarm_noop_when_reviewer_is_generator():
    step = _bare_step()
    shared = MagicMock()
    step.llm = shared
    step.reviewer_llm = shared  # fallback case: no separate reviewer
    t = step._prewarm_reviewer()
    assert t is None
    assert not shared.chat.called


def test_prewarm_swallows_reviewer_errors():
    step = _bare_step()
    step.llm = MagicMock()
    step.reviewer_llm = MagicMock()
    step.reviewer_llm.chat.side_effect = RuntimeError("cold start failed")
    t = step._prewarm_reviewer()  # must not raise
    t.join(timeout=5)
    assert step.reviewer_llm.chat.called  # attempted, error swallowed inside the thread
