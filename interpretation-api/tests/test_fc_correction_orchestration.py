"""
Tests for the step04 orchestration of fold-change correction
(`Step04HypothesisGeneration._correct_fold_change_citations`) — the method the pipeline
actually calls. Covers: hypothesis/summary/keyPredictions magnitude fixes, and the
central-model sign-conflict flow (regenerate once → re-check → drop the contradicting
sentence).
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration


DE = {'HEATR1': 1.10, 'NOP56': 1.24, 'ACADS': -1.41}


def _bare_step04():
    step = Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)
    step.step_number = 4
    step.llm = MagicMock()
    return step


def test_empty_gene_fc_is_noop():
    step = _bare_step04()
    result = {'centralMechanisticModel': 'HEATR1 +1.99-fold'}
    assert step._correct_fold_change_citations(result, {}, seed=42) == set()
    assert result['centralMechanisticModel'] == 'HEATR1 +1.99-fold'  # unchanged


def test_magnitude_fix_in_central_and_hypotheses_and_predictions():
    step = _bare_step04()
    result = {
        'centralMechanisticModel': 'Up-regulation of HEATR1 +1.15-fold expands capacity.',
        'hypotheses': [{
            'hypothesis': 'HEATR1 +1.15-fold drives it',
            'keyPlayers': ['HEATR1 +1.15-fold'],
            'testability': {'approach1': 'measure NOP56 +1.20-fold'},
        }],
        'keyPredictions': [{'prediction': 'HEATR1 +1.15-fold rises'}],
        'hypothesesSummary': 'Overall HEATR1 +1.15-fold up.',
    }
    conflicts = step._correct_fold_change_citations(result, DE, seed=42)
    assert conflicts == set()
    assert '+1.10-fold' in result['centralMechanisticModel']
    assert result['hypotheses'][0]['hypothesis'] == 'HEATR1 +1.10-fold drives it'
    assert result['hypotheses'][0]['keyPlayers'] == ['HEATR1 +1.10-fold']
    assert result['hypotheses'][0]['testability']['approach1'] == 'measure NOP56 +1.24-fold'
    assert result['keyPredictions'][0]['prediction'] == 'HEATR1 +1.10-fold rises'
    assert '+1.10-fold' in result['hypothesesSummary']


def test_sign_conflict_regenerates_once_and_resolves():
    step = _bare_step04()
    # Regeneration returns a clean, direction-correct sentence (no conflict).
    step._regenerate_central_model_with_fcs = MagicMock(
        return_value='Down-regulation of ACADS -1.41-fold suppresses catabolism.')
    result = {'centralMechanisticModel': 'Increased ACADS +1.40-fold supports catabolism.'}
    conflicts = step._correct_fold_change_citations(result, DE, seed=42)
    step._regenerate_central_model_with_fcs.assert_called_once()
    # Resolved by regeneration → not reported as an unresolved conflict.
    assert conflicts == set()
    assert result['centralMechanisticModel'].startswith('Down-regulation of ACADS -1.41')


def test_sign_conflict_surviving_regeneration_drops_sentence():
    step = _bare_step04()
    # Regeneration STILL contradicts the data → offending sentence must be dropped.
    step._regenerate_central_model_with_fcs = MagicMock(
        return_value='Cells proliferate. Increased ACADS +1.40-fold persists. Overall aggressive.')
    result = {'centralMechanisticModel': 'Increased ACADS +1.40-fold supports catabolism.'}
    conflicts = step._correct_fold_change_citations(result, DE, seed=42)
    assert 'ACADS' in conflicts
    final = result['centralMechanisticModel']
    assert 'ACADS' not in final                     # contradicting sentence removed
    assert 'Cells proliferate.' in final
    assert 'Overall aggressive.' in final


def test_hypothesis_sign_conflict_is_not_sentence_dropped():
    # Sign conflicts in hypothesis prose are reported but wording is not rewritten/dropped
    # (the existing direction validator surfaces them); only the central model is rewritten.
    step = _bare_step04()
    result = {'hypotheses': [{'hypothesis': 'increased ACADS +1.40-fold drives growth'}]}
    conflicts = step._correct_fold_change_citations(result, DE, seed=42)
    assert 'ACADS' in conflicts
    assert 'drives growth' in result['hypotheses'][0]['hypothesis']


def test_resolved_central_conflict_not_reported_as_unresolved():
    # If regeneration produces a clean model, the conflict is resolved and must NOT be
    # reported in the returned (unresolved) conflict set.
    step = _bare_step04()
    step._regenerate_central_model_with_fcs = MagicMock(
        return_value='ACADS -1.41-fold suppresses catabolism.')
    result = {'centralMechanisticModel': 'increased ACADS +1.40-fold supports it.'}
    conflicts = step._correct_fold_change_citations(result, DE, seed=42)
    assert conflicts == set()  # resolved → not reported


def test_non_dict_hypothesis_entry_does_not_crash():
    step = _bare_step04()
    result = {'hypotheses': ['a bare string', {'hypothesis': 'HEATR1 +1.15-fold up'}]}
    conflicts = step._correct_fold_change_citations(result, DE, seed=42)
    assert conflicts == set()
    assert result['hypotheses'][1]['hypothesis'] == 'HEATR1 +1.10-fold up'


# ---------------------------------------------------------------------------
# _regenerate_central_model_with_fcs — prompt/parse plumbing
# ---------------------------------------------------------------------------

def test_regenerate_parses_llm_json():
    step = _bare_step04()
    step.llm.chat = MagicMock(return_value='{"centralMechanisticModel": "Clean model."}')
    out = step._regenerate_central_model_with_fcs(
        {'centralMechanisticModel': 'old'}, DE, seed=1)
    assert out == 'Clean model.'


def test_regenerate_returns_none_on_unparseable():
    step = _bare_step04()
    step.llm.chat = MagicMock(return_value='not json at all')
    out = step._regenerate_central_model_with_fcs(
        {'centralMechanisticModel': 'old'}, DE, seed=1)
    assert out is None
