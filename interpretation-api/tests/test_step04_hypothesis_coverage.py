"""
Fix 1 / AC1 & AC3 — effect-size coverage of hypotheses.

Every dominant enrichment direction (|NES| >= threshold) must be addressed by at
least one hypothesis (AC1), and the top-|NES| cluster must be referenced even when
it is peripheral in the PPI network (AC3). These tests cover the deterministic
detection helper and the prompt-construction guarantees (no LLM required).
"""

import json
from unittest.mock import MagicMock

from src.pipeline.steps.step04_hypothesis_generation import (
    Step04HypothesisGeneration,
    _nes_axis_signals,
    DOMINANT_NES_THRESHOLD,
)


def _bare_step04():
    return Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)


UP_HYP = {'hypothesis': 'CDK1 accelerates mitosis',
          'mechanisticModel': 'CDK1 and CCNB1 drive G2/M in the cell cycle.',
          'keyPlayers': ['CDK1']}
DOWN_HYP = {'hypothesis': 'Loss of fatty-acid oxidation in HCC',
            'mechanisticModel': 'ACADS down-regulation impairs beta-oxidation.',
            'keyPlayers': ['ACADS']}


# Strong down-regulated metabolic axis (peripheral in PPI) + strong up axis.
PATHWAYS = [
    {'name': 'Fatty acid degradation', 'NES': -3.4,
     'genes': ['ACADS', 'ALDH2', 'CPT1A']},
    {'name': 'Cell cycle', 'NES': 2.1, 'genes': ['CDK1', 'CCNB1']},
]

GENES = [
    {'geneSymbol': 'ACADS', 'foldChange': -1.4},
    {'geneSymbol': 'ALDH2', 'foldChange': -1.5},
    {'geneSymbol': 'CPT1A', 'foldChange': -2.1},
    {'geneSymbol': 'CDK1', 'foldChange': +2.4},
    {'geneSymbol': 'CCNB1', 'foldChange': +3.1},
]


def test_axis_signals_rank_both_directions():
    ups, downs = _nes_axis_signals(PATHWAYS)
    assert ups[0][1] == 'Cell cycle'
    assert downs[0][1] == 'Fatty acid degradation'


def test_up_only_hypotheses_flag_the_down_axis():
    """AC1/AC3: proliferation-only output must flag the strong down metabolic axis."""
    step = _bare_step04()
    result = {
        'centralMechanisticModel': 'HRAS drives E2F1 and CDK1 in the cell cycle.',
        'hypotheses': [
            {'hypothesis': 'CDK1 accelerates mitosis',
             'mechanisticModel': 'CDK1 and CCNB1 drive G2/M in the cell cycle.',
             'keyPlayers': ['CDK1', 'CCNB1']},
        ],
    }
    unmet = step._validate_hypothesis_coverage(result, PATHWAYS, GENES)
    directions = {u['direction'] for u in unmet}
    assert 'down' in directions
    # the flagged cluster is the top-|NES| cluster (AC3)
    down = [u for u in unmet if u['direction'] == 'down'][0]
    assert down['pathway'] == 'Fatty acid degradation'
    assert abs(down['nes']) >= DOMINANT_NES_THRESHOLD


def test_balanced_hypotheses_have_no_unmet_axis():
    step = _bare_step04()
    result = {
        'centralMechanisticModel': 'Proliferation up, catabolism down.',
        'hypotheses': [
            {'hypothesis': 'CDK1 accelerates mitosis',
             'mechanisticModel': 'CDK1/CCNB1 drive the cell cycle.',
             'keyPlayers': ['CDK1']},
            {'hypothesis': 'Loss of fatty-acid oxidation',
             'mechanisticModel': 'ACADS down-regulation impairs beta-oxidation.',
             'keyPlayers': ['ACADS']},
        ],
    }
    unmet = step._validate_hypothesis_coverage(result, PATHWAYS, GENES)
    assert unmet == []


def test_moderate_signals_are_not_forced():
    """A direction below the |NES| threshold is not required to have a hypothesis."""
    step = _bare_step04()
    weak_pathways = [
        {'name': 'Cell cycle', 'NES': 2.1, 'genes': ['CDK1']},
        {'name': 'Some mild program', 'NES': -1.2, 'genes': ['ACADS']},  # |NES| < 2
    ]
    result = {'hypotheses': [
        {'hypothesis': 'CDK1 drives proliferation',
         'mechanisticModel': 'CDK1 in cell cycle', 'keyPlayers': ['CDK1']}]}
    unmet = step._validate_hypothesis_coverage(result, weak_pathways, GENES)
    assert unmet == []  # the weak down signal is not a dominant axis


def test_priority_signals_block_surfaces_down_axis():
    step = _bare_step04()
    block = step._priority_signals_block(PATHWAYS)
    assert 'Fatty acid degradation' in block
    assert 'DOWN-regulated' in block
    # the dominant down axis is marked as requiring a hypothesis
    assert 'MUST' in block


def _coverage_step(regen_hypotheses, biochem_errors=None):
    step = _bare_step04()
    step.llm = MagicMock()
    step.llm.chat.return_value = json.dumps({'hypotheses': regen_hypotheses})
    step._validate_biochemistry_with_llm = MagicMock(return_value=biochem_errors or [])
    return step


def test_coverage_regen_appends_and_preserves_originals():
    """H1 fix: a model that returns only the new (delta) hypothesis must not drop
    the already-validated originals — coverage APPENDS, never replaces."""
    step = _coverage_step([DOWN_HYP])
    result = {'centralMechanisticModel': 'CDK1 drives the cell cycle.',
              'hypotheses': [dict(UP_HYP)]}
    warnings = step._enforce_hypothesis_coverage(result, PATHWAYS, GENES, '', 'sys', 'usr')
    stmts = [h['hypothesis'] for h in result['hypotheses']]
    assert UP_HYP['hypothesis'] in stmts          # original preserved
    assert DOWN_HYP['hypothesis'] in stmts        # new appended
    assert len(result['hypotheses']) == 2
    assert warnings == []                          # converged
    # synthesis fields left untouched (not overwritten by regen)
    assert result['centralMechanisticModel'] == 'CDK1 drives the cell cycle.'


def test_coverage_regen_dedups_echoed_full_set():
    """If the model echoes the full set (original + new), no duplicate is created."""
    step = _coverage_step([dict(UP_HYP), DOWN_HYP])
    result = {'hypotheses': [dict(UP_HYP)]}
    step._enforce_hypothesis_coverage(result, PATHWAYS, GENES, '', 'sys', 'usr')
    assert len(result['hypotheses']) == 2


def test_coverage_regen_drops_biochem_failing_new_hypothesis():
    """A coverage hypothesis that fails biochemistry review is not appended."""
    step = _coverage_step(
        [DOWN_HYP],
        biochem_errors=['Biochemistry review (error) - Hypothesis 1: implausible'])
    result = {'hypotheses': [dict(UP_HYP)]}
    warnings = step._enforce_hypothesis_coverage(result, PATHWAYS, GENES, '', 'sys', 'usr')
    assert len(result['hypotheses']) == 1          # nothing bad appended
    assert warnings                                 # unmet axis recorded


def test_priority_block_labels_classic_es_as_es():
    """M4 fix: a genuine classic ES (|ES| <= 1, no NES field) is labeled 'ES', not 'NES'."""
    step = _bare_step04()
    block = step._priority_signals_block([{'name': 'Weak up pathway', 'ES': 0.55}])
    assert 'ES +0.55' in block
    assert 'NES +0.55' not in block


def test_nes_axis_signals_carries_metric():
    ups, downs = _nes_axis_signals([
        {'name': 'A', 'NES': 2.2},
        {'name': 'B', 'ES': 0.4},        # classic ES
        {'name': 'C', 'ES': -3.1},       # |ES|>1 -> promoted to NES scale
    ])
    metric_by_name = {nm: metric for _v, nm, metric in ups + downs}
    assert metric_by_name['A'] == 'NES'
    assert metric_by_name['B'] == 'ES'
    assert metric_by_name['C'] == 'NES'


def test_pathway_table_not_truncating_strong_down_signal():
    """AC3 input-side: the strong down pathway survives the top-15 table cap even
    when many up-regulated pathways are present."""
    step = _bare_step04()
    structures = [
        {'pathway': f'Up pathway {i}', 'enrichment_score': 3.0 - i * 0.05,
         'enrichment_direction': 'upregulated', 'de_genes_count': 5,
         'mapped_de_genes': [], 'p_value_fdr': 1e-3}
        for i in range(20)
    ]
    structures.append({
        'pathway': 'Fatty acid degradation', 'enrichment_score': -3.4,
        'enrichment_direction': 'downregulated', 'de_genes_count': 11,
        'mapped_de_genes': [{'gene_symbol': 'ACADS', 'fold_change': -1.4}],
        'p_value_fdr': 1e-2,
    })
    prompt = step._build_user_prompt(
        genes=GENES, pathways=PATHWAYS, themes=None, hub_genes_result=None,
        mechanisms_result={'pathway_structures': structures, 'pathway_overlaps': []},
        experiment_context='HCC',
    )
    assert 'Fatty acid degradation' in prompt


def test_all_upstream_regulators_reach_the_prompt():
    """Step 3 already caps and rank-orders upstream regulators, so Step 4 must pass
    every one it receives to the LLM. A second cap here silently dropped candidates
    that had passed Step 3's significance gate, and disagreed with the report, which
    lists them all."""
    step = _bare_step04()
    regulators = [
        {'tf': f'TF{i}', 'overlap_count': 10 - i, 'is_de': False,
         'inferred_tf_activity': 'decreased', 'targets': ['ACADS', 'CPT1A']}
        for i in range(8)
    ]
    prompt = step._build_user_prompt(
        genes=GENES, pathways=PATHWAYS, themes=None, hub_genes_result=None,
        mechanisms_result={'upstream_regulators': regulators},
        experiment_context='HCC',
    )
    for i in range(8):
        assert f'**TF{i}**' in prompt, f'TF{i} was dropped from the Step 4 prompt'
