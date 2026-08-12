"""
Tests for bug 4: Step 4 must not lose its pathway tools when Step 3 retrieved
structures but its mechanism-interpretation layer came back empty.

Two parts:
- Part A: `Step04HypothesisGeneration._has_pathway_data` gates tool calling on
  pathway_structures OR pathway_mechanisms (not mechanisms alone).
- Part B: `PathwayQueryService.get_pathway_mechanism` falls back to the structure's
  curated/inferred relations when the mechanism layer is empty, so the tool still
  returns mechanistic grounding.
"""

import pytest
from unittest.mock import MagicMock

from src.pipeline.steps.step04_hypothesis_generation import Step04HypothesisGeneration
from src.pipeline.services.pathway_query_service import PathwayQueryService


# --------------------------------------------------------------------------- #
# Part A: tool-calling gate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mechanisms_result, expected", [
    # Normal case: mechanisms present.
    ({'pathway_mechanisms': [{'pathway': 'p'}], 'pathway_structures': [{'pathway': 'p'}]}, True),
    # The bug-4 case: structures present, mechanism interpretation empty.
    ({'pathway_mechanisms': [], 'pathway_structures': [{'pathway': 'p'}]}, True),
    # Mechanisms present but no structures (defensive).
    ({'pathway_mechanisms': [{'pathway': 'p'}], 'pathway_structures': []}, True),
    # Step 3 skipped (no genes) — must stay off even if a stray structure leaks in.
    ({'pathway_structures': [{'pathway': 'p'}], 'metadata': {'skipped': True}}, False),
    # Nothing usable.
    ({'pathway_mechanisms': [], 'pathway_structures': []}, False),
    ({}, False),
    (None, False),
])
def test_has_pathway_data(mechanisms_result, expected):
    assert Step04HypothesisGeneration._has_pathway_data(mechanisms_result) is expected


# --------------------------------------------------------------------------- #
# Part B: PathwayQueryService works (and surfaces relations) with empty mechanisms
# --------------------------------------------------------------------------- #

def _structures_only_step3():
    """A Step 3 output where retrieval succeeded but interpretation produced nothing."""
    structure = {
        'pathway': 'Butanoate metabolism',
        'pathway_id': 'hsa00650',
        'source': 'kegg',
        'confidence': 'high',
        'enrichment_score': -1.9,
        'enrichment_direction': 'downregulated',
        'p_value': 1e-3,
        'p_value_fdr': 1.75e-3,
        'de_genes_count': 2,
        'mapped_de_genes': [
            {'gene_symbol': 'ACAT1', 'fold_change': -1.2, 'p_value': 0.01, 'direction': 'down'},
            {'gene_symbol': 'DBT', 'fold_change': -1.06, 'p_value': 0.02, 'direction': 'down'},
        ],
        'de_relations': [
            {'source': 'ACAT1', 'target': 'DBT', 'type': 'ECrel', 'subtype': 'activation'},
        ],
    }
    return {
        'pathway_mechanisms': [],          # interpretation layer empty
        'pathway_structures': [structure],
        'pathway_overlaps': [],
    }


def test_get_pathway_mechanism_falls_back_to_structure_relations():
    svc = PathwayQueryService(_structures_only_step3())
    result = svc.get_pathway_mechanism('Butanoate metabolism')

    assert 'error' not in result
    assert result['pathway'] == 'Butanoate metabolism'
    assert result['source'] == 'kegg'
    assert result['de_genes_count'] == 2

    # Relations surfaced from the structure even though pathway_mechanisms was empty.
    rels = result['curatedRelations']
    assert len(rels) == 1
    assert (rels[0]['source'], rels[0]['target']) == ('ACAT1', 'DBT')
    assert rels[0]['type'] == 'activation'  # mapped from de_relations' subtype


def test_get_pathway_mechanism_derives_de_genes_from_structure():
    """With an empty mechanism layer, deGeneInvolvement is derived from the structure's
    real DE genes (symmetric with the curatedRelations fallback)."""
    svc = PathwayQueryService(_structures_only_step3())
    result = svc.get_pathway_mechanism('Butanoate metabolism')
    assert {g['gene'] for g in result['deGeneInvolvement']} == {'ACAT1', 'DBT'}


def test_get_pathway_mechanism_fallback_relations_isDE():
    """Fallback relations carry isDE, True only when both endpoints are DE genes."""
    svc = PathwayQueryService(_structures_only_step3())
    rels = svc.get_pathway_mechanism('Butanoate metabolism')['curatedRelations']
    assert rels[0]['isDE'] is True  # ACAT1 and DBT are both DE


def test_get_pathway_mechanism_fallback_isDE_false_for_non_de_endpoint():
    step3 = _structures_only_step3()
    step3['pathway_structures'][0]['de_relations'] = [
        {'source': 'ACAT1', 'target': 'ZZZ9', 'type': 'ECrel', 'subtype': 'activation'}
    ]
    svc = PathwayQueryService(step3)
    rels = svc.get_pathway_mechanism('Butanoate metabolism')['curatedRelations']
    assert rels[0]['isDE'] is False  # ZZZ9 is not a DE gene


def test_get_pathway_mechanism_excludes_placeholder_de_genes():
    """Placeholder genes (direction 'unknown') are excluded from the derived DE list."""
    step3 = _structures_only_step3()
    step3['pathway_structures'][0]['mapped_de_genes'] = [
        {'gene_symbol': 'ACAT1', 'fold_change': -1.2, 'p_value': 0.01, 'direction': 'down'},
        {'gene_symbol': 'PLACE', 'fold_change': 0.0, 'p_value': 1.0, 'direction': 'unknown'},
    ]
    svc = PathwayQueryService(step3)
    genes = {g['gene'] for g in svc.get_pathway_mechanism('Butanoate metabolism')['deGeneInvolvement']}
    assert genes == {'ACAT1'}


def test_search_pathways_by_gene_works_without_mechanisms():
    svc = PathwayQueryService(_structures_only_step3())
    result = svc.search_pathways_by_gene('ACAT1')
    assert result['pathways_found'] == 1
    assert result['pathways'][0]['pathway'] == 'Butanoate metabolism'
    assert result['pathways'][0]['direction'] == 'down'


def test_list_available_pathways_works_without_mechanisms():
    svc = PathwayQueryService(_structures_only_step3())
    result = svc.list_available_pathways()
    assert result['total_pathways'] == 1
    assert result['pathways'][0]['pathway_id'] == 'hsa00650'


def test_get_pathway_mechanism_uses_curated_when_present():
    """When the mechanism layer HAS curatedRelations, the structure fallback must NOT
    clobber them — locks in the `if not relations and structure` guard."""
    step3 = _structures_only_step3()
    step3['pathway_mechanisms'] = [{
        'pathway': 'Butanoate metabolism',
        'biologicalFunction': 'SCFA catabolism.',
        'curatedRelations': [
            {'source': 'ACAT1', 'target': 'OXCT1', 'type': 'expression', 'interpretation': 'x'}
        ],
    }]
    svc = PathwayQueryService(step3)
    rels = svc.get_pathway_mechanism('Butanoate metabolism')['curatedRelations']
    # The curated relation is returned, NOT the structure's de_relations fallback.
    assert len(rels) == 1
    assert (rels[0]['source'], rels[0]['target']) == ('ACAT1', 'OXCT1')
    assert rels[0]['type'] == 'expression'


def test_get_pathway_mechanism_fallback_type_falls_through_to_type():
    """If a structure relation has no subtype, the mapped type falls through to `type`."""
    step3 = _structures_only_step3()
    step3['pathway_structures'][0]['de_relations'] = [
        {'source': 'ACAT1', 'target': 'DBT', 'type': 'PPrel'},          # no subtype
        {'source': '', 'target': 'DBT', 'type': 'PPrel', 'subtype': 'x'},  # missing source -> dropped
    ]
    svc = PathwayQueryService(step3)
    rels = svc.get_pathway_mechanism('Butanoate metabolism')['curatedRelations']
    assert len(rels) == 1                       # the endpoint-less relation is filtered out
    assert rels[0]['type'] == 'PPrel'


def test_get_pathway_mechanism_not_found():
    svc = PathwayQueryService(_structures_only_step3())
    result = svc.get_pathway_mechanism('No Such Pathway 12345')
    assert 'error' in result


# --------------------------------------------------------------------------- #
# Execute-level wiring: _has_pathway_data must actually drive tool selection
# --------------------------------------------------------------------------- #

_EMPTY_HYP_JSON = ('{"hypotheses": [], "centralMechanisticModel": "", '
                   '"keyPredictions": [], "hypothesesSummary": ""}')


def _bare_step04():
    """Step 4 without its network-touching __init__; reviewer == llm so prewarm no-ops."""
    step = Step04HypothesisGeneration.__new__(Step04HypothesisGeneration)
    step.step_number = 4
    step.step_name = 'Mechanistic Hypothesis Generation'
    step.llm = MagicMock()
    step.reviewer_llm = step.llm
    return step


def test_execute_enables_tools_when_structures_only():
    """The bug-4 end-to-end: structures present + empty mechanisms -> tools enabled."""
    step = _bare_step04()
    step.llm.chat_with_tools.return_value = {
        'content': _EMPTY_HYP_JSON, 'tool_calls': None, 'finish_reason': 'stop'
    }
    mechanisms_result = {
        'pathway_mechanisms': [],
        'pathway_structures': _structures_only_step3()['pathway_structures'],
        'pathway_overlaps': [],
    }
    step.execute(
        genes=[], pathways=[], themes=None, hub_genes_result=None,
        mechanisms_result=mechanisms_result, analyses=None, context=None,
    )
    assert step.llm.chat_with_tools.called      # tool-calling path taken
    assert not step.llm.chat.called


def test_execute_disables_tools_when_step3_skipped():
    """Step 3 skipped -> no tools even if a stray structure is present."""
    step = _bare_step04()
    step.llm.chat.return_value = _EMPTY_HYP_JSON
    mechanisms_result = {
        'pathway_mechanisms': [],
        'pathway_structures': [{'pathway': 'p'}],
        'metadata': {'skipped': True},
    }
    step.execute(
        genes=[], pathways=[], themes=None, hub_genes_result=None,
        mechanisms_result=mechanisms_result, analyses=None, context=None,
    )
    assert step.llm.chat.called                 # plain-chat path taken
    assert not step.llm.chat_with_tools.called
