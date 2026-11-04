"""Tests for the step-1c tissue-specificity filter PROMPT (issue #15).

Two properties are locked here:

1. **Decoupling** — the filter prompt must not contain the step-1b theme name. That name is
   LLM-generated, none of the filter's own rules reference it, and it was the only channel by
   which upstream wording noise could change a keep/delete decision. Same pathways + same
   context must render a byte-identical prompt regardless of how step 1b happened to word the
   theme.

2. **Self-consistency** — the prompt must not put the same example on both the KEEP and the
   FILTER side. `MITOCHONDRIAL DYSFUNCTION` is a Rule 3 KEEP example, so it must not also be
   cited as the archetype of "do not keep this for a shared mechanism"; and the target study's
   own organ must not be listed as a FILTER exemplar.

Filtering itself must still work — the no-regression test at the bottom is what stops these
edits turning the filter into a no-op.
"""

import json
from unittest.mock import MagicMock

from src.pipeline.services.pathway_clustering_service import PathwayClusteringService
from src.pipeline.steps.step01_pathway_themes import Step01PathwayThemes

# The experimental context of the three divergent v3 runs.
HCC_CONTEXT = {'disease': 'Hepatocellular carcinoma (LIHC)', 'tissue': 'Liver'}

# The theme whose keep/delete decision was the ONLY thing that differed across those runs:
# run 2 kept Thermogenesis, runs 1 and 3 deleted it and lost the theme entirely.
MITO_PATHWAYS = [
    {'name': 'Thermogenesis', 'p_value_fdr': 1.23e-02, 'gene_count': 25},
    {'name': 'Alzheimer disease', 'p_value_fdr': 2.10e-02, 'gene_count': 31},
    {'name': 'Parkinson disease', 'p_value_fdr': 3.40e-02, 'gene_count': 22},
]


def _bare_step01():
    """Step 1 without __init__ — no LLM client, no network. Mirrors test_offtissue_filter.py."""
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    step.llm = MagicMock()
    # Tissue-filtering recomputes theme significance via the clustering classifier.
    step.clustering_service = PathwayClusteringService()
    return step


def _render_prompt(theme_name, pathways=MITO_PATHWAYS, context=HCC_CONTEXT, response=None):
    """Run the filter with a stubbed LLM and return the (system, user) prompt it built."""
    captured = []
    step = _bare_step01()

    def fake_chat_json(messages):
        captured.append(messages)
        return response if response is not None else json.dumps({'filtered_pathways': []})

    step._chat_json = fake_chat_json
    step._filter_off_tissue_pathways(
        themes=[{'name': theme_name, 'pathways': pathways,
                 'pathway_count': len(pathways), 'avg_p_value_fdr': 1e-2}],
        context=context)

    assert len(captured) == 1, 'expected exactly one filter call'
    messages = captured[0]
    system = next(m['content'] for m in messages if m['role'] == 'system')
    user = next(m['content'] for m in messages if m['role'] == 'user')
    return system, user


# --------------------------------------------------------------- decoupling

def test_prompt_is_byte_identical_under_different_theme_names():
    """The direct proof that the filter no longer depends on step 1b's wording.

    Both names below are real: the first is run 2's actual name for this cluster, the second
    is run 3's actual name for a different cluster in the same report set.
    """
    a = _render_prompt('Mitochondrial respiratory chain biogenesis and unfolded protein response')
    b = _render_prompt('Histone-driven chromatin remodeling and MAPK signaling activation')

    assert a[0] == b[0], 'system prompt varies with the theme name'
    assert a[1] == b[1], 'user prompt varies with the theme name'


def test_theme_name_absent_from_prompt():
    theme_name = 'Mitochondrial respiratory chain biogenesis and unfolded protein response'
    system, user = _render_prompt(theme_name)
    assert theme_name not in system
    assert theme_name not in user
    # The old wording, verbatim, must not come back.
    assert 'from the theme' not in user


def test_pathway_names_and_context_still_reach_the_prompt():
    """Decoupling must not have removed the inputs the filter actually decides from."""
    _, user = _render_prompt('any theme name at all')
    for pw in MITO_PATHWAYS:
        assert pw['name'] in user
    assert 'Hepatocellular carcinoma (LIHC)' in user
    assert 'Liver' in user


# ---------------------------------------------------------- self-consistency

def test_mitochondrial_dysfunction_is_only_a_keep_example():
    """Rule 3 lists MITOCHONDRIAL DYSFUNCTION as KEEP, so nothing may cite it as FILTER.

    Before the fix the motif appeared three times on the FILTER side (Rule 2 rationale,
    STRICT RULE 2, and 'mitochondrial genes' in the IMPORTANT block) against that one KEEP
    example — pointing both ways for exactly the cluster that failed.
    """
    system, user = _render_prompt('t')

    assert 'MITOCHONDRIAL DYSFUNCTION' in system, 'Rule 3 KEEP example was removed'

    combined = (system + user).lower()
    assert combined.count('mitochondrial') == 1, (
        'mitochondrial is cited more than once; the KEEP/FILTER collision is back')
    assert 'mitochondrial genes' not in combined


def test_target_organ_is_not_listed_as_a_filter_exemplar():
    """The DECISION CRITERIA organ list must not name the study's own organ as a FILTER trigger."""
    _, user = _render_prompt('t')
    assert 'PANCREATIC, HEPATOCELLULAR, GLIOMA' not in user
    assert "never filter the target's own organ" in user


# --------------------------------------------------------------- no-regression

def test_off_target_disease_pathways_are_still_filtered():
    """The fix must not turn the filter into a no-op."""
    step = _bare_step01()
    step._chat_json = lambda messages: json.dumps({'filtered_pathways': [
        {'pathway_name': 'Parkinson disease', 'decision': 'FILTER', 'rationale': 'off-disease'},
        {'pathway_name': 'Bladder cancer', 'decision': 'FILTER', 'rationale': 'off-organ'},
        {'pathway_name': 'Cell cycle', 'decision': 'KEEP', 'rationale': 'general mechanism'},
    ]})

    cleaned, filtered = step._filter_off_tissue_pathways(
        themes=[{'name': 'T', 'avg_p_value_fdr': 1e-3, 'pathway_count': 3, 'pathways': [
            {'name': 'Parkinson disease', 'p_value_fdr': 3.4e-2, 'gene_count': 22},
            {'name': 'Bladder cancer', 'p_value_fdr': 1.0e-2, 'gene_count': 5},
            {'name': 'Cell cycle', 'p_value_fdr': 3.76e-4, 'gene_count': 40},
        ]}],
        context=HCC_CONTEXT)

    assert filtered == 2
    assert [p['name'] for p in cleaned[0]['pathways']] == ['Cell cycle']
