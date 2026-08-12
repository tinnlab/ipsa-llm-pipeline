"""
Tests for report bug 5: theme blurbs inverted regulation direction (a down-regulated
beta-oxidation cluster described as "a shift toward oxidative fuel utilization") because
the computed ``dominant_direction`` was never passed into the theme-naming prompt. The
fix threads direction + signed fold changes into the cluster description and instructs
the model to respect it.
"""

from unittest.mock import MagicMock

from src.pipeline.steps.step01_pathway_themes import Step01PathwayThemes


def _bare_step01():
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    step.llm = MagicMock()
    return step


def _down_cluster():
    return {
        'pathways': [{'name': 'Fatty acid degradation'}],
        'pathway_names': ['Fatty acid degradation'],
        'pathway_count': 1,
        'shared_genes': ['ACADS', 'ACAT1', 'ACAA2'],
        'shared_gene_count': 3,
        'avg_jaccard_overlap': 0.3,
        'avg_p_value_fdr': 1.45e-2,
        'significance': 'medium',
        'key_genes_with_fc': [
            {'gene': 'ACADS', 'fold_change': -1.41, 'direction': 'down'},
            {'gene': 'ACAT1', 'fold_change': -1.02, 'direction': 'down'},
        ],
        'dominant_direction': 'down',
    }


def _captured_user_prompt(step):
    # _name_clusters_with_llm calls self._chat_json([system, user]); capture the user msg.
    captured = {}

    def fake_chat_json(messages):
        captured['messages'] = messages
        return '{"themes": []}'

    step._chat_json = fake_chat_json
    step._name_clusters_with_llm([_down_cluster()], {'disease': 'HCC'}, 'Homo sapiens')
    user_msg = next(m['content'] for m in captured['messages'] if m['role'] == 'user')
    return user_msg


def test_direction_passed_into_naming_prompt():
    prompt = _captured_user_prompt(_bare_step01())
    assert 'DOWN-regulated' in prompt
    assert 'DECREASED' in prompt


def test_signed_fold_changes_passed_into_prompt():
    prompt = _captured_user_prompt(_bare_step01())
    assert 'ACADS (-1.41)' in prompt


def test_naming_system_prompt_instructs_direction_respect():
    # The direction-respect instruction is carried in the system prompt.
    step = _bare_step01()
    captured = {}

    def fake_chat_json(messages):
        captured['messages'] = messages
        return '{"themes": []}'

    step._chat_json = fake_chat_json
    step._name_clusters_with_llm([_down_cluster()], None, 'Homo sapiens')
    system_msg = next(m['content'] for m in captured['messages'] if m['role'] == 'system')
    assert 'regulation direction' in system_msg.lower()


def test_fallback_themes_carry_direction():
    # On total LLM-naming failure the fallback must still carry dominant_direction/FCs,
    # or downstream steps regress to 'mixed' and reintroduce the inversion bug.
    step = _bare_step01()
    themes = step._generate_fallback_themes([_down_cluster()])
    assert themes[0]['dominant_direction'] == 'down'
    assert themes[0]['key_genes_with_fc']  # non-empty, carried through
