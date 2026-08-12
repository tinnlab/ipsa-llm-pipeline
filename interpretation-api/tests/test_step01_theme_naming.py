"""
Tests for Step 1 pathway theme naming robustness.

Regression coverage for the bug where every theme rendered as "Pathway Cluster N"
with "LLM interpretation unavailable": the Step-1 LLM naming call failed to parse
(reasoning model wrapped JSON in fences / added prose) and the whole-batch fallback
fired. Also covers the milder case where *some* clusters got generic names because
the LLM returned cluster_number as a string or omitted it.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.steps.step01_pathway_themes import (
    Step01PathwayThemes,
    _parse_llm_json,
    _coerce_int,
    _fallback_theme_name,
)


def _bare_step01():
    step = Step01PathwayThemes.__new__(Step01PathwayThemes)
    step.llm = MagicMock()
    return step


def _cluster(shared_genes, pathway_names=('Cell cycle',)):
    return {
        'pathways': [{'name': n, 'p_value_fdr': 1e-3, 'gene_count': 10} for n in pathway_names],
        'pathway_names': list(pathway_names),
        'pathway_count': len(pathway_names),
        'shared_genes': list(shared_genes),
        'shared_gene_count': len(shared_genes),
        'avg_jaccard_overlap': 0.3,
        'avg_p_value_fdr': 1e-3,
        'significance': 'high',
        'key_genes_with_fc': [],
        'dominant_direction': 'up',
    }


# ---------------------------------------------------------------------------
# _parse_llm_json
# ---------------------------------------------------------------------------

def test_parse_plain_json():
    assert _parse_llm_json('{"themes": [1]}') == {'themes': [1]}


def test_parse_fenced_json():
    resp = '```json\n{"themes": [{"cluster_number": 1, "name": "Cell cycle"}]}\n```'
    out = _parse_llm_json(resp)
    assert out['themes'][0]['name'] == 'Cell cycle'


def test_parse_prose_prefixed_json():
    resp = 'Here is the JSON you asked for:\n{"themes": [], "themes_summary": "ok"}'
    assert _parse_llm_json(resp) == {'themes': [], 'themes_summary': 'ok'}


def test_parse_only_reasoning_text_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_llm_json('I was unable to produce valid output for this request.')


def test_parse_empty_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_llm_json('')


def test_coerce_int_variants():
    assert _coerce_int(1) == 1
    assert _coerce_int('2') == 2
    assert _coerce_int(3.0) == 3
    assert _coerce_int(None) is None
    assert _coerce_int('x') is None
    assert _coerce_int(True) is None  # bool must not be treated as a number


# ---------------------------------------------------------------------------
# _name_clusters_with_llm — end to end with mocked LLM
# ---------------------------------------------------------------------------

def test_fenced_response_still_names_themes():
    step = _bare_step01()
    clusters = [_cluster(['CDK1', 'CCNB1'])]
    step.llm.chat.return_value = (
        '```json\n{"themes": [{"cluster_number": 1, "name": "Cell cycle control", '
        '"description": "d", "significance": "high", "key_genes": ["CDK1"]}]}\n```'
    )
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert themes[0]['name'] == 'Cell cycle control'
    assert 'Pathway Cluster' not in themes[0]['name']


def test_response_format_is_requested():
    step = _bare_step01()
    step.llm.chat.return_value = '{"themes": []}'
    step._name_clusters_with_llm([_cluster(['CDK1'])], context=None, organism='Homo sapiens')
    _, kwargs = step.llm.chat.call_args
    assert kwargs.get('response_format') == {'type': 'json_object'}


def test_string_cluster_number_still_matches():
    step = _bare_step01()
    clusters = [_cluster(['CDK1']), _cluster(['COX6C'], pathway_names=('OXPHOS',))]
    step.llm.chat.return_value = json.dumps({'themes': [
        {'cluster_number': '1', 'name': 'Cell cycle'},
        {'cluster_number': '2', 'name': 'Oxidative phosphorylation'},
    ]})
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert [t['name'] for t in themes] == ['Cell cycle', 'Oxidative phosphorylation']


def test_missing_cluster_number_positional_fallback():
    step = _bare_step01()
    clusters = [_cluster(['CDK1']), _cluster(['COX6C'], pathway_names=('OXPHOS',))]
    # LLM omitted cluster_number entirely but returned one theme per cluster in order
    step.llm.chat.return_value = json.dumps({'themes': [
        {'name': 'Cell cycle'},
        {'name': 'Oxidative phosphorylation'},
    ]})
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert [t['name'] for t in themes] == ['Cell cycle', 'Oxidative phosphorylation']


def test_fewer_themes_than_clusters_only_unmatched_falls_back():
    step = _bare_step01()
    clusters = [_cluster(['CDK1']), _cluster(['TP53', 'MDM2'], pathway_names=('p53',))]
    # LLM only named cluster 1
    step.llm.chat.return_value = json.dumps({'themes': [
        {'cluster_number': 1, 'name': 'Cell cycle'},
    ]})
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert themes[0]['name'] == 'Cell cycle'
    # unmatched cluster gets an informative (shared-gene) fallback, not a bare number
    assert themes[1]['name'].startswith('Pathway Cluster 2 (shared genes:')
    assert 'TP53' in themes[1]['name']


def test_total_parse_failure_uses_shared_gene_fallback():
    step = _bare_step01()
    clusters = [_cluster(['CDK1', 'CCNB1', 'AURKB'])]
    step.llm.chat.return_value = 'sorry, no json here {{{'
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert themes[0]['name'].startswith('Pathway Cluster 1 (shared genes:')
    assert 'CDK1' in themes[0]['name']
    assert themes[0]['biological_context'] == 'LLM interpretation unavailable'


def test_empty_name_falls_back():
    step = _bare_step01()
    clusters = [_cluster(['CDK1'])]
    step.llm.chat.return_value = json.dumps({'themes': [
        {'cluster_number': 1, 'name': '   '},  # blank name
    ]})
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert themes[0]['name'].startswith('Pathway Cluster 1 (shared genes:')


# ---------------------------------------------------------------------------
# response_format reaches the HTTP payload (llm_client)
# ---------------------------------------------------------------------------

def test_llm_client_forwards_response_format():
    from src.agents.llm_client import UnifiedLLMClient

    client = UnifiedLLMClient(provider='local')
    fake = MagicMock()
    fake.json.return_value = {'choices': [{'message': {'content': '{}'}}]}
    fake.raise_for_status.return_value = None
    with patch('src.agents.llm_client.requests.post', return_value=fake) as post:
        client.chat([{'role': 'user', 'content': 'hi'}],
                    response_format={'type': 'json_object'})
    sent = post.call_args.kwargs['json']
    assert sent['response_format'] == {'type': 'json_object'}


def test_llm_client_omits_response_format_by_default():
    from src.agents.llm_client import UnifiedLLMClient

    client = UnifiedLLMClient(provider='local')
    fake = MagicMock()
    fake.json.return_value = {'choices': [{'message': {'content': '{}'}}]}
    fake.raise_for_status.return_value = None
    with patch('src.agents.llm_client.requests.post', return_value=fake) as post:
        client.chat([{'role': 'user', 'content': 'hi'}])
    assert 'response_format' not in post.call_args.kwargs['json']


# ---------------------------------------------------------------------------
# Audit regressions
# ---------------------------------------------------------------------------

def test_none_response_does_not_crash():
    """F1: a None LLM response must degrade to fallback names, not TypeError."""
    step = _bare_step01()
    clusters = [_cluster(['CDK1', 'CCNB1'])]
    step.llm.chat.return_value = None  # e.g. reasoning model returned no content
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert themes[0]['name'].startswith('Pathway Cluster 1 (shared genes:')


def test_chat_json_retries_without_response_format_on_error():
    """F1/F4: if the provider rejects response_format, retry without it (no hard failure)."""
    step = _bare_step01()
    calls = []

    def chat(messages, **kwargs):
        calls.append(kwargs)
        if 'response_format' in kwargs:
            raise Exception('response_format not supported')
        return '{"themes": []}'

    step.llm.chat.side_effect = chat
    out = step._chat_json([{'role': 'user', 'content': 'x'}])
    assert out == '{"themes": []}'
    assert 'response_format' in calls[0] and 'response_format' not in calls[1]


def test_mixed_cluster_number_no_duplicate_or_drop():
    """F4: some themes numbered, some not — must not assign one theme to two clusters."""
    step = _bare_step01()
    clusters = [_cluster(['CDK1']), _cluster(['COX6C'], pathway_names=('OXPHOS',))]
    # theme for cluster 2 is numbered; the other has no number. Positional fallback
    # must NOT fire (mixed), so cluster 1 gets a real fallback and B is not duplicated.
    step.llm.chat.return_value = json.dumps({'themes': [
        {'name': 'A-no-number'},
        {'cluster_number': 2, 'name': 'Oxidative phosphorylation'},
    ]})
    themes = step._name_clusters_with_llm(clusters, context=None, organism='Homo sapiens')
    assert themes[1]['name'] == 'Oxidative phosphorylation'
    # cluster 0 must NOT be labelled 'A-no-number' via a bogus positional match,
    # and 'Oxidative phosphorylation' must appear exactly once
    names = [t['name'] for t in themes]
    assert names.count('Oxidative phosphorylation') == 1
    assert themes[0]['name'].startswith('Pathway Cluster 1')


def test_singleton_filter_runs_on_no_cluster_path():
    """F2: when no clusters form, off-tissue singletons must still be filtered."""
    step = _bare_step01()
    step.clustering_service = MagicMock()
    step.clustering_service.cluster_pathways_by_gene_overlap.return_value = {
        'clusters': [],
        'singletons': [
            {'name': 'Bladder cancer', 'p_value_fdr': 0.01, 'gene_count': 5},
            {'name': 'p53 signaling', 'p_value_fdr': 1e-3, 'gene_count': 14},
        ],
        'metadata': {},
    }
    step.llm.chat.return_value = json.dumps({'filtered_pathways': [
        {'pathway_name': 'Bladder cancer', 'decision': 'FILTER', 'rationale': 'off-organ'},
        {'pathway_name': 'p53 signaling', 'decision': 'KEEP', 'rationale': 'general'},
    ]})
    result = step.execute(
        pathways=[{'name': 'Bladder cancer', 'p_value_fdr': 0.01, 'genes': ['X']},
                  {'name': 'p53 signaling', 'p_value_fdr': 1e-3, 'genes': ['TP53']}],
        genes=[{'gene': 'TP53', 'foldChange': 2.0}],
        context={'disease': 'Hepatocellular carcinoma', 'tissue': 'liver'},
    )
    names = [p['name'] for p in result['ungrouped']]
    assert 'Bladder cancer' not in names
    assert 'p53 signaling' in names
