"""chat(require_complete=True) never hands back truncated or reasoning-only text; the
default path is exactly the old behaviour (the single route depends on it)."""
import json
from unittest.mock import MagicMock, patch

import pytest

import src.agents.llm_client as llm_mod
from src.agents.llm_client import (
    LLMEmptyContentError, LLMTruncatedError, UnifiedLLMClient,
)


def _response(content, finish_reason='stop', reasoning=None):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    message = {'content': content}
    if reasoning is not None:
        message['reasoning_content'] = reasoning
    body = {'choices': [{'message': message, 'finish_reason': finish_reason}]}
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _stream_response(chunks, finish_reason):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    lines = [f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}'.encode() for c in chunks]
    lines.append(f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": finish_reason}]})}'.encode())
    lines.append(b'data: [DONE]')
    resp.iter_lines.return_value = lines
    return resp


def _chat(responses, **kwargs):
    client = UnifiedLLMClient(provider='local')
    payloads = []

    def fake_post(url, **kw):
        payloads.append(kw['json'])
        return responses.pop(0)

    with patch.object(llm_mod.requests, 'post', side_effect=fake_post):
        result = client.chat([{'role': 'user', 'content': 'x'}], max_tokens=100, **kwargs)
    return result, payloads


# --- default path: unchanged ---------------------------------------------------------

def test_default_returns_truncated_content_as_before():
    result, payloads = _chat([_response('cut off mid', finish_reason='length')])
    assert result == 'cut off mid'
    assert len(payloads) == 1


def test_default_still_falls_back_to_reasoning_when_content_is_none():
    result, _ = _chat([_response(None, reasoning='thinking')])
    assert result == 'thinking'


def test_default_stream_unchanged():
    result, _ = _chat([_stream_response(['a', 'b'], 'length')], stream=True)
    assert result == 'ab'


# --- require_complete ------------------------------------------------------------------

def test_complete_reply_is_returned():
    result, payloads = _chat([_response('done')], require_complete=True)
    assert result == 'done'
    assert len(payloads) == 1


def test_length_is_retried_once_with_double_budget():
    result, payloads = _chat([_response('cut', finish_reason='length'), _response('full answer')],
                             require_complete=True)
    assert result == 'full answer'
    assert [p['max_tokens'] for p in payloads] == [100, 200]


def test_length_twice_raises():
    with pytest.raises(LLMTruncatedError):
        _chat([_response('cut', finish_reason='length'), _response('cut', finish_reason='length')],
              require_complete=True)


def test_empty_content_raises_and_never_returns_reasoning():
    with pytest.raises(LLMEmptyContentError):
        _chat([_response(None, reasoning='internal reasoning')], require_complete=True)
    with pytest.raises(LLMEmptyContentError):
        _chat([_response('   ')], require_complete=True)


def test_stream_length_is_detected():
    with pytest.raises(LLMTruncatedError):
        _chat([_stream_response(['a'], 'length'), _stream_response(['a'], 'length')],
              stream=True, require_complete=True)
    result, _ = _chat([_stream_response(['a', 'b'], 'stop')], stream=True, require_complete=True)
    assert result == 'ab'


def test_complete_passes_require_complete_through():
    client = UnifiedLLMClient(provider='local')
    with patch.object(llm_mod.requests, 'post', return_value=_response(None, reasoning='r')):
        assert client.complete('x') == 'r'
        with pytest.raises(LLMEmptyContentError):
            client.complete('x', require_complete=True)
