"""Fix 1a: the LLM request timeout is configurable and applied to the actual requests.post
calls (so a reviewer call can wait through an on-demand cold start instead of the old 120 s)."""
from unittest.mock import MagicMock, patch

import src.agents.llm_client as llm_mod
from src.agents.llm_client import UnifiedLLMClient
from src.config import settings


def _fake_post_capturing(captured):
    def fake_post(url, **kwargs):
        captured.update(kwargs)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
        return resp
    return fake_post


def test_config_default_timeout_is_600():
    assert settings.LLM_REQUEST_TIMEOUT == 600


def test_client_picks_up_configured_timeout():
    assert UnifiedLLMClient(provider="local").timeout == settings.LLM_REQUEST_TIMEOUT


def test_chat_passes_configured_timeout_to_requests():
    client = UnifiedLLMClient(provider="reviewer")  # the bio-model path
    captured = {}
    with patch.object(llm_mod.requests, "post", side_effect=_fake_post_capturing(captured)):
        try:
            client.chat([{"role": "user", "content": "x"}])
        except Exception:
            pass  # response parsing details are irrelevant; we asserted the call kwargs
    assert captured.get("timeout") == settings.LLM_REQUEST_TIMEOUT == 600


def test_chat_with_tools_passes_configured_timeout():
    client = UnifiedLLMClient(provider="local")
    captured = {}
    with patch.object(llm_mod.requests, "post", side_effect=_fake_post_capturing(captured)):
        try:
            client.chat_with_tools([{"role": "user", "content": "x"}], tools=[])
        except Exception:
            pass
    assert captured.get("timeout") == 600
