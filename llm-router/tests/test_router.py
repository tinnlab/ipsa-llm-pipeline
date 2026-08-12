"""Tests for the sleep-aware llm-router.

Two layers:
  * HTTP route tests (health, /v1/models, unknown-model 400) via FastAPI TestClient —
    these never touch an upstream.
  * SleepRouter logic tests (is_sleeping / ensure_awake / maybe_sleep / forward) driven
    directly with respx mocking the vLLM dev endpoints. We avoid mixing TestClient with
    respx so respx never intercepts the test client's own request to the app.

The default fixture uses the production shape — a **single** upstream, the local OpenBioLLM
reviewer (gpt-oss is served externally and is not routed here). The router is still generic
over N upstreams, so ``test_forward_routes_by_model`` builds its own two-upstream map to
keep the route-by-model behaviour covered.
"""

import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from starlette.requests import Request

import app as app_module
from app import SleepRouter, create_app, parse_upstreams

BIO = "aaditya/Llama3-OpenBioLLM-70B"
BIO_URL = "http://vllm-openbio:8002"
UPSTREAMS = {BIO: BIO_URL}  # production shape: one local upstream

# A second, hypothetical upstream — used only where multi-upstream behaviour is under test.
OTHER = "vendor/other-model"
OTHER_URL = "http://vllm-other:8003"


def make_router(**kwargs) -> SleepRouter:
    client = httpx.AsyncClient()
    return SleepRouter(UPSTREAMS, client, **kwargs)


def make_request(body: bytes, query_string: bytes = b"") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": query_string,
        "headers": [(b"content-type", b"application/json")],
    }
    state = {"sent": False}

    async def receive():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


async def read_stream(streaming_response) -> bytes:
    return b"".join([chunk async for chunk in streaming_response.body_iterator])


# --------------------------------------------------------------------------- #
# parse_upstreams
# --------------------------------------------------------------------------- #
def test_parse_upstreams_single_pair():
    """The deployed UPSTREAMS value — one model, no comma."""
    assert parse_upstreams(f"{BIO}={BIO_URL}") == {BIO: BIO_URL}


def test_parse_upstreams_ok():
    parsed = parse_upstreams(f"{BIO}={BIO_URL}, {OTHER}={OTHER_URL}/")
    assert parsed == {BIO: BIO_URL, OTHER: OTHER_URL}  # trailing slash stripped


@pytest.mark.parametrize("raw", ["", "  ", "no-equals-sign", "=http://x", "model="])
def test_parse_upstreams_invalid(raw):
    with pytest.raises(ValueError):
        parse_upstreams(raw)


# --------------------------------------------------------------------------- #
# HTTP routes (no upstream involved)
# --------------------------------------------------------------------------- #
@pytest.fixture
def client():
    router = SleepRouter(UPSTREAMS, httpx.AsyncClient())
    with TestClient(create_app(router=router, run_sleeper=False)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_lists_configured_ids_without_waking(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    ids = {m["id"] for m in r.json()["data"]}
    assert ids == {BIO}


def test_unknown_model_returns_400(client):
    r = client.post("/v1/chat/completions", json={"model": "does-not-exist", "messages": []})
    assert r.status_code == 400


def test_non_json_body_returns_400(client):
    r = client.post("/v1/chat/completions", content=b"not json",
                    headers={"content-type": "application/json"})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# is_sleeping / ensure_awake
# --------------------------------------------------------------------------- #
@respx.mock
async def test_is_sleeping_true_false():
    router = make_router()
    up = router.upstreams[BIO]
    route = respx.get(f"{BIO_URL}/is_sleeping")
    route.return_value = httpx.Response(200, json={"is_sleeping": True})
    assert await router.is_sleeping(up) is True
    route.return_value = httpx.Response(200, json={"is_sleeping": False})
    assert await router.is_sleeping(up) is False


@respx.mock
async def test_is_sleeping_error_assumes_awake():
    router = make_router()
    respx.get(f"{BIO_URL}/is_sleeping").mock(side_effect=httpx.ConnectError("down"))
    assert await router.is_sleeping(router.upstreams[BIO]) is False


@respx.mock
async def test_ensure_awake_wakes_when_sleeping():
    router = make_router()
    sleeping = respx.get(f"{BIO_URL}/is_sleeping").mock(
        side_effect=[
            httpx.Response(200, json={"is_sleeping": True}),   # initial check -> wake
            httpx.Response(200, json={"is_sleeping": False}),  # poll after wake -> done
        ]
    )
    wake = respx.post(f"{BIO_URL}/wake_up").mock(return_value=httpx.Response(200, json={}))
    await router.ensure_awake(router.upstreams[BIO])
    assert wake.called
    assert sleeping.call_count == 2


@respx.mock
async def test_ensure_awake_noop_when_awake():
    router = make_router()
    respx.get(f"{BIO_URL}/is_sleeping").mock(return_value=httpx.Response(200, json={"is_sleeping": False}))
    wake = respx.post(f"{BIO_URL}/wake_up")
    await router.ensure_awake(router.upstreams[BIO])
    assert not wake.called  # no needless wake


# --------------------------------------------------------------------------- #
# maybe_sleep
# --------------------------------------------------------------------------- #
@respx.mock
async def test_maybe_sleep_sleeps_when_idle():
    router = make_router(idle_timeout=10.0, sleep_level=1)
    up = router.upstreams[BIO]
    up.last_used = up._clock() - 1000  # long idle
    respx.get(f"{BIO_URL}/is_sleeping").mock(return_value=httpx.Response(200, json={"is_sleeping": False}))
    sleep = respx.post(f"{BIO_URL}/sleep").mock(return_value=httpx.Response(200, json={}))
    await router.maybe_sleep(up)
    assert sleep.called
    assert "level=1" in str(sleep.calls.last.request.url)


@respx.mock
async def test_maybe_sleep_skips_when_inflight():
    router = make_router(idle_timeout=10.0)
    up = router.upstreams[BIO]
    up.last_used = up._clock() - 1000
    up.inflight = 1  # a request is in progress
    sleep = respx.post(f"{BIO_URL}/sleep")
    await router.maybe_sleep(up)
    assert not sleep.called


@respx.mock
async def test_maybe_sleep_skips_when_recently_used():
    router = make_router(idle_timeout=600.0)
    up = router.upstreams[BIO]  # last_used = now -> not idle
    sleep = respx.post(f"{BIO_URL}/sleep")
    await router.maybe_sleep(up)
    assert not sleep.called


@respx.mock
async def test_maybe_sleep_skips_when_already_sleeping():
    router = make_router(idle_timeout=10.0)
    up = router.upstreams[BIO]
    up.last_used = up._clock() - 1000
    respx.get(f"{BIO_URL}/is_sleeping").mock(return_value=httpx.Response(200, json={"is_sleeping": True}))
    sleep = respx.post(f"{BIO_URL}/sleep")
    await router.maybe_sleep(up)
    assert not sleep.called


# --------------------------------------------------------------------------- #
# forward — routing + streaming passthrough
# --------------------------------------------------------------------------- #
@respx.mock
async def test_forward_routes_by_model():
    """With more than one upstream, the request goes to the one matching its "model"."""
    router = SleepRouter({BIO: BIO_URL, OTHER: OTHER_URL}, httpx.AsyncClient())
    respx.get(f"{BIO_URL}/is_sleeping").mock(return_value=httpx.Response(200, json={"is_sleeping": False}))
    chat = respx.post(f"{BIO_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"served_by": "bio"})
    )
    # the other upstream should NOT be touched
    other_chat = respx.post(f"{OTHER_URL}/v1/chat/completions")

    req = make_request(json.dumps({"model": BIO, "messages": []}).encode())
    resp = await router.forward(req, "v1/chat/completions")
    body = json.loads(await read_stream(resp))

    assert resp.status_code == 200
    assert body == {"served_by": "bio"}
    assert chat.called and not other_chat.called


@respx.mock
async def test_forward_wakes_then_forwards():
    router = make_router()
    respx.get(f"{BIO_URL}/is_sleeping").mock(
        side_effect=[
            httpx.Response(200, json={"is_sleeping": True}),
            httpx.Response(200, json={"is_sleeping": False}),
        ]
    )
    wake = respx.post(f"{BIO_URL}/wake_up").mock(return_value=httpx.Response(200, json={}))
    chat = respx.post(f"{BIO_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    req = make_request(json.dumps({"model": BIO, "messages": []}).encode())
    resp = await router.forward(req, "v1/chat/completions")
    await read_stream(resp)

    assert wake.called and chat.called


@respx.mock
async def test_forward_decrements_inflight_after_stream():
    router = make_router()
    up = router.upstreams[BIO]
    respx.get(f"{BIO_URL}/is_sleeping").mock(return_value=httpx.Response(200, json={"is_sleeping": False}))
    respx.post(f"{BIO_URL}/v1/chat/completions").mock(return_value=httpx.Response(200, json={"ok": True}))

    req = make_request(json.dumps({"model": BIO, "messages": []}).encode())
    resp = await router.forward(req, "v1/chat/completions")
    assert up.inflight == 1  # held while the response streams
    await read_stream(resp)
    assert up.inflight == 0  # released once fully streamed


@respx.mock
async def test_forward_propagates_upstream_error_status():
    router = make_router()
    respx.get(f"{BIO_URL}/is_sleeping").mock(return_value=httpx.Response(200, json={"is_sleeping": False}))
    respx.post(f"{BIO_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": "max_tokens too large"})
    )
    req = make_request(json.dumps({"model": BIO, "messages": []}).encode())
    resp = await router.forward(req, "v1/chat/completions")
    assert resp.status_code == 400  # upstream status propagated, not turned into a 5xx
    assert json.loads(await read_stream(resp)) == {"error": "max_tokens too large"}
    assert router.upstreams[BIO].inflight == 0  # released after the error body is streamed
