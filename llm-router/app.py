"""Sleep-aware OpenAI-compatible router for the IPSA vLLM models.

Rather than starting and stopping whole containers to free a GPU (a multi-minute cold
start), the local vLLM models run **always-on** with vLLM sleep mode enabled
(`--enable-sleep-mode` + `VLLM_SERVER_DEV_MODE=1`). This router keeps the same
`:9000` OpenAI-compatible entrypoint and:

  * routes each request to the right upstream by its ``model`` field,
  * wakes a sleeping model on demand (weights RAM->GPU, ~seconds) before forwarding,
  * sleeps a model after ``IDLE_TIMEOUT_SECONDS`` of inactivity (level 1: weights->RAM).

Consumers POST to ``/v1/chat/completions`` on :9000 and are routed by the request's
``model`` field.

The router is generic over any number of upstreams; in the default deployment it fronts a
**single** one, the local OpenBioLLM reviewer. The main reasoning model (gpt-oss) is called
directly by pipeline-api and is deliberately NOT an upstream here: the router owns a
sleep/wake lifecycle and must never manage a server this stack does not own.

Config (env):
  UPSTREAMS              comma-separated ``model=url`` pairs, e.g.
                         ``aaditya/Llama3-OpenBioLLM-70B=http://vllm-openbio:8002``
  SLEEP_LEVEL            vLLM sleep level (default 1 — offload weights to CPU RAM)
  IDLE_TIMEOUT_SECONDS   idle seconds before a model is auto-slept (default 600)
  ROUTER_PORT            listen port (default 9000; used by the uvicorn entrypoint)
  WAKE_TIMEOUT_SECONDS   max seconds to wait for a wake to complete (default 300)
  SLEEPER_INTERVAL_SECONDS  how often the idle checker runs (default 30)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("llm-router")

# Headers we must not blindly copy between the client<->router<->upstream hops.
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailers", "transfer-encoding", "upgrade"}


def parse_upstreams(raw: str) -> Dict[str, str]:
    """Parse ``model=url,model=url`` into ``{model: url}`` (urls have trailing / stripped)."""
    upstreams: Dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"Invalid UPSTREAMS entry (expected model=url): {pair!r}")
        model, url = pair.split("=", 1)
        model, url = model.strip(), url.strip().rstrip("/")
        if not model or not url:
            raise ValueError(f"Invalid UPSTREAMS entry (empty model or url): {pair!r}")
        upstreams[model] = url
    if not upstreams:
        raise ValueError("UPSTREAMS is empty — set it to 'model=url,model=url'")
    return upstreams


class Upstream:
    """Tracks the liveness/usage state of a single vLLM backend."""

    def __init__(self, model: str, url: str, clock: Callable[[], float] = time.monotonic):
        self.model = model
        self.url = url
        self._clock = clock
        self.last_used = clock()
        self.inflight = 0
        # Serializes sleep<->wake transitions so a wake and a sleep can't race.
        self.wake_lock = asyncio.Lock()

    def touch(self) -> None:
        self.last_used = self._clock()

    def idle_seconds(self) -> float:
        return self._clock() - self.last_used


class SleepRouter:
    def __init__(
        self,
        upstreams: Dict[str, str],
        client: httpx.AsyncClient,
        sleep_level: int = 1,
        idle_timeout: float = 600.0,
        wake_timeout: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.upstreams: Dict[str, Upstream] = {
            model: Upstream(model, url, clock) for model, url in upstreams.items()
        }
        self.client = client
        self.sleep_level = sleep_level
        self.idle_timeout = idle_timeout
        self.wake_timeout = wake_timeout
        self._clock = clock

    # -- routing ---------------------------------------------------------------
    def get_upstream(self, model: Optional[str]) -> Upstream:
        up = self.upstreams.get(model) if model else None
        if up is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model {model!r}. Known models: {list(self.upstreams)}",
            )
        return up

    # -- sleep/wake primitives -------------------------------------------------
    async def is_sleeping(self, up: Upstream) -> bool:
        """Best-effort sleep check. On error assume awake — forwarding surfaces real errors."""
        try:
            r = await self.client.get(f"{up.url}/is_sleeping", timeout=10.0)
            r.raise_for_status()
            return bool(r.json().get("is_sleeping", False))
        except Exception as exc:  # noqa: BLE001
            logger.warning("is_sleeping check failed for %s: %s", up.model, exc)
            return False

    async def ensure_awake(self, up: Upstream) -> None:
        """Wake the upstream if it is sleeping, blocking until it reports awake."""
        async with up.wake_lock:
            if not await self.is_sleeping(up):
                return
            logger.info("Waking %s ...", up.model)
            await self.client.post(f"{up.url}/wake_up", timeout=self.wake_timeout)
            deadline = self._clock() + self.wake_timeout
            while await self.is_sleeping(up):
                if self._clock() > deadline:
                    raise HTTPException(status_code=503, detail=f"Timed out waking {up.model}")
                await asyncio.sleep(0.5)
            logger.info("%s is awake", up.model)

    async def maybe_sleep(self, up: Upstream) -> None:
        """Sleep the upstream if it is idle and has no in-flight requests."""
        if up.inflight > 0 or up.idle_seconds() < self.idle_timeout:
            return
        if await self.is_sleeping(up):
            return
        async with up.wake_lock:
            # Re-check under the lock — a request may have arrived meanwhile.
            if up.inflight > 0 or up.idle_seconds() < self.idle_timeout:
                return
            logger.info("Sleeping %s (idle %.0fs, level %d)", up.model, up.idle_seconds(), self.sleep_level)
            try:
                await self.client.post(
                    f"{up.url}/sleep", params={"level": self.sleep_level}, timeout=self.wake_timeout
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("sleep failed for %s: %s", up.model, exc)

    async def sleeper_loop(self, interval: float = 30.0) -> None:
        while True:
            await asyncio.sleep(interval)
            for up in self.upstreams.values():
                try:
                    await self.maybe_sleep(up)
                except Exception:  # noqa: BLE001
                    logger.exception("sleeper error for %s", up.model)

    # -- request forwarding ----------------------------------------------------
    async def forward(self, request: Request, path: str) -> StreamingResponse:
        body = await request.body()
        model = _extract_model(body)
        up = self.get_upstream(model)
        await self.ensure_awake(up)

        target = f"{up.url}/{path}"
        upstream_req = self.client.build_request(
            request.method,
            target,
            content=body,
            headers=_forward_request_headers(request.headers),
            params=request.query_params,
        )
        up.inflight += 1
        up.touch()
        try:
            upstream_resp = await self.client.send(upstream_req, stream=True)
        except Exception:
            up.inflight -= 1
            up.touch()
            raise

        async def body_iter():
            try:
                async for chunk in upstream_resp.aiter_raw():
                    yield chunk
            finally:
                await upstream_resp.aclose()
                up.inflight -= 1
                up.touch()

        return StreamingResponse(
            body_iter(),
            status_code=upstream_resp.status_code,
            headers=_forward_response_headers(upstream_resp.headers),
            media_type=upstream_resp.headers.get("content-type"),
        )


def _extract_model(body: bytes) -> Optional[str]:
    try:
        return json.loads(body).get("model")
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="Request body must be JSON containing a 'model' field")


def _forward_request_headers(headers) -> Dict[str, str]:
    # Drop hop-by-hop + host/content-length; httpx recomputes them for the new request.
    skip = _HOP_BY_HOP | {"host", "content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in skip}


def _forward_response_headers(headers) -> Dict[str, str]:
    # Drop hop-by-hop + content-length (we re-stream chunked, so the length no longer holds).
    skip = _HOP_BY_HOP | {"content-length"}
    return {k: v for k, v in headers.items() if k.lower() not in skip}


def build_router_from_env() -> SleepRouter:
    upstreams = parse_upstreams(os.environ.get("UPSTREAMS", ""))
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None))
    return SleepRouter(
        upstreams,
        client,
        sleep_level=int(os.environ.get("SLEEP_LEVEL", "1")),
        idle_timeout=float(os.environ.get("IDLE_TIMEOUT_SECONDS", "600")),
        wake_timeout=float(os.environ.get("WAKE_TIMEOUT_SECONDS", "300")),
    )


def create_app(router: Optional[SleepRouter] = None, run_sleeper: bool = True) -> FastAPI:
    sleeper_interval = float(os.environ.get("SLEEPER_INTERVAL_SECONDS", "30"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.router is None:
            app.state.router = build_router_from_env()
        r: SleepRouter = app.state.router
        task = asyncio.create_task(r.sleeper_loop(sleeper_interval)) if run_sleeper else None
        logger.info("llm-router ready — models: %s", list(r.upstreams))
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
            await r.client.aclose()

    app = FastAPI(title="ipsa-llm-router", lifespan=lifespan)
    app.state.router = router

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(request: Request):
        r: SleepRouter = request.app.state.router
        return JSONResponse(
            {"object": "list",
             "data": [{"id": m, "object": "model", "owned_by": "ipsa"} for m in r.upstreams]}
        )

    @app.api_route("/v1/{path:path}", methods=["POST"])
    async def proxy(request: Request, path: str):
        r: SleepRouter = request.app.state.router
        return await r.forward(request, f"v1/{path}")

    return app


app = create_app()
