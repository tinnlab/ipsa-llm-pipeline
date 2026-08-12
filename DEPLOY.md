# Production deployment

This is the operational companion to the README. The README gets you running; this covers
deploying on a single GPU host and keeping it running.

The stack is three services (`pipeline-api`, `llm-router`, `vllm-openbio`), plus the
optional `vllm-gptoss`. Two images are **built** on the host (`pipeline-api`,
`llm-router`); `vllm/vllm-openai` is **pulled**; model weights download once to the
directory you set as `HF_CACHE`.

## 1. One-time prep on the deploy host

```bash
mkdir -p /data/ipsa/llm/hf-cache     # persistent weights cache (~130GB first run)
docker info | grep -i runtime        # expect 'nvidia'
nvidia-smi -L                        # note which GPU indices are free
```

That path is the default for `HF_CACHE`; use any directory you like and set `HF_CACHE` to
match in `.env`. Put it on a volume with room to grow — the reviewer model alone is about
130 GB, and enabling the optional local gpt-oss adds another large download.

If you are not running the optional local gpt-oss profile, confirm the host can reach the
endpoint you configured. `.env` is read by Compose, not by your shell, so pass the URL
explicitly:

```bash
curl -fsS https://your-server.example.com/v1/models
```

`GPTOSS_MODEL` must be one of the ids that returns.

## 2. Configuration

Copy `.env.prod.example` to `.env` and edit it. Every variable is documented there and in
the README's configuration table. The only one with no usable default is
`GPTOSS_API_URL`.

If your CI system injects configuration as environment variables rather than a `.env`
file, note that Compose reads `.env` from the project directory — either write the file
during the deploy step or export the same names into the build agent's environment.

## 3. Deploy

```bash
# Pre-pull the vLLM image so model startup doesn't also wait on an image download.
docker pull "vllm/vllm-openai:${VLLM_TAG:-latest}"

# Build and start. Add `--profile local-gptoss` if this host also serves gpt-oss.
docker compose -f docker-compose.prod.yml up -d --build --remove-orphans
```

## 4. Health gates

These are fast and do **not** force a model load, so they are safe to run immediately:

```bash
curl -fsS --retry 30 --retry-delay 5 --retry-all-errors \
  "http://localhost:${ROUTER_PORT:-9000}/health" >/dev/null && echo "llm-router ready"
curl -fsS --retry 30 --retry-delay 5 --retry-all-errors \
  "http://localhost:${PIPELINE_PORT:-6000}/health" >/dev/null && echo "pipeline-api ready"
```

Then pre-warm the reviewer model, which forces its weights onto the GPU so the first real
user request isn't slow. Allow several minutes: the initial load is much slower than a
wake from sleep.

```bash
m="aaditya/Llama3-OpenBioLLM-70B"
curl -fsS --max-time 900 "http://localhost:${ROUTER_PORT:-9000}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
  >/dev/null && echo "$m warmed"
```

Do **not** pre-warm an external gpt-oss endpoint you do not own — this stack must not
manage its lifecycle.

## Operating notes

- **GPU allocation.** `OPENBIO_GPUS` (and `GPTOSS_GPUS` under the optional profile) must
  point at indices nothing else is using. If other GPU workloads share the host, pick free
  indices from `nvidia-smi -L` and lower `OPENBIO_MEM_UTIL` to leave them headroom.
- **Idle/sleep behaviour** is set by `IDLE_TIMEOUT_SECONDS` and `SLEEP_LEVEL` — change
  them in `.env`, not in code. A sleeping model's GPU memory is freed and the next request
  wakes it in seconds. This applies to the reviewer model only; gpt-oss is never
  sleep-managed.
- **The gpt-oss dependency is external by default.** `pipeline-api` needs network access
  to `GPTOSS_API_URL`. If that server is down or renames its model id, the pipeline's
  reasoning steps fail even though every container here is healthy. Enable the
  `local-gptoss` profile if you would rather own it.
- **Rolling versions.** `VLLM_TAG` rolls vLLM; `PIPELINE_TAG` and `ROUTER_TAG` tag the
  built images so you can roll back. Verify a new `VLLM_TAG` still supports
  `--enable-sleep-mode` for the fp8 reviewer model before rolling it out widely.
- **The raw vLLM containers are not host-published** (except the optional gpt-oss one,
  which publishes `GPTOSS_PORT` so you can query it). Only `llm-router` reaches the
  reviewer model, in-network.
- **Tests** should run in CI or pre-deploy — see the README's Tests section for both
  suites.
