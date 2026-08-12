# ipsa-llm-pipeline

Turn pathway-enrichment and differential-expression results into a mechanistic
interpretation report, using large language models grounded in curated pathway databases.

You POST the output of an enrichment analysis — enriched pathways, and optionally your
differentially expressed genes — and get back a markdown report: which biological themes
the pathways fall into, which genes are network hubs, what the curated molecular
interactions inside those pathways actually say, and a set of testable mechanistic
hypotheses drawn from all of it.

The pipeline runs five steps:

| Step | What it does | Grounded in |
|------|--------------|-------------|
| 1 | Clusters enriched pathways into biological themes | Markov clustering over gene overlap |
| 2 | Finds network hub genes and master regulators | STRING protein-protein interactions |
| 3 | Extracts molecular mechanisms within pathways | KEGG pathway maps (KGML), Pathway Commons |

| 4 | Generates testable mechanistic hypotheses | Steps 1-3, with tool-assisted lookups |
| 5 | Compiles everything into a markdown report | Steps 1-4 |

Steps 2 and 3 need differentially expressed (DE) genes; without them the pipeline runs
pathway-only and says so. Two models are used throughout: a **reasoning** model
(gpt-oss-120b) that does the analysis, and a **reviewer** model (OpenBioLLM-70B, a
biomedical model) that checks biochemical claims.

There is also a **meta-analysis** mode, `POST /api/meta-pipeline`, which runs the same
five steps over combined data from several datasets and then adds reproducibility and
comparative analysis across them. This README covers the single-dataset pipeline; see
`/docs` for the meta-analysis request schema.

> **Where your data goes.** Everything runs locally except the calls to the reasoning
> endpoint you configure — so if you point `GPTOSS_API_URL` at a third party, your input
> goes there. Steps 3 and 2 also query the public KEGG, STRING and Pathway Commons APIs
> with pathway and gene identifiers. The optional baseline-comparison feature
> (`ENABLE_BASELINE_COMPARISON`, off by default) would additionally send input to the
> OpenAI and Anthropic APIs.

---

## Prerequisites

- **Docker** and **Docker Compose**.
- **An NVIDIA GPU with enough memory for a 70B model at fp8 — about 80 GB** — plus the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  so containers can see the GPU. Verify with `docker info | grep -i runtime` — you want
  `nvidia` in the list. Self-hosting the reasoning model as well (Route A below) needs a
  **second** GPU; pointing at an endpoint you already have (Route B) does not.
- **Disk for model weights: about 130 GB** for the reviewer model, downloaded once on
  first run and cached. Route A needs a second, comparable download. This is *not* in the
  container images.
- **A reasoning-model endpoint.** This is the one thing the stack does not give you for
  free — see the next section.

Everything else, including the reviewer model, is started for you by `docker compose`.

## Serving the models

**The reviewer model needs no setup.** `vllm-openbio` is part of the stack: `docker compose
up -d` pulls the upstream vLLM image, downloads `aaditya/Llama3-OpenBioLLM-70B` on first
run, and serves it. It occupies a GPU, sleeps when idle, and wakes on demand.

**The reasoning model is your choice of two routes.**

### Route A — let this stack run it

Requires a second GPU. The compose file ships an optional `vllm-gptoss` service, off
unless you ask for it:

```bash
docker compose -f docker-compose.prod.yml --profile local-gptoss up -d
```

Then in your `.env`:

```bash
GPTOSS_API_URL=http://vllm-gptoss:8001/v1/chat/completions
GPTOSS_MODEL=openai/gpt-oss-120b
```

That URL is a Compose DNS name, resolved inside the Docker network — which is where
`pipeline-api` calls it from. Set `GPTOSS_GPUS` to a free GPU index that does not overlap
`OPENBIO_GPUS`.

### Route B — point at a server you already have

Any OpenAI-compatible `/v1/chat/completions` endpoint works — another vLLM instance, a
hosted provider, anything speaking that API:

```bash
GPTOSS_API_URL=https://your-server.example.com/v1/chat/completions
GPTOSS_MODEL=<id from GET /v1/models>
```

**Get `GPTOSS_MODEL` from the server itself**, don't guess it:

```bash
curl -s https://your-server.example.com/v1/models
```

vLLM reports whichever id it was launched with, so a self-hosted instance serving
`--model openai/gpt-oss-120b` reports exactly `openai/gpt-oss-120b`, while another server
may expose a shorter alias like `gpt-oss-120b`. Sending an id the server doesn't recognise
gets the request rejected as an unknown model. If the model id is wrong, every reasoning
step fails while the stack itself looks perfectly healthy.

---

## Quickstart

```bash
git clone https://github.com/tinnlab/ipsa-llm-pipeline.git
cd ipsa-llm-pipeline
cp .env.prod.example .env
```

Edit `.env`. Only one value has no working default:

- **`GPTOSS_API_URL`** — required, see *Serving the models* above.
- `HF_TOKEN` — a Hugging Face token, needed if the reviewer model's Llama base is gated
  for your account.
- `HF_CACHE` — where the ~130 GB of weights live. Point it at a disk with room.
- `OPENBIO_GPUS` — GPU index for the reviewer model. Pick a free one from `nvidia-smi -L`.

Then start it:

```bash
# Pre-pull the vLLM image so model startup doesn't also wait on an image download.
docker pull vllm/vllm-openai:latest

docker compose -f docker-compose.prod.yml up -d --build
```

Add `--profile local-gptoss` to that last command if you are using Route A.

**The first run downloads about 130 GB**, which can take a long time. The API answers
immediately, but the reviewer model is unusable until the download *and* the subsequent
weight load both finish. Watch it, and wait for vLLM to report the server is up before
running the warm-up call below:

```bash
docker compose -f docker-compose.prod.yml logs -f vllm-openbio
```

Until then, requests routed to the reviewer model fail with a connection error rather
than waiting.

## Verify it works

The two CPU services come up in seconds:

```bash
curl http://localhost:6000/health    # {"status":"healthy","service":"IPSA Interpretation Pipeline API","version":"1.0.0"}
curl http://localhost:9000/health    # {"status":"ok"}
```

If either fails with **"port is already allocated"** at `up -d`, something else on the
host holds 6000 or 9000. Set `PIPELINE_PORT` and/or `ROUTER_PORT` in `.env` to free ports
and start again — the container-internal ports do not change, so nothing else needs
editing.

Check the router can see the reviewer model — this lists it without loading it:

```bash
curl http://localhost:9000/v1/models
```

Under Route A, check the reasoning model too:

```bash
curl http://localhost:8001/v1/models
```

Finally, once the logs show vLLM serving, force the reviewer model onto the GPU. This is
the real readiness gate, and it takes minutes rather than seconds:

```bash
curl -s --max-time 900 http://localhost:9000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"aaditya/Llama3-OpenBioLLM-70B","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
```

A JSON response with a `choices` array means the model is warm and the stack is ready. A
connection error means the weights are still loading — check the logs and try again.

Interactive API docs are at <http://localhost:6000/docs>.

## Run a pipeline job

The service is driven entirely over HTTP — no front-end required. This repository ships a
small, valid example input so you can run one end to end:

```bash
curl -X POST http://localhost:6000/api/pipeline \
  -H 'content-type: application/json' \
  -d @examples/example_input.json
```

You get a job id back straight away; the pipeline runs in the background:

```json
{
  "success": true,
  "job_id": "a1b2c3d4-...",
  "status": "pending",
  "message": "Pipeline job submitted successfully. Use GET /api/pipeline/job/a1b2c3d4-... to check status."
}
```

Poll it:

```bash
curl http://localhost:6000/api/pipeline/job/<job_id>
```

While it runs, `status` is `running` and you can watch progress:

```json
{
  "success": true,
  "status": "running",
  "current_step": 3,
  "current_step_message": "Fetching molecular interactions from pathway databases...",
  "total_steps": 5
}
```

When `status` becomes `completed`, the report is at
`results.steps.step5.markdown_content`:

```bash
curl -s http://localhost:6000/api/pipeline/job/<job_id> | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['results']['steps']['step5']['markdown_content'] if d.get('results') else 'Job is ' + d['status'] + ' — no report yet.')"
```

(The status response also carries `job_id`, timestamps, `steps_info`, `step_results` and
`step_execution_times`; the fields above are the ones you need to follow a run.)

If `status` becomes `failed`, `error` says why. The most common cause is an unreachable or
misconfigured reasoning endpoint — see *Common problems*.

The report itself opens like this:

```markdown
# Molecular Pathway Analysis Report

## Executive Summary

... 2-3 paragraphs synthesising the strongest signals in both directions ...

**Key Metrics:**
- **Major biological theme:** Cell cycle control
- **Dominant enrichment axes:** Cell cycle (↑ NES +2.41); Fatty acid degradation (↓ NES -2.28)
- **Network hub genes (master regulators):** CDK1, CCNB1 (2 of 5 hub genes)
- **Testable hypotheses:** 4

## 1. Pathway Theme Analysis
## 2. Network Hub Genes
## 3. Pathway Mechanisms and Molecular Interactions
## 4. Mechanistic Hypotheses and Predictions
## 5. Summary and Conclusions
## Appendices
```

**A full run takes several minutes** and makes many LLM calls plus live queries to KEGG,
STRING and Pathway Commons.

To try a single step instead, `POST /api/pipeline/step/{n}` runs one step at a time. The
path number selects the step; the body needs `step_number` and `input_data`, plus
`previous_results` for any step after the first:

```bash
curl -X POST http://localhost:6000/api/pipeline/step/1 \
  -H 'content-type: application/json' \
  -d "{\"step_number\": 1, \"input_data\": $(cat examples/example_input.json)}"
```

### Your own input

`examples/example_input.json` shows the required shape. Briefly:

- `pathways[]` — **required**. Each needs `name`, `source`, `pathwayId`, `pValue`,
  `pValueFDR` and `genes` (the pathway's gene set). `NES` — the normalised enrichment
  score from your enrichment tool — is optional but strongly recommended: its sign and
  magnitude are what let the report talk about direction and strength.
  Set `source` to `KEGG` for KEGG pathways; anything else routes to Pathway Commons and
  the LLM. **Pathway names matter**: KEGG structures are looked up by *name*, not by
  `pathwayId`, so use the database's own naming (`Cell cycle`, not `cell-cycle`).
- `genes[]` — optional, your differentially expressed genes. Each needs `geneSymbol`,
  `foldChange`, `pValue`, `pValueFDR`. **`foldChange` is a signed log2 value** — negative
  means down-regulated. Omit this list entirely and steps 2 and 3 are skipped, and the
  pipeline runs pathway-only.
- `metadata` — free-form. `organism`, `disease` and `tissue` are used for context. Give
  `organism` as the binomial name (`Homo sapiens`, not `human`) — KEGG lookups key on it
  and are skipped for an unrecognised value.

Full schemas are at <http://localhost:6000/docs>.

## Optional: the CollecTRI regulon

Step 3 can infer **upstream transcription-factor regulators** by testing which TF's target
set is over-represented among your down-regulated genes. That needs the CollecTRI TF-target
network, which is **not shipped with this repository** — it is third-party data with its own
terms.

Without it, nothing breaks: the pipeline falls back to asking the LLM for candidate TFs and
labels them as hypotheses rather than database hits. To install it:

```python
import decoupler as dc

net = dc.get_collectri(organism='human', split_complexes=False)
net[['source', 'target', 'weight']].to_csv(
    'interpretation-api/src/pipeline/data/collectri_human.tsv',
    sep='\t', index=False,
)
```

See `interpretation-api/src/pipeline/data/README.md` for the exact format, an OmniPath
alternative, and the citations.

## Configuration

Set these in `.env`. Defaults come from `docker-compose.prod.yml` and apply if you leave a
variable unset.

| Variable | Default | Required | What it does |
|----------|---------|----------|--------------|
| `GPTOSS_API_URL` | *(placeholder)* | **yes** | Full chat-completions URL of the reasoning model |
| `GPTOSS_MODEL` | `openai/gpt-oss-120b` | if your server differs | Model id to send; must match `GET /v1/models` |
| `HF_TOKEN` | *(empty)* | if the model is gated | Hugging Face token for weight downloads |
| `HF_CACHE` | `/data/ipsa/llm/hf-cache` | no | Host directory for the model cache (~130 GB) |
| `OPENBIO_GPUS` | `0` | no | GPU index(es) for the reviewer model |
| `OPENBIO_TP` | `1` | no | Tensor-parallel size for the reviewer model |
| `OPENBIO_MEM_UTIL` | `0.90` | no | Fraction of its GPU the reviewer model may reserve; needs ~66 GB for weights |
| `VLLM_TAG` | `latest` | no | vLLM image tag; must support `--enable-sleep-mode` |
| `SLEEP_LEVEL` | `1` | no | vLLM sleep level: 1 = weights to RAM, 2 = discard |
| `IDLE_TIMEOUT_SECONDS` | `600` | no | Idle seconds before the reviewer model sleeps |
| `ROUTER_PORT` | `9000` | no | Host port for the router |
| `ROUTER_TAG` | `latest` | no | Image tag for the built router image |
| `PIPELINE_PORT` | `6000` | no | Host port for the pipeline API |
| `PIPELINE_TAG` | `latest` | no | Image tag for the built pipeline-api image |

These four apply **only** when the `local-gptoss` profile is enabled, and are ignored
otherwise:

| Variable | Default | What it does |
|----------|---------|--------------|
| `GPTOSS_GPUS` | `1` | GPU index(es) for the local reasoning model; must not overlap `OPENBIO_GPUS` |
| `GPTOSS_TP` | `1` | Tensor-parallel size |
| `GPTOSS_MEM_UTIL` | `0.90` | Fraction of its GPU the model may reserve |
| `GPTOSS_PORT` | `8001` | Host port the local reasoning model publishes |

`pipeline-api` reads further settings from `interpretation-api/src/config.py` — model
temperatures, request timeouts, clustering parameters, validation toggles. Compose sets the
ones that matter for deployment (`LOCAL_LLM_*`, `REVIEWER_LLM_*`, `API_HOST`, `API_PORT`)
from the table above; the rest keep their defaults unless you add them to the service's
`environment:` block.

Note that CORS is permissive by default (`allow_origins=["*"]`, in
`interpretation-api/src/api/server.py`) so the service works behind any front-end out of the
box. Restrict it before exposing the API beyond a trusted network.

## Common problems

**The first run seems to hang.** It is downloading ~130 GB of weights. `docker compose -f
docker-compose.prod.yml logs -f vllm-openbio` shows progress. Nothing routed to the
reviewer model works until that finishes.

**A job is accepted but then fails.** The API returns a `job_id` before any model is
contacted, so a misconfigured reasoning endpoint only surfaces once the job runs. Check
`error` on the job status. If you never set `GPTOSS_API_URL`, it reads:

```
LOCAL API error: HTTPConnectionPool(host='replace-me-see-readme', port=8001):
Max retries exceeded with url: /v1/chat/completions
(Caused by NameResolutionError(... Failed to resolve 'replace-me-see-readme' ...))
```

Any other connection error means `GPTOSS_API_URL` is set but unreachable *from inside the
container* — note that `localhost` there means the container itself, not your host. Use
the Compose service name (`http://vllm-gptoss:8001/...`) or an address routable from the
container.

Note that steps 1 and 2 can complete before this surfaces: pathway clustering and network
analysis do not need the reasoning model, so a run with no working endpoint typically
fails at step 3 with partial `step_results` already populated.

**The model id is rejected.** If `error` mentions an unknown or not-found model, your
`GPTOSS_MODEL` is not what the server serves. Ask it: `curl <base>/v1/models`, and use an
id from the response verbatim. `openai/gpt-oss-120b` and `gpt-oss-120b` are *different
ids*, and a server that serves one will reject the other.

**vLLM exits at startup with a KV-cache or out-of-memory error.** `OPENBIO_MEM_UTIL` is
below what the weights need. The reviewer model needs roughly 66 GB at fp8; raise the
fraction (0.90 or higher on an 80 GB card) or move it to a larger GPU.

**A GPU index is wrong or already busy.** `OPENBIO_GPUS` and `GPTOSS_GPUS` are indices
into `nvidia-smi -L`, and they must not overlap each other or anything else on the host.
The container fails immediately if the index does not exist.

**Reports have no upstream regulators.** That section needs the optional CollecTRI
regulon — see above. Without it the pipeline substitutes LLM-proposed candidates and
labels them as hypotheses.

---

## Architecture

| Service | What it is | Port | GPU |
|---------|-----------|------|-----|
| `pipeline-api` | The 5-step interpretation API (FastAPI) | 6000 | no |
| `llm-router`   | Sleep-aware OpenAI-compatible router; routes by model, wakes and sleeps the reviewer | 9000 | no |
| `vllm-openbio` | `aaditya/Llama3-OpenBioLLM-70B` reviewer model (vLLM) | *internal* 8002 | `OPENBIO_GPUS` |
| `vllm-gptoss`  | *Optional* `openai/gpt-oss-120b` reasoning model (vLLM) | 8001 | `GPTOSS_GPUS` |

```
client → :6000 pipeline-api ┬→ ${GPTOSS_API_URL}              (reasoning model)
                            │    either vllm-gptoss:8001 (profile local-gptoss)
                            │    or an OpenAI-compatible server you provide
                            └→ llm-router:9000 → vllm-openbio  (reviewer model)
```

`vllm-openbio` is not published to the host — only `llm-router` reaches it, over the
Compose network.

The reasoning model is called **directly, bypassing the router**, on purpose: the router
owns a sleep/wake lifecycle, and it must never sleep a server this stack doesn't own. The
trade-off is that a locally-run gpt-oss holds its GPU for as long as the profile is up.

### Model sleep and wake

The reviewer model runs always-on with vLLM **sleep mode** (`--enable-sleep-mode`,
`VLLM_SERVER_DEV_MODE=1`), and `llm-router` owns its lifecycle:

- **Wake on demand** — when a request arrives for a sleeping model, the router calls
  vLLM's `/wake_up`, waits until it reports awake, then forwards. Waking copies weights
  from RAM back to the GPU and takes a few **seconds**, because the engine, CUDA context
  and compiled graphs all stay warm.
- **Sleep when idle** — after `IDLE_TIMEOUT_SECONDS` with no requests (default 10 minutes)
  the router calls `/sleep?level=1`, which offloads weights to CPU RAM and drops the KV
  cache, **freeing the GPU** for other work.
- **No per-request cold start** — the container and engine stay alive throughout; only the
  weights move.

`SLEEP_LEVEL=1` keeps offloaded weights in host RAM, so waking is near-instant; it costs
roughly the model's size in RAM while it sleeps. Level 2 discards the weights and reloads
them from disk instead — slower to wake, but no RAM held. Use level 2 if host RAM is tight.

## Tests

Both suites mock all LLM and network calls, so neither needs a GPU, a model, or network
access.

```bash
# Pipeline (interpretation-api)
cd interpretation-api
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -v

# Router (llm-router)
cd ../llm-router
pip install -r requirements-dev.txt
python -m pytest
```

One test is skipped unless you have installed the optional CollecTRI regulon; that is
expected.

## Deploying it properly

[DEPLOY.md](DEPLOY.md) covers running this on a dedicated host: one-time prep, health
gates, pre-warming, GPU allocation and rolling image versions.

## License

MIT — see [LICENSE](LICENSE).

The **code** in this repository is MIT-licensed, but it runs models and queries databases
that carry their own terms, and those are not covered by it. In particular OpenBioLLM is a
Llama 3 derivative under Meta's community licence, and KEGG requires a licence for
programmatic or commercial use. See [NOTICE](NOTICE) before deploying commercially.
