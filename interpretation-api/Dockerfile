# 5-step LLM interpretation pipeline — FastAPI service. Listens on :6000.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. The steps add the app root to sys.path at import time, so the
# package must be copied with src/ at the top level (see .dockerignore for exclusions).
COPY . .

# Runtime dirs used by the pipeline (job outputs), the services' HTTP caches, and the
# LLM prompt/response log. /app/logs/llm_prompts is where docker-compose.prod.yml mounts a
# host volume — and it is also the code default (config.LLM_PROMPT_LOG_DIR, relative to this
# WORKDIR), so the logs land on the persisted volume either way.
RUN mkdir -p /app/output/pipeline_jobs /app/cache /app/logs/llm_prompts

EXPOSE 6000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "6000"]
