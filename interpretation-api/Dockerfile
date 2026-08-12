# 5-step LLM interpretation pipeline — FastAPI service. Listens on :6000.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. The steps add the app root to sys.path at import time, so the
# package must be copied with src/ at the top level (see .dockerignore for exclusions).
COPY . .

# Runtime dirs used by the pipeline (job outputs) and the services' HTTP caches.
RUN mkdir -p /app/output/pipeline_jobs /app/cache

EXPOSE 6000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "6000"]
