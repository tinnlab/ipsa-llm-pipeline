# Sleep-aware OpenAI-compatible router on :9000.
# Routes by the request's "model" field, wakes a sleeping vLLM model on demand,
# and sleeps idle models (vLLM sleep level 1). CPU-only; reaches the vLLM
# services by compose DNS (no Docker socket needed).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 9000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
