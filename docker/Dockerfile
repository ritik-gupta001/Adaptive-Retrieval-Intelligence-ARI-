# ── Stage 1: dependency install ───────────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /build

# System deps needed by some Python packages (chromadb, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.lock


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user for production security
RUN addgroup --system ari && adduser --system --ingroup ari ari

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy app source
COPY app ./app
COPY evals ./evals
COPY ingestion ./ingestion

# Data directory for Chroma + BM25 index + checkpoints
RUN mkdir -p /app/data && chown -R ari:ari /app

USER ari

EXPOSE 8000

# Gunicorn for production; --workers 1 because LangGraph's in-memory
# checkpointer and store are process-local (use Postgres backends for
# multi-worker deployments).
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
