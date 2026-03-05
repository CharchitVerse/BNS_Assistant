# ============================================================================
# BNS Legal RAG — FastAPI Backend Dockerfile
# Python 3.12 | Multi-stage build | Render-ready
# ============================================================================

# ── Stage 1: Build dependencies ──────────────────────────────────────────────

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# System deps for building Python packages (psycopg2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency file first (better Docker layer caching)
COPY pyproject.toml .

# Install deps into a clean prefix directory
RUN pip install --prefix=/install .

# ── Stage 2: Production image ────────────────────────────────────────────────

FROM python:3.12-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production

WORKDIR /app

# Runtime-only system deps (no build-essential — keeps image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /install/bin /usr/local/bin

# Copy application code
COPY app/ app/
COPY pyproject.toml .

# Create data directory for ChromaDB persistence
# Copy pre-computed embeddings from local Mac ingestion
COPY data/ data/
RUN mkdir -p /app/data/chroma_db

# Create non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check using curl (lighter than importing Python)
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Single worker — Render free tier has 512MB RAM limit
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
