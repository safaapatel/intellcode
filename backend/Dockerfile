# IntelliCode Review — Backend Dockerfile
# Multi-stage build: install deps in builder, copy to slim runtime image.

# ── Stage 1: dependency installer ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System packages needed to compile some wheels (e.g. numpy, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# libgomp is required by XGBoost / scikit-learn at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY . .

# Non-root user for least-privilege operation
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Checkpoints are expected to be mounted at /app/checkpoints via a volume.
# Set LOG_LEVEL to control uvicorn verbosity (default: info).
ENV LOG_LEVEL=info

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=8)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info", "--workers", "1"]
