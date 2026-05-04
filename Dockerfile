FROM python:3.11-slim

LABEL maintainer="CSIT5970 Project"
LABEL description="Custom autonomous agent with RAG and persona injection"

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────
# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────────
COPY . .

# ── Data volume for ChromaDB persistence ──────────────────────────────────
VOLUME ["/app/data/chroma_db", "/app/benchmark/results"]

# ── Environment variables (override at runtime or via .env) ───────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Default command: interactive agent REPL ───────────────────────────────
ENTRYPOINT ["python", "main.py"]
CMD []
