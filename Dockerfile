# ==============================================================================
# Indic Voice RAG — production image
# ==============================================================================

# ---------- Stage 1: frontend ----------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci || npm install

COPY frontend/ ./
RUN npm run build


# ---------- Stage 2: backend ----------
FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r ./backend/requirements.txt

# Pre-bake embedding models so the container never downloads at boot.
#
# The previous image baked intfloat/multilingual-e5-large (2.24GB) even though
# the shipped index was 384-dim MiniLM. Both models used by this project are
# baked instead, and together they are still under a third of e5-large's size:
#   multilingual-e5-small  0.47GB  - new index (build_index_gpu.py)
#   paraphrase-MiniLM-L12  0.22GB  - legacy index fallback
# BM25 is tiny and needed for hybrid retrieval.
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.common.model_description import PoolingType, ModelSource; \
TextEmbedding.add_custom_model(model='intfloat/multilingual-e5-small', \
    pooling=PoolingType.MEAN, normalization=True, \
    sources=ModelSource(hf='intfloat/multilingual-e5-small'), \
    dim=384, model_file='onnx/model.onnx'); \
list(TextEmbedding('intfloat/multilingual-e5-small').embed(['warmup'])); \
list(TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').embed(['warmup'])); \
list(SparseTextEmbedding('Qdrant/bm25', disable_stemmer=True).embed(['warmup']))"

# Application code. .dockerignore keeps .env and __pycache__ out of the image;
# secrets are supplied at runtime by the platform, never baked in.
COPY backend/ ./backend/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# The index, if it exists in the build context. The bracket glob makes each COPY
# optional — Docker fails on a missing literal path but tolerates a glob that
# matches nothing. A build without an index still succeeds; entrypoint.sh then
# refuses to start and says exactly what is missing.
COPY backend/qdrant_dat[a] ./backend/qdrant_data
COPY backend/parents.sqlit[e] ./backend/parents.sqlite
COPY backend/index_manifest.jso[n] ./backend/index_manifest.json

WORKDIR /app/backend

ENV PORT=3004 \
    QDRANT_PATH=/app/backend/qdrant_data \
    PYTHONUNBUFFERED=1

EXPOSE 3004

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:3004/health || exit 1

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
