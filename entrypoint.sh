#!/bin/bash
set -e

echo "======================================================================"
echo "PRODUCTION ENTRYPOINT: INDIC VOICE RAG"
echo "======================================================================"

# If collection does not exist or qdrant_data is empty, run ingestion automatically
if [ ! -d "/app/backend/qdrant_data" ] || [ ! -f "/app/backend/qdrant_data/meta.json" ]; then
    echo "==> No existing vector database found. Starting automated ingestion..."
    python ingest_pipeline.py --model ${EMBEDDING_MODEL:-intfloat/multilingual-e5-large} --sample-limit 1000 --recreate-collection
    echo "==> Ingestion completed successfully!"
else
    echo "==> Existing vector database detected in qdrant_data."
fi

echo "==> Starting FastAPI Voice RAG Server on port 3004..."
exec uvicorn server:app --host 0.0.0.0 --port 3004
