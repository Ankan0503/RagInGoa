#!/bin/bash
set -e

echo "======================================================================"
echo "PRODUCTION ENTRYPOINT: INDIC VOICE RAG"
echo "======================================================================"

# Check if Qdrant collection directory exists with indexed segments
COLLECTION_DIR="/app/backend/qdrant_data/collections/indic_rag_msmarco_hi"

if [ ! -d "$COLLECTION_DIR" ]; then
    echo "==> Vector database not indexed yet. Running automated ingestion..."
    python /app/backend/ingest_pipeline.py --model "${EMBEDDING_MODEL:-intfloat/multilingual-e5-large}" --sample-limit 1000 --recreate-collection
    echo "==> Ingestion finished successfully!"
else
    echo "==> Vector collection 'indic_rag_msmarco_hi' detected. Starting server directly."
fi

echo "==> Starting FastAPI Voice RAG Server on port 3004..."
exec uvicorn server:app --host 0.0.0.0 --port 3004
