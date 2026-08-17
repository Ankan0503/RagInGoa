#!/bin/bash
set -e

echo "======================================================================"
echo "PRODUCTION ENTRYPOINT: INDIC VOICE RAG"
echo "======================================================================"

# Check if Qdrant collection directory exists with indexed data
if [ -d "/app/backend/qdrant_data/collection/indic_rag_msmarco_hi" ] || [ -d "/app/backend/qdrant_data/collections/indic_rag_msmarco_hi" ]; then
    echo "==> Vector collection 'indic_rag_msmarco_hi' detected. Starting server directly."
else
    echo "==> Vector database not indexed yet. Running automated ingestion..."
    python /app/backend/ingest_pipeline.py --model "${EMBEDDING_MODEL:-intfloat/multilingual-e5-large}" --sample-limit 1000 --recreate-collection
    echo "==> Ingestion finished successfully!"
fi

echo "==> Starting FastAPI Voice RAG Server on port 3004..."
exec uvicorn server:app --host 0.0.0.0 --port 3004
