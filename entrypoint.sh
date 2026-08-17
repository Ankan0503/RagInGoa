#!/bin/bash
set -e

echo "======================================================================"
echo "PRODUCTION ENTRYPOINT: INDIC VOICE RAG"
echo "======================================================================"

# Check if Qdrant storage.sqlite exists with actual data (>1MB)
SQLITE_FILE1="/app/backend/qdrant_data/collection/indic_rag_msmarco_hi/storage.sqlite"
SQLITE_FILE2="/app/backend/qdrant_data/collections/indic_rag_msmarco_hi/storage.sqlite"

if ([ -f "$SQLITE_FILE1" ] && [ $(stat -c%s "$SQLITE_FILE1" 2>/dev/null || echo 0) -gt 1000000 ]) || ([ -f "$SQLITE_FILE2" ] && [ $(stat -c%s "$SQLITE_FILE2" 2>/dev/null || echo 0) -gt 1000000 ]); then
    echo "==> Valid indexed vector database detected in qdrant_data. Starting server directly."
else
    echo "==> Vector database not indexed yet or empty. Running automated ingestion..."
    python /app/backend/ingest_pipeline.py --model "${EMBEDDING_MODEL:-intfloat/multilingual-e5-large}" --sample-limit 1000 --recreate-collection
    echo "==> Ingestion finished successfully!"
fi

echo "==> Starting FastAPI Voice RAG Server on port 3004..."
exec uvicorn server:app --host 0.0.0.0 --port 3004
