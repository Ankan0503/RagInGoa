#!/bin/bash
set -e

echo "======================================================================"
echo "PRODUCTION ENTRYPOINT: INDIC VOICE RAG"
echo "======================================================================"

# Direct Python check: connect to Qdrant and check if collection has indexed points (>0)
HAS_POINTS=$(python3 -c "
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(path='/app/backend/qdrant_data')
    cols = [c.name for c in client.get_collections().collections]
    if 'indic_rag_msmarco_hi' in cols:
        count = client.count(collection_name='indic_rag_msmarco_hi').count
        print(1 if count > 0 else 0)
    else:
        print(0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)

if [ "$HAS_POINTS" = "1" ]; then
    echo "==> Vector collection 'indic_rag_msmarco_hi' detected with indexed vectors. Starting server directly."
else
    echo "==> Vector database not indexed yet. Running automated ingestion..."
    python /app/backend/ingest_pipeline.py --model "${EMBEDDING_MODEL:-intfloat/multilingual-e5-large}" --sample-limit 1000 --recreate-collection
    echo "==> Ingestion finished successfully!"
fi

echo "==> Starting FastAPI Voice RAG Server on port 3004..."
exec uvicorn server:app --host 0.0.0.0 --port 3004


