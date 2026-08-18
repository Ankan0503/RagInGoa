#!/bin/bash
set -euo pipefail

echo "======================================================================"
echo "INDIC VOICE RAG — STARTUP"
echo "======================================================================"

QDRANT_PATH="${QDRANT_PATH:-/app/backend/qdrant_data}"
QDRANT_URL="${QDRANT_URL:-}"
PORT="${PORT:-3004}"
MANIFEST="/app/backend/index_manifest.json"

# ---------------------------------------------------------------------------
# Preflight. The original entrypoint started uvicorn unconditionally, so a
# container with no index came up "healthy" and answered every question with
# "no information available". Fail here instead, naming the actual reason.
# ---------------------------------------------------------------------------

fail() {
    echo ""
    echo "STARTUP ABORTED: $1"
    echo ""
    shift
    for line in "$@"; do echo "  $line"; done
    echo ""
    exit 1
}

# ---------------------------------------------------------------------------
# Manifest: declares which transport built this index. Server storage and
# local-mode storage are different formats, so a mismatch is fatal, not a
# fallback. retriever.py enforces this too; catching it here gives a clearer
# message before a 90 second model load.
# ---------------------------------------------------------------------------

MANIFEST_TRANSPORT=""
if [ -f "${MANIFEST}" ]; then
    MANIFEST_TRANSPORT="$(python -c "
import json
m = json.load(open('${MANIFEST}', encoding='utf-8'))
print(m.get('transport') or 'local')
" 2>/dev/null || echo "")"

    echo "==> Index manifest:"
    python - <<'PY'
import json
m = json.load(open("/app/backend/index_manifest.json", encoding="utf-8"))
s = m.get("stats", {})
def n(v):
    return f"{v:,}" if isinstance(v, int) else str(v)
print(f"    model      : {m.get('model_name')} ({m.get('embed_dim')}-dim)")
print(f"    built      : {m.get('built_at')}")
print(f"    transport  : {m.get('transport') or 'local'}")
print(f"    vectors    : {n(s.get('vectors', '?'))}")
print(f"    passages   : {n(s.get('parents', '?'))}")
print(f"    queries    : {n(m.get('queries_done', '?'))}")
print(f"    hybrid bm25: {bool(m.get('sparse_vector_name'))}")
PY

    if [ ! -f "/app/backend/parents.sqlite" ]; then
        fail "index_manifest.json is present but parents.sqlite is missing" \
             "Passage text lives in parents.sqlite; without it every answer would" \
             "have no context. Mount it alongside the manifest."
    fi
else
    echo "==> No index_manifest.json — legacy index mode."
    echo "    EMBEDDING_MODEL must match whatever built the index. Both candidate"
    echo "    models are 384-dim, so a mismatch will NOT raise; it returns wrong results."
fi

# ---------------------------------------------------------------------------
# Transport-specific checks
# ---------------------------------------------------------------------------

if [ -n "${QDRANT_URL}" ]; then
    echo "==> Transport : SERVER (${QDRANT_URL})"

    if [ "${MANIFEST_TRANSPORT}" = "local" ] && [ -f "${MANIFEST}" ]; then
        fail "QDRANT_URL is set but the index was built for local file mode" \
             "These storage formats are not interchangeable." \
             "Either unset QDRANT_URL, or rebuild with build_index_gpu.py in server mode."
    fi

    echo "==> Waiting for Qdrant to become ready..."
    ready=0
    for i in $(seq 1 60); do
        if curl -fsS -H "api-key: ${QDRANT_API_KEY:-}" "${QDRANT_URL}/readyz" >/dev/null 2>&1; then
            ready=1
            echo "    Qdrant ready after ${i}s"
            break
        fi
        sleep 1
    done

    if [ "${ready}" -ne 1 ]; then
        fail "Qdrant at ${QDRANT_URL} did not become ready within 60s" \
             "Check:  docker compose logs qdrant" \
             "If QDRANT__SERVICE__API_KEY is set, QDRANT_API_KEY must match it."
    fi

    # Report what the server actually holds. A collection with zero points is the
    # failure mode this whole preflight exists to catch.
    curl -fsS -H "api-key: ${QDRANT_API_KEY:-}" \
        "${QDRANT_URL}/collections/${QDRANT_COLLECTION:-indic_rag_msmarco_hi}" 2>/dev/null \
        | python -c "
import json, sys
try:
    r = json.load(sys.stdin).get('result', {})
    pts = r.get('points_count')
    print(f'    collection points : {pts:,}' if isinstance(pts, int) else f'    collection points : {pts}')
    print(f'    status            : {r.get(\"status\")}')
    if pts == 0:
        print('    WARNING: collection is EMPTY — restore the snapshot before serving.')
except Exception:
    print('    (could not read collection info — has the snapshot been restored?)')
" || echo "    (collection not found — restore the snapshot)"

else
    echo "==> Transport : LOCAL FILE MODE (${QDRANT_PATH})"
    echo "    WARNING: local mode is brute-force search. Measured at 680k vectors:"
    echo "             retrieval p50 ~11s, startup ~75min. Set QDRANT_URL for production."

    if [ "${MANIFEST_TRANSPORT}" = "server" ]; then
        fail "the index was built for a Qdrant server but QDRANT_URL is not set" \
             "Set QDRANT_URL and restore the snapshot into that server."
    fi

    [ -d "${QDRANT_PATH}" ] || fail "no index directory at ${QDRANT_PATH}" \
        "qdrant_data/ is gitignored, so a clean build has no index. Provide it by" \
        "mounting a volume, baking it into the image, or setting QDRANT_URL."

    [ -n "$(ls -A "${QDRANT_PATH}" 2>/dev/null)" ] || \
        fail "index directory ${QDRANT_PATH} is empty"
fi

# ---------------------------------------------------------------------------
# Non-fatal warnings — the server degrades rather than failing on these.
# ---------------------------------------------------------------------------

[ -z "${GROQ_API_KEY:-}${SARVAM_API_KEY:-}" ] && \
    echo "==> WARNING: no LLM key — retrieval-only mode, no generated answers."
[ -z "${SARVAM_API_KEY:-}" ] && \
    echo "==> WARNING: no Sarvam key — voice input disabled, text queries still work."
[ -z "${ADMIN_TOKEN:-}" ] && \
    echo "==> Admin endpoints disabled (ADMIN_TOKEN unset). Safe default."

case "${GROQ_MODEL:-}" in
    *compound*)
        echo "==> WARNING: GROQ_MODEL='${GROQ_MODEL}' is an AGENTIC model with tool-use"
        echo "    loops. Measured multi-second latency even at max_tokens=1."
        echo "    Use llama-3.1-8b-instant for the sub-200ms path."
        ;;
esac

echo "======================================================================"
echo "==> Starting FastAPI on port ${PORT}"
echo "======================================================================"

exec uvicorn server:app --host 0.0.0.0 --port "${PORT}" --workers 1
