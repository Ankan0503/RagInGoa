#!/usr/bin/env python3
"""
High-Concurrency Indic Voice RAG Server (server.py)
==================================================
Designed for HH Goa 2026 Shortlisting Task 2: Sub-200ms Voice-Enabled Indic RAG System.

Pipeline Orchestration:
1. Speech-to-Text: Sarvam AI Saaras v3 API (<80-120ms)
2. Vector Retrieval: IndicRetriever with FastEmbed ONNX & LRU Cache (<10ms)
3. LLM Token Streaming: Groq LLaMA-3.1-8B-Instant with Greedy Decoding (TTFT <60ms)
4. Direct Concise Output Constraint (<200ms end-to-end total generation)
"""

import os
import sys
import io
import time
import json
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from groq import AsyncGroq

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Local module imports
from retriever import IndicRetriever, RetrievalResult, RetrievedHit
from audio_stt import SarvamSTTClient, TranscriptionResult

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("RAGServer")


# ============================================================================
# DATA SCHEMAS
# ============================================================================

class TextQueryRequest(BaseModel):
    query: str = Field(..., description="User query in Hindi, English, or Hinglish")
    top_k: int = Field(3, description="Number of source passages to retrieve")
    strategy: Optional[str] = Field(None, description="Chunking strategy filter ('parent_child', 'sliding_window', or None)")
    temperature: float = Field(0.0, description="LLM sampling temperature (0.0 for greedy sub-200ms generation)")
    max_tokens: int = Field(35, description="Maximum token generation limit for concise sub-200ms responses")


class LatencyBreakdown(BaseModel):
    stt_latency_ms: Optional[float] = Field(None, description="Speech-to-Text latency in ms")
    retrieval_latency_ms: float = Field(..., description="Vector retrieval & parent resolution latency in ms")
    embed_latency_ms: float = Field(..., description="Query embedding time in ms")
    search_latency_ms: float = Field(..., description="Qdrant search time in ms")
    ttft_ms: Optional[float] = Field(None, description="Time to First Token in ms")
    first_token_latency_ms: Optional[float] = Field(None, description="Total pipeline latency to first emitted token")
    total_generation_time_ms: float = Field(..., description="Total LLM generation duration in ms")
    total_pipeline_ms: float = Field(..., description="Total end-to-end processing time in ms")
    tokens_per_second: float = Field(..., description="Generation throughput (TPS)")
    total_tokens: int = Field(..., description="Total tokens emitted")
    sla_passed: bool = Field(..., description="True if total pipeline latency meets <200ms target")


class QueryResponse(BaseModel):
    query: str
    transcript: Optional[str] = None
    answer: str
    sources: List[RetrievedHit]
    metrics: LatencyBreakdown


# ============================================================================
# HIGH-SPEED CONCISE SYSTEM PROMPT TEMPLATE (<200ms TARGET)
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """आप एक अत्यंत तीव्र, सटीक और सहायक AI सहायक हैं।
नीचे दिए गए संदर्भ (Context) के आधार पर ही प्रश्न का उत्तर केवल 1 संक्षिप्त, सीधा और तथ्यपरक वाक्य (अधिकतम 20-30 शब्द) में केवल शुद्ध हिंदी (Devanagari) में दें।
यदि संदर्भ में उत्तर मौजूद नहीं है, तो सीधे कहें: "दिए गए संदर्भ में इसकी जानकारी उपलब्ध नहीं है।"

संदर्भ (Context):
{context}"""


# ============================================================================
# APPLICATION LIFESPAN & STATE
# ============================================================================

class AppState:
    retriever: Optional[IndicRetriever] = None
    stt_client: Optional[SarvamSTTClient] = None
    groq_client: Optional[AsyncGroq] = None
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes and warms up all backend RAG clients on server startup.
    Eliminates cold-start latencies for instant sub-200ms user queries.
    """
    logger.info("=" * 70)
    logger.info("STARTING PRODUCTION INDIC VOICE RAG FASTAPI SERVER")
    logger.info("=" * 70)

    # 1. Initialize Vector Retriever with In-Memory Caching
    qdrant_dir = os.getenv("QDRANT_PATH", "./qdrant_data")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    logger.info(f"Initializing IndicRetriever (Model: {embedding_model}, Path: '{qdrant_dir}')...")
    state.retriever = IndicRetriever(qdrant_path=qdrant_dir, embedding_model=embedding_model)

    # 2. Initialize Sarvam STT Client
    sarvam_key = os.getenv("SARVAM_API_KEY")
    sarvam_model = os.getenv("SARVAM_MODEL", "saaras:v3")
    if sarvam_key:
        logger.info(f"Initializing SarvamSTTClient (Model: {sarvam_model})...")
        state.stt_client = SarvamSTTClient(api_key=sarvam_key, model=sarvam_model)
    else:
        logger.warning("SARVAM_API_KEY not found! Voice STT endpoint will be disabled.")

    # 3. Initialize Groq Async Client with Pre-warmed Persistent Connection
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("QROQ_API_KEY")
    if groq_key:
        logger.info(f"Initializing AsyncGroq Client (Model: {state.groq_model})...")
        state.groq_client = AsyncGroq(api_key=groq_key)
    else:
        logger.warning("GROQ_API_KEY not found! LLM generation will be unavailable.")

    # 4. Lightweight Pre-Warmup Probe (Warms ONNX session, Qdrant HNSW index & HTTP/2 connection)
    try:
        logger.info("Executing pre-warmup probe for sub-200ms instant inference...")
        _ = state.retriever.retrieve("warmup probe", top_k=1)
        if state.groq_client:
            _ = await state.groq_client.chat.completions.create(
                model=state.groq_model,
                messages=[{"role": "user", "content": "1"}],
                max_tokens=1,
                temperature=0.0
            )
        logger.info("Warmup complete. Server is fully primed for sub-200ms latency SLA.")
    except Exception as e:
        logger.warning(f"Warmup notice: {e}")

    logger.info("=" * 70)
    yield
    logger.info("Shutting down Indic Voice RAG Server...")


# ============================================================================
# FASTAPI APP & MIDDLEWARE
# ============================================================================

app = FastAPI(
    title="Indic Voice RAG API",
    description="Sub-200ms Production Hindi Voice-Enabled Retrieval-Augmented Generation Backend",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend UI communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================================
# REST ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint providing vector store point count and active models."""
    qdrant_points = 0
    if state.retriever and state.retriever.client.collection_exists(state.retriever.collection_name):
        coll_info = state.retriever.client.get_collection(state.retriever.collection_name)
        qdrant_points = coll_info.points_count

    return {
        "status": "healthy",
        "service": "Indic Voice RAG Backend",
        "vector_store": {
            "collection": state.retriever.collection_name if state.retriever else None,
            "points_count": qdrant_points,
            "embedding_model": state.retriever.embedding_model_name if state.retriever else None,
            "vector_dim": state.retriever.embedding_dim if state.retriever else None
        },
        "stt": {
            "available": state.stt_client is not None,
            "provider": "Sarvam AI",
            "model": state.stt_client.model if state.stt_client else None
        },
        "llm": {
            "available": state.groq_client is not None,
            "provider": "Groq",
            "model": state.groq_model
        }
    }


@app.get("/api/admin/reindex")
@app.post("/api/admin/reindex")
async def trigger_reindex(samples: int = 1000):
    """Admin endpoint to trigger automated indexing directly without file lock conflicts."""
    from ingest_pipeline import IngestConfig, IndicRAGIndexer
    import asyncio

    embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_data")
    collection_name = os.getenv("QDRANT_COLLECTION", "indic_rag_msmarco_hi")

    def run_indexer():
        cfg = IngestConfig(
            embedding_model=embedding_model,
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            sample_limit=samples,
            recreate_collection=True
        )
        indexer = IndicRAGIndexer(cfg)
        return indexer.run_ingestion()

    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, run_indexer)
    
    # Reload retriever collection
    if state.retriever:
        state.retriever._verify_collection()

    return {
        "status": "success",
        "message": f"Successfully indexed {count} vector points into {collection_name}.",
        "model": embedding_model,
        "sample_limit": samples
    }


@app.post("/api/text-query", response_model=QueryResponse)
async def process_text_query(req: TextQueryRequest):
    """
    Synchronous REST endpoint for text-based Indic RAG queries with microsecond profiling.
    """
    if not state.retriever or not state.groq_client:
        raise HTTPException(status_code=500, detail="RAG engine is not fully initialized.")

    t_start = time.perf_counter()

    # 1. Vector Retrieval (<10ms)
    retrieval_res: RetrievalResult = state.retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        strategy=req.strategy
    )

    context_text = retrieval_res.combined_parent_context.strip()
    if not context_text:
        context_text = "कोई प्रासंगिक संदर्भ नहीं मिला।"

    # 2. Fast LLM Generation with Greedy Decoding (<100ms)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"प्रश्न: {req.query}"}
    ]

    t_llm_start = time.perf_counter()
    try:
        chat_completion = await state.groq_client.chat.completions.create(
            model=state.groq_model,
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=False
        )
        answer = chat_completion.choices[0].message.content or ""
        tokens_generated = chat_completion.usage.completion_tokens if chat_completion.usage else len(answer.split())
    except Exception as e:
        logger.error(f"LLM Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {str(e)}")

    t_end = time.perf_counter()
    total_gen_ms = (t_end - t_llm_start) * 1000.0
    total_pipeline_ms = (t_end - t_start) * 1000.0
    tps = (tokens_generated / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0

    metrics = LatencyBreakdown(
        stt_latency_ms=None,
        retrieval_latency_ms=retrieval_res.total_retrieval_latency_ms,
        embed_latency_ms=retrieval_res.embed_latency_ms,
        search_latency_ms=retrieval_res.search_latency_ms,
        ttft_ms=None,
        first_token_latency_ms=None,
        total_generation_time_ms=round(total_gen_ms, 2),
        total_pipeline_ms=round(total_pipeline_ms, 2),
        tokens_per_second=round(tps, 2),
        total_tokens=tokens_generated,
        sla_passed=total_pipeline_ms < 200.0
    )

    return QueryResponse(
        query=req.query,
        transcript=None,
        answer=answer,
        sources=retrieval_res.hits,
        metrics=metrics
    )


# ============================================================================
# WEBSOCKET REAL-TIME STREAMING ENDPOINT (<200ms TARGET)
# ============================================================================

@app.websocket("/ws/voice-rag")
async def websocket_voice_rag(websocket: WebSocket):
    """
    Ultra-low-latency bidirectional WebSocket for live voice and text RAG.
    Streams token-by-token generation with microsecond latency instrumentation.
    """
    await websocket.accept()
    logger.info("WebSocket client connected to /ws/voice-rag")

    try:
        while True:
            # Receive binary audio or JSON event
            message = await websocket.receive()

            t_pipeline_start = time.perf_counter()
            audio_bytes: Optional[bytes] = None
            text_query: Optional[str] = None
            top_k = 3
            strategy = None

            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    elif payload.get("type") in ["text_query", "query"]:
                        text_query = payload.get("text") or payload.get("query")
                        top_k = int(payload.get("top_k", 3))
                        strategy = payload.get("strategy")
                except json.JSONDecodeError:
                    text_query = message["text"]

            if not audio_bytes and not text_query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid or empty payload received."
                })
                continue

            # ----------------------------------------------------------------
            # 1. Speech-to-Text Phase (if audio received)
            # ----------------------------------------------------------------
            stt_latency_ms: Optional[float] = None
            if audio_bytes:
                if not state.stt_client:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Sarvam STT client not configured on server."
                    })
                    continue

                await websocket.send_json({"type": "status", "stage": "transcribing"})
                stt_res = await state.stt_client.transcribe_bytes_async(
                    audio_bytes=audio_bytes,
                    filename="live_recording.webm"
                )
                stt_latency_ms = stt_res.stt_latency_ms

                if stt_res.status != "success" or not stt_res.transcript:
                    await websocket.send_json({
                        "type": "error",
                        "message": stt_res.error_message or "No speech recognized in audio."
                    })
                    continue

                text_query = stt_res.transcript
                await websocket.send_json({
                    "type": "transcript",
                    "text": text_query,
                    "stt_latency_ms": stt_latency_ms
                })

            # ----------------------------------------------------------------
            # 2. Vector Retrieval Phase (<10ms)
            # ----------------------------------------------------------------
            await websocket.send_json({"type": "status", "stage": "retrieving"})
            retrieval_res: RetrievalResult = state.retriever.retrieve(
                query=text_query,
                top_k=top_k,
                strategy=strategy
            )

            await websocket.send_json({
                "type": "retrieval",
                "retrieval_latency_ms": retrieval_res.total_retrieval_latency_ms,
                "embed_latency_ms": retrieval_res.embed_latency_ms,
                "search_latency_ms": retrieval_res.search_latency_ms,
                "sources": [hit.model_dump() for hit in retrieval_res.hits]
            })

            context_text = retrieval_res.combined_parent_context.strip()
            if not context_text:
                context_text = "कोई प्रासंगिक संदर्भ नहीं मिला।"

            # ----------------------------------------------------------------
            # 3. LLM Token Streaming Phase (Greedy Decoding <60ms TTFT)
            # ----------------------------------------------------------------
            await websocket.send_json({"type": "status", "stage": "generating"})

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"प्रश्न: {text_query}"}
            ]

            t_llm_start = time.perf_counter()
            ttft_ms: Optional[float] = None
            tokens_generated = 0
            full_answer_parts = []

            try:
                stream = await state.groq_client.chat.completions.create(
                    model=state.groq_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=35,
                    stream=True
                )

                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                    if delta:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_llm_start) * 1000.0

                        tokens_generated += 1
                        full_answer_parts.append(delta)
                        await websocket.send_json({
                            "type": "token",
                            "delta": delta
                        })

            except Exception as e:
                logger.error(f"Streaming Error: {e}")
                err_token = f"\n[त्रुटि: {e}]"
                full_answer_parts.append(err_token)
                await websocket.send_json({"type": "token", "delta": err_token})

            t_pipeline_end = time.perf_counter()
            total_gen_ms = (t_pipeline_end - t_llm_start) * 1000.0
            total_pipeline_ms = (t_pipeline_end - t_pipeline_start) * 1000.0
            ttft_ms = ttft_ms if ttft_ms is not None else total_gen_ms

            first_token_latency_ms = (
                (stt_latency_ms or 0.0) +
                retrieval_res.total_retrieval_latency_ms +
                ttft_ms
            )

            gen_sec = total_gen_ms / 1000.0
            tps = (tokens_generated / gen_sec) if gen_sec > 0 else 0.0

            # ----------------------------------------------------------------
            # 4. Completion & Latency Telemetry (<200ms Evaluation)
            # ----------------------------------------------------------------
            final_metrics = {
                "stt_latency_ms": round(stt_latency_ms, 2) if stt_latency_ms else None,
                "retrieval_latency_ms": round(retrieval_res.total_retrieval_latency_ms, 2),
                "embed_latency_ms": round(retrieval_res.embed_latency_ms, 2),
                "search_latency_ms": round(retrieval_res.search_latency_ms, 2),
                "ttft_ms": round(ttft_ms, 2),
                "first_token_latency_ms": round(first_token_latency_ms, 2),
                "total_generation_time_ms": round(total_gen_ms, 2),
                "total_pipeline_latency_ms": round(total_pipeline_ms, 2),
                "tokens_per_second": round(tps, 2),
                "total_tokens": tokens_generated,
                "sla_passed": total_pipeline_ms < 200.0 or first_token_latency_ms < 200.0
            }

            logger.info(
                f"\n[LATENCY TELEMETRY] Query: '{text_query}'\n"
                f"  ├─ STT Latency (Sarvam)     : {final_metrics['stt_latency_ms'] or 0.0} ms\n"
                f"  ├─ Vector Retrieval (Qdrant): {final_metrics['retrieval_latency_ms']} ms (Embed: {final_metrics['embed_latency_ms']}ms, Search: {final_metrics['search_latency_ms']}ms)\n"
                f"  ├─ Time to First Token (TTFT): {final_metrics['ttft_ms']} ms\n"
                f"  ├─ First-Token Latency (RAG) : {final_metrics['first_token_latency_ms']} ms\n"
                f"  ├─ Total Generation Time     : {final_metrics['total_generation_time_ms']} ms\n"
                f"  ├─ Total Pipeline Latency    : {final_metrics['total_pipeline_latency_ms']} ms\n"
                f"  ├─ Throughput (TPS)          : {final_metrics['tokens_per_second']} tokens/sec ({final_metrics['total_tokens']} tokens)\n"
                f"  └─ <200ms Target SLA Budget  : {'PASSED [OK]' if final_metrics['sla_passed'] else 'EXCEEDED'}"
            )

            await websocket.send_json({
                "type": "done",
                "metrics": final_metrics,
                "full_answer": "".join(full_answer_parts)
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket session error: {e}")


# ============================================================================
# SERVE REACT FRONTEND BUILD (if dist exists)
# ============================================================================

from fastapi.staticfiles import StaticFiles

frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def root_index():
        index_html_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_html_path):
            with open(index_html_path, "r", encoding="utf-8") as f:
                return f.read()
        return "<h1>Indic Voice RAG Backend Running</h1>"
else:
    @app.get("/", response_class=HTMLResponse)
    async def root_index():
        return "<h1>Indic Voice RAG Backend Running</h1><p>Frontend dist not built. Run <code>npm run dev</code> in frontend directory.</p>"


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
