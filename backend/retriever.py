#!/usr/bin/env python3
"""
Hybrid Indic Retrieval Engine (retriever.py)
============================================
HH Goa 2026 Task 2. Replaces single-shot dense search with a fused multi-strategy
retriever.

WHAT CHANGED AND WHY
--------------------
1. Embedding prefixes are now model-aware. E5 models are trained with
   "query: " / "passage: " markers; MiniLM and BGE are not. Applying E5 prefixes
   to a MiniLM index adds noise to one side of the comparison and degrades every
   search. The prefix is derived from the model name, never hardcoded.

2. Two index formats are supported. If index_manifest.json is present the
   retriever runs in HYBRID mode (named vectors, BM25 sparse, SQLite parent
   store). Otherwise it falls back to LEGACY mode and reads the old index
   unchanged. This means the repo keeps working before the new index is built.

3. Multi-strategy fusion. The old code searched once and let parent_child and
   sliding_window chunks of the same passage fight for the same top-k slots.
   Now chunks are fused with Reciprocal Rank Fusion, then grouped by parent, so
   a passage that ranks well under several chunkings rises instead of crowding
   itself out.

4. Dense + sparse hybrid. ~40% of MSMARCO-XI queries are NUMERIC / ENTITY /
   PERSON / LOCATION lookups where exact token match beats semantic similarity.
   BM25 runs alongside the dense search and both lists are fused.

5. Failures are typed. The old code caught every Qdrant exception and returned an
   empty result, so a dimension mismatch and a genuine no-match were
   indistinguishable. Now a broken index raises.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import sqlite3
import logging
from typing import List, Dict, Any, Optional, Sequence, Tuple

from pydantic import BaseModel, Field
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("IndicRetriever")


# ============================================================================
# ERRORS  (distinguishable failure modes)
# ============================================================================

class RetrieverError(RuntimeError):
    """Base class for retrieval failures."""


class IndexUnavailableError(RetrieverError):
    """Collection missing or empty. The server must not report itself healthy."""


class IndexMismatchError(RetrieverError):
    """Serving model disagrees with the model the index was built with."""


class SearchFailedError(RetrieverError):
    """Qdrant raised during a query. Never silently downgraded to 'no results'."""


# ============================================================================
# SCHEMAS
# ============================================================================

class RetrievedHit(BaseModel):
    score: float = Field(..., description="Fused relevance score (RRF, scale-free)")
    raw_score: float = Field(0.0, description="Best DENSE cosine similarity, 0..1. "
                                              "This is the semantic-relevance signal the "
                                              "retrieval guardrail thresholds on.")
    sparse_score: float = Field(0.0, description="Best BM25 score. Unbounded — never "
                                                 "mix with raw_score or threshold on it.")
    strategy: str = Field(..., description="Chunking strategy of the best-matching chunk")
    strategies_matched: List[str] = Field(default_factory=list,
                                          description="All strategies whose chunks matched this passage")
    child_text: str = Field("", description="Best matching chunk text")
    parent_id: str
    parent_text: str = Field("", description="Full passage used as LLM context")
    chunk_index: int = 0
    total_chunks: int = 1
    query_id: Optional[str] = None
    query_type: Optional[str] = Field(None, description="DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION")
    language: str = "hi"


class RetrievalResult(BaseModel):
    query: str
    combined_parent_context: str
    hits: List[RetrievedHit] = Field(default_factory=list)
    top_score: float = 0.0
    embed_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    parent_fetch_latency_ms: float = 0.0
    total_retrieval_latency_ms: float = 0.0
    mode: str = "legacy"
    transport: str = "local"
    cache_hit: bool = False
    stage_timings: List[Dict[str, Any]] = Field(default_factory=list,
        description="Per-stage latency, from profiling.Profiler")


# ============================================================================
# MODEL-AWARE PREFIXES
# ============================================================================

def prefixes_for_model(model_name: str) -> Tuple[str, str]:
    """
    Returns (query_prefix, passage_prefix).

    E5 is trained on asymmetric pairs and REQUIRES the markers. Everything else
    in play here (paraphrase-MiniLM, mpnet, BGE-M3) is trained without them, and
    adding them measurably hurts. Getting this wrong is silent — the vectors are
    still valid, just worse — which is exactly why it is derived, not configured.
    """
    n = (model_name or "").lower()
    if "e5" in n:
        return "query: ", "passage: "
    return "", ""


# ============================================================================
# QUERY NORMALISATION
# ============================================================================

# Sarvam returns Devanagari transliterations of English question words when the
# speaker code-mixes. Mapping them to native Hindi puts the query vector in the
# same region as the passages.
#
# NOTE ON BOUNDARIES: \b is defined in terms of \w, and Devanagari combining
# marks (nukta, matras) sit inside \w, so \b does NOT fire where a human would
# expect a word edge. The original patterns used \b and silently half-matched --
# "हाउ डज़" became "कैसे डज़" instead of "कैसे" because the trailing \b after the
# nukta in डज़ never matched. These lookarounds are the correct boundary for
# Devanagari.
_L = r'(?<![ऀ-ॿ])'      # not preceded by a Devanagari char
_R = r'(?![ऀ-ॿ])'       # not followed by a Devanagari char

PHONETIC_PATTERNS = [
    (r'(व्हाट\s+(?:इज़|इज|वाज़|आर|वर|अबाउट))', 'क्या है'),
    (r'(व्हाट्स|व्हाट)', 'क्या'),
    (r'(हाउ\s+(?:डज़|डस|डू|टू|कैन|इज़|इज))', 'कैसे'),
    (r'(हाउ)', 'कैसे'),
    (r'(हू\s+(?:इज़|इज|वाज़|वर))', 'कौन है'),
    (r'(हू)', 'कौन'),
    (r'(व्हेयर\s+(?:इज़|इज|वाज़|वर))', 'कहाँ है'),
    (r'(व्हेयर)', 'कहाँ'),
    (r'(व्हाय\s+(?:इज़|इज|डज़|डस))', 'क्यों'),
    (r'(व्हाय)', 'क्यों'),
    (r'(व्हेन\s+(?:इज़|इज|वाज़|वर))', 'कब'),
    (r'(व्हेन)', 'कब'),
    (r'(टेल\s+मी\s+अबाउट)', 'के बारे में बताएं'),
    (r'(एक्सप्लेन)', 'व्याख्या करें'),
    (r'(मीनिंग\s+ऑफ)', 'का अर्थ'),
    (r'(डेफिनेशन\s+ऑफ)', 'की परिभाषा'),
]
_COMPILED_PHONETIC = [
    (re.compile(_L + p + _R, re.IGNORECASE), r) for p, r in PHONETIC_PATTERNS
]
_WS = re.compile(r'\s+')


def normalize_query_text(query: str) -> str:
    norm = _WS.sub(" ", (query or "").strip())
    for rx, repl in _COMPILED_PHONETIC:
        norm = rx.sub(repl, norm)
    return norm.strip()


# ============================================================================
# PARENT STORE
# ============================================================================

class ParentStore:
    """
    Passage text lives here once instead of being copied onto every child vector.
    A primary-key lookup is ~50 microseconds, so this costs nothing measurable
    while removing several hundred megabytes from a full-size index.
    """

    def __init__(self, path: str):
        self.path = path
        # check_same_thread=False: FastAPI runs sync handlers on a threadpool.
        # Reads only, so concurrent access is safe.
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.count = self.conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def fetch_many(self, parent_ids: Sequence[str]) -> Dict[str, sqlite3.Row]:
        if not parent_ids:
            return {}
        marks = ",".join("?" * len(parent_ids))
        rows = self.conn.execute(
            f"SELECT parent_id, passage, query_id, query_type, gold_answer "
            f"FROM parents WHERE parent_id IN ({marks})",
            tuple(parent_ids),
        ).fetchall()
        return {r["parent_id"]: r for r in rows}


# ============================================================================
# RETRIEVER
# ============================================================================

class IndicRetriever:

    RRF_K = 60          # standard RRF damping constant
    OVERFETCH = 8       # chunks pulled per requested passage, before fusion

    def __init__(
        self,
        qdrant_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        manifest_path: Optional[str] = None,
        parent_db_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        strict: bool = True,
    ):
        base = os.path.dirname(os.path.abspath(__file__))
        self.qdrant_path = qdrant_path or os.getenv("QDRANT_PATH", "./qdrant_data")
        self.strict = strict

        # --- manifest decides which mode we run in -------------------------
        self.manifest_path = manifest_path or os.getenv(
            "INDEX_MANIFEST", os.path.join(base, "index_manifest.json"))
        self.manifest = self._load_manifest(self.manifest_path)
        self.mode = "hybrid" if self.manifest else "legacy"

        if self.manifest:
            self.collection_name = collection_name or self.manifest["collection"]

            # The manifest WINS over env/args. The index is the ground truth about
            # which model produced it, and a wrong model here is the worst class of
            # bug available: multilingual-e5-small and paraphrase-MiniLM are both
            # 384-dim, so a dimension check cannot catch the swap. Every search
            # would return confident nonsense with no error anywhere.
            self.embedding_model_name = self.manifest["model_name"]
            requested = embedding_model or os.getenv("EMBEDDING_MODEL")
            if requested and requested != self.embedding_model_name:
                raise IndexMismatchError(
                    f"EMBEDDING_MODEL is '{requested}' but the index was built with "
                    f"'{self.embedding_model_name}'. These may share a dimension, so "
                    f"nothing would raise at query time — results would just be wrong. "
                    f"Either unset EMBEDDING_MODEL or rebuild the index."
                )

            self.dense_vector_name = self.manifest.get("dense_vector_name", "dense")
            self.sparse_vector_name = self.manifest.get("sparse_vector_name")
            self.expected_dim = self.manifest.get("embed_dim")
            parent_db = parent_db_path or os.path.join(
                base, self.manifest.get("parent_db", "parents.sqlite"))
        else:
            self.collection_name = collection_name or os.getenv(
                "QDRANT_COLLECTION", "indic_rag_msmarco_hi")
            self.embedding_model_name = embedding_model or os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.dense_vector_name = None      # unnamed / default vector
            self.sparse_vector_name = None
            self.expected_dim = None
            parent_db = None

        self.query_prefix, self.passage_prefix = prefixes_for_model(self.embedding_model_name)

        self._embed_cache: Dict[str, List[float]] = {}
        self._max_cache_size = 2000

        try:
            from qdrant_client import QdrantClient, models
            self._models = models
        except ImportError as e:
            raise RetrieverError(f"qdrant-client missing: {e}. pip install -r requirements.txt")

        self._init_encoder()

        # --- transport: server URL beats local path ------------------------
        # qdrant-client's local (path=) mode is documented for <20,000 points and
        # does BRUTE-FORCE numpy search -- hnsw_config is accepted and ignored. At
        # 720k vectors that is a linear scan per query and the 200ms budget is
        # gone. A real server builds an actual HNSW index, so search stays in
        # single-digit ms as the corpus grows.
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL") or None
        if self.qdrant_url:
            logger.info(f"Connecting to Qdrant server at {self.qdrant_url} (mode={self.mode})...")
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=os.getenv("QDRANT_API_KEY") or None,
                prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true",
                timeout=float(os.getenv("QDRANT_TIMEOUT_S", "10")),
            )
            self.transport = "server"
        else:
            logger.warning(
                f"No QDRANT_URL set — falling back to local file mode at '{self.qdrant_path}'. "
                f"This does brute-force search and is only viable below ~20,000 points."
            )
            self.client = QdrantClient(path=self.qdrant_path)
            self.transport = "local"

        # HNSW ef: higher = better recall, slower search. Server mode only.
        self.hnsw_ef = int(os.getenv("HNSW_EF", "128"))

        # The two storage formats are not interchangeable. Catch the mismatch here
        # rather than letting it surface as a confusing "collection not found".
        expected = (self.manifest or {}).get("transport")
        if expected and expected != self.transport:
            raise IndexUnavailableError(
                f"The index was built for transport '{expected}' but this process is "
                f"using '{self.transport}'. Qdrant server storage and local-mode "
                f"storage are different formats.\n"
                + ("Set QDRANT_URL to a running Qdrant server and restore the snapshot."
                   if expected == "server" else
                   "Unset QDRANT_URL to read this local-mode index from disk.")
            )

        self._verify_collection()

        # --- parent store (hybrid mode only) -------------------------------
        self.parents: Optional[ParentStore] = None
        if parent_db and os.path.exists(parent_db):
            self.parents = ParentStore(parent_db)
            logger.info(f"Parent store loaded: {self.parents.count:,} passages")
        elif self.mode == "hybrid":
            raise IndexUnavailableError(
                f"Manifest declares parent store '{parent_db}' but the file is missing. "
                f"Download parents.sqlite alongside qdrant_data/."
            )

        # --- sparse encoder (hybrid mode only) -----------------------------
        self.bm25 = None
        if self.sparse_vector_name:
            try:
                from fastembed import SparseTextEmbedding
                self.bm25 = SparseTextEmbedding(model_name="Qdrant/bm25", disable_stemmer=True)
                list(self.bm25.embed(["warmup"]))
                logger.info("BM25 sparse encoder ready (hybrid retrieval enabled)")
            except Exception as e:
                logger.warning(f"BM25 unavailable, falling back to dense-only: {e}")
                self.sparse_vector_name = None

    # ------------------------------------------------------------------ init

    @staticmethod
    def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            logger.info(f"Index manifest found: model={m.get('model_name')} "
                        f"dim={m.get('embed_dim')} built={m.get('built_at')}")
            return m
        except Exception as e:
            logger.error(f"Manifest at '{path}' is unreadable: {e}")
            return None

    @staticmethod
    def _builtin_model_names() -> set:
        """
        fastembed changed this return type across versions: older releases yield
        dicts, 0.5+ yields dataclasses. Handle both rather than pinning a version.
        """
        from fastembed import TextEmbedding
        names = set()
        try:
            for m in TextEmbedding.list_supported_models():
                names.add(m["model"] if isinstance(m, dict) else getattr(m, "model", ""))
        except Exception as e:
            logger.warning(f"Could not enumerate fastembed built-ins: {e}")
        return names

    def _init_encoder(self):
        from fastembed import TextEmbedding

        # fastembed ships only a handful of multilingual models by default.
        # e5-small/base have official ONNX exports but are not in the built-in
        # list, so they are registered on the fly.
        if self.embedding_model_name not in self._builtin_model_names():
            try:
                from fastembed.common.model_description import PoolingType, ModelSource
                dim = self.expected_dim or (384 if "small" in self.embedding_model_name else 768)
                TextEmbedding.add_custom_model(
                    model=self.embedding_model_name,
                    pooling=PoolingType.MEAN,
                    normalization=True,
                    sources=ModelSource(hf=self.embedding_model_name),
                    dim=dim,
                    model_file="onnx/model.onnx",
                )
                logger.info(f"Registered custom model '{self.embedding_model_name}' (dim={dim})")
            except Exception as e:
                raise IndexMismatchError(
                    f"Cannot load embedding model '{self.embedding_model_name}': {e}"
                )

        logger.info(f"Loading encoder '{self.embedding_model_name}' "
                    f"(query_prefix={self.query_prefix!r})...")
        t0 = time.perf_counter()
        self.embed_model = TextEmbedding(model_name=self.embedding_model_name)
        warm = list(self.embed_model.embed([self.query_prefix + "warmup"]))[0]
        self.embedding_dim = len(warm)
        logger.info(f"Encoder ready in {time.perf_counter() - t0:.2f}s | dim={self.embedding_dim}")

        if self.expected_dim and self.embedding_dim != self.expected_dim:
            raise IndexMismatchError(
                f"Model '{self.embedding_model_name}' produces {self.embedding_dim}-dim vectors "
                f"but the index was built at {self.expected_dim} dims."
            )

    def _verify_collection(self):
        """
        Loud on failure. The old version logged a warning and carried on, so a
        missing index produced a healthy-looking server that answered every
        question with 'no information available'.
        """
        if not self.client.collection_exists(self.collection_name):
            where = self.qdrant_url or self.qdrant_path
            msg = (f"Collection '{self.collection_name}' not found at '{where}'. "
                   f"Build it with build_index_gpu.py and restore the snapshot. "
                   f"(ingest_pipeline.py still works but produces the LEGACY format: "
                   f"unnamed vectors, no manifest, no BM25 — usable only in local mode.)")
            if self.strict:
                raise IndexUnavailableError(msg)
            logger.error(msg)
            return

        info = self.client.get_collection(self.collection_name)
        self.points_count = info.points_count or 0

        if self.points_count == 0:
            msg = f"Collection '{self.collection_name}' exists but holds 0 vectors."
            if self.strict:
                raise IndexUnavailableError(msg)
            logger.error(msg)
            return

        # Dimension check against what the collection actually stores.
        try:
            vec_cfg = info.config.params.vectors
            stored_dim = (vec_cfg.size if hasattr(vec_cfg, "size")
                          else vec_cfg[self.dense_vector_name].size)
            if stored_dim != self.embedding_dim:
                raise IndexMismatchError(
                    f"Index stores {stored_dim}-dim vectors but the encoder produces "
                    f"{self.embedding_dim}. The index was built with a different model — "
                    f"every search would return meaningless results. Rebuild or switch models."
                )
        except IndexMismatchError:
            raise
        except Exception as e:
            logger.warning(f"Could not verify stored vector dimension: {e}")

        logger.info(f"Collection '{self.collection_name}' ready | {self.points_count:,} vectors")

    # ------------------------------------------------------------- embedding

    def _get_embedding(self, text: str) -> Tuple[List[float], bool]:
        if text in self._embed_cache:
            return self._embed_cache[text], True

        emb = list(self.embed_model.embed([self.query_prefix + text]))[0].tolist()

        if len(self._embed_cache) >= self._max_cache_size:
            del self._embed_cache[next(iter(self._embed_cache))]
        self._embed_cache[text] = emb
        return emb, False

    # --------------------------------------------------------------- fusion

    @staticmethod
    def _rrf(rank: int, k: int = RRF_K) -> float:
        return 1.0 / (k + rank + 1)

    def _fuse(self, ranked_lists: Sequence[Sequence[Any]]) -> Dict[Any, float]:
        """
        Reciprocal Rank Fusion over N ranked lists of point ids.

        RRF is used rather than score averaging because dense cosine scores and
        BM25 scores live on incompatible scales; ranks are directly comparable.
        """
        fused: Dict[Any, float] = {}
        for lst in ranked_lists:
            for rank, pid in enumerate(lst):
                fused[pid] = fused.get(pid, 0.0) + self._rrf(rank)
        return fused

    # -------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        strategy: Optional[str] = None,
        score_threshold: Optional[float] = None,
        query_type: Optional[str] = None,
        normalize: bool = True,
        profiler: Optional["Profiler"] = None,
    ) -> RetrievalResult:
        """
        Args:
            query:           question text (Hindi / English / code-mixed)
            top_k:           number of distinct PASSAGES to return
            strategy:        optional filter to one chunking strategy
            score_threshold: minimum raw cosine similarity; hits below are dropped.
                             This is the off-topic gate — wire it to
                             guardrails.RetrievalGate rather than leaving it unused.
            query_type:      optional metadata filter (DESCRIPTION / NUMERIC / ...)
            profiler:        optional Profiler; each phase is recorded as its own
                             stage so the 200ms budget can be attributed exactly.
        """
        from profiling import Profiler  # local import keeps the module dependency-free

        prof = profiler if profiler is not None else Profiler(label="retrieval")
        t_start = time.perf_counter()

        with prof.stage("query_normalize"):
            clean = normalize_query_text(query) if normalize else (query or "").strip()

        if not clean:
            return RetrievalResult(query=query or "", combined_parent_context="",
                                   mode=self.mode)

        # 1. embed -----------------------------------------------------------
        t0 = time.perf_counter()
        with prof.stage("embed_query") as st:
            qvec, cache_hit = self._get_embedding(clean)
            st.detail = "cache hit" if cache_hit else "encoded"
        embed_ms = (time.perf_counter() - t0) * 1000

        # 2. filters ---------------------------------------------------------
        conditions = []
        if strategy and strategy.lower() not in ("all", "none", "", "best match"):
            conditions.append(self._models.FieldCondition(
                key="strategy",
                match=self._models.MatchValue(value=self._canonical_strategy(strategy))))
        if query_type:
            conditions.append(self._models.FieldCondition(
                key="query_type",
                match=self._models.MatchValue(value=query_type.upper())))
        qfilter = self._models.Filter(must=conditions) if conditions else None

        # 3. search ----------------------------------------------------------
        limit = max(top_k * self.OVERFETCH, 24)
        # ef only applies to a real HNSW index; local brute-force mode ignores it.
        search_params = (self._models.SearchParams(hnsw_ef=self.hnsw_ef)
                         if self.transport == "server" else None)

        t0 = time.perf_counter()
        try:
            with prof.stage("search_dense") as st:
                dense_points = self.client.query_points(
                    collection_name=self.collection_name,
                    query=qvec,
                    using=self.dense_vector_name,
                    query_filter=qfilter,
                    limit=limit,
                    score_threshold=score_threshold,
                    search_params=search_params,
                    with_payload=True,
                ).points
                st.detail = f"{len(dense_points)} chunks"

            sparse_points = []
            if self.sparse_vector_name and self.bm25 is not None:
                with prof.stage("bm25_encode"):
                    sv = list(self.bm25.embed([clean]))[0]
                if len(sv.indices):
                    with prof.stage("search_sparse") as st:
                        sparse_points = self.client.query_points(
                            collection_name=self.collection_name,
                            query=self._models.SparseVector(
                                indices=sv.indices.tolist(), values=sv.values.tolist()),
                            using=self.sparse_vector_name,
                            query_filter=qfilter,
                            limit=limit,
                            with_payload=True,
                        ).points
                        st.detail = f"{len(sparse_points)} chunks"
        except Exception as e:
            raise SearchFailedError(f"Qdrant query failed: {e}") from e
        search_ms = (time.perf_counter() - t0) * 1000

        # 4. fuse ------------------------------------------------------------
        t0 = time.perf_counter()
        prof_fusion = prof.stage("fusion_rrf")
        prof_fusion.__enter__()
        by_id: Dict[Any, Any] = {}
        for p in list(dense_points) + list(sparse_points):
            by_id.setdefault(p.id, p)

        # Dense and sparse scores live on INCOMPATIBLE SCALES: dense is cosine in
        # [0,1], BM25 is unbounded (observed up to 85). Keeping them in one field
        # made raw_score meaningless and silently broke the retrieval guardrail —
        # MIN_RETRIEVAL_SCORE=0.80 was being compared against values from 0.84 to
        # 85, so nothing was ever refused. They are tracked separately now, and
        # only the cosine is used as the semantic-relevance signal.
        dense_scores = {p.id: float(p.score or 0.0) for p in dense_points}
        sparse_scores = {p.id: float(p.score or 0.0) for p in sparse_points}

        fused = self._fuse([
            [p.id for p in dense_points],
            [p.id for p in sparse_points],
        ])

        # Group chunks under their passage. A passage matched by several
        # strategies accumulates several RRF contributions, which is exactly the
        # signal multi-strategy chunking is supposed to produce.
        parents: Dict[str, Dict[str, Any]] = {}
        for pid, fscore in fused.items():
            p = by_id[pid]
            payload = p.payload or {}
            parent_id = str(payload.get("parent_id", pid))
            entry = parents.setdefault(parent_id, {
                "score": 0.0, "raw": 0.0, "sparse": 0.0, "best_fused": -1.0,
                "strategies": set(), "best_chunk": "", "best_strategy": "",
                "chunk_index": 0, "total_chunks": 1, "payload": payload,
            })
            entry["score"] += fscore
            entry["raw"] = max(entry["raw"], dense_scores.get(pid, 0.0))
            entry["sparse"] = max(entry["sparse"], sparse_scores.get(pid, 0.0))

            # Representative chunk is picked by FUSED rank, not by raw score:
            # fused rank is the only scale-free comparison available across a
            # dense hit and a sparse one.
            if fscore > entry["best_fused"]:
                entry["best_fused"] = fscore
                entry["best_chunk"] = payload.get("chunk_text", "")
                entry["best_strategy"] = payload.get("strategy", "unknown")
                entry["chunk_index"] = payload.get("chunk_index", 0)
                entry["total_chunks"] = payload.get(
                    "n_chunks", payload.get("total_chunks_in_parent", 1))
                entry["payload"] = payload
            st = payload.get("strategy")
            if st:
                entry["strategies"].add(st)

        ordered = sorted(parents.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]
        prof_fusion.__exit__(None, None, None)
        fusion_ms = (time.perf_counter() - t0) * 1000

        # 5. resolve passage text --------------------------------------------
        t0 = time.perf_counter()
        prof_parent = prof.stage("parent_fetch")
        prof_parent.__enter__()
        texts: Dict[str, str] = {}
        if self.parents is not None:
            rows = self.parents.fetch_many([pid for pid, _ in ordered])
            texts = {k: r["passage"] for k, r in rows.items()}
        else:
            # legacy index carries the full passage in every payload
            texts = {pid: (e["payload"].get("parent_text") or e["payload"].get("chunk_text", ""))
                     for pid, e in ordered}
        prof_parent.__exit__(None, None, None)
        parent_ms = (time.perf_counter() - t0) * 1000

        hits: List[RetrievedHit] = []
        contexts: List[str] = []
        for parent_id, e in ordered:
            ptext = (texts.get(parent_id) or e["best_chunk"] or "").strip()
            payload = e["payload"]
            hits.append(RetrievedHit(
                score=round(e["score"], 6),
                raw_score=round(e["raw"], 4),
                sparse_score=round(e["sparse"], 4),
                strategy=e["best_strategy"],
                strategies_matched=sorted(e["strategies"]),
                child_text=e["best_chunk"],
                parent_id=parent_id,
                parent_text=ptext,
                chunk_index=e["chunk_index"],
                total_chunks=e["total_chunks"],
                query_id=str(payload.get("query_id", "")) or None,
                query_type=payload.get("query_type"),
                language=payload.get("lang", payload.get("language", "hi")),
            ))
            if ptext:
                contexts.append(ptext)

        total_ms = (time.perf_counter() - t_start) * 1000
        return RetrievalResult(
            query=clean,
            combined_parent_context="\n\n".join(contexts),
            hits=hits,
            top_score=round(max((h.raw_score for h in hits), default=0.0), 4),
            embed_latency_ms=round(embed_ms, 2),
            search_latency_ms=round(search_ms, 2),
            fusion_latency_ms=round(fusion_ms, 2),
            parent_fetch_latency_ms=round(parent_ms, 2),
            total_retrieval_latency_ms=round(total_ms, 2),
            mode=self.mode,
            transport=self.transport,
            cache_hit=cache_hit,
            stage_timings=[st.to_dict() for st in prof.stages],
        )

    @staticmethod
    def _canonical_strategy(s: str) -> str:
        s = s.lower()
        if "parent" in s or "sentence" in s:
            return "sentence"
        if "window" in s or "sliding" in s:
            return "window"
        if "semantic" in s:
            return "semantic"
        return "passage"


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  INDIC RETRIEVER — VERIFICATION")
    print("=" * 72)

    # Prefix derivation is pure logic and worth checking without loading a model.
    cases = [
        ("intfloat/multilingual-e5-small", ("query: ", "passage: ")),
        ("intfloat/multilingual-e5-large", ("query: ", "passage: ")),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ("", "")),
        ("BAAI/bge-m3", ("", "")),
    ]
    print("\nPrefix derivation:")
    for name, want in cases:
        got = prefixes_for_model(name)
        print(f"  {'OK  ' if got == want else 'FAIL'} {name:<58} -> {got}")

    print("\nQuery normalisation:")
    for q in ["व्हाट इज कॉर्पोरेशन", "हाउ डज़ कंप्यूटर वर्क", "  भारत   की राजधानी  "]:
        print(f"  {q!r:<40} -> {normalize_query_text(q)!r}")

    print("\nConnecting to index...")
    try:
        r = IndicRetriever(strict=True)
    except RetrieverError as e:
        print(f"\n  {type(e).__name__}: {e}\n")
        sys.exit(1)

    for q in ["दवा कैसे काम करती है?", "कॉर्पोरेशन क्या है?", "संगतता की परिभाषा क्या है?"]:
        res = r.retrieve(q, top_k=3)
        print(f"\nQuery: {q}")
        print(f"  mode={res.mode} | embed={res.embed_latency_ms}ms "
              f"search={res.search_latency_ms}ms fuse={res.fusion_latency_ms}ms "
              f"parent={res.parent_fetch_latency_ms}ms TOTAL={res.total_retrieval_latency_ms}ms")
        for i, h in enumerate(res.hits, 1):
            print(f"   #{i} raw={h.raw_score:.4f} fused={h.score:.5f} "
                  f"strategies={h.strategies_matched} | {h.parent_text[:80]}...")

    print("\n" + "=" * 72)
