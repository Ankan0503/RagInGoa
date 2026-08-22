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


class RetrieverError(RuntimeError):
    pass


class IndexUnavailableError(RetrieverError):
    pass


class IndexMismatchError(RetrieverError):
    pass


class SearchFailedError(RetrieverError):
    pass


class RetrievedHit(BaseModel):
    score: float = Field(..., description="Fused relevance score (RRF)")
    raw_score: float = Field(0.0, description="Best dense cosine similarity, 0..1")
    sparse_score: float = Field(0.0, description="Best BM25 score")
    strategy: str = Field(..., description="Chunking strategy of the best-matching chunk")
    strategies_matched: List[str] = Field(default_factory=list, description="All strategies whose chunks matched")
    child_text: str = Field("", description="Best matching chunk text")
    parent_id: str
    parent_text: str = Field("", description="Full passage used as LLM context")
    chunk_index: int = 0
    total_chunks: int = 1
    query_id: Optional[str] = None
    query_type: Optional[str] = Field(None, description="Query type metadata")
    language: str = "hi"


class RetrievalResult(BaseModel):
    query: str
    combined_parent_context: str
    hits: List[RetrievedHit] = Field(default_factory=list)
    top_score: float = 0.0
    score_margin: float = Field(0.0, description="Margin between top score and candidate field mean")
    margin_candidates: int = Field(0, description="Distinct passages margin was computed over")
    embed_latency_ms: float = 0.0
    search_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    parent_fetch_latency_ms: float = 0.0
    total_retrieval_latency_ms: float = 0.0
    mode: str = "legacy"
    transport: str = "local"
    cache_hit: bool = False
    stage_timings: List[Dict[str, Any]] = Field(default_factory=list, description="Per-stage latency")


def prefixes_for_model(model_name: str) -> Tuple[str, str]:
    n = (model_name or "").lower()
    if "e5" in n:
        return "query: ", "passage: "
    return "", ""


_L = r'(?<![ऀ-ॿ])'
_R = r'(?![ऀ-ॿ])'

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


class ParentStore:
    def __init__(self, path: str):
        self.path = path
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


class IndicRetriever:
    RRF_K = 60
    OVERFETCH = 8
    MARGIN_WINDOW = 10

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

        self.manifest_path = manifest_path or os.getenv(
            "INDEX_MANIFEST", os.path.join(base, "index_manifest.json"))
        self.manifest = self._load_manifest(self.manifest_path)
        self.mode = "hybrid" if self.manifest else "legacy"

        if self.manifest:
            self.collection_name = collection_name or self.manifest["collection"]
            self.embedding_model_name = self.manifest["model_name"]
            requested = embedding_model or os.getenv("EMBEDDING_MODEL")
            if requested and requested != self.embedding_model_name:
                raise IndexMismatchError(
                    f"EMBEDDING_MODEL is '{requested}' but index was built with '{self.embedding_model_name}'."
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
            self.dense_vector_name = None
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
            logger.warning(f"No QDRANT_URL set — falling back to local file mode at '{self.qdrant_path}'.")
            self.client = QdrantClient(path=self.qdrant_path)
            self.transport = "local"

        self.hnsw_ef = int(os.getenv("HNSW_EF", "128"))

        # Retrieval cost knobs, all env-tunable so they can be A/B'd against
        # benchmark.py's answer_recall_mean without a redeploy.
        #
        # Measured on this deployment (3.43M vectors, 2 vCPU, cold cache --
        # i.e. a different query vector each time, which is what production
        # actually does):
        #     search_dense   ~40ms
        #     search_sparse  ~104ms   <- 2.6x dense, the dominant cost
        # Warm/repeated vectors measure ~9ms, so any benchmark that reuses one
        # vector will badly understate all of this.
        self.candidate_limit = int(os.getenv("CANDIDATE_LIMIT", "24"))

        # on   -- always run BM25 (default; unchanged behaviour)
        # off  -- never run it; dense-only retrieval
        # auto -- skip it when dense retrieval is already confident, paying
        #         the ~104ms only for queries that actually need lexical
        #         matching (rare terms, proper nouns, numbers).
        self.sparse_mode = (os.getenv("SPARSE_MODE", "on") or "on").lower().strip()
        self.sparse_skip_score = float(os.getenv("SPARSE_SKIP_SCORE", "0.90"))

        # Qdrant re-reads the original fp32 vectors from disk to rescore
        # quantized results. Measured: median 6.6ms -> 6.1ms, but the tail
        # collapses from 77.5ms to 6.4ms, so disabling it mainly buys
        # consistency. Costs a little ranking accuracy (int8 scores).
        self.quant_rescore = (os.getenv("QUANT_RESCORE", "true").lower() != "false")

        expected = (self.manifest or {}).get("transport")
        if expected and expected != self.transport:
            raise IndexUnavailableError(
                f"The index was built for transport '{expected}' but process is using '{self.transport}'."
            )

        self._verify_collection()

        self.parents: Optional[ParentStore] = None
        if parent_db and os.path.exists(parent_db):
            self.parents = ParentStore(parent_db)
            logger.info(f"Parent store loaded: {self.parents.count:,} passages")
        elif self.mode == "hybrid":
            raise IndexUnavailableError(f"Parent store '{parent_db}' missing.")

        self.bm25 = None
        if self.sparse_vector_name:
            try:
                from fastembed import SparseTextEmbedding
                self.bm25 = SparseTextEmbedding(model_name="Qdrant/bm25", disable_stemmer=True)
                list(self.bm25.embed(["warmup"]))
                logger.info("BM25 sparse encoder ready")
            except Exception as e:
                logger.warning(f"BM25 unavailable: {e}")
                self.sparse_vector_name = None

    @staticmethod
    def _load_manifest(path: str) -> Optional[Dict[str, Any]]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                m = json.load(f)
            logger.info(f"Index manifest found: model={m.get('model_name')} dim={m.get('embed_dim')}")
            return m
        except Exception as e:
            logger.error(f"Manifest at '{path}' is unreadable: {e}")
            return None

    @staticmethod
    def _builtin_model_names() -> set:
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
                raise IndexMismatchError(f"Cannot load embedding model '{self.embedding_model_name}': {e}")

        logger.info(f"Loading encoder '{self.embedding_model_name}'...")
        t0 = time.perf_counter()
        self.embed_model = TextEmbedding(model_name=self.embedding_model_name)
        warm = list(self.embed_model.embed([self.query_prefix + "warmup"]))[0]
        self.embedding_dim = len(warm)
        logger.info(f"Encoder ready in {time.perf_counter() - t0:.2f}s | dim={self.embedding_dim}")

        if self.expected_dim and self.embedding_dim != self.expected_dim:
            raise IndexMismatchError(
                f"Model '{self.embedding_model_name}' produces {self.embedding_dim}-dim vectors but index was built at {self.expected_dim} dims."
            )

    def _verify_collection(self):
        if not self.client.collection_exists(self.collection_name):
            where = self.qdrant_url or self.qdrant_path
            msg = f"Collection '{self.collection_name}' not found at '{where}'."
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

        try:
            vec_cfg = info.config.params.vectors
            stored_dim = (vec_cfg.size if hasattr(vec_cfg, "size")
                          else vec_cfg[self.dense_vector_name].size)
            if stored_dim != self.embedding_dim:
                raise IndexMismatchError(
                    f"Index stores {stored_dim}-dim vectors but encoder produces {self.embedding_dim}."
                )
        except IndexMismatchError:
            raise
        except Exception as e:
            logger.warning(f"Could not verify stored vector dimension: {e}")

        logger.info(f"Collection '{self.collection_name}' ready | {self.points_count:,} vectors")

    def _get_embedding(self, text: str) -> Tuple[List[float], bool]:
        if text in self._embed_cache:
            return self._embed_cache[text], True

        emb = list(self.embed_model.embed([self.query_prefix + text]))[0].tolist()

        if len(self._embed_cache) >= self._max_cache_size:
            del self._embed_cache[next(iter(self._embed_cache))]
        self._embed_cache[text] = emb
        return emb, False

    @staticmethod
    def _rrf(rank: int, k: int = RRF_K) -> float:
        return 1.0 / (k + rank + 1)

    def _fuse(self, ranked_lists: Sequence[Sequence[Any]]) -> Dict[Any, float]:
        fused: Dict[Any, float] = {}
        for lst in ranked_lists:
            for rank, pid in enumerate(lst):
                fused[pid] = fused.get(pid, 0.0) + self._rrf(rank)
        return fused

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
        from profiling import Profiler

        prof = profiler if profiler is not None else Profiler(label="retrieval")
        t_start = time.perf_counter()

        with prof.stage("query_normalize"):
            clean = normalize_query_text(query) if normalize else (query or "").strip()

        if not clean:
            return RetrievalResult(query=query or "", combined_parent_context="", mode=self.mode)

        t0 = time.perf_counter()
        with prof.stage("embed_query") as st:
            qvec, cache_hit = self._get_embedding(clean)
            st.detail = "cache hit" if cache_hit else "encoded"
        embed_ms = (time.perf_counter() - t0) * 1000

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

        limit = max(top_k * self.OVERFETCH, self.candidate_limit)
        search_params = None
        if self.transport == "server":
            search_params = self._models.SearchParams(
                hnsw_ef=self.hnsw_ef,
                quantization=self._models.QuantizationSearchParams(
                    rescore=self.quant_rescore),
            )

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
            run_sparse = (self.sparse_vector_name is not None
                          and self.bm25 is not None
                          and self.sparse_mode != "off")
            skip_reason = ""
            if run_sparse and self.sparse_mode == "auto" and dense_points:
                # Dense already found something it is confident about, so the
                # ~104ms lexical scan is unlikely to change the answer. Uses
                # the dense cosine directly (not the fused score, which does
                # not exist yet at this point).
                if dense_points[0].score >= self.sparse_skip_score:
                    run_sparse = False
                    skip_reason = f"skipped: dense {dense_points[0].score:.3f} >= {self.sparse_skip_score}"

            if not run_sparse:
                # Still emit the stage so the Insights breakdown shows why it
                # was not run, rather than the row silently disappearing.
                with prof.stage("search_sparse") as st:
                    st.detail = skip_reason or f"skipped: SPARSE_MODE={self.sparse_mode}"

            if run_sparse:
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

        t0 = time.perf_counter()
        prof_fusion = prof.stage("fusion_rrf")
        prof_fusion.__enter__()
        by_id: Dict[Any, Any] = {}
        for p in list(dense_points) + list(sparse_points):
            by_id.setdefault(p.id, p)

        dense_scores = {p.id: float(p.score or 0.0) for p in dense_points}
        sparse_scores = {p.id: float(p.score or 0.0) for p in sparse_points}

        fused = self._fuse([
            [p.id for p in dense_points],
            [p.id for p in sparse_points],
        ])

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

        all_raw = sorted((e["raw"] for e in parents.values()), reverse=True)
        top_raw = all_raw[0] if all_raw else 0.0
        _rest = all_raw[1:1 + self.MARGIN_WINDOW]
        mean_rest = (sum(_rest) / len(_rest)) if _rest else top_raw
        score_margin = round(max(0.0, top_raw - mean_rest), 4)
        margin_candidates = len(all_raw)

        ordered = sorted(parents.items(), key=lambda kv: kv[1]["score"], reverse=True)[:top_k]
        prof_fusion.__exit__(None, None, None)
        fusion_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        prof_parent = prof.stage("parent_fetch")
        prof_parent.__enter__()
        texts: Dict[str, str] = {}
        if self.parents is not None:
            rows = self.parents.fetch_many([pid for pid, _ in ordered])
            texts = {k: r["passage"] for k, r in rows.items()}
        else:
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
            score_margin=score_margin,
            margin_candidates=margin_candidates,
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


if __name__ == "__main__":
    print("=" * 72)
    print("  INDIC RETRIEVER — VERIFICATION")
    print("=" * 72)

    cases = [
        ("intfloat/multilingual-e5-small", ("query: ", "passage: ")),
        ("intfloat/multilingual-e5-large", ("query: ", "passage: ")),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ("", "")),
        ("BAAI/bge-m3", ("", "")),
    ]
    for name, want in cases:
        got = prefixes_for_model(name)
        print(f"  {'OK  ' if got == want else 'FAIL'} {name:<58} -> {got}")

    try:
        r = IndicRetriever(strict=True)
        for q in ["दवा कैसे काम करती है?", "कॉर्पोरेशन क्या है?"]:
            res = r.retrieve(q, top_k=3)
            print(f"\nQuery: {q}")
            print(f"  mode={res.mode} | TOTAL={res.total_retrieval_latency_ms}ms | top_score={res.top_score}")
    except RetrieverError as e:
        print(f"\n  {type(e).__name__}: {e}\n")
