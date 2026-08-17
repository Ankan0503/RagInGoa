#!/usr/bin/env python3
"""
Production-Grade Indic RAG Ingestion & Indexing Pipeline (MSMARCO-XI Hindi)
===========================================================================
Designed for HH Goa 2026 Shortlisting Task 2: Sub-200ms Voice-Enabled RAG System.

Key Engineering Highlights:
1. Indic-Aware Sentence Segmentation: Handles '।', '॥', '?', '!', '.', and '\n'.
2. Strategy Pattern for Multi-Strategy Chunking:
   - Hierarchical Parent-Child: High-precision child sentence vectors linked to full parent passages.
   - Sliding Window with Overlap: Context-preserving overlapping windows (e.g. 2-sentence window, 1 overlap).
3. Metadata-Aware Qdrant Payload Design for instant parent context retrieval & filtering.
4. Ultra-Fast ONNX Dense Embeddings via FastEmbed.
5. Local On-Disk Qdrant Vector Store with Cosine distance.
6. Built-in Benchmark Harness with P50 / P70 / P90 / P100 latency analytics (<200ms target).
"""

import os
import sys
import io
import re
import time
import uuid
import logging
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("IndicRAGIngest")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IngestConfig:
    """Configuration settings for MSMARCO-XI Indic Ingestion & Indexing."""
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_split: str = "validation"  # 'validation' (hinval.parquet) or 'train' (hintrain.parquet)
    sample_limit: int = 100  # Number of samples to ingest (0 for all)
    collection_name: str = "indic_rag_msmarco_hi"
    qdrant_path: str = "./qdrant_data"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: Optional[int] = None  # Auto-detected if None
    embedding_batch_size: int = 16
    upsert_batch_size: int = 32
    window_sentences: int = 2
    overlap_sentences: int = 1
    min_chunk_chars: int = 15
    strategy: str = "all"  # 'parent_child', 'sliding_window', or 'all'
    index_all_passages: bool = False  # If True, index all 10 passages per query; If False, index positive/selected passages


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ChunkPayload:
    """Rich metadata payload stored alongside each vector point in Qdrant."""
    chunk_id: str
    strategy: str
    parent_id: str
    parent_text: str
    chunk_text: str
    chunk_index: int
    total_chunks_in_parent: int
    char_count: int
    language: str = "hi"
    source_dataset: str = "ai4bharat/MSMARCO-XI"
    query_id: str = ""
    is_selected_passage: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# INDIC SENTENCE SEGMENTER
# ============================================================================

class IndicSentenceSplitter:
    """
    Indic-Aware Sentence Segmenter.
    Splits text on Devanagari danda ('।'), double danda ('॥'),
    as well as standard punctuation ('?', '!', '.', '\n').
    """
    
    BOUNDARY_REGEX = re.compile(r'([।॥?!.\n]+)')

    @classmethod
    def split(cls, text: str, min_chars: int = 15) -> List[str]:
        """
        Segments raw Hindi text into clean sentences while preserving punctuation
        and filtering out noise/fragments with character count < min_chars.
        """
        if not text or not text.strip():
            return []

        parts = cls.BOUNDARY_REGEX.split(text)
        sentences: List[str] = []
        current_sent = ""

        for part in parts:
            if not part:
                continue
            if cls.BOUNDARY_REGEX.fullmatch(part):
                current_sent += part
                clean_sent = current_sent.strip()
                if len(clean_sent) >= min_chars:
                    sentences.append(clean_sent)
                current_sent = ""
            else:
                current_sent += part

        if current_sent.strip():
            clean_sent = current_sent.strip()
            if len(clean_sent) >= min_chars:
                sentences.append(clean_sent)
            elif sentences:
                sentences[-1] += " " + clean_sent

        if not sentences and text.strip():
            sentences = [text.strip()]

        return sentences


# ============================================================================
# CHUNKING STRATEGY PATTERN IMPLEMENTATIONS
# ============================================================================

class BaseChunker(ABC):
    """Abstract Strategy Interface for passage chunking."""
    
    @abstractmethod
    def chunk(self, parent_text: str, parent_id: str, query_id: str, is_selected: bool = True) -> List[ChunkPayload]:
        pass


class HierarchicalParentChildChunker(BaseChunker):
    """
    Strategy A: Hierarchical Parent-Child Chunking
    ----------------------------------------------
    Splits parent passage into fine-grained sentence chunks (child vectors)
    while storing the complete original passage as parent context in the payload.
    """

    def __init__(self, min_chunk_chars: int = 15):
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, parent_text: str, parent_id: str, query_id: str, is_selected: bool = True) -> List[ChunkPayload]:
        sentences = IndicSentenceSplitter.split(parent_text, min_chars=self.min_chunk_chars)
        total_chunks = len(sentences)
        chunks: List[ChunkPayload] = []

        for idx, sentence in enumerate(sentences):
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_pc_{idx}"))
            payload = ChunkPayload(
                chunk_id=chunk_uuid,
                strategy="parent_child",
                parent_id=parent_id,
                parent_text=parent_text,
                chunk_text=sentence,
                chunk_index=idx,
                total_chunks_in_parent=total_chunks,
                char_count=len(sentence),
                language="hi",
                source_dataset="ai4bharat/MSMARCO-XI",
                query_id=query_id,
                is_selected_passage=is_selected
            )
            chunks.append(payload)

        return chunks


class SlidingWindowChunker(BaseChunker):
    """
    Strategy B: Sliding Window with Overlap Chunking
    ------------------------------------------------
    Groups N sentences together with an overlapping step (e.g., 2 sentences, 1 overlap).
    Preserves continuous narrative and discourse coherence across sentence splits.
    """

    def __init__(self, window_size: int = 2, overlap: int = 1, min_chunk_chars: int = 15):
        self.window_size = max(1, window_size)
        self.overlap = max(0, min(overlap, self.window_size - 1))
        self.step_size = self.window_size - self.overlap
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, parent_text: str, parent_id: str, query_id: str, is_selected: bool = True) -> List[ChunkPayload]:
        sentences = IndicSentenceSplitter.split(parent_text, min_chars=self.min_chunk_chars)
        
        if len(sentences) <= self.window_size:
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_sw_0"))
            return [
                ChunkPayload(
                    chunk_id=chunk_uuid,
                    strategy="sliding_window",
                    parent_id=parent_id,
                    parent_text=parent_text,
                    chunk_text=parent_text,
                    chunk_index=0,
                    total_chunks_in_parent=1,
                    char_count=len(parent_text),
                    language="hi",
                    source_dataset="ai4bharat/MSMARCO-XI",
                    query_id=query_id,
                    is_selected_passage=is_selected
                )
            ]

        windows: List[str] = []
        for i in range(0, len(sentences), self.step_size):
            window_slice = sentences[i : i + self.window_size]
            if not window_slice:
                continue
            window_text = " ".join(window_slice)
            windows.append(window_text)
            if i + self.window_size >= len(sentences):
                break

        total_chunks = len(windows)
        chunks: List[ChunkPayload] = []
        for idx, win_text in enumerate(windows):
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{parent_id}_sw_{idx}"))
            payload = ChunkPayload(
                chunk_id=chunk_uuid,
                strategy="sliding_window",
                parent_id=parent_id,
                parent_text=parent_text,
                chunk_text=win_text,
                chunk_index=idx,
                total_chunks_in_parent=total_chunks,
                char_count=len(win_text),
                language="hi",
                source_dataset="ai4bharat/MSMARCO-XI",
                query_id=query_id,
                is_selected_passage=is_selected
            )
            chunks.append(payload)

        return chunks


# ============================================================================
# DATASET STREAMING & INGESTION MANAGER
# ============================================================================

class IndicDatasetLoader:
    """Handles high-performance streaming & loading of MSMARCO-XI Hindi parquet files."""

    @staticmethod
    def get_parquet_path(split: str = "validation") -> str:
        """Retrieves or downloads the parquet file from HuggingFace Hub cache."""
        from huggingface_hub import hf_hub_download
        
        filename = "validation/hinval.parquet" if split in ["val", "validation"] else "train/hintrain.parquet"
        logger.info(f"Locating dataset file '{filename}' from 'ai4bharat/MSMARCO-XI'...")
        
        cached_file = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=filename,
            repo_type="dataset"
        )
        logger.info(f"Dataset path ready: {cached_file}")
        return cached_file

    @classmethod
    def iterate_samples(cls, split: str = "validation", max_samples: int = 100) -> Generator[Dict[str, Any], None, None]:
        """Streams dataset rows efficiently in batches from parquet file."""
        import pyarrow.parquet as pq

        parquet_path = cls.get_parquet_path(split=split)
        pf = pq.ParquetFile(parquet_path)
        logger.info(f"Parquet opened: {pf.metadata.num_rows} total dataset rows, {pf.num_row_groups} row groups.")

        yielded_count = 0
        for rg_idx in range(pf.num_row_groups):
            df_rg = pf.read_row_group(rg_idx).to_pandas()
            for _, row in df_rg.iterrows():
                yield row.to_dict()
                yielded_count += 1
                if 0 < max_samples <= yielded_count:
                    return


# ============================================================================
# RAG INDEXER
# ============================================================================

class IndicRAGIndexer:
    """
    Orchestrates embedding generation and batch insertion into Qdrant vector database.
    """

    def __init__(self, config: IngestConfig, client=None):
        self.config = config

        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient, models
        except ImportError as e:
            logger.error(f"Required library missing: {e}. Please run: pip install -r requirements.txt")
            sys.exit(1)

        self._models = models

        logger.info(f"Loading FastEmbed model '{self.config.embedding_model}'...")
        t0 = time.perf_counter()
        self.embed_model = TextEmbedding(model_name=self.config.embedding_model)
        
        # Determine embedding dimension by testing a single probe vector
        probe_emb = list(self.embed_model.embed(["probe"]))[0]
        self.config.embedding_dim = len(probe_emb)
        logger.info(f"Model loaded in {(time.perf_counter() - t0):.2f}s | Vector Dimension: {self.config.embedding_dim}")

        if client is not None:
            self.client = client
        else:
            logger.info(f"Connecting to local Qdrant vector store at '{self.config.qdrant_path}'...")
            os.makedirs(self.config.qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=self.config.qdrant_path)

        # Setup chunking strategies
        self.chunkers: List[BaseChunker] = []
        if self.config.strategy in ["parent_child", "all"]:
            self.chunkers.append(HierarchicalParentChildChunker(min_chunk_chars=self.config.min_chunk_chars))
        if self.config.strategy in ["sliding_window", "all"]:
            self.chunkers.append(
                SlidingWindowChunker(
                    window_size=self.config.window_sentences,
                    overlap=self.config.overlap_sentences,
                    min_chunk_chars=self.config.min_chunk_chars
                )
            )

    def init_collection(self, recreate: bool = False) -> None:
        """Initializes or resets the target Qdrant collection."""
        exists = self.client.collection_exists(self.config.collection_name)
        if exists:
            if recreate:
                logger.warning(f"Recreating collection '{self.config.collection_name}' (dropping existing)...")
                self.client.delete_collection(self.config.collection_name)
            else:
                logger.info(f"Collection '{self.config.collection_name}' exists. New vectors will be added.")
                return

        logger.info(f"Creating collection '{self.config.collection_name}' (Dim={self.config.embedding_dim}, Distance=Cosine)...")
        self.client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=self._models.VectorParams(
                size=self.config.embedding_dim,
                distance=self._models.Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=self._models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                on_disk=True
            )
        )

        # Create indexing on payload fields for instant filtering
        for field in ["strategy", "parent_id", "query_id", "is_selected_passage"]:
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name=field,
                field_schema=self._models.PayloadSchemaType.KEYWORD
            )
        logger.info("Collection and fast payload indexes configured successfully.")

    def run_ingestion(self) -> int:
        """Runs the streaming chunking, embedding, and indexing loop."""
        from tqdm import tqdm

        total_limit = self.config.sample_limit if self.config.sample_limit > 0 else 97941
        pbar = tqdm(total=total_limit, desc="Ingesting MSMARCO-XI (hi)", unit="sample")

        chunk_buffer: List[ChunkPayload] = []
        total_indexed = 0
        total_passages_processed = 0

        def flush_buffer(buffer: List[ChunkPayload]) -> int:
            if not buffer:
                return 0
            
            texts = [c.chunk_text for c in buffer]
            embeddings = list(self.embed_model.embed(texts, batch_size=self.config.embedding_batch_size))
            
            points = []
            for chunk_meta, emb in zip(buffer, embeddings):
                point = self._models.PointStruct(
                    id=chunk_meta.chunk_id,
                    vector=emb.tolist(),
                    payload=chunk_meta.to_dict()
                )
                points.append(point)

            for i in range(0, len(points), self.config.upsert_batch_size):
                sub = points[i : i + self.config.upsert_batch_size]
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=sub
                )
            
            count = len(points)
            buffer.clear()
            import gc
            gc.collect()
            return count

        sample_count = 0
        try:
            for item in IndicDatasetLoader.iterate_samples(split=self.config.dataset_split, max_samples=self.config.sample_limit):
                query_id = str(item.get("query_id") or f"sample_{sample_count}")
                passages_dict = item.get("passages") or {}
                
                trans_passages = []
                is_selected_flags = []
                if isinstance(passages_dict, dict):
                    raw_tp = passages_dict.get("Translated_passages")
                    if raw_tp is not None:
                        trans_passages = list(raw_tp)
                    raw_sel = passages_dict.get("is_selected")
                    if raw_sel is not None:
                        is_selected_flags = list(raw_sel)

                if len(trans_passages) == 0 and "passage" in item:
                    trans_passages = [item["passage"]]
                    is_selected_flags = [1]

                for p_idx, p_text in enumerate(trans_passages):
                    if not p_text or not str(p_text).strip():
                        continue
                    
                    is_sel = bool(is_selected_flags[p_idx]) if p_idx < len(is_selected_flags) else True
                    
                    # By default index selected (positive) passages; if index_all_passages=True, index all distractors too
                    if not self.config.index_all_passages and not is_sel:
                        continue

                    total_passages_processed += 1
                    parent_id = f"{query_id}_p{p_idx}"

                    for chunker in self.chunkers:
                        generated_chunks = chunker.chunk(
                            parent_text=str(p_text).strip(),
                            parent_id=parent_id,
                            query_id=query_id,
                            is_selected=is_sel
                        )
                        chunk_buffer.extend(generated_chunks)

                    if len(chunk_buffer) >= self.config.embedding_batch_size * 2:
                        total_indexed += flush_buffer(chunk_buffer)

                sample_count += 1
                pbar.update(1)

        except KeyboardInterrupt:
            logger.warning("Ingestion interrupted by user! Flushing active buffer...")
        finally:
            total_indexed += flush_buffer(chunk_buffer)
            pbar.close()

        logger.info(f"Ingestion finished: {sample_count} dataset queries, {total_passages_processed} parent passages -> {total_indexed} indexed vector points in Qdrant.")
        return total_indexed


# ============================================================================
# RETRIEVAL SANITY CHECK & LATENCY BENCHMARK
# ============================================================================

def benchmark_retrieval(
    indexer: IndicRAGIndexer,
    test_queries: Optional[List[str]] = None,
    top_k: int = 3,
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Measures end-to-end vector retrieval latency + parent context resolution.
    Calculates P50, P70, P90, P100 percentiles against the <200ms target budget.
    """
    if not test_queries:
        test_queries = [
            "कॉर्पोरेशन क्या है?",
            "कंप्यूटर कैसे काम करता है?",
            "स्वस्थ रहने के लिए क्या खाना चाहिए?",
            "मौसम में बदलाव के क्या कारण हैं?",
            "भारत की राजधानी क्या है?"
        ]

    logger.info("\n" + "=" * 70)
    logger.info(f"RUNNING RETRIEVAL BENCHMARK ({len(test_queries)} queries, {num_iterations} iterations each)")
    logger.info("=" * 70)

    collection_info = indexer.client.get_collection(indexer.config.collection_name)
    logger.info(f"Collection: '{indexer.config.collection_name}' | Indexed Vectors: {collection_info.points_count}")

    if collection_info.points_count == 0:
        logger.warning("No vectors found in collection. Ingest samples first.")
        return {}

    all_latencies_ms: List[float] = []
    sample_retrievals: List[Dict[str, Any]] = []

    for q_idx, query in enumerate(test_queries):
        for it in range(num_iterations):
            t0 = time.perf_counter()

            # 1. Generate query vector
            query_emb = list(indexer.embed_model.embed([query]))[0].tolist()

            # 2. Query Qdrant vector store
            search_results = indexer.client.query_points(
                collection_name=indexer.config.collection_name,
                query=query_emb,
                limit=top_k,
                with_payload=True
            ).points

            # 3. Resolve parent contexts (Parent-Child Resolution)
            resolved_hits = []
            for hit in search_results:
                payload = hit.payload or {}
                resolved_hits.append({
                    "score": round(hit.score, 4),
                    "strategy": payload.get("strategy"),
                    "chunk_text": payload.get("chunk_text"),
                    "parent_text": payload.get("parent_text"),
                    "parent_id": payload.get("parent_id"),
                    "chunk_index": payload.get("chunk_index"),
                    "total_chunks": payload.get("total_chunks_in_parent"),
                    "query_id": payload.get("query_id")
                })

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            all_latencies_ms.append(elapsed_ms)

            if it == 0 and q_idx < 2:
                sample_retrievals.append({
                    "query": query,
                    "latency_ms": elapsed_ms,
                    "hits": resolved_hits
                })

    p50 = float(np.percentile(all_latencies_ms, 50))
    p70 = float(np.percentile(all_latencies_ms, 70))
    p90 = float(np.percentile(all_latencies_ms, 90))
    p100 = float(np.percentile(all_latencies_ms, 100))
    mean_lat = float(np.mean(all_latencies_ms))

    logger.info("\n" + "-" * 50)
    logger.info("RETRIEVAL & RESOLUTION LATENCY ANALYTICS (ms)")
    logger.info("-" * 50)
    logger.info(f"Total Query Iterations : {len(all_latencies_ms)}")
    logger.info(f"Mean Latency           : {mean_lat:.2f} ms")
    logger.info(f"P50 Latency (Median)   : {p50:.2f} ms")
    logger.info(f"P70 Latency            : {p70:.2f} ms")
    logger.info(f"P90 Latency            : {p90:.2f} ms")
    logger.info(f"P100 Latency (Max)     : {p100:.2f} ms")
    logger.info(f"Target Budget (<200ms) : {'PASSED [OK]' if p100 < 200 else 'NEEDS OPTIMIZATION'}")
    logger.info("-" * 50)

    logger.info("\n--- SAMPLE PARENT-CHILD RETRIEVAL DEMO ---")
    for sample in sample_retrievals:
        logger.info(f"\nQuery: \"{sample['query']}\" (Search + Parent Resolution: {sample['latency_ms']:.2f}ms)")
        for rank, hit in enumerate(sample["hits"], 1):
            chunk_display = (hit['chunk_text'][:90] + '...') if len(hit['chunk_text'] or '') > 90 else hit['chunk_text']
            parent_display = (hit['parent_text'][:130] + '...') if len(hit['parent_text'] or '') > 130 else hit['parent_text']
            logger.info(f"  Rank #{rank} [Cosine Similarity: {hit['score']:.4f} | Strategy: {hit['strategy']}]")
            logger.info(f"    - Matched Child Chunk  : \"{chunk_display}\"")
            logger.info(f"    - Resolved Parent Context : \"{parent_display}\"")
            logger.info(f"    - Parent ID: {hit['parent_id']} (Segment {hit['chunk_index'] + 1}/{hit['total_chunks']})")

    logger.info("\n" + "=" * 70)

    return {
        "mean_ms": mean_lat,
        "p50_ms": p50,
        "p70_ms": p70,
        "p90_ms": p90,
        "p100_ms": p100,
        "total_vectors": collection_info.points_count
    }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production-Grade Indic RAG Ingestion & Indexing Pipeline (MSMARCO-XI Hindi)"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=100,
        help="Number of dataset samples to ingest (default: 100, set to 0 for full dataset)."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["parent_child", "sliding_window", "all"],
        default="all",
        help="Chunking strategy: 'parent_child', 'sliding_window', or 'all' (default: 'all')."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="indic_rag_msmarco_hi",
        help="Qdrant collection name (default: 'indic_rag_msmarco_hi')."
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="./qdrant_data",
        help="Local on-disk Qdrant storage path (default: './qdrant_data')."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="FastEmbed embedding model name."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "train"],
        help="Dataset split to ingest from: 'validation' or 'train'."
    )
    parser.add_argument(
        "--index-all-passages",
        action="store_true",
        help="Index all passages (including negative distractors), not just positive selected passages."
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Drop and recreate the collection before ingesting."
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion and immediately run retrieval benchmark on existing index."
    )
    parser.add_argument(
        "--window-sentences",
        type=int,
        default=2,
        help="Window size in sentences for sliding-window chunking (default: 2)."
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help="Sentence overlap for sliding-window chunking (default: 1)."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = IngestConfig(
        dataset_split=args.split,
        sample_limit=args.sample_limit,
        strategy=args.strategy,
        collection_name=args.collection,
        qdrant_path=args.qdrant_path,
        embedding_model=args.model,
        window_sentences=args.window_sentences,
        overlap_sentences=args.overlap_sentences,
        index_all_passages=args.index_all_passages
    )

    logger.info("=" * 70)
    logger.info("INDIC VOICE RAG INGESTION & INDEXING PIPELINE")
    logger.info(f"Target Collection   : {config.collection_name}")
    logger.info(f"Qdrant Path         : {config.qdrant_path}")
    logger.info(f"Embedding Model     : {config.embedding_model}")
    logger.info(f"Chunking Strategy   : {config.strategy}")
    logger.info(f"Dataset Split       : {config.dataset_split}")
    logger.info(f"Sample Limit        : {config.sample_limit}")
    logger.info("=" * 70)

    # Initialize Indexer
    indexer = IndicRAGIndexer(config)

    # Ingestion Phase
    if not args.skip_ingest:
        indexer.init_collection(recreate=args.recreate_collection)
        t_start = time.perf_counter()
        indexed_count = indexer.run_ingestion()
        t_total = time.perf_counter() - t_start
        logger.info(f"Total Ingestion & Indexing time: {t_total:.2f}s ({indexed_count} vector points)")
    else:
        logger.info("Skipping ingestion phase (--skip-ingest flag provided).")

    # Retrieval Sanity Check & Latency Benchmark Phase
    benchmark_retrieval(indexer)


if __name__ == "__main__":
    main()
