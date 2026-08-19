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
from typing import List, Dict, Any, Generator, Optional
from dataclasses import dataclass, asdict
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("IndicRAGIngest")


@dataclass
class IngestConfig:
    dataset_name: str = "ai4bharat/MSMARCO-XI"
    dataset_split: str = "validation"
    sample_limit: int = 100
    collection_name: str = "indic_rag_msmarco_hi"
    qdrant_path: str = "./qdrant_data"
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: Optional[int] = None
    embedding_batch_size: int = 16
    upsert_batch_size: int = 32
    window_sentences: int = 2
    overlap_sentences: int = 1
    min_chunk_chars: int = 15
    strategy: str = "all"
    index_all_passages: bool = False


@dataclass
class ChunkPayload:
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


class IndicSentenceSplitter:
    BOUNDARY_REGEX = re.compile(r'([।॥?!.\n]+)')

    @classmethod
    def split(cls, text: str, min_chars: int = 15) -> List[str]:
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


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, parent_text: str, parent_id: str, query_id: str, is_selected: bool = True) -> List[ChunkPayload]:
        pass


class HierarchicalParentChildChunker(BaseChunker):
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


class IndicDatasetLoader:
    @staticmethod
    def get_parquet_path(split: str = "validation") -> str:
        from huggingface_hub import hf_hub_download
        filename = "validation/hinval.parquet" if split in ["val", "validation"] else "train/hintrain.parquet"
        logger.info(f"Locating dataset file '{filename}'...")
        cached_file = hf_hub_download(
            repo_id="ai4bharat/MSMARCO-XI",
            filename=filename,
            repo_type="dataset"
        )
        return cached_file

    @classmethod
    def iterate_samples(cls, split: str = "validation", max_samples: int = 100,
                        batch_size: int = 1000) -> Generator[Dict[str, Any], None, None]:
        import pyarrow.parquet as pq

        parquet_path = cls.get_parquet_path(split=split)
        pf = pq.ParquetFile(parquet_path)
        columns = ["query_id", "query", "query_type", "Answer", "passages"]

        yielded = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield row
                yielded += 1
                if 0 < max_samples <= yielded:
                    return


class IndicRAGIndexer:
    def __init__(self, config: IngestConfig, client=None):
        self.config = config

        try:
            from fastembed import TextEmbedding
            from qdrant_client import QdrantClient, models
        except ImportError as e:
            logger.error(f"Missing library: {e}")
            sys.exit(1)

        self._models = models
        from retriever import prefixes_for_model
        self.query_prefix, self.passage_prefix = prefixes_for_model(self.config.embedding_model)

        logger.info(f"Loading FastEmbed model '{self.config.embedding_model}'...")
        t0 = time.perf_counter()
        self.embed_model = TextEmbedding(model_name=self.config.embedding_model)
        probe_emb = list(self.embed_model.embed(["probe"]))[0]
        self.config.embedding_dim = len(probe_emb)
        logger.info(f"Model loaded in {(time.perf_counter() - t0):.2f}s | Dim: {self.config.embedding_dim}")

        if client is not None:
            self.client = client
        else:
            os.makedirs(self.config.qdrant_path, exist_ok=True)
            self.client = QdrantClient(path=self.config.qdrant_path)

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
        exists = self.client.collection_exists(self.config.collection_name)
        if exists:
            if recreate:
                logger.warning(f"Recreating collection '{self.config.collection_name}'...")
                self.client.delete_collection(self.config.collection_name)
            else:
                logger.info(f"Collection '{self.config.collection_name}' exists.")
                return

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

        for field in ["strategy", "parent_id", "query_id", "is_selected_passage"]:
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name=field,
                field_schema=self._models.PayloadSchemaType.KEYWORD
            )

    def run_ingestion(self) -> int:
        from tqdm import tqdm

        total_limit = self.config.sample_limit if self.config.sample_limit > 0 else 97941
        pbar = tqdm(total=total_limit, desc="Ingesting MSMARCO-XI (hi)", unit="sample")

        chunk_buffer: List[ChunkPayload] = []
        total_indexed = 0
        total_passages_processed = 0

        def flush_buffer(buffer: List[ChunkPayload]) -> int:
            if not buffer:
                return 0
            
            texts = [f"{self.passage_prefix}{c.chunk_text}" for c in buffer]
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
            logger.warning("Interrupted by user, flushing buffer...")
        finally:
            total_indexed += flush_buffer(chunk_buffer)
            pbar.close()

        logger.info(f"Ingestion finished: {sample_count} queries -> {total_indexed} points.")
        return total_indexed


def benchmark_retrieval(
    indexer: IndicRAGIndexer,
    test_queries: Optional[List[str]] = None,
    top_k: int = 3,
    num_iterations: int = 10
) -> Dict[str, Any]:
    if not test_queries:
        test_queries = [
            "कॉर्पोरेशन क्या है?",
            "कंप्यूटर कैसे काम करता है?",
            "स्वस्थ रहने के लिए क्या खाना चाहिए?",
            "मौसम में बदलाव के क्या कारण हैं?",
            "भारत की राजधानी क्या है?"
        ]

    logger.info("RUNNING RETRIEVAL BENCHMARK")
    collection_info = indexer.client.get_collection(indexer.config.collection_name)

    if collection_info.points_count == 0:
        logger.warning("No vectors found in collection.")
        return {}

    all_latencies_ms: List[float] = []
    sample_retrievals: List[Dict[str, Any]] = []

    for q_idx, query in enumerate(test_queries):
        for it in range(num_iterations):
            t0 = time.perf_counter()

            query_emb = list(indexer.embed_model.embed([f"{indexer.query_prefix}{query}"]))[0].tolist()

            search_results = indexer.client.query_points(
                collection_name=indexer.config.collection_name,
                query=query_emb,
                limit=top_k,
                with_payload=True
            ).points

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

    logger.info(f"Mean Latency : {mean_lat:.2f} ms | P50: {p50:.2f} ms | P70: {p70:.2f} ms | P100: {p100:.2f} ms")

    return {
        "mean_ms": mean_lat,
        "p50_ms": p50,
        "p70_ms": p70,
        "p90_ms": p90,
        "p100_ms": p100,
        "total_vectors": collection_info.points_count
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indic RAG Ingestion & Indexing Pipeline")
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--strategy", type=str, choices=["parent_child", "sliding_window", "all"], default="all")
    parser.add_argument("--collection", type=str, default="indic_rag_msmarco_hi")
    parser.add_argument("--qdrant-path", type=str, default="./qdrant_data")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-large")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "train"])
    parser.add_argument("--index-all-passages", action="store_true")
    parser.add_argument("--recreate-collection", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--window-sentences", type=int, default=2)
    parser.add_argument("--overlap-sentences", type=int, default=1)
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

    indexer = IndicRAGIndexer(config)

    if not args.skip_ingest:
        indexer.init_collection(recreate=args.recreate_collection)
        t_start = time.perf_counter()
        indexed_count = indexer.run_ingestion()
        t_total = time.perf_counter() - t_start
        logger.info(f"Ingestion time: {t_total:.2f}s ({indexed_count} points)")

    benchmark_retrieval(indexer)


if __name__ == "__main__":
    main()
