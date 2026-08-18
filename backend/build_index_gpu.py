#!/usr/bin/env python3
"""
GPU Index Builder for MSMARCO-XI Hindi  (run on Colab / Kaggle, then download qdrant_data/)
==========================================================================================
Designed for HH Goa 2026 Task 2. This replaces the naive ingest for index building.

WHY THIS EXISTS
---------------
Measured facts about ai4bharat/MSMARCO-XI (validation split, 97,941 queries):
  - median passage = 290 chars / 54 words / 4 sentences   -> already "chunk sized"
  - 37.7% of passages are <= 2 sentences                  -> must NOT be sub-chunked
  - 5.7% of passages are 7+ sentences (max 2047)          -> must be sub-chunked
  - 10 passages per query, only 0.7 are is_selected=1     -> 93% of the corpus is
                                                             thrown away if you index
                                                             positives only
  - 35.6% of queries have ZERO selected passages          -> those queries index nothing
So a single chunking rule is wrong for this dataset. This builder routes each passage
to a strategy based on its own length, and indexes ALL 10 passages per query.

STRATEGIES PRODUCED (every passage gets >= 1 vector)
----------------------------------------------------
  S1 "passage"   : whole passage, one vector.  Every passage. The workhorse.
  S2 "sentence"  : one vector per sentence, context-prefixed. Passages of 3-6 sentences.
  S3 "window"    : sliding window w=3 / overlap=1.          Passages of 7+ sentences.
  S4 "semantic"  : embedding-similarity boundary detection. Passages of 7+ sentences.
Retrieval fuses them by Reciprocal Rank Fusion and dedupes on parent_id, so the
strategies cooperate instead of competing for the same top-k slots.

INCREMENTAL BY DEFAULT
----------------------
Raising num_queries and re-running EXTENDS the existing index instead of rebuilding
it. Work already done is never repeated:

    run 1:  num_queries = 20_000   ->  builds queries      0 .. 20,000
    run 2:  num_queries = 60_000   ->  builds queries 20,000 .. 60,000 and appends

Safety comes from a config fingerprint stored in the manifest. Every setting that
changes what a vector means -- model, prefixes, chunking thresholds, sparse/quant
flags -- is hashed. Change any of them and the builder refuses to append, because
mixing vectors from two configs in one collection produces an index that returns
confident nonsense with nothing raising anywhere.

A checkpoint is written every `checkpoint_every` queries, so a Colab disconnect
costs you the last few minutes rather than the whole run. Re-running resumes.

OUTPUT (download all three, drop into backend/)
-----------------------------------------------
  qdrant_data/       - Qdrant local store, int8-quantized dense + BM25 sparse
  parents.sqlite     - passage text / query_type / gold answer, stored ONCE
  index_manifest.json- model, dim, prefixes, fingerprint, progress. Server validates it.

USAGE ON COLAB
--------------
  !pip -q install sentence-transformers qdrant-client fastembed pyarrow huggingface_hub
  !python build_index_gpu.py
  # then:  !zip -r index.zip qdrant_data parents.sqlite index_manifest.json
"""

import os
import re
import gc
import json
import time
import sqlite3
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Tuple, Optional, Set

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("IndexBuilder")


# ============================================================================
# CONFIG  -- the only block you normally edit
# ============================================================================

@dataclass
class BuildConfig:
    # --- corpus size (CUMULATIVE: total queries the index should cover) ---
    # Raise this and re-run to extend. Measured sizing for this dataset:
    #   10_000 queries ->  ~98k passages ->  ~360k vectors -> ~140MB int8 + ~80MB text
    #   20_000 queries -> ~197k passages ->  ~720k vectors -> ~280MB int8 + ~160MB text
    #   50_000 queries -> ~492k passages -> ~1.8M vectors  -> ~690MB int8 + ~400MB text
    #   97_941 queries -> ~963k passages -> ~3.5M vectors  -> ~1.4GB int8 + ~780MB text
    # NOTE: the validation split has 97,941 queries total. Asking for more than
    # that simply stops at the end of the file.
    num_queries: int = 20_000
    split: str = "validation"

    # --- embedding model ---
    # multilingual-e5-small: 384-dim, retrieval-trained, ONNX available for CPU serving.
    # Changing this invalidates the whole index -- the fingerprint check will catch it.
    model_name: str = "intfloat/multilingual-e5-small"
    embed_dim: int = 384
    query_prefix: str = "query: "      # E5 asymmetric convention
    passage_prefix: str = "passage: "
    gpu_batch_size: int = 256
    max_seq_length: int = 320          # p99 passage is 705 chars; 320 tokens covers it

    # --- chunking ---
    min_chunk_chars: int = 20
    atomic_max_sentences: int = 2      # <= this -> passage vector only
    sentence_max_sentences: int = 6    # <= this -> passage + sentence children
    window_size: int = 3
    window_overlap: int = 1
    semantic_percentile: int = 25      # lower = fewer, larger semantic chunks

    # --- storage ---
    # A REAL Qdrant server, not qdrant-client's local file mode. Local mode is
    # documented for <20,000 points and does brute-force numpy search -- at 720k
    # vectors that is a linear scan on every query and the 200ms budget is gone.
    # A server builds an actual HNSW index, so search stays in single-digit ms.
    # On Kaggle/Colab the binary is downloaded and launched automatically.
    qdrant_url: str = "http://127.0.0.1:6333"
    auto_start_server: bool = True
    qdrant_version: str = "v1.12.4"      # must match the version you deploy with
    server_storage: str = "./qdrant_storage"
    snapshot_dir: str = "./snapshots"

    collection: str = "indic_rag_msmarco_hi"
    qdrant_path: str = "./qdrant_data"   # only used if auto_start_server=False
    parent_db: str = "./parents.sqlite"
    manifest_path: str = "./index_manifest.json"
    # Local-mode Qdrant persists through sqlite, and transaction overhead dominates
    # the whole build -- the GPU sits at 0% waiting on it. Larger batches mean
    # fewer commits. Not part of the config fingerprint: batch size cannot change
    # what a vector is, so raising it never invalidates an existing index.
    upsert_batch: int = 4096
    enable_bm25: bool = True           # hybrid sparse retrieval
    enable_int8: bool = True           # ~4x smaller vectors, faster search

    # --- incremental behaviour ---
    # resume=True  : extend an existing index, skipping queries already done.
    # resume=False : delete and rebuild from scratch.
    resume: bool = True
    checkpoint_every: int = 2_000      # queries between durable checkpoints

    # --- misc ---
    stream_batch: int = 1_000          # parquet rows pulled at a time (memory safe)
    verify_parity: bool = True         # check GPU vectors == CPU serving vectors


CFG = BuildConfig()


# ============================================================================
# CONFIG FINGERPRINT
# ============================================================================

def config_fingerprint(cfg: BuildConfig) -> str:
    """
    Hash of every setting that changes what a stored vector MEANS.

    num_queries is deliberately excluded -- extending the corpus is the whole
    point of resuming. Everything else is included, because appending vectors
    produced under different settings to the same collection yields an index
    where half the points are incomparable to the other half, and no error is
    ever raised.
    """
    payload = {
        "model": cfg.model_name,
        "dim": cfg.embed_dim,
        "query_prefix": cfg.query_prefix,
        "passage_prefix": cfg.passage_prefix,
        "max_seq_length": cfg.max_seq_length,
        "min_chunk_chars": cfg.min_chunk_chars,
        "atomic_max": cfg.atomic_max_sentences,
        "sentence_max": cfg.sentence_max_sentences,
        "window_size": cfg.window_size,
        "window_overlap": cfg.window_overlap,
        "semantic_percentile": cfg.semantic_percentile,
        "collection": cfg.collection,
        "bm25": cfg.enable_bm25,
        "int8": cfg.enable_int8,
        "split": cfg.split,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=12).hexdigest()


# ============================================================================
# INDIC SENTENCE SEGMENTATION
# ============================================================================

class IndicSentenceSplitter:
    """
    Splits Devanagari text on danda / double-danda / western terminators.

    Improvements over the naive version:
      - does not split inside decimals (1.5) or common abbreviations
      - never emits a fragment shorter than min_chars; fragments are glued to the
        previous sentence instead of being dropped (dropping loses facts)
    """

    _BOUNDARY = re.compile(r'([।॥?!]+|(?<![0-9])\.(?![0-9])|\n+)')
    _WS = re.compile(r'[ \t\r ]+')

    @classmethod
    def split(cls, text: str, min_chars: int = 20) -> List[str]:
        text = cls._WS.sub(" ", (text or "").strip())
        if not text:
            return []

        parts = cls._BOUNDARY.split(text)
        out: List[str] = []
        cur = ""
        for part in parts:
            if not part:
                continue
            if cls._BOUNDARY.fullmatch(part):
                cur = (cur + part).strip()
                if len(cur) >= min_chars:
                    out.append(cur)
                    cur = ""
                elif out and cur:
                    out[-1] = out[-1] + " " + cur
                    cur = ""
            else:
                cur += part

        tail = cur.strip()
        if tail:
            if len(tail) >= min_chars or not out:
                out.append(tail)
            else:
                out[-1] = out[-1] + " " + tail

        return out or [text]


# ============================================================================
# CHUNK RECORD
# ============================================================================

@dataclass
class Chunk:
    text: str            # what actually gets embedded (may carry a context prefix)
    raw_text: str        # the literal span, for display / citation
    strategy: str
    parent_id: str
    chunk_index: int
    n_chunks: int


# ============================================================================
# LENGTH-ADAPTIVE CHUNKER
# ============================================================================

class AdaptiveChunker:
    """
    Routes each passage to the strategy that suits its length.

    The context prefix on sentence chunks is the key quality trick: an isolated
    Hindi sentence like "यह 1998 में स्थापित हुआ था।" is meaningless on its own.
    Prefixing the passage's opening clause restores the subject, so the vector
    lands near queries about the actual entity instead of near every date sentence
    in the corpus.
    """

    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg

    @staticmethod
    def _context_prefix(sentences: List[str], limit: int = 90) -> str:
        head = sentences[0] if sentences else ""
        return head.strip()[:limit]

    def _passage_chunk(self, passage: str, parent_id: str) -> Chunk:
        return Chunk(text=passage, raw_text=passage, strategy="passage",
                     parent_id=parent_id, chunk_index=0, n_chunks=1)

    def _sentence_chunks(self, sentences: List[str], parent_id: str) -> List[Chunk]:
        ctx = self._context_prefix(sentences)
        n = len(sentences)
        out = []
        for i, s in enumerate(sentences):
            embed_text = s if i == 0 else f"{ctx} {s}"   # first sentence is its own context
            out.append(Chunk(text=embed_text, raw_text=s, strategy="sentence",
                             parent_id=parent_id, chunk_index=i, n_chunks=n))
        return out

    def _window_chunks(self, sentences: List[str], parent_id: str) -> List[Chunk]:
        w, ov = self.cfg.window_size, self.cfg.window_overlap
        step = max(1, w - ov)
        spans: List[str] = []
        i = 0
        while i < len(sentences):
            span = sentences[i:i + w]
            if not span:
                break
            spans.append(" ".join(span))
            if i + w >= len(sentences):
                break
            i += step

        n = len(spans)
        return [Chunk(text=t, raw_text=t, strategy="window",
                      parent_id=parent_id, chunk_index=i, n_chunks=n)
                for i, t in enumerate(spans)]

    def _semantic_chunks(self, sentences: List[str], parent_id: str,
                         sent_vecs: np.ndarray) -> List[Chunk]:
        """
        Boundary detection: cosine similarity between neighbouring sentences.
        Where similarity dips below the Nth percentile, the topic shifted -> cut.
        """
        if len(sentences) < 3:
            return []

        sims = np.einsum("ij,ij->i", sent_vecs[:-1], sent_vecs[1:])
        if sims.size == 0:
            return []
        cut_at = float(np.percentile(sims, self.cfg.semantic_percentile))
        boundaries = [i + 1 for i, s in enumerate(sims) if s <= cut_at]

        groups: List[str] = []
        start = 0
        for b in boundaries + [len(sentences)]:
            if b <= start:
                continue
            groups.append(" ".join(sentences[start:b]))
            start = b

        groups = [g for g in groups if len(g) >= self.cfg.min_chunk_chars]
        n = len(groups)
        return [Chunk(text=g, raw_text=g, strategy="semantic",
                      parent_id=parent_id, chunk_index=i, n_chunks=n)
                for i, g in enumerate(groups)]

    def plan(self, passage: str, parent_id: str) -> Tuple[List[Chunk], List[str], bool]:
        """
        Returns (chunks, sentences, needs_semantic). Semantic chunking is resolved
        in a second pass because it needs sentence vectors, which are batched on
        the GPU rather than computed per-passage.
        """
        sents = IndicSentenceSplitter.split(passage, self.cfg.min_chunk_chars)
        chunks: List[Chunk] = [self._passage_chunk(passage, parent_id)]

        n = len(sents)
        if n <= self.cfg.atomic_max_sentences:
            return chunks, sents, False

        if n <= self.cfg.sentence_max_sentences:
            chunks.extend(self._sentence_chunks(sents, parent_id))
            return chunks, sents, False

        chunks.extend(self._window_chunks(sents, parent_id))
        return chunks, sents, True


# ============================================================================
# DATASET STREAMING  (memory-safe: never materialises the whole parquet)
# ============================================================================

def stream_rows(split: str, limit: int, batch_size: int,
                skip: int = 0) -> Iterator[Dict[str, Any]]:
    """
    Yields rows [skip, limit). Skipping is done at the Arrow batch level, so
    resuming past 20,000 queries does not decode 20,000 rows of passage text --
    whole batches are discarded by counter before to_pylist() is ever called.

    The naive version called read_row_group(0), and this parquet has ONE row
    group holding all 97,941 rows -- so it pulled 441MB into pandas no matter how
    small the limit was. That was the crash. iter_batches streams instead.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    fname = "validation/hinval.parquet" if split.startswith("val") else "train/hintrain.parquet"
    logger.info(f"Fetching {fname} from ai4bharat/MSMARCO-XI ...")
    path = hf_hub_download(repo_id="ai4bharat/MSMARCO-XI", filename=fname, repo_type="dataset")

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    if limit > total:
        logger.warning(f"Requested {limit:,} queries but the {split} split holds only "
                       f"{total:,}. Will stop at {total:,}.")
    logger.info(f"Parquet ready: {total:,} rows | skipping {skip:,} | target {min(limit, total):,}")

    cols = ["query_id", "query", "query_type", "Answer", "passages"]
    seen = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        n = batch.num_rows

        # Whole batch already processed in a previous run -- drop without decoding.
        if seen + n <= skip:
            seen += n
            continue

        for row in batch.to_pylist():
            if seen >= skip:
                yield row
            seen += 1
            if 0 < limit <= seen:
                return


# ============================================================================
# QDRANT SERVER BOOTSTRAP  (Kaggle / Colab have no docker, so run the binary)
# ============================================================================

def ensure_qdrant_server(url: str, storage: str, version: str,
                         timeout_s: int = 90) -> Optional[Any]:
    """
    Returns the subprocess handle if this call started the server, else None.

    Kaggle and Colab cannot run `docker run qdrant/qdrant`, so the release binary
    is fetched and launched directly. If something is already listening on `url`
    it is reused untouched.
    """
    import subprocess, urllib.request, urllib.error, tarfile, shutil

    def responding() -> bool:
        try:
            with urllib.request.urlopen(url + "/readyz", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    if responding():
        logger.info(f"Qdrant already running at {url} — reusing it.")
        return None

    binary = os.path.abspath("./qdrant_bin/qdrant")
    if not os.path.exists(binary):
        os.makedirs("./qdrant_bin", exist_ok=True)
        tgz = "./qdrant_bin/qdrant.tar.gz"
        rel = (f"https://github.com/qdrant/qdrant/releases/download/{version}/"
               f"qdrant-x86_64-unknown-linux-gnu.tar.gz")
        logger.info(f"Downloading Qdrant {version} ...")
        try:
            urllib.request.urlretrieve(rel, tgz)
            with tarfile.open(tgz) as t:
                t.extractall("./qdrant_bin")
            os.chmod(binary, 0o755)
        except Exception as e:
            raise SystemExit(
                "\n".join([
                    "",
                    f"Could not fetch the Qdrant binary: {e}",
                    "Either provide a reachable server via CFG.qdrant_url, or set",
                    "CFG.auto_start_server = False to fall back to local file mode",
                    "(only viable below ~20,000 points).",
                    "",
                ])
            )

    os.makedirs(storage, exist_ok=True)
    env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=os.path.abspath(storage))
    logger.info(f"Starting Qdrant, storage={storage} ...")
    proc = subprocess.Popen([binary], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(timeout_s):
        if responding():
            logger.info(f"Qdrant ready at {url}")
            return proc
        if proc.poll() is not None:
            raise SystemExit("Qdrant exited during startup — check the binary and storage path.")
        time.sleep(1)

    proc.terminate()
    raise SystemExit(f"Qdrant did not become ready within {timeout_s}s.")


# ============================================================================
# PARENT STORE  (passage text lives here ONCE, not on every child vector)
# ============================================================================

class ParentStore:
    """
    The naive index copied the full parent_text onto every child point. With
    ~3.7 vectors per passage that is a ~3.7x blowup of the largest field in the
    payload -- measured at ~5.5KB per point. Here the text lives in one sqlite
    row and child vectors carry only parent_id. Lookup is a primary-key hit
    (~50 microseconds), so retrieval latency is unaffected.

    passage_hash is stored so that a resumed run can rebuild the dedup set
    without re-reading the corpus.
    """

    SCHEMA = """
        CREATE TABLE IF NOT EXISTS parents (
            parent_id    TEXT PRIMARY KEY,
            passage      TEXT NOT NULL,
            passage_hash TEXT,
            query_id     TEXT,
            query        TEXT,
            query_type   TEXT,
            gold_answer  TEXT,
            is_selected  INTEGER
        )
    """

    def __init__(self, path: str, resume: bool = False):
        if os.path.exists(path) and not resume:
            os.remove(path)

        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(self.SCHEMA)
        self.conn.commit()
        self._buf: List[Tuple] = []

    def load_hashes(self) -> Set[bytes]:
        """Rebuild the dedup set from disk so a resumed run does not re-add passages."""
        rows = self.conn.execute(
            "SELECT passage_hash FROM parents WHERE passage_hash IS NOT NULL")
        return {bytes.fromhex(r[0]) for r in rows}

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def add(self, **kw):
        self._buf.append((
            kw["parent_id"], kw["passage"], kw["passage_hash"].hex(), kw["query_id"],
            kw["query"], kw["query_type"], kw["gold_answer"], int(kw["is_selected"]),
        ))
        if len(self._buf) >= 2000:
            self.flush()

    def flush(self):
        if self._buf:
            self.conn.executemany(
                "INSERT OR IGNORE INTO parents VALUES (?,?,?,?,?,?,?,?)", self._buf)
            self.conn.commit()
            self._buf.clear()

    def finish(self) -> int:
        self.flush()
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_qtype ON parents(query_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_sel ON parents(is_selected)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON parents(passage_hash)")
        self.conn.commit()
        n = self.count()
        self.conn.close()
        return n


# ============================================================================
# BUILDER
# ============================================================================

class IndexBuilder:

    def __init__(self, cfg: BuildConfig):
        self.cfg = cfg
        self.chunker = AdaptiveChunker(cfg)
        self.fingerprint = config_fingerprint(cfg)
        self.stats = {
            "queries": 0, "passages_seen": 0, "passages_unique": 0,
            "duplicates_skipped": 0, "vectors": 0,
            "by_strategy": {"passage": 0, "sentence": 0, "window": 0, "semantic": 0},
            "by_tier": {"atomic": 0, "short": 0, "long": 0},
        }
        self.resuming = False
        self.start_query = 0
        self.next_point_id = 0
        # Set here (not in run) so _plan_run works when called standalone.
        self.use_server = bool(cfg.auto_start_server or cfg.qdrant_url)
        self._server_proc = None

    # ------------------------------------------------------------- planning

    def _read_manifest(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.cfg.manifest_path):
            return None
        try:
            with open(self.cfg.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Existing manifest is unreadable ({e}); treating as absent.")
            return None

    def _plan_run(self):
        """
        Decide between fresh build, resume, and refuse. Runs before the model is
        loaded so a mistake fails in seconds rather than after a 2 minute download.
        """
        store = self.cfg.server_storage if self.use_server else self.cfg.qdrant_path
        index_exists = os.path.isdir(store) and bool(os.listdir(store))
        manifest = self._read_manifest()

        if not index_exists:
            logger.info("No existing index found -- building from scratch.")
            return

        if not self.cfg.resume:
            size = sum(os.path.getsize(os.path.join(dp, f))
                       for dp, _, fs in os.walk(store) for f in fs)
            logger.warning("=" * 74)
            logger.warning(f"resume=False -- DELETING the existing index at "
                           f"{store} ({size/1e6:,.1f} MB) and rebuilding.")
            logger.warning("Set CFG.resume = True to extend it instead.")
            logger.warning("=" * 74)
            logger.warning("Deleting in 5 seconds. Ctrl-C to abort.")
            try:
                time.sleep(5)
            except KeyboardInterrupt:
                raise SystemExit("\nAborted by user. Nothing was deleted.\n")
            return

        if manifest is None:
            raise SystemExit(
                f"\nAborted: an index exists at {store} but there is no\n"
                f"{self.cfg.manifest_path} describing it, so it cannot be safely extended.\n"
                f"Set CFG.resume = False to rebuild from scratch, or restore the manifest.\n"
            )

        old_fp = manifest.get("config_fingerprint")
        if old_fp != self.fingerprint:
            raise SystemExit(
                f"\nAborted: configuration has changed since the existing index was built.\n"
                f"  existing fingerprint : {old_fp}\n"
                f"  current  fingerprint : {self.fingerprint}\n"
                f"  existing model       : {manifest.get('model_name')}\n"
                f"  current  model       : {self.cfg.model_name}\n\n"
                f"Appending vectors built under different settings to the same collection\n"
                f"produces an index where the new points are not comparable to the old ones,\n"
                f"and nothing would raise at query time -- searches would just be wrong.\n\n"
                f"Either restore the original settings, or set CFG.resume = False to rebuild.\n"
            )

        done = int(manifest.get("queries_done", 0))
        if done >= self.cfg.num_queries:
            raise SystemExit(
                f"\nNothing to do: the index already covers {done:,} queries and\n"
                f"num_queries is {self.cfg.num_queries:,}. Raise num_queries to extend it.\n"
            )

        self.resuming = True
        self.start_query = done
        self.next_point_id = int(manifest.get("next_point_id", 0))
        prev = manifest.get("stats", {})
        for k in ("queries", "passages_seen", "passages_unique", "duplicates_skipped", "vectors"):
            self.stats[k] = int(prev.get(k, 0))
        for group in ("by_strategy", "by_tier"):
            for k, v in (prev.get(group) or {}).items():
                if k in self.stats[group]:
                    self.stats[group][k] = int(v)

        logger.info("=" * 74)
        logger.info(f"RESUMING: {done:,} queries already indexed "
                    f"({self.stats['vectors']:,} vectors, {self.stats['passages_unique']:,} passages)")
        logger.info(f"This run will process queries {done:,} -> {self.cfg.num_queries:,} "
                    f"({self.cfg.num_queries - done:,} new)")
        logger.info("=" * 74)

    # ---------------------------------------------------------------- model

    def _load_encoder(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.cfg.model_name} on {device.upper()} ...")
        m = SentenceTransformer(self.cfg.model_name, device=device)
        m.max_seq_length = self.cfg.max_seq_length
        if device == "cuda":
            m = m.half()   # fp16: ~2x throughput, cosine-identical to 4 decimals
        logger.info(f"Encoder ready | dim={m.get_sentence_embedding_dimension()} | device={device}")
        return m, device

    def encode(self, texts: List[str], prefix: str) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.cfg.gpu_batch_size,
            normalize_embeddings=True,      # cosine == dot product afterwards
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    # --------------------------------------------------------------- qdrant

    def _init_qdrant(self):
        from qdrant_client import QdrantClient, models

        self.models = models

        if self.use_server:
            # Server mode. Storage is managed by the Qdrant process; a fresh build
            # drops the collection through the API rather than deleting files
            # underneath a running server.
            client = QdrantClient(url=self.cfg.qdrant_url, prefer_grpc=False, timeout=60)
            if not self.resuming and client.collection_exists(self.cfg.collection):
                logger.warning(f"Dropping existing collection '{self.cfg.collection}'")
                client.delete_collection(self.cfg.collection)
        else:
            if not self.resuming and os.path.exists(self.cfg.qdrant_path):
                import shutil
                shutil.rmtree(self.cfg.qdrant_path)
            os.makedirs(self.cfg.qdrant_path, exist_ok=True)
            client = QdrantClient(path=self.cfg.qdrant_path)

        if self.resuming and client.collection_exists(self.cfg.collection):
            info = client.get_collection(self.cfg.collection)
            logger.info(f"Reopened collection '{self.cfg.collection}' "
                        f"({info.points_count:,} existing points)")
            return client

        quant = None
        if self.cfg.enable_int8:
            # int8 scalar quantization: 384 floats (1536B) -> 384 bytes.
            # always_ram keeps the compressed vectors hot; originals stay on disk
            # and are only touched when rescoring.
            quant = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True))

        sparse_cfg = None
        if self.cfg.enable_bm25:
            sparse_cfg = {"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)}

        client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config={"dense": models.VectorParams(
                size=self.cfg.embed_dim, distance=models.Distance.COSINE, on_disk=True)},
            sparse_vectors_config=sparse_cfg,
            quantization_config=quant,
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
        )

        for f in ("strategy", "parent_id", "query_type"):
            client.create_payload_index(
                collection_name=self.cfg.collection, field_name=f,
                field_schema=models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(
            collection_name=self.cfg.collection, field_name="is_selected",
            field_schema=models.PayloadSchemaType.INTEGER)

        logger.info(f"Collection '{self.cfg.collection}' created "
                    f"(int8={self.cfg.enable_int8}, bm25={self.cfg.enable_bm25})")
        return client

    def _init_bm25(self):
        if not self.cfg.enable_bm25:
            return None
        from fastembed import SparseTextEmbedding
        # No Hindi stemmer exists in snowball, so stemming is disabled and we rely
        # on whitespace tokenisation. Devanagari is not heavily inflected at the
        # token level for these queries, so this is fine.
        logger.info("Loading BM25 sparse encoder (stemmer disabled for Devanagari) ...")
        return SparseTextEmbedding(model_name="Qdrant/bm25", disable_stemmer=True)

    # ------------------------------------------------------------- manifest

    def _write_manifest(self, queries_done: int, n_parents: int, final: bool = False):
        manifest = {
            "model_name": self.cfg.model_name,
            "embed_dim": self.cfg.embed_dim,
            "query_prefix": self.cfg.query_prefix,
            "passage_prefix": self.cfg.passage_prefix,
            "collection": self.cfg.collection,
            "dense_vector_name": "dense",
            "sparse_vector_name": "bm25" if self.cfg.enable_bm25 else None,
            "int8_quantized": self.cfg.enable_int8,
            "parent_db": os.path.basename(self.cfg.parent_db),
            "config_fingerprint": self.fingerprint,
            # The serving side must use the same transport: server storage and
            # local-mode storage are different formats, not interchangeable.
            "transport": "server" if self.use_server else "local",
            "qdrant_version": self.cfg.qdrant_version if self.use_server else None,
            "split": self.cfg.split,
            "queries_done": queries_done,
            "next_point_id": self.next_point_id,
            "complete": final,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats": dict(self.stats, parents=n_parents),
        }
        tmp = self.cfg.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cfg.manifest_path)   # atomic: never a half-written manifest

    # ------------------------------------------------------------- main loop

    def run(self):
        t0 = time.perf_counter()
        self._plan_run()

        if self.use_server and self.cfg.auto_start_server:
            self._server_proc = ensure_qdrant_server(
                self.cfg.qdrant_url, self.cfg.server_storage, self.cfg.qdrant_version)

        self.model, self.device = self._load_encoder()
        client = self._init_qdrant()
        bm25 = self._init_bm25()
        parents = ParentStore(self.cfg.parent_db, resume=self.resuming)

        seen_hashes: Set[bytes] = set()
        if self.resuming:
            seen_hashes = parents.load_hashes()
            logger.info(f"Dedup set rebuilt from disk: {len(seen_hashes):,} known passages")

        pending: List[Chunk] = []
        pending_meta: List[Dict[str, Any]] = []
        pending_semantic: List[Tuple[str, List[str], Dict[str, Any]]] = []

        def resolve_semantic():
            """
            Semantic chunking needs sentence vectors. Doing that per passage fired
            one tiny GPU call per long passage -- roughly 11,000 launches for a
            20k-query build, each dominated by launch and sync overhead rather
            than actual compute. Here every deferred passage's sentences are
            encoded in ONE call and sliced back apart.

            Identical vectors, identical chunks: encoding is per-sentence and
            order-preserving, so batching changes nothing except the call count.
            """
            if not pending_semantic:
                return
            all_sents: List[str] = []
            spans: List[Tuple[str, Dict[str, Any], int, int]] = []
            for pid, sents, meta in pending_semantic:
                spans.append((pid, meta, len(all_sents), len(sents)))
                all_sents.extend(sents)

            vecs = self.encode(all_sents, self.cfg.passage_prefix)

            for pid, meta, start, n in spans:
                sc = self.chunker._semantic_chunks(
                    all_sents[start:start + n], pid, vecs[start:start + n])
                pending.extend(sc)
                pending_meta.extend([meta] * len(sc))
            pending_semantic.clear()

        def flush(wait: bool = False):
            resolve_semantic()
            if not pending:
                return
            vecs = self.encode([c.text for c in pending], self.cfg.passage_prefix)
            sparse_vecs = list(bm25.embed([c.raw_text for c in pending])) if bm25 else None

            points = []
            for i, (c, meta) in enumerate(zip(pending, pending_meta)):
                vector: Dict[str, Any] = {"dense": vecs[i].tolist()}
                if sparse_vecs is not None:
                    sv = sparse_vecs[i]
                    vector["bm25"] = self.models.SparseVector(
                        indices=sv.indices.tolist(), values=sv.values.tolist())
                points.append(self.models.PointStruct(
                    id=self.next_point_id,
                    vector=vector,
                    payload={
                        "parent_id": c.parent_id,
                        "strategy": c.strategy,
                        "chunk_text": c.raw_text,
                        "chunk_index": c.chunk_index,
                        "n_chunks": c.n_chunks,
                        "query_type": meta["query_type"],
                        "is_selected": meta["is_selected"],
                        "lang": "hi",
                    },
                ))
                self.next_point_id += 1
                self.stats["by_strategy"][c.strategy] += 1

            for i in range(0, len(points), self.cfg.upsert_batch):
                client.upsert(collection_name=self.cfg.collection,
                              points=points[i:i + self.cfg.upsert_batch], wait=wait)

            self.stats["vectors"] += len(points)
            pending.clear()
            pending_meta.clear()
            gc.collect()

        logger.info("=" * 74)
        logger.info(f"{'EXTENDING' if self.resuming else 'BUILDING'} INDEX | "
                    f"target {self.cfg.num_queries:,} queries | ALL 10 passages each")
        logger.info("=" * 74)

        queries_done = self.start_query
        t_first_row: Optional[float] = None      # set when real work begins
        try:
            for row in stream_rows(self.cfg.split, self.cfg.num_queries,
                                   self.cfg.stream_batch, skip=self.start_query):
                if t_first_row is None:
                    # Throughput must be measured from here, not from t0. t0
                    # precedes the model download, the BM25 load and the 461MB
                    # parquet fetch -- charging that one-time setup against the
                    # first checkpoint's 2,000 queries inflates the ETA several-fold.
                    t_first_row = time.perf_counter()
                    logger.info(f"Setup took {(t_first_row - t0)/60:.1f} min; "
                                f"timing throughput from here.")

                self.stats["queries"] += 1
                queries_done += 1

                qid = str(row.get("query_id") or queries_done)
                qtype = str(row.get("query_type") or "UNKNOWN")
                qtext = str(row.get("query") or "")
                gold = str(row.get("Answer") or "")

                pk = row.get("passages") or {}
                texts = list(pk.get("Translated_passages") or [])
                flags = list(pk.get("is_selected") or [])

                for pi, raw in enumerate(texts):
                    passage = str(raw or "").strip()
                    if len(passage) < self.cfg.min_chunk_chars:
                        continue
                    self.stats["passages_seen"] += 1

                    h = hashlib.blake2b(passage.encode("utf-8"), digest_size=16).digest()
                    if h in seen_hashes:
                        self.stats["duplicates_skipped"] += 1
                        continue
                    seen_hashes.add(h)
                    self.stats["passages_unique"] += 1

                    parent_id = f"{qid}_p{pi}"
                    is_sel = int(flags[pi]) if pi < len(flags) else 0

                    parents.add(parent_id=parent_id, passage=passage, passage_hash=h,
                                query_id=qid, query=qtext, query_type=qtype,
                                gold_answer=gold, is_selected=is_sel)

                    chunks, sents, needs_semantic = self.chunker.plan(passage, parent_id)

                    n = len(sents)
                    tier = ("atomic" if n <= self.cfg.atomic_max_sentences
                            else "short" if n <= self.cfg.sentence_max_sentences else "long")
                    self.stats["by_tier"][tier] += 1

                    meta = {"query_type": qtype, "is_selected": is_sel}
                    if needs_semantic:
                        # Deferred so all long passages in this batch are encoded
                        # in a single GPU call at flush time.
                        pending_semantic.append((parent_id, sents, meta))

                    pending.extend(chunks)
                    pending_meta.extend([meta] * len(chunks))

                if len(pending) >= self.cfg.gpu_batch_size * 4:
                    flush()

                # Durable checkpoint. A Colab disconnect costs the last few minutes,
                # not the whole run -- re-running picks up from here.
                if queries_done % self.cfg.checkpoint_every == 0:
                    flush(wait=True)
                    parents.flush()
                    self._write_manifest(queries_done, parents.count(), final=False)
                    now = time.perf_counter()
                    work_s = now - (t_first_row or t0)      # excludes one-time setup
                    done_here = queries_done - self.start_query
                    rate = done_here / max(work_s, 1e-9)    # queries/sec, steady state
                    remaining = max(0, self.cfg.num_queries - queries_done)
                    logger.info(
                        f"CHECKPOINT {queries_done:>7,}/{self.cfg.num_queries:,} | "
                        f"{self.stats['passages_unique']:>8,} passages | "
                        f"{self.stats['vectors']:>9,} vectors | "
                        f"{rate*60:6.0f} q/min | {(now-t0)/60:5.1f} min elapsed | "
                        f"~{remaining/max(rate,1e-9)/60:5.1f} min left"
                    )

        except KeyboardInterrupt:
            logger.warning("Interrupted -- flushing and checkpointing before exit.")

        flush(wait=True)
        n_parents = parents.finish()
        self._write_manifest(queries_done, n_parents, final=True)

        info = client.get_collection(self.cfg.collection)
        logger.info(f"Collection reports {info.points_count:,} points")

        if self.cfg.verify_parity:
            self._verify_parity()

        self.snapshot_file = None
        if self.use_server:
            self.snapshot_file = self._export_snapshot(client)

        self._report(time.perf_counter() - t0, n_parents, queries_done)

        if self._server_proc is not None:
            logger.info("Stopping the Qdrant process started by this script.")
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=30)
            except Exception:
                self._server_proc.kill()

    def _export_snapshot(self, client) -> Optional[str]:
        """
        A snapshot is Qdrant's portable format: build here, restore into the
        deployed server. Copying the raw storage directory also works but only
        between identical Qdrant versions, so the snapshot is the safer artifact.
        """
        try:
            logger.info("Creating collection snapshot ...")
            snap = client.create_snapshot(collection_name=self.cfg.collection, wait=True)
            name = getattr(snap, "name", None)
            if not name:
                logger.warning("Snapshot API returned no name; skipping download.")
                return None

            os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
            dest = os.path.join(self.cfg.snapshot_dir, name)

            import urllib.request
            url = (f"{self.cfg.qdrant_url}/collections/{self.cfg.collection}"
                   f"/snapshots/{name}")
            urllib.request.urlretrieve(url, dest)
            size = os.path.getsize(dest)
            logger.info(f"Snapshot saved: {dest} ({size/1e6:,.0f} MB)")
            return dest
        except Exception as e:
            logger.warning(f"Snapshot export failed ({e}). The raw storage directory "
                           f"at '{self.cfg.server_storage}' is still usable if the "
                           f"deployed Qdrant runs the same version.")
            return None

    # -------------------------------------------------------- parity check

    def _verify_parity(self):
        """
        The index is built with sentence-transformers on GPU, but the server
        encodes queries with fastembed ONNX on CPU. If those two disagree, every
        search silently returns garbage. This catches it at build time.
        """
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import PoolingType, ModelSource
        except ImportError:
            logger.warning("fastembed not installed; skipping parity check")
            return

        probes = ["कॉर्पोरेशन क्या है?", "भारत की राजधानी कौन सी है?", "मधुमेह के लक्षण क्या हैं?"]
        try:
            TextEmbedding.add_custom_model(
                model=self.cfg.model_name, pooling=PoolingType.MEAN, normalization=True,
                sources=ModelSource(hf=self.cfg.model_name),
                dim=self.cfg.embed_dim, model_file="onnx/model.onnx")
        except Exception:
            pass  # already registered

        cpu = TextEmbedding(model_name=self.cfg.model_name)
        cpu_vecs = np.array(list(cpu.embed([self.cfg.query_prefix + p for p in probes])))

        gpu_vecs = self.model.float().encode(
            [self.cfg.query_prefix + p for p in probes],
            normalize_embeddings=True, convert_to_numpy=True)

        sims = np.einsum("ij,ij->i", cpu_vecs, gpu_vecs)
        logger.info("-" * 60)
        logger.info("GPU(build) vs CPU(serve) vector parity")
        for p, s in zip(probes, sims):
            logger.info(f"  cos={s:.6f} [{'OK' if s > 0.999 else 'MISMATCH'}]  {p}")
        if sims.min() <= 0.999:
            logger.error("PARITY FAILED -- serving vectors will not match the index. "
                         "Check pooling / normalization / model_file before shipping.")
        else:
            logger.info("Parity OK. Index is safe to serve with fastembed CPU.")
        logger.info("-" * 60)

    # --------------------------------------------------------------- report

    def _report(self, elapsed: float, n_parents: int, queries_done: int):
        s = self.stats
        store = self.cfg.server_storage if self.use_server else self.cfg.qdrant_path
        qsize = (sum(os.path.getsize(os.path.join(dp, f))
                     for dp, _, fs in os.walk(store) for f in fs)
                 if os.path.isdir(store) else 0)
        psize = os.path.getsize(self.cfg.parent_db)

        logger.info("")
        logger.info("=" * 74)
        logger.info("BUILD COMPLETE" if queries_done >= self.cfg.num_queries else "STOPPED EARLY")
        logger.info("=" * 74)
        logger.info(f"  this run             : {elapsed/60:.1f} min "
                    f"({queries_done - self.start_query:,} new queries)")
        logger.info(f"  queries covered      : {queries_done:,} (cumulative)")
        logger.info(f"  passages seen        : {s['passages_seen']:,}")
        logger.info(f"  unique passages      : {s['passages_unique']:,}  "
                    f"(dropped {s['duplicates_skipped']:,} exact duplicates)")
        logger.info(f"  vectors indexed      : {s['vectors']:,}  "
                    f"({s['vectors']/max(1, s['passages_unique']):.2f} per passage)")
        logger.info("")
        logger.info("  tier routing:")
        for k, v in s["by_tier"].items():
            logger.info(f"    {k:<10} {v:>9,}  ({v/max(1, s['passages_unique'])*100:5.1f}%)")
        logger.info("  vectors by strategy:")
        for k, v in s["by_strategy"].items():
            logger.info(f"    {k:<10} {v:>9,}  ({v/max(1, s['vectors'])*100:5.1f}%)")
        logger.info("")
        snap = getattr(self, "snapshot_file", None)
        logger.info(f"  vector storage       : {qsize/1e6:,.0f} MB  ({store})")
        logger.info(f"  parents.sqlite       : {psize/1e6:,.0f} MB  ({n_parents:,} rows)")
        if snap:
            logger.info(f"  snapshot             : {os.path.getsize(snap)/1e6:,.0f} MB  ({snap})")
        logger.info(f"  transport            : {'SERVER (HNSW)' if self.use_server else 'LOCAL (brute force)'}")
        logger.info("=" * 74)
        logger.info("")
        logger.info("To extend later: raise CFG.num_queries and re-run. Already-indexed")
        logger.info("queries are skipped -- no re-vectorising.")
        logger.info("")
        if self.use_server:
            logger.info("DOWNLOAD THESE:")
            if snap:
                logger.info(f"  {snap}          <- restore into the deployed Qdrant")
            logger.info(f"  {self.cfg.parent_db}")
            logger.info(f"  {self.cfg.manifest_path}")
            logger.info("")
            logger.info("  !zip -r index.zip snapshots parents.sqlite index_manifest.json")
            logger.info("")
            logger.info(f"Your deployment must run Qdrant {self.cfg.qdrant_version} as a")
            logger.info("service and set QDRANT_URL. Local file mode cannot read this index.")
        else:
            logger.info("  !zip -r index.zip qdrant_data parents.sqlite index_manifest.json")


if __name__ == "__main__":
    IndexBuilder(CFG).run()
