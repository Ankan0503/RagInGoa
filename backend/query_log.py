#!/usr/bin/env python3
"""
Local, persistent log of every question asked and the answer given, with the
full latency breakdown for each -- the data behind the site's Insights page.

Every request that reaches run_rag_pipeline[_streaming] gets one row here,
refused or not. This is deliberately a separate SQLite file from
parents.sqlite: that file is mounted read-only in production (it's shipped
index data, not meant to be written to), so a writable store for live traffic
needs its own path and its own volume.
"""

from __future__ import annotations

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("QueryLog")

DEFAULT_PATH = os.getenv("QUERY_LOG_PATH", "./query_log.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    query         TEXT NOT NULL,
    transcript    TEXT,
    answer        TEXT NOT NULL,
    refused       INTEGER NOT NULL,
    refusal_reason TEXT,
    provider      TEXT,
    model         TEXT,
    retrieval_ms  REAL,
    generation_ms REAL,
    guardrail_ms  REAL,
    end_to_end_ms REAL,
    grounding_score REAL,
    ttft_ms       REAL,
    stages_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_query_log_ts ON query_log(ts);
"""

# Columns added after the table first shipped. CREATE TABLE IF NOT EXISTS does
# nothing to an existing table, so a deployment that already has rows would
# keep the old shape and every INSERT naming a new column would fail. Each
# entry is applied only when the column is genuinely absent.
_MIGRATIONS = (
    ("ttft_ms", "ALTER TABLE query_log ADD COLUMN ttft_ms REAL"),
)


@dataclass
class QueryLogEntry:
    query: str
    answer: str
    refused: bool
    stages: List[Dict[str, Any]] = field(default_factory=list)
    transcript: Optional[str] = None
    refusal_reason: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    retrieval_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    guardrail_ms: Optional[float] = None
    end_to_end_ms: Optional[float] = None
    grounding_score: Optional[float] = None
    # Time to first token. Only the streaming path can measure this -- the
    # REST path calls generate() and sees the whole completion at once -- so
    # it is None for non-streamed requests rather than 0.
    ttft_ms: Optional[float] = None


class QueryLog:
    """Not safe to share across threads without external locking -- each
    call site opens/uses its connection from a single async executor thread
    (see server.py), matching how the rest of the app already isolates
    blocking SQLite/Qdrant calls off the event loop."""

    def __init__(self, path: str = DEFAULT_PATH):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()
        logger.info(f"Query log ready at {path}")

    def _migrate(self) -> None:
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(query_log)")}
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                logger.info(f"Adding missing query_log column: {column}")
                self.conn.execute(ddl)

    def record(self, entry: QueryLogEntry) -> None:
        self.conn.execute(
            """INSERT INTO query_log
               (ts, query, transcript, answer, refused, refusal_reason,
                provider, model, retrieval_ms, generation_ms, guardrail_ms,
                end_to_end_ms, grounding_score, ttft_ms, stages_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                entry.query, entry.transcript, entry.answer,
                1 if entry.refused else 0, entry.refusal_reason,
                entry.provider, entry.model,
                entry.retrieval_ms, entry.generation_ms, entry.guardrail_ms,
                entry.end_to_end_ms, entry.grounding_score, entry.ttft_ms,
                json.dumps(entry.stages, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        limit = max(1, min(limit, 500))
        rows = self.conn.execute(
            """SELECT id, ts, query, transcript, answer, refused, refusal_reason,
                      provider, model, retrieval_ms, generation_ms, guardrail_ms,
                      end_to_end_ms, grounding_score, ttft_ms, stages_json
               FROM query_log ORDER BY id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0], "ts": r[1], "query": r[2], "transcript": r[3],
                "answer": r[4], "refused": bool(r[5]), "refusal_reason": r[6],
                "provider": r[7], "model": r[8], "retrieval_ms": r[9],
                "generation_ms": r[10], "guardrail_ms": r[11],
                "end_to_end_ms": r[12], "grounding_score": r[13],
                "ttft_ms": r[14],
                "stages": json.loads(r[15]) if r[15] else [],
            })
        return out

    def stats(self) -> Dict[str, Any]:
        row = self.conn.execute(
            """SELECT COUNT(*), SUM(refused),
                      AVG(retrieval_ms), AVG(generation_ms), AVG(end_to_end_ms),
                      AVG(ttft_ms)
               FROM query_log"""
        ).fetchone()
        total = row[0] or 0
        return {
            "total_queries": total,
            "total_refused": row[1] or 0,
            "avg_retrieval_ms": round(row[2], 2) if row[2] is not None else None,
            "avg_generation_ms": round(row[3], 2) if row[3] is not None else None,
            "avg_end_to_end_ms": round(row[4], 2) if row[4] is not None else None,
            # AVG() ignores NULLs, so this averages only streamed requests --
            # REST ones cannot measure a first token and must not be counted
            # as zero, which would understate it.
            "avg_ttft_ms": round(row[5], 2) if row[5] is not None else None,
        }

    def clear(self) -> int:
        """Permanently delete every row. Returns the number removed. Callers
        are responsible for authorization -- this method itself has none."""
        cur = self.conn.execute("SELECT COUNT(*) FROM query_log")
        n = cur.fetchone()[0] or 0
        self.conn.execute("DELETE FROM query_log")
        self.conn.commit()
        self.conn.execute("VACUUM")
        return n

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    import sys
    import tempfile
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ok = fail = 0

    def check(label, cond):
        global ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {label}")
        else:
            fail += 1
            print(f"  FAIL  {label}")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test_log.sqlite")
        log = QueryLog(path)

        check("file created", os.path.exists(path))
        check("empty recent() on fresh db", log.recent() == [])

        log.record(QueryLogEntry(
            query="कॉर्पोरेशन क्या है?", answer="एक व्यावसायिक इकाई।",
            refused=False, provider="groq", model="qwen/qwen3-32b",
            retrieval_ms=90.1, generation_ms=560.2, guardrail_ms=0.1,
            end_to_end_ms=651.0, grounding_score=0.71, ttft_ms=180.5,
            stages=[{"stage": "embed_query", "ms": 25.6, "in_budget": True, "detail": ""}],
        ))
        log.record(QueryLogEntry(
            query="मौसम कैसा है", answer="क्षमा करें, यह जानकारी उपलब्ध नहीं है।",
            refused=True, refusal_reason="off_topic", transcript="मौसम कैसा है",
            stages=[],
        ))

        recent = log.recent()
        check("two rows recorded", len(recent) == 2)
        check("newest first", recent[0]["query"] == "मौसम कैसा है")
        check("refused flag correct", recent[0]["refused"] is True)
        check("non-refused row intact", recent[1]["refused"] is False)
        check("stages round-trip through JSON",
              recent[1]["stages"][0]["stage"] == "embed_query")
        check("transcript nullable", recent[1]["transcript"] is None)

        limited = log.recent(limit=1)
        check("limit respected", len(limited) == 1)

        s = log.stats()
        check("stats total_queries", s["total_queries"] == 2)
        check("stats total_refused", s["total_refused"] == 1)
        check("stats avg_retrieval_ms averages only non-null rows",
              s["avg_retrieval_ms"] == 90.1)
        check("ttft round-trips", recent[1]["ttft_ms"] == 180.5)
        check("ttft is None on the row that had none",
              recent[0]["ttft_ms"] is None)
        check("avg_ttft_ms ignores rows without a first token",
              s["avg_ttft_ms"] == 180.5)

        log.close()

        reopened = QueryLog(path)
        check("data survives reconnect", len(reopened.recent()) == 2)

        removed = reopened.clear()
        check("clear() returns the count removed", removed == 2)
        check("clear() actually empties the table", reopened.recent() == [])
        check("clear() resets stats", reopened.stats()["total_queries"] == 0)
        reopened.close()

        # A database created before ttft_ms existed must upgrade in place and
        # keep its rows -- the deployed log already has traffic in it, so a
        # migration that dropped or failed on old data would lose real history.
        legacy = os.path.join(tmp, "legacy.sqlite")
        lc = sqlite3.connect(legacy)
        lc.executescript("""
            CREATE TABLE query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
                query TEXT NOT NULL, transcript TEXT, answer TEXT NOT NULL,
                refused INTEGER NOT NULL, refusal_reason TEXT, provider TEXT,
                model TEXT, retrieval_ms REAL, generation_ms REAL,
                guardrail_ms REAL, end_to_end_ms REAL, grounding_score REAL,
                stages_json TEXT NOT NULL);
        """)
        lc.execute("INSERT INTO query_log (ts, query, answer, refused, stages_json)"
                   " VALUES ('2026-01-01T00:00:00Z', 'पुराना प्रश्न', 'पुराना उत्तर', 0, '[]')")
        lc.commit()
        lc.close()

        upgraded = QueryLog(legacy)
        rows = upgraded.recent()
        check("legacy db keeps its existing row", len(rows) == 1)
        check("legacy row still readable", rows[0]["query"] == "पुराना प्रश्न")
        check("legacy row reports ttft_ms as None", rows[0]["ttft_ms"] is None)
        upgraded.record(QueryLogEntry(query="नया", answer="उत्तर", refused=False,
                                      ttft_ms=99.9))
        check("insert works after migration",
              upgraded.recent()[0]["ttft_ms"] == 99.9)
        upgraded.close()

    print(f"\n  {ok} passed, {fail} failed\n")
    sys.exit(1 if fail else 0)
