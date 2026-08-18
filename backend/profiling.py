#!/usr/bin/env python3
"""
Stage-Level Latency Profiler (profiling.py)
===========================================
HH Goa 2026 Task 2, requirement 3/4.

The brief scopes the 200ms budget as "chunking + vector DB retrieval + everything
through to final output". Speech-to-text sits BEFORE that boundary, so it is
measured and reported but excluded from the budget total. Every stage declares
which side of the line it is on, so the two numbers can never be quietly mixed --
which is exactly the failure the old `total < 200 or first_token < 200` check had.

Every stage is timed individually and the sum is reconciled against wall clock.
The difference is reported as `unaccounted`, so time spent somewhere nobody
instrumented shows up instead of hiding.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Iterator


DEFAULT_BUDGET_MS = 200.0


@dataclass
class Stage:
    name: str
    ms: float
    in_budget: bool = True
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.name, "ms": round(self.ms, 3),
                "in_budget": self.in_budget, "detail": self.detail}


class Profiler:
    """
    Usage:
        p = Profiler()
        with p.stage("embed_query"):
            vec = encode(q)
        with p.stage("stt", in_budget=False):
            text = transcribe(audio)
        print(p.table())
    """

    def __init__(self, budget_ms: float = DEFAULT_BUDGET_MS, label: str = ""):
        self.budget_ms = budget_ms
        self.label = label
        self.stages: List[Stage] = []
        self._wall_start = time.perf_counter()
        self._closed_at: Optional[float] = None

    # ---------------------------------------------------------------- record

    @contextmanager
    def stage(self, name: str, in_budget: bool = True, detail: str = "") -> Iterator[Stage]:
        t0 = time.perf_counter()
        st = Stage(name=name, ms=0.0, in_budget=in_budget, detail=detail)
        try:
            yield st
        finally:
            st.ms = (time.perf_counter() - t0) * 1000.0
            self.stages.append(st)

    def record(self, name: str, ms: float, in_budget: bool = True, detail: str = "") -> None:
        """Add a stage timed elsewhere (e.g. returned by a sub-component)."""
        self.stages.append(Stage(name=name, ms=float(ms), in_budget=in_budget, detail=detail))

    def merge(self, stages: List[Stage], prefix: str = "") -> None:
        """Fold a sub-profiler's stages in, optionally namespaced."""
        for s in stages:
            self.stages.append(Stage(name=f"{prefix}{s.name}", ms=s.ms,
                                     in_budget=s.in_budget, detail=s.detail))

    def close(self) -> None:
        if self._closed_at is None:
            self._closed_at = time.perf_counter()

    # ------------------------------------------------------------- totals

    @property
    def wall_ms(self) -> float:
        end = self._closed_at if self._closed_at is not None else time.perf_counter()
        return (end - self._wall_start) * 1000.0

    @property
    def budget_ms_used(self) -> float:
        """Sum of stages inside the 200ms boundary. This is the number that matters."""
        return sum(s.ms for s in self.stages if s.in_budget)

    @property
    def excluded_ms(self) -> float:
        return sum(s.ms for s in self.stages if not s.in_budget)

    @property
    def unaccounted_ms(self) -> float:
        """Wall clock minus every measured stage. Large values mean missing instrumentation."""
        return max(0.0, self.wall_ms - (self.budget_ms_used + self.excluded_ms))

    @property
    def passed(self) -> bool:
        return self.budget_ms_used < self.budget_ms

    def slowest(self, n: int = 3) -> List[Stage]:
        return sorted([s for s in self.stages if s.in_budget],
                      key=lambda s: s.ms, reverse=True)[:n]

    # -------------------------------------------------------------- output

    def report(self) -> Dict[str, Any]:
        budget = self.budget_ms_used
        return {
            "label": self.label,
            "budget_ms": self.budget_ms,
            "stages": [s.to_dict() for s in self.stages],
            "in_budget_ms": round(budget, 3),
            "excluded_ms": round(self.excluded_ms, 3),
            "unaccounted_ms": round(self.unaccounted_ms, 3),
            "wall_ms": round(self.wall_ms, 3),
            "budget_used_pct": round(budget / self.budget_ms * 100, 1) if self.budget_ms else 0.0,
            "passed": self.passed,
            "bottleneck": self.slowest(1)[0].name if self.slowest(1) else None,
        }

    def table(self, width: int = 74) -> str:
        lines: List[str] = []
        add = lines.append

        title = f"LATENCY BREAKDOWN{f'  [{self.label}]' if self.label else ''}"
        add("=" * width)
        add(f"  {title}")
        add("=" * width)
        add(f"  {'stage':<26}{'ms':>10}{'% budget':>11}{'cumulative':>13}")
        add("  " + "-" * (width - 4))

        cum = 0.0
        for s in self.stages:
            if not s.in_budget:
                continue
            cum += s.ms
            pct = (s.ms / self.budget_ms * 100) if self.budget_ms else 0.0
            add(f"  {s.name:<26}{s.ms:>10.2f}{pct:>10.1f}%{cum:>13.2f}")

        add("  " + "-" * (width - 4))
        used = self.budget_ms_used
        pct = (used / self.budget_ms * 100) if self.budget_ms else 0.0
        add(f"  {'IN-BUDGET TOTAL':<26}{used:>10.2f}{pct:>10.1f}%")

        excluded = [s for s in self.stages if not s.in_budget]
        if excluded:
            add("")
            add("  excluded from the 200ms budget (upstream of chunking):")
            for s in excluded:
                add(f"    {s.name:<24}{s.ms:>10.2f}")

        if self.unaccounted_ms > 0.5:
            add("")
            add(f"  {'unaccounted':<26}{self.unaccounted_ms:>10.2f}"
                f"   <- time not covered by any stage")

        add("")
        add(f"  {'wall clock':<26}{self.wall_ms:>10.2f}")
        add("")
        verdict = "PASS" if self.passed else "OVER BUDGET"
        add(f"  200ms target (in-budget only) : {used:.2f}ms  [{verdict}]")
        slow = self.slowest(3)
        if slow:
            add(f"  slowest stages                : "
                + ", ".join(f"{s.name} {s.ms:.1f}ms" for s in slow))
        add("=" * width)
        return "\n".join(lines)

    def one_line(self) -> str:
        parts = " ".join(f"{s.name}={s.ms:.1f}" for s in self.stages if s.in_budget)
        ex = " ".join(f"{s.name}={s.ms:.1f}" for s in self.stages if not s.in_budget)
        out = f"budget={self.budget_ms_used:.1f}ms [{parts}]"
        if ex:
            out += f" | excluded [{ex}]"
        return out + f" | wall={self.wall_ms:.1f}ms {'PASS' if self.passed else 'OVER'}"


# ============================================================================
# AGGREGATION  (used by benchmark.py for per-stage percentiles)
# ============================================================================

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class ProfileAggregator:
    """Collects many Profiler runs and produces per-stage percentiles."""

    def __init__(self, budget_ms: float = DEFAULT_BUDGET_MS):
        self.budget_ms = budget_ms
        self.by_stage: Dict[str, List[float]] = {}
        self.stage_order: List[str] = []
        self.in_budget_flags: Dict[str, bool] = {}
        self.totals: List[float] = []
        self.walls: List[float] = []

    def add(self, p: Profiler) -> None:
        for s in p.stages:
            if s.name not in self.by_stage:
                self.by_stage[s.name] = []
                self.stage_order.append(s.name)
                self.in_budget_flags[s.name] = s.in_budget
            self.by_stage[s.name].append(s.ms)
        self.totals.append(p.budget_ms_used)
        self.walls.append(p.wall_ms)

    def report(self) -> Dict[str, Any]:
        rows = []
        for name in self.stage_order:
            v = self.by_stage[name]
            rows.append({
                "stage": name,
                "in_budget": self.in_budget_flags[name],
                "n": len(v),
                "mean": round(sum(v) / len(v), 2),
                "p50": round(percentile(v, 50), 2),
                "p70": round(percentile(v, 70), 2),
                "p90": round(percentile(v, 90), 2),
                "p95": round(percentile(v, 95), 2),
                "p100": round(percentile(v, 100), 2),
            })
        return {
            "budget_ms": self.budget_ms,
            "runs": len(self.totals),
            "stages": rows,
            "in_budget_total": {
                "p50": round(percentile(self.totals, 50), 2),
                "p70": round(percentile(self.totals, 70), 2),
                "p90": round(percentile(self.totals, 90), 2),
                "p100": round(percentile(self.totals, 100), 2),
            },
            "wall_total": {
                "p50": round(percentile(self.walls, 50), 2),
                "p70": round(percentile(self.walls, 70), 2),
                "p100": round(percentile(self.walls, 100), 2),
            },
        }

    def table(self, width: int = 88) -> str:
        r = self.report()
        lines = ["=" * width,
                 f"  PER-STAGE LATENCY PERCENTILES  ({r['runs']} runs)",
                 "=" * width,
                 f"  {'stage':<26}{'n':>6}{'mean':>9}{'P50':>9}{'P70':>9}{'P90':>9}{'P100':>10}",
                 "  " + "-" * (width - 4)]

        for row in r["stages"]:
            if not row["in_budget"]:
                continue
            lines.append(f"  {row['stage']:<26}{row['n']:>6}{row['mean']:>9.2f}"
                         f"{row['p50']:>9.2f}{row['p70']:>9.2f}{row['p90']:>9.2f}{row['p100']:>10.2f}")

        t = r["in_budget_total"]
        lines.append("  " + "-" * (width - 4))
        lines.append(f"  {'IN-BUDGET TOTAL':<26}{'':>6}{'':>9}"
                     f"{t['p50']:>9.2f}{t['p70']:>9.2f}{t['p90']:>9.2f}{t['p100']:>10.2f}")

        excluded = [row for row in r["stages"] if not row["in_budget"]]
        if excluded:
            lines.append("")
            lines.append("  excluded from budget (upstream of chunking):")
            for row in excluded:
                lines.append(f"  {row['stage']:<26}{row['n']:>6}{row['mean']:>9.2f}"
                             f"{row['p50']:>9.2f}{row['p70']:>9.2f}{row['p90']:>9.2f}{row['p100']:>10.2f}")

        lines.append("")
        for p in ("p50", "p70", "p100"):
            v = t[p]
            lines.append(f"  {p.upper():<6} in-budget: {v:>8.2f}ms   "
                         f"[{'PASS' if v < r['budget_ms'] else 'OVER BUDGET'}]")
        lines.append("=" * width)
        return "\n".join(lines)


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    import sys
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

    print("\n-- profiler --")
    p = Profiler(label="selftest")
    with p.stage("stt", in_budget=False):
        time.sleep(0.030)
    with p.stage("embed_query"):
        time.sleep(0.010)
    with p.stage("search_dense"):
        time.sleep(0.005)
    p.record("fusion", 0.4)
    p.close()

    check("excluded stage not counted in budget", p.budget_ms_used < 25)
    check("excluded stage recorded separately", 25 < p.excluded_ms < 60)
    check("wall >= budget + excluded", p.wall_ms >= p.budget_ms_used + p.excluded_ms - 1)
    check("passes 200ms with fast stages", p.passed)
    check("bottleneck identified", p.report()["bottleneck"] == "embed_query")

    slow = Profiler()
    slow.record("search_dense", 260.0)
    check("over-budget detected", not slow.passed)

    print("\n" + p.table())

    print("\n-- aggregator --")
    agg = ProfileAggregator()
    for i in range(50):
        q = Profiler()
        q.record("embed_query", 8 + i * 0.1)
        q.record("search_dense", 3 + i * 0.05)
        q.record("stt", 400 + i, in_budget=False)
        q.close()
        agg.add(q)
    rep = agg.report()
    emb = next(r for r in rep["stages"] if r["stage"] == "embed_query")
    check("aggregator collects all runs", emb["n"] == 50)
    check("p100 >= p50", emb["p100"] >= emb["p50"])
    check("stt excluded from in-budget total", rep["in_budget_total"]["p50"] < 50)

    print("\n" + agg.table())
    print(f"\n  {ok} passed, {fail} failed\n")
    sys.exit(1 if fail else 0)
