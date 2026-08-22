#!/usr/bin/env python3
"""
Retrieval Gate Diagnostics (diagnose_gates.py)
=============================================
Measures the two corrections the retrieval guardrail gained on top of a single
absolute cosine floor, so their constants are numbers off this index rather
than numbers off a whiteboard.

WHY THIS EXISTS
---------------
calibrate_threshold.py answers "where should MIN_RETRIEVAL_SCORE sit?" and its
answer is uncomfortable: on this index real and out-of-domain scores genuinely
overlap (real p50 0.896, out-of-domain p50 0.872), so no single value separates
them. Both signals below exist because the honest response to that is to stop
asking one number to do the whole job.

    --strategies   Pinning a chunking strategy filters the search to one chunk
                   type, so the best reachable chunk is not the best chunk in
                   the corpus and every score shifts DOWN. One absolute floor
                   therefore answers a question under one dropdown setting and
                   refuses the same question under another. This mode measures
                   the shift per strategy and prints the offsets that restore
                   a consistent verdict.

    --rescue       The gate reads raw_score, which is dense-only on purpose, so
                   BM25 could never vouch for anything -- and the false
                   refusals are overwhelmingly entity lookups, precisely what
                   BM25 is in the pipeline to serve. retriever.py now reports
                   `lexical_agreement`: dense and BM25 independently ranked the
                   SAME passage first. This mode measures how many real queries
                   that recovers against how many out-of-domain queries it lets
                   through, at each width of rescue band.

Both modes report against the floor this process actually has
(MIN_RETRIEVAL_SCORE), so run it in the same environment as the server.

USAGE
-----
    python diagnose_gates.py                        # both modes
    python diagnose_gates.py --strategies -n 60
    python diagnose_gates.py --rescue -n 200 --json gates.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Sequence

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from calibrate_threshold import load_positives, OOD_QUERIES, NONSENSE

W = 78

# The labels the UI can actually send, plus the two the index holds but the
# dropdown does not expose yet. Left in deliberately: the offsets for those are
# provisional in the shipped default, and this is what would replace the guess.
STRATEGY_LABELS = ["parent_child", "sliding_window", "semantic", "passage"]


def _q(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * p))]


def _rate(values: Sequence[float], floor: float) -> float:
    return (sum(1 for v in values if v >= floor) / len(values)) if values else 0.0


# ----------------------------------------------------------------------
#  MODE A -- per-strategy floor offsets
# ----------------------------------------------------------------------

def run_strategies(r, queries: List[str], floor: float) -> Dict[str, Any]:
    from retriever import canonical_strategy

    base: List[float] = []
    per_strategy: Dict[str, List[float]] = {lbl: [] for lbl in STRATEGY_LABELS}

    n_runs = len(queries) * (len(STRATEGY_LABELS) + 1)
    print(f"\nScoring {len(queries)} queries under {len(STRATEGY_LABELS) + 1} "
          f"settings ({n_runs} retrievals)...")
    for i, q in enumerate(queries, 1):
        try:
            unfiltered = r.retrieve(q, top_k=3).top_score
            row = {lbl: r.retrieve(q, top_k=3, strategy=lbl).top_score
                   for lbl in STRATEGY_LABELS}
        except Exception as e:
            print(f"  query failed, skipping: {e}")
            continue
        # Appended together so base[] and every per_strategy[] list stay index
        # aligned -- the shift is a per-query difference, and a half-recorded
        # query would silently pair one query's filtered score with another
        # query's unfiltered one.
        base.append(unfiltered)
        for lbl, v in row.items():
            per_strategy[lbl].append(v)
        if i % 20 == 0:
            print(f"  {i}/{len(queries)}")

    if not base:
        return {}

    base_rate = _rate(base, floor)
    print("\n" + "=" * W)
    print("  MODE A: PER-STRATEGY FLOOR OFFSETS")
    print("=" * W)
    print(f"  floor MIN_RETRIEVAL_SCORE = {floor:.4f}")
    print(f"  unfiltered answer rate    = {base_rate * 100:.1f}%  (n={len(base)})")
    print(f"  unfiltered score          = p10 {_q(base, 0.10):.4f}  "
          f"p50 {_q(base, 0.50):.4f}  p90 {_q(base, 0.90):.4f}")
    print()
    print(f"  {'strategy':<18}{'canonical':<12}{'p50 score':>11}{'p50 shift':>11}"
          f"{'answers':>10}{'offset':>10}{'then':>8}")
    print("  " + "-" * (W - 4))

    out: Dict[str, Any] = {"floor": floor, "n": len(base),
                           "unfiltered_answer_rate": round(base_rate, 4),
                           "strategies": {}}

    for lbl in STRATEGY_LABELS:
        vals = per_strategy[lbl]
        if not vals:
            continue
        canon = canonical_strategy(lbl)
        shifts = [v - b for v, b in zip(vals, base)]
        rate_now = _rate(vals, floor)

        # Pick the offset that restores the unfiltered answer rate: the
        # smallest downward shift at which this strategy answers as often as an
        # unfiltered search does. Chosen by matching rates rather than by taking
        # a fixed percentile of the shift, because the rate is the thing that
        # actually misbehaves in front of a user.
        step, max_steps = 0.005, 40
        chosen, achieved = round(-max_steps * step, 4), _rate(vals, floor - max_steps * step)
        for k in range(0, max_steps + 1):
            cand = -k * step
            if _rate(vals, floor + cand) >= base_rate - 1e-9:
                chosen, achieved = round(cand, 4), _rate(vals, floor + cand)
                break

        print(f"  {lbl:<18}{str(canon):<12}{_q(vals, 0.50):>11.4f}"
              f"{_q(shifts, 0.50):>11.4f}{rate_now * 100:>9.1f}%"
              f"{chosen:>10.3f}{achieved * 100:>7.1f}%")
        out["strategies"][lbl] = {
            "canonical": canon,
            "p50_score": round(_q(vals, 0.50), 4),
            "p50_shift": round(_q(shifts, 0.50), 4),
            "p10_shift": round(_q(shifts, 0.10), 4),
            "answer_rate_now": round(rate_now, 4),
            "offset": chosen,
            "answer_rate_with_offset": round(achieved, 4),
        }

    spec = ",".join(
        f"{v['canonical']}={v['offset']:.3f}"
        for v in out["strategies"].values()
        if v["canonical"] and v["offset"] < 0)
    out["recommended_spec"] = spec
    print("\n  " + "-" * (W - 4))
    print(f"  STRATEGY_FLOOR_DELTA={spec or '(no offset needed)'}")
    return out


# ----------------------------------------------------------------------
#  MODE B -- BM25 rescue band
# ----------------------------------------------------------------------

BANDS = (0.0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10)
SHIPPED_BAND = 0.02


def run_rescue(r, positives: List[str], negatives: List[str],
               floor: float) -> Dict[str, Any]:
    def collect(queries: List[str], label: str) -> List[Dict[str, Any]]:
        rows = []
        for i, q in enumerate(queries, 1):
            try:
                res = r.retrieve(q, top_k=3)
                rows.append({"query": q, "top": res.top_score,
                             "agree": bool(res.lexical_agreement),
                             "sparse": res.top_sparse_score})
            except Exception as e:
                print(f"  {label} query failed: {e}")
            if i % 50 == 0:
                print(f"  {label}: {i}/{len(queries)}")
        return rows

    print(f"\nScoring {len(positives)} real queries...")
    pos = collect(positives, "positive")
    print(f"\nScoring {len(negatives)} out-of-domain queries...")
    neg = collect(negatives, "negative")
    if not pos:
        return {}

    # Every positive is a query whose own passage is in the index, so every
    # refusal in this set is a FALSE refusal by construction. That is what makes
    # this set usable as ground truth without hand labelling.
    refused = [p for p in pos if p["top"] < floor]
    leakable = [n for n in neg if n["top"] < floor]

    print("\n" + "=" * W)
    print("  MODE B: BM25 RESCUE BAND")
    print("=" * W)
    print(f"  floor MIN_RETRIEVAL_SCORE = {floor:.4f}")
    print(f"  real queries              n={len(pos)}, {len(refused)} refused "
          f"({len(refused) / len(pos) * 100:.1f}%) -- all false by construction")
    print(f"  out-of-domain             n={len(neg)}, "
          f"{len(neg) - len(leakable)} already answered above the floor")
    if refused:
        print(f"  dense/BM25 agreement      "
              f"{sum(1 for p in refused if p['agree'])}/{len(refused)} of the refused "
              f"real queries, {sum(1 for n in leakable if n['agree'])}/{len(leakable)} "
              f"of the below-floor out-of-domain ones")
    print()
    print(f"  {'band':<10}{'real recovered':>20}{'OOD admitted':>18}{'net':>8}")
    print("  " + "-" * (W - 4))

    rows = []
    for band in BANDS:
        rec = sum(1 for p in refused if p["agree"] and p["top"] >= floor - band)
        leak = sum(1 for n in leakable if n["agree"] and n["top"] >= floor - band)
        rows.append({"band": band, "recovered": rec, "admitted": leak})
        mark = "  <- shipped default" if abs(band - SHIPPED_BAND) < 1e-9 else ""
        print(f"  {band:<10.3f}{rec:>14} /{len(refused):<5}"
              f"{leak:>12} /{len(leakable):<5}{rec - leak:>8}{mark}")

    clean = [x for x in rows if x["admitted"] == 0 and x["recovered"] > 0]
    best = max(clean, key=lambda x: x["band"]) if clean else None
    print("\n  " + "-" * (W - 4))
    if best:
        print(f"  SPARSE_RESCUE_DELTA={best['band']:.3f}   recovers "
              f"{best['recovered']} real queries, admits 0 out-of-domain")
    else:
        print("  No band recovers real queries without admitting out-of-domain ones.")
        print("  Keep SPARSE_RESCUE_DELTA=0 and treat the floor itself as the defect.")

    if refused:
        print("\n  Refused real queries closest to the floor:")
        for p in sorted(refused, key=lambda x: -x["top"])[:10]:
            flag = "AGREE" if p["agree"] else "  -  "
            print(f"    {p['top']:.4f}  {flag}  bm25={p['sparse']:>8.3f}  {p['query'][:44]}")

    return {"floor": floor, "n_positive": len(pos), "n_negative": len(neg),
            "n_refused": len(refused), "bands": rows,
            "recommended_band": best["band"] if best else 0.0,
            "refused_examples": sorted(refused, key=lambda x: -x["top"])[:25]}


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose the retrieval gate's two correction signals")
    ap.add_argument("--strategies", action="store_true", help="only run mode A")
    ap.add_argument("--rescue", action="store_true", help="only run mode B")
    ap.add_argument("-n", "--num-queries", type=int, default=0,
                    help="query count (default 40 for mode A, 200 for mode B)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    do_a = args.strategies or not args.rescue
    do_b = args.rescue or not args.strategies

    from retriever import IndicRetriever, RetrieverError
    floor = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.80"))
    print(f"Initialising retriever... (MIN_RETRIEVAL_SCORE={floor}, "
          f"SPARSE_MODE={os.getenv('SPARSE_MODE', 'on')})")
    try:
        r = IndicRetriever(strict=True)
    except RetrieverError as e:
        sys.exit(f"\nFAILED: {e}\n")

    out: Dict[str, Any] = {"floor": floor, "seed": args.seed}
    if do_a:
        n = args.num_queries or 40
        out["strategies"] = run_strategies(r, load_positives(n, seed=args.seed), floor)
    if do_b:
        n = args.num_queries or 200
        out["rescue"] = run_rescue(r, load_positives(n, seed=args.seed),
                                   OOD_QUERIES + NONSENSE, floor)

    print("\n" + "=" * W)
    print("  APPLY (Dokploy environment panel, then recreate the container)")
    print("=" * W)
    if out.get("strategies", {}).get("recommended_spec"):
        print(f"  STRATEGY_FLOOR_DELTA={out['strategies']['recommended_spec']}")
    if do_b and out.get("rescue"):
        print(f"  SPARSE_RESCUE_DELTA={out['rescue']['recommended_band']:.3f}")
    print("=" * W + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Written to {args.json}\n")


if __name__ == "__main__":
    main()
