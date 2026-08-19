#!/usr/bin/env python3
"""
Retrieval Threshold Calibration (calibrate_threshold.py)
========================================================
MIN_RETRIEVAL_SCORE decides when the system refuses to answer. Set it too high
and it rejects questions it could have answered; too low and it hallucinates from
irrelevant passages. The correct value is a property of the index and the
embedding model, so it must be measured, not guessed.

WHY THIS EXISTS
---------------
The first benchmark run refused 88 of 100 real dataset queries at the default
0.80 -- while `answer_recall_mean` was 0.75, meaning retrieval was actually
finding the right passages and the gate was throwing them away. That is a
threshold problem, not a retrieval problem, and it is invisible unless you look
at the score distribution.

A second, harder problem showed up once the absolute score itself was fixed
(commit c749b1f): on this index, real-query and out-of-domain scores overlap
heavily (measured 2026-08-19: real p50=0.896, out-of-domain p50=0.872). E5's
contrastive training compresses cosine into a narrow high band, so no single
MIN_RETRIEVAL_SCORE value cleanly separates them. This script now ALSO
calibrates MIN_SCORE_MARGIN -- retriever.py's relative-margin signal (does the
top passage stand out from the field, or is everything uniformly mediocre) --
since that is expected to separate better than the absolute score alone.

METHOD
------
Two labelled sets:
  POSITIVE  real queries from parents.sqlite. Their passages ARE indexed, so
            retrieval should succeed and the system SHOULD answer.
  NEGATIVE  out-of-domain / unsafe queries from benchmark.GUARDRAIL_SUITE plus
            generated nonsense. The system SHOULD refuse.

Retrieval runs over both, and BOTH signals are collected per query: the top
absolute score and the relative margin. Each is scored independently by how
cleanly it separates the two sets across every candidate threshold. Youden's J
(TPR - FPR) picks the balanced optimum for each; a coverage-first value is also
reported for anyone who would rather answer more and lean on the grounding gate
to catch the rest. The report states plainly which signal separates better —
that is measured here, not assumed.

USAGE
-----
    python calibrate_threshold.py                # 300 positives, 60 negatives
    python calibrate_threshold.py -n 500 --json calib.json
"""

from __future__ import annotations

import os
import sys
import json
import math
import random
import sqlite3
import argparse
from typing import List, Dict, Any, Tuple, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))


def load_positives(n: int) -> List[str]:
    db = os.path.join(BASE, "parents.sqlite")
    if not os.path.exists(db):
        sys.exit(f"parents.sqlite not found at {db}.")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT DISTINCT query FROM parents WHERE query IS NOT NULL "
        "AND length(query) > 8 ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


OOD_QUERIES = [
    "मंगल ग्रह पर आज का तापमान क्या है?",
    "कल भारत और ऑस्ट्रेलिया का मैच कौन जीतेगा?",
    "मेरे बैंक खाते में कितना पैसा है?",
    "2027 का शेयर बाजार कैसा रहेगा?",
    "मेरा पासवर्ड क्या है?",
    "आज दिल्ली में सोने का भाव क्या है?",
    "मेरी अगली मीटिंग कब है?",
    "इस समय कितने बजे हैं?",
    "अगले हफ्ते मौसम कैसा रहेगा?",
    "मेरे फोन का IMEI नंबर क्या है?",
    "क्वांटम कंप्यूटर में मेरा शोध पत्र कहाँ प्रकाशित हुआ?",
    "मेरी बिल्ली का नाम क्या है?",
]

NONSENSE = [
    "बैंगनी हाथी गणित खाता है",
    "सात नीला सोमवार दौड़ता पत्थर",
    "खिड़की संगीत आलू क्यों उड़ता",
    "चंद्रमा कुर्सी तैरना पीला शब्दकोश",
    "हरा समय चम्मच नाचता बादल",
    "किताब पहाड़ हँसती मछली दरवाज़ा",
]


def collect_scores(retriever, queries: List[str], top_k: int,
                   label: str) -> Tuple[List[float], List[float]]:
    scores, margins = [], []
    for i, q in enumerate(queries, 1):
        try:
            res = retriever.retrieve(q, top_k=top_k)
            scores.append(res.top_score)
            margins.append(res.score_margin)
        except Exception as e:
            print(f"  {label} query failed: {e}")
        if i % 50 == 0:
            print(f"  {label}: {i}/{len(queries)}")
    return scores, margins


def histogram(scores: List[float], label: str, lo: float = 0.60, hi: float = 1.0,
              bins: int = 20) -> str:
    if not scores:
        return f"  {label}: no data"
    width = (hi - lo) / bins
    counts = [0] * bins
    for s in scores:
        idx = min(bins - 1, max(0, int((s - lo) / width)))
        counts[idx] += 1
    peak = max(counts) or 1
    out = [f"  {label}  (n={len(scores)})"]
    for i, c in enumerate(counts):
        edge = lo + i * width
        bar = "#" * int(c / peak * 42)
        out.append(f"    {edge:.3f}  {bar:<42} {c}")
    return "\n".join(out)


def auto_range(values: List[float], lo: float = 0.0, pad: float = 0.01,
               n_steps: int = 120) -> Tuple[float, float, float]:
    hi = round((max(values) if values else 0.0) + pad, 4)
    step = max((hi - lo) / n_steps, 1e-4)
    return lo, hi, step


def evaluate(positives: List[float], negatives: List[float],
             lo: float = 0.60, hi: float = 0.99, step: float = 0.005) -> List[Dict[str, Any]]:
    rows = []
    t = lo
    while t <= hi + 1e-9:
        tpr = sum(1 for s in positives if s >= t) / len(positives) if positives else 0.0
        fpr = sum(1 for s in negatives if s >= t) / len(negatives) if negatives else 0.0
        rows.append({"threshold": round(t, 4), "answer_rate": round(tpr, 4),
                     "leak_rate": round(fpr, 4), "youden_j": round(tpr - fpr, 4)})
        t += step
    return rows


def _coverage_tolerance(n: int, min_answer_rate: float) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(min_answer_rate * (1 - min_answer_rate) / n)


def pick(rows: List[Dict[str, Any]], min_answer_rate: float = 0.95,
        n_positives: Optional[int] = None) -> Dict[str, Any]:
    balanced = max(rows, key=lambda r: r["youden_j"])
    tol = _coverage_tolerance(n_positives, min_answer_rate) if n_positives else 0.0
    eligible = [r for r in rows if r["answer_rate"] >= min_answer_rate - tol]
    coverage = (min(eligible, key=lambda r: (r["leak_rate"], -r["threshold"]))
               if eligible else rows[0])
    return {"balanced": balanced, "coverage_first": coverage}


def pair_stats(pos_score: List[float], neg_score: List[float],
               pos_margin: List[float], neg_margin: List[float],
               score_thr: float, margin_thr: float) -> Dict[str, float]:
    n_pos, n_neg = len(pos_score), len(neg_score)
    ans = sum(1 for s, m in zip(pos_score, pos_margin) if s >= score_thr and m >= margin_thr)
    leak = sum(1 for s, m in zip(neg_score, neg_margin) if s >= score_thr and m >= margin_thr)
    return {"answer_rate": round(ans / n_pos, 4) if n_pos else 0.0,
            "leak_rate": round(leak / n_neg, 4) if n_neg else 0.0}


def joint_search(pos_score: List[float], neg_score: List[float],
                 pos_margin: List[float], neg_margin: List[float],
                 score_grid: List[float], margin_grid: List[float],
                 min_answer_rate: float) -> Dict[str, Any]:
    n_pos, n_neg = len(pos_score), len(neg_score)
    eff_min = min_answer_rate - _coverage_tolerance(n_pos, min_answer_rate)
    best_balanced = best_coverage = None
    for ts in score_grid:
        pos_pass_s = [s >= ts for s in pos_score]
        neg_pass_s = [s >= ts for s in neg_score]
        for tm in margin_grid:
            ans = sum(1 for i in range(n_pos) if pos_pass_s[i] and pos_margin[i] >= tm)
            leak = sum(1 for i in range(n_neg) if neg_pass_s[i] and neg_margin[i] >= tm)
            ar = round(ans / n_pos, 4) if n_pos else 0.0
            lr = round(leak / n_neg, 4) if n_neg else 0.0
            row = {"score_threshold": round(ts, 4), "margin_threshold": round(tm, 4),
                  "answer_rate": ar, "leak_rate": lr, "youden_j": round(ar - lr, 4)}
            if best_balanced is None or row["youden_j"] > best_balanced["youden_j"]:
                best_balanced = row
            if ar >= eff_min and (
                    best_coverage is None
                    or lr < best_coverage["leak_rate"]
                    or (lr == best_coverage["leak_rate"] and ts > best_coverage["score_threshold"])):
                best_coverage = row
    return {"balanced": best_balanced, "coverage_first": best_coverage}


def _quantiles(values: List[float]) -> str:
    s = sorted(values)
    def q(p):
        return s[min(len(s) - 1, int(len(s) * p))]
    return f"min={s[0]:.4f}  p5={q(0.05):.4f}  p50={q(0.5):.4f}  max={s[-1]:.4f}"


def _print_threshold_table(rows, best, current, env_name: str, W: int,
                           min_answer_rate: float):
    cur_row = next((x for x in rows if abs(x["threshold"] - current) < (rows[1]["threshold"] - rows[0]["threshold"]) / 2 + 1e-9), None)
    print(f"  {'threshold':<12}{'answers':>12}{'leaks OOD':>12}   note")
    print("  " + "-" * (W - 4))
    if cur_row:
        print(f"  {cur_row['threshold']:<12.4f}{cur_row['answer_rate']*100:>11.1f}%"
              f"{cur_row['leak_rate']*100:>11.1f}%   <- CURRENT {env_name}={current}")
    b, c = best["balanced"], best["coverage_first"]
    print(f"  {b['threshold']:<12.4f}{b['answer_rate']*100:>11.1f}%"
          f"{b['leak_rate']*100:>11.1f}%   <- balanced")
    print(f"  {c['threshold']:<12.4f}{c['answer_rate']*100:>11.1f}%"
          f"{c['leak_rate']*100:>11.1f}%   <- coverage-first (>={min_answer_rate*100:.0f}%)")
    return cur_row


def main():
    ap = argparse.ArgumentParser(description="Calibrate MIN_RETRIEVAL_SCORE and MIN_SCORE_MARGIN")
    ap.add_argument("-n", "--num-positives", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-answer-rate", type=float, default=0.95)
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    from retriever import IndicRetriever, RetrieverError
    print("Initialising retriever...")
    try:
        r = IndicRetriever(strict=True)
    except RetrieverError as e:
        sys.exit(f"\nFAILED: {e}\n")

    current_score = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.80"))
    current_margin = float(os.getenv("MIN_SCORE_MARGIN", "0.0"))
    positives = load_positives(args.num_positives)
    negatives = OOD_QUERIES + NONSENSE

    print(f"\nScoring {len(positives)} real queries...")
    pos, pos_margin = collect_scores(r, positives, args.top_k, "positive")
    print(f"\nScoring {len(negatives)} out-of-domain queries...")
    neg, neg_margin = collect_scores(r, negatives, args.top_k, "negative")

    if not pos:
        sys.exit("No positive scores collected.")

    out_of_range = [s for s in pos + neg if s > 1.0001 or s < -0.0001]
    if out_of_range:
        sys.exit(f"\nABORT: {len(out_of_range)} score(s) outside valid cosine range [0,1].\n")

    bad_margins = [m for m in pos_margin + neg_margin if m < -0.0001]
    if bad_margins:
        sys.exit(f"\nABORT: {len(bad_margins)} negative score_margin value(s) seen.\n")

    rows = evaluate(pos, neg)
    best = pick(rows, args.min_answer_rate, n_positives=len(pos))

    m_lo, m_hi, m_step = auto_range(pos_margin + neg_margin)
    rows_m = evaluate(pos_margin, neg_margin, lo=m_lo, hi=m_hi, step=m_step)
    best_m = pick(rows_m, args.min_answer_rate, n_positives=len(pos_margin))

    W = 76
    print("\n" + "=" * W)
    print("  SIGNAL 1: ABSOLUTE SCORE  (MIN_RETRIEVAL_SCORE)")
    print("=" * W)
    print(histogram(pos, "REAL QUERIES"))
    print()
    print(histogram(neg, "OUT-OF-DOMAIN"))
    print("\n" + "-" * W)
    print(f"  real queries      {_quantiles(pos)}")
    if neg:
        print(f"  out-of-domain     {_quantiles(neg)}")
    print("\n" + "-" * W)
    cur = _print_threshold_table(rows, best, current_score, "MIN_RETRIEVAL_SCORE", W,
                                 args.min_answer_rate)

    print("\n" + "=" * W)
    print("  SIGNAL 2: RELATIVE MARGIN  (MIN_SCORE_MARGIN)")
    print("=" * W)
    print(histogram(pos_margin, "REAL QUERIES", lo=m_lo, hi=m_hi))
    print()
    print(histogram(neg_margin, "OUT-OF-DOMAIN", lo=m_lo, hi=m_hi))
    print("\n" + "-" * W)
    print(f"  real queries      {_quantiles(pos_margin)}")
    if neg_margin:
        print(f"  out-of-domain     {_quantiles(neg_margin)}")
    print("\n" + "-" * W)
    cur_m = _print_threshold_table(rows_m, best_m, current_margin, "MIN_SCORE_MARGIN", W,
                                   args.min_answer_rate)

    j_score = max(x["youden_j"] for x in rows)
    j_margin = max(x["youden_j"] for x in rows_m)

    score_grid = sorted({row["threshold"] for row in rows})
    margin_grid = sorted({row["threshold"] for row in rows_m})
    joint = joint_search(pos, neg, pos_margin, neg_margin,
                         score_grid, margin_grid, args.min_answer_rate)
    now = pair_stats(pos, neg, pos_margin, neg_margin, current_score, current_margin)

    print("\n" + "=" * W)
    print("  RECOMMENDATION")
    print("=" * W)
    print(f"  MIN_RETRIEVAL_SCORE alone best J = {j_score:.4f}")
    print(f"  MIN_SCORE_MARGIN alone    best J = {j_margin:.4f}")
    print(f"\n  JOINT EFFECT (AND-gate)")
    print("  " + "-" * (W - 4))
    print(f"  currently deployed   MIN_RETRIEVAL_SCORE={current_score:<7.3f}"
          f"MIN_SCORE_MARGIN={current_margin:<8.4f}"
          f"-> answers {now['answer_rate']*100:5.1f}%  leaks {now['leak_rate']*100:5.1f}%")
    if joint["coverage_first"]:
        jc = joint["coverage_first"]
        print(f"  coverage-first pair  MIN_RETRIEVAL_SCORE={jc['score_threshold']:<7.3f}"
              f"MIN_SCORE_MARGIN={jc['margin_threshold']:<8.4f}"
              f"-> answers {jc['answer_rate']*100:5.1f}%  leaks {jc['leak_rate']*100:5.1f}%")
    if joint["balanced"]:
        jb = joint["balanced"]
        print(f"  balanced pair        MIN_RETRIEVAL_SCORE={jb['score_threshold']:<7.3f}"
              f"MIN_SCORE_MARGIN={jb['margin_threshold']:<8.4f}"
              f"-> answers {jb['answer_rate']*100:5.1f}%  leaks {jb['leak_rate']*100:5.1f}%")

    print("=" * W + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "positives": pos, "negatives": neg,
                "positives_margin": pos_margin, "negatives_margin": neg_margin,
                "curve_score": rows, "curve_margin": rows_m,
                "recommended_score": best, "recommended_margin": best_m,
                "current_score": current_score, "current_margin": current_margin,
                "best_j_score": j_score, "best_j_margin": j_margin,
                "joint_search": joint, "joint_current": now,
            }, f, indent=2)
        print(f"Written to {args.json}\n")


if __name__ == "__main__":
    main()
