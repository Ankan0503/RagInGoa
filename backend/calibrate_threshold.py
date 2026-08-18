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

METHOD
------
Two labelled sets:
  POSITIVE  real queries from parents.sqlite. Their passages ARE indexed, so
            retrieval should succeed and the system SHOULD answer.
  NEGATIVE  out-of-domain / unsafe queries from benchmark.GUARDRAIL_SUITE plus
            generated nonsense. The system SHOULD refuse.

Retrieval runs over both, top scores are collected, and every candidate threshold
is scored by how cleanly it separates them. Youden's J (TPR - FPR) picks the
balanced optimum; a coverage-first value is also reported for anyone who would
rather answer more and lean on the grounding gate to catch the rest.

USAGE
-----
    python calibrate_threshold.py                # 300 positives, 60 negatives
    python calibrate_threshold.py -n 500 --json calib.json
"""

from __future__ import annotations

import os
import sys
import json
import random
import sqlite3
import argparse
from typing import List, Dict, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# QUERY SETS
# ============================================================================

def load_positives(n: int) -> List[str]:
    db = os.path.join(BASE, "parents.sqlite")
    if not os.path.exists(db):
        sys.exit(f"parents.sqlite not found at {db}. Build or download the index first.")
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT DISTINCT query FROM parents WHERE query IS NOT NULL "
        "AND length(query) > 8 ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# Out-of-domain. MSMARCO-XI is a static 2018-era web corpus, so live data,
# personal data and future events genuinely have no answer in it.
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

# Nonsense: real Hindi words in meaningless combinations. Nothing should match.
NONSENSE = [
    "बैंगनी हाथी गणित खाता है",
    "सात नीला सोमवार दौड़ता पत्थर",
    "खिड़की संगीत आलू क्यों उड़ता",
    "चंद्रमा कुर्सी तैरना पीला शब्दकोश",
    "हरा समय चम्मच नाचता बादल",
    "किताब पहाड़ हँसती मछली दरवाज़ा",
]


# ============================================================================
# SCORING
# ============================================================================

def collect_scores(retriever, queries: List[str], top_k: int, label: str) -> List[float]:
    scores = []
    for i, q in enumerate(queries, 1):
        try:
            res = retriever.retrieve(q, top_k=top_k)
            scores.append(res.top_score)
        except Exception as e:
            print(f"  {label} query failed: {e}")
        if i % 50 == 0:
            print(f"  {label}: {i}/{len(queries)}")
    return scores


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


def evaluate(positives: List[float], negatives: List[float],
             lo: float = 0.60, hi: float = 0.99, step: float = 0.005) -> List[Dict[str, Any]]:
    """
    For each candidate threshold: what fraction of real queries survive (TPR) and
    what fraction of out-of-domain queries wrongly survive (FPR)?
    """
    rows = []
    t = lo
    while t <= hi + 1e-9:
        tpr = sum(1 for s in positives if s >= t) / len(positives) if positives else 0.0
        fpr = sum(1 for s in negatives if s >= t) / len(negatives) if negatives else 0.0
        rows.append({"threshold": round(t, 4), "answer_rate": round(tpr, 4),
                     "leak_rate": round(fpr, 4), "youden_j": round(tpr - fpr, 4)})
        t += step
    return rows


def pick(rows: List[Dict[str, Any]], min_answer_rate: float = 0.95) -> Dict[str, Any]:
    balanced = max(rows, key=lambda r: r["youden_j"])
    # Coverage-first: the strictest threshold that still answers min_answer_rate
    # of real queries. The grounding gate is the second line of defence.
    eligible = [r for r in rows if r["answer_rate"] >= min_answer_rate]
    coverage = max(eligible, key=lambda r: r["threshold"]) if eligible else rows[0]
    return {"balanced": balanced, "coverage_first": coverage}


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Calibrate MIN_RETRIEVAL_SCORE against the real index")
    ap.add_argument("-n", "--num-positives", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-answer-rate", type=float, default=0.95,
                    help="Coverage-first target: fraction of real queries that must be answerable")
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

    current = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.80"))
    positives = load_positives(args.num_positives)
    negatives = OOD_QUERIES + NONSENSE

    print(f"\nScoring {len(positives)} real queries (should be ANSWERED)...")
    pos = collect_scores(r, positives, args.top_k, "positive")
    print(f"\nScoring {len(negatives)} out-of-domain queries (should be REFUSED)...")
    neg = collect_scores(r, negatives, args.top_k, "negative")

    if not pos:
        sys.exit("No positive scores collected — is the index populated?")

    rows = evaluate(pos, neg)
    best = pick(rows, args.min_answer_rate)

    W = 76
    print("\n" + "=" * W)
    print("  RETRIEVAL SCORE DISTRIBUTION")
    print("=" * W)
    print(histogram(pos, "REAL QUERIES (want to answer these)"))
    print()
    print(histogram(neg, "OUT-OF-DOMAIN (want to refuse these)"))

    pos_s, neg_s = sorted(pos), sorted(neg)
    def q(v, p):
        return v[min(len(v) - 1, int(len(v) * p))]

    print("\n" + "-" * W)
    print(f"  real queries      min={pos_s[0]:.4f}  p5={q(pos_s,0.05):.4f}  "
          f"p50={q(pos_s,0.5):.4f}  max={pos_s[-1]:.4f}")
    if neg_s:
        print(f"  out-of-domain     min={neg_s[0]:.4f}  p50={q(neg_s,0.5):.4f}  "
              f"p95={q(neg_s,0.95):.4f}  max={neg_s[-1]:.4f}")

    cur = next((x for x in rows if abs(x["threshold"] - current) < 0.0026), None)
    print("\n" + "=" * W)
    print("  THRESHOLD OPTIONS")
    print("=" * W)
    print(f"  {'threshold':<12}{'answers':>12}{'leaks OOD':>12}   note")
    print("  " + "-" * (W - 4))
    if cur:
        print(f"  {cur['threshold']:<12.3f}{cur['answer_rate']*100:>11.1f}%"
              f"{cur['leak_rate']*100:>11.1f}%   <- CURRENT ({current})")
    b, c = best["balanced"], best["coverage_first"]
    print(f"  {b['threshold']:<12.3f}{b['answer_rate']*100:>11.1f}%"
          f"{b['leak_rate']*100:>11.1f}%   <- balanced (best separation)")
    print(f"  {c['threshold']:<12.3f}{c['answer_rate']*100:>11.1f}%"
          f"{c['leak_rate']*100:>11.1f}%   <- coverage-first "
          f"(>={args.min_answer_rate*100:.0f}% answered)")

    print("\n" + "=" * W)
    print(f"  RECOMMENDED:  MIN_RETRIEVAL_SCORE={c['threshold']:.3f}")
    print("=" * W)
    print("  Coverage-first is the right default here: the grounding gate already")
    print("  catches answers unsupported by context, so the retrieval gate does not")
    print("  have to be the only defence. Refusing real questions is the more")
    print("  visible failure in a demo.")
    if cur and cur["answer_rate"] < 0.5:
        print()
        print(f"  WARNING: the current {current} answers only "
              f"{cur['answer_rate']*100:.0f}% of real queries.")
    print("=" * W + "\n")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"positives": pos, "negatives": neg, "curve": rows,
                       "recommended": best, "current": current}, f, indent=2)
        print(f"Written to {args.json}\n")


if __name__ == "__main__":
    main()
