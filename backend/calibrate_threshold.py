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

def collect_scores(retriever, queries: List[str], top_k: int,
                   label: str) -> Tuple[List[float], List[float]]:
    """Returns (top_scores, score_margins) — same retrieval call, both signals."""
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
    """
    score_margin has no natural scale the way cosine has [0,1] — it depends on
    how tightly this index's scores cluster. Derive (lo, hi, step) from what was
    actually observed rather than guessing bounds.
    """
    hi = round((max(values) if values else 0.0) + pad, 4)
    step = max((hi - lo) / n_steps, 1e-4)
    return lo, hi, step


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
          f"{b['leak_rate']*100:>11.1f}%   <- balanced (best separation)")
    print(f"  {c['threshold']:<12.4f}{c['answer_rate']*100:>11.1f}%"
          f"{c['leak_rate']*100:>11.1f}%   <- coverage-first "
          f"(>={min_answer_rate*100:.0f}% answered)")
    return cur_row


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate MIN_RETRIEVAL_SCORE and MIN_SCORE_MARGIN against the real index")
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

    current_score = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.80"))
    current_margin = float(os.getenv("MIN_SCORE_MARGIN", "0.0"))
    positives = load_positives(args.num_positives)
    negatives = OOD_QUERIES + NONSENSE

    print(f"\nScoring {len(positives)} real queries (should be ANSWERED)...")
    pos, pos_margin = collect_scores(r, positives, args.top_k, "positive")
    print(f"\nScoring {len(negatives)} out-of-domain queries (should be REFUSED)...")
    neg, neg_margin = collect_scores(r, negatives, args.top_k, "negative")

    if not pos:
        sys.exit("No positive scores collected — is the index populated?")

    # Cosine similarity is bounded to [0,1]. Anything above that means a
    # non-cosine score (BM25 is unbounded) has leaked into raw_score, which
    # makes every threshold below meaningless. This exact bug shipped once and
    # silently disabled the retrieval guardrail entirely.
    out_of_range = [s for s in pos + neg if s > 1.0001 or s < -0.0001]
    if out_of_range:
        sys.exit(
            f"\nABORT: {len(out_of_range)} score(s) outside the valid cosine range "
            f"[0,1] (max seen {max(out_of_range):.4f}).\n\n"
            f"raw_score must be the DENSE cosine only. A value above 1.0 means a\n"
            f"BM25 score has leaked into it, in which case MIN_RETRIEVAL_SCORE is\n"
            f"being compared against an unbounded scale and nothing will ever be\n"
            f"refused. Fix retriever.retrieve() before calibrating.\n"
        )

    # score_margin is top minus a mean of lower scores, so it is never negative
    # by construction. A negative value means retriever.py's margin computation
    # is broken, not a calibration problem — fix it before trusting anything else.
    bad_margins = [m for m in pos_margin + neg_margin if m < -0.0001]
    if bad_margins:
        sys.exit(
            f"\nABORT: {len(bad_margins)} negative score_margin value(s) seen "
            f"(min {min(bad_margins):.4f}). score_margin cannot be negative by "
            f"construction — check retriever.retrieve()'s margin computation.\n"
        )

    rows = evaluate(pos, neg)
    best = pick(rows, args.min_answer_rate)

    m_lo, m_hi, m_step = auto_range(pos_margin + neg_margin)
    rows_m = evaluate(pos_margin, neg_margin, lo=m_lo, hi=m_hi, step=m_step)
    best_m = pick(rows_m, args.min_answer_rate)

    W = 76
    print("\n" + "=" * W)
    print("  SIGNAL 1: ABSOLUTE SCORE  (MIN_RETRIEVAL_SCORE)")
    print("=" * W)
    print(histogram(pos, "REAL QUERIES (want to answer these)"))
    print()
    print(histogram(neg, "OUT-OF-DOMAIN (want to refuse these)"))
    print("\n" + "-" * W)
    print(f"  real queries      {_quantiles(pos)}")
    if neg:
        print(f"  out-of-domain     {_quantiles(neg)}")
    print("\n" + "-" * W)
    cur = _print_threshold_table(rows, best, current_score, "MIN_RETRIEVAL_SCORE", W,
                                 args.min_answer_rate)

    print("\n" + "=" * W)
    print("  SIGNAL 2: RELATIVE MARGIN  (MIN_SCORE_MARGIN)")
    print("  top score minus the mean of the next candidates — does the best")
    print("  match stand out, or is everything about equally (ir)relevant?")
    print("=" * W)
    print(histogram(pos_margin, "REAL QUERIES (want to answer these)", lo=m_lo, hi=m_hi))
    print()
    print(histogram(neg_margin, "OUT-OF-DOMAIN (want to refuse these)", lo=m_lo, hi=m_hi))
    print("\n" + "-" * W)
    print(f"  real queries      {_quantiles(pos_margin)}")
    if neg_margin:
        print(f"  out-of-domain     {_quantiles(neg_margin)}")
    print("\n" + "-" * W)
    cur_m = _print_threshold_table(rows_m, best_m, current_margin, "MIN_SCORE_MARGIN", W,
                                   args.min_answer_rate)

    # Which signal ACTUALLY separates better here? Youden's J at its best point
    # is a single number for "how cleanly can any threshold on this signal split
    # real from out-of-domain" — measured, not assumed, because the whole reason
    # this script grew a second signal is that the first one was assumed to work
    # and measurably didn't.
    j_score = max(x["youden_j"] for x in rows)
    j_margin = max(x["youden_j"] for x in rows_m)
    winner = "MIN_SCORE_MARGIN" if j_margin > j_score else "MIN_RETRIEVAL_SCORE"

    b, c = best["balanced"], best["coverage_first"]
    bm, cm = best_m["balanced"], best_m["coverage_first"]
    print("\n" + "=" * W)
    print("  RECOMMENDATION")
    print("=" * W)
    print(f"  separation quality (Youden's J, higher = cleaner split):")
    print(f"    MIN_RETRIEVAL_SCORE   best J = {j_score:.4f}")
    print(f"    MIN_SCORE_MARGIN      best J = {j_margin:.4f}   <- {winner} separates better"
          if j_margin != j_score else f"    MIN_SCORE_MARGIN      best J = {j_margin:.4f}")
    print()
    print(f"  Set BOTH — they are independent AND-ed checks (guardrails.RetrievalGate):")
    print(f"    MIN_RETRIEVAL_SCORE={c['threshold']:.3f}   "
          f"(coverage-first: answers {c['answer_rate']*100:.1f}%, leaks {c['leak_rate']*100:.1f}%)")
    print(f"    MIN_SCORE_MARGIN={cm['threshold']:.4f}   "
          f"(coverage-first: answers {cm['answer_rate']*100:.1f}%, leaks {cm['leak_rate']*100:.1f}%)")
    print()
    print("  Coverage-first is the right default here: the grounding gate already")
    print("  catches answers unsupported by context, so the retrieval gate does not")
    print("  have to be the only defence. Refusing real questions is the more")
    print("  visible failure in a demo.")
    if cur and cur["answer_rate"] < 0.5:
        print()
        print(f"  WARNING: the current MIN_RETRIEVAL_SCORE={current_score} answers only "
              f"{cur['answer_rate']*100:.0f}% of real queries.")
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
            }, f, indent=2)
        print(f"Written to {args.json}\n")


if __name__ == "__main__":
    main()
