/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState, useCallback } from "react";
import { ChevronDown, RefreshCw, AlertCircle } from "lucide-react";
import { resolveApiBaseUrl } from "../../context/RagContext";

interface StageTiming {
  stage: string;
  ms: number;
  in_budget: boolean;
  detail?: string;
}

interface QueryLogEntry {
  id: number;
  ts: string;
  query: string;
  transcript: string | null;
  answer: string;
  refused: boolean;
  refusal_reason: string | null;
  provider: string | null;
  model: string | null;
  retrieval_ms: number | null;
  generation_ms: number | null;
  guardrail_ms: number | null;
  end_to_end_ms: number | null;
  grounding_score: number | null;
  stages: StageTiming[];
}

interface InsightsResponse {
  entries: QueryLogEntry[];
  stats: {
    total_queries: number;
    total_refused: number;
    avg_retrieval_ms: number | null;
    avg_generation_ms: number | null;
    avg_end_to_end_ms: number | null;
  };
}

const RETRIEVAL_STAGE_NAMES = new Set([
  "query_normalize", "embed_query", "search_dense",
  "bm25_encode", "search_sparse", "fusion_rrf", "parent_fetch",
]);

function fmtMs(v: number | null | undefined): string {
  if (v == null) return "—";
  return v < 10 ? `${v.toFixed(2)}ms` : `${v.toFixed(1)}ms`;
}

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex-1 min-w-[120px] rounded-[12px] bg-[#F7F9F6] border border-[#E6EAE3] px-4 py-3 flex flex-col gap-0.5">
      <span className="font-sans text-[19px] font-semibold text-[#176B4F] leading-none">
        {value}
      </span>
      <span className="font-sans text-[11px] text-[#727873] uppercase tracking-wide mt-1">
        {label}
      </span>
    </div>
  );
}

function StageRow({ stage }: { stage: StageTiming }) {
  const isRetrieval = RETRIEVAL_STAGE_NAMES.has(stage.stage);
  return (
    <div className="flex items-center justify-between py-1.5 px-3 rounded-[6px] hover:bg-[#F7F9F6]">
      <div className="flex items-center gap-2">
        {isRetrieval && (
          <span className="w-[5px] h-[5px] rounded-full bg-[#8AAE97]" aria-hidden="true" />
        )}
        <span className="font-sans text-[12.5px] text-[#3E453F]">{stage.stage}</span>
        {stage.detail && (
          <span className="font-sans text-[11px] text-[#9AA39D]">({stage.detail})</span>
        )}
      </div>
      <span className={`font-sans text-[12.5px] tabular-nums ${stage.in_budget ? "text-[#3E453F]" : "text-[#C4622D]"}`}>
        {fmtMs(stage.ms)}
      </span>
    </div>
  );
}

function EntryRow({ entry }: { entry: QueryLogEntry }) {
  const [expanded, setExpanded] = useState(false);
  const displayQuestion = entry.transcript || entry.query;

  return (
    <div className="border border-[#EAE8E2] rounded-[14px] bg-white overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full text-left px-4 py-3 flex items-start gap-3 cursor-pointer hover:bg-[#FAFBF9] transition-colors duration-150 outline-none"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-sans text-[11px] text-[#9AA39D]">{fmtTime(entry.ts)}</span>
            {entry.refused ? (
              <span className="font-sans text-[10.5px] font-medium text-[#B8480F] bg-[#FCEEE6] px-2 py-[1px] rounded-full">
                Refused{entry.refusal_reason ? `: ${entry.refusal_reason}` : ""}
              </span>
            ) : (
              <span className="font-sans text-[10.5px] font-medium text-[#176B4F] bg-[#EEF5ED] px-2 py-[1px] rounded-full">
                Answered
              </span>
            )}
            {entry.model && (
              <span className="font-sans text-[10.5px] text-[#727873] bg-[#F3F2ED] px-2 py-[1px] rounded-full">
                {entry.model}
              </span>
            )}
          </div>
          <p className="font-sans text-[14px] text-[#151A17] mt-1.5 leading-[1.4] break-words">
            {displayQuestion}
          </p>
          {!entry.refused && (
            <p className="font-sans text-[13px] text-[#5C635D] mt-1 leading-[1.4] break-words line-clamp-2">
              {entry.answer}
            </p>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          <div className="hidden sm:flex items-center gap-2.5 font-sans text-[11.5px] text-[#727873] tabular-nums">
            <span>R {fmtMs(entry.retrieval_ms)}</span>
            <span>·</span>
            <span>L {fmtMs(entry.generation_ms)}</span>
            <span>·</span>
            <span className="font-medium text-[#3E453F]">E2E {fmtMs(entry.end_to_end_ms)}</span>
          </div>
          <ChevronDown
            className={`w-[16px] h-[16px] text-[#727873] transition-transform duration-200 ${expanded ? "rotate-180" : ""}`}
          />
        </div>
      </button>

      {expanded && (
        <div className="border-t border-[#EEEEEA] px-4 py-3 bg-[#FCFDFB]">
          <div className="sm:hidden flex items-center gap-3 font-sans text-[12px] text-[#727873] tabular-nums mb-2">
            <span>Retrieval {fmtMs(entry.retrieval_ms)}</span>
            <span>LLM {fmtMs(entry.generation_ms)}</span>
            <span>End-to-end {fmtMs(entry.end_to_end_ms)}</span>
          </div>
          {entry.stages.length > 0 ? (
            <div className="flex flex-col">
              <span className="font-sans text-[11px] font-medium text-[#176B4F] uppercase tracking-wide px-3 mb-1">
                Stage breakdown
              </span>
              {entry.stages.map((s, i) => <StageRow key={`${s.stage}-${i}`} stage={s} />)}

              {/* Totals -- the same numbers already shown inline on the
                  collapsed row (R / L / E2E), repeated here as an explicit
                  sum under the sub-stages they add up from, since a reader
                  expanding the breakdown wants the total right where the
                  parts are, not just up in the header. */}
              <div className="flex flex-col mt-1.5 pt-1.5 border-t border-[#E6EAE3]">
                <div className="flex items-center justify-between py-1 px-3">
                  <span className="font-sans text-[12.5px] font-medium text-[#3E453F]">Total retrieval</span>
                  <span className="font-sans text-[12.5px] font-semibold text-[#176B4F] tabular-nums">
                    {fmtMs(entry.retrieval_ms)}
                  </span>
                </div>
                <div className="flex items-center justify-between py-1 px-3">
                  <span className="font-sans text-[12.5px] font-medium text-[#3E453F]">LLM generation</span>
                  <span className="font-sans text-[12.5px] font-semibold text-[#176B4F] tabular-nums">
                    {fmtMs(entry.generation_ms)}
                  </span>
                </div>
                <div className="flex items-center justify-between py-1 px-3">
                  <span className="font-sans text-[12.5px] font-semibold text-[#151A17]">Total end-to-end</span>
                  <span className="font-sans text-[13px] font-bold text-[#176B4F] tabular-nums">
                    {fmtMs(entry.end_to_end_ms)}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <p className="font-sans text-[12.5px] text-[#9AA39D] px-3">No stage data recorded.</p>
          )}
          {entry.grounding_score != null && (
            <p className="font-sans text-[12px] text-[#727873] px-3 mt-2">
              Grounding overlap: {(entry.grounding_score * 100).toFixed(0)}%
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default function InsightsSection() {
  const [data, setData] = useState<InsightsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setError(null);
    try {
      const res = await fetch(`${resolveApiBaseUrl()}/api/insights?limit=200`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: InsightsResponse = await res.json();
      setData(json);
    } catch (e: any) {
      setError(e?.message || "Could not load insights.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const stats = data?.stats;
  const entries = data?.entries || [];

  return (
    <div
      id="insights-section"
      className="flex-1 w-full h-full overflow-y-auto flex flex-col items-center px-6 py-8"
    >
      <div className="w-full max-w-[820px] flex flex-col gap-6">

        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1">
            <span className="font-sans text-[13px] font-semibold text-[#176B4F] tracking-[0.12em] uppercase">
              Insights
            </span>
            <h1 className="font-serif text-[28px] sm:text-[34px] font-semibold text-[#151A17] leading-[1.05] tracking-tight">
              Every question, every answer, every millisecond
            </h1>
            <p className="font-sans text-[14.5px] text-[#686D69] leading-relaxed mt-1 max-w-[560px]">
              A live, local record of what this deployment has been asked and how it answered —
              retrieval, generation and end-to-end latency for each request. Expand a row for the
              full per-stage breakdown.
            </p>
          </div>
          <button
            onClick={load}
            disabled={loading}
            className="shrink-0 w-[36px] h-[36px] rounded-full flex items-center justify-center border border-[#DADDD7] text-[#59635D] hover:bg-[#F3F2ED] transition-colors duration-150 cursor-pointer outline-none disabled:opacity-50"
            aria-label="Refresh"
          >
            <RefreshCw className={`w-[15px] h-[15px] ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>

        {stats && (
          <div className="flex flex-wrap gap-2.5">
            <StatTile label="Total queries" value={String(stats.total_queries)} />
            <StatTile label="Refused" value={String(stats.total_refused)} />
            <StatTile label="Avg retrieval" value={fmtMs(stats.avg_retrieval_ms)} />
            <StatTile label="Avg generation" value={fmtMs(stats.avg_generation_ms)} />
            <StatTile label="Avg end-to-end" value={fmtMs(stats.avg_end_to_end_ms)} />
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 rounded-[10px] border border-[#F0D9A8] bg-[#FDF7EA] px-3.5 py-2.5">
            <AlertCircle className="w-[15px] h-[15px] text-[#B8801F] shrink-0" />
            <span className="font-sans text-[13px] text-[#8A5F12]">{error}</span>
          </div>
        )}

        {!loading && !error && entries.length === 0 && (
          <p className="font-sans text-[14px] text-[#9AA39D] text-center py-10">
            No questions have been asked yet on this deployment.
          </p>
        )}

        <div className="flex flex-col gap-2.5 pb-6">
          {entries.map((e) => <EntryRow key={e.id} entry={e} />)}
        </div>
      </div>
    </div>
  );
}
