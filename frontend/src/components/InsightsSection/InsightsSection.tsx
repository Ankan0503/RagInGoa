/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useState, useCallback } from "react";
import { ChevronDown, RefreshCw, AlertCircle, Lock, Unlock, Trash2 } from "lucide-react";
import { resolveApiBaseUrl, readLocalLog, clearLocalLog, LocalLogEntry } from "../../context/RagContext";

interface StageTiming {
  stage: string;
  ms: number;
  in_budget: boolean;
  detail?: string;
}

interface QueryLogEntry {
  id: string | number;
  ts: string;
  query: string;
  transcript: string | null;
  answer: string;
  refused: boolean;
  refusal_reason?: string | null;
  provider: string | null;
  model: string | null;
  retrieval_ms: number | null;
  generation_ms: number | null;
  guardrail_ms: number | null;
  end_to_end_ms: number | null;
  grounding_score: number | null;
  // Time to first token: when the answer STARTS appearing, which is what a
  // user actually perceives as the wait. Only streamed (voice/WebSocket)
  // requests can measure it, so it is null for REST ones rather than 0.
  ttft_ms?: number | null;
  stages: StageTiming[];
}

interface SharedStats {
  total_queries: number;
  total_refused: number;
  avg_retrieval_ms: number | null;
  avg_generation_ms: number | null;
  avg_end_to_end_ms: number | null;
  avg_ttft_ms?: number | null;
}

interface InsightsResponse {
  entries: QueryLogEntry[];
  stats: SharedStats;
  admin: boolean;
}

const ADMIN_TOKEN_KEY = "rag_admin_token";
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
    return new Date(iso).toLocaleString(undefined, {
      month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  } catch {
    return iso;
  }
}

function StatTile({ label, value }: { label: string; value: string; key?: React.Key }) {
  return (
    <div className="flex-1 min-w-[110px] rounded-[12px] bg-[#F7F9F6] border border-[#E6EAE3] px-3.5 py-2.5 flex flex-col gap-0.5">
      <span className="font-sans text-[17px] font-semibold text-[#176B4F] leading-none">
        {value}
      </span>
      <span className="font-sans text-[10.5px] text-[#727873] uppercase tracking-wide mt-1">
        {label}
      </span>
    </div>
  );
}

function StageRow({ stage }: { stage: StageTiming; key?: React.Key }) {
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

function EntryRow({ entry }: { entry: QueryLogEntry; key?: React.Key }) {
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
            {entry.ttft_ms != null && (
              <>
                <span title="Time to first token — when the answer starts appearing">
                  TTFT {fmtMs(entry.ttft_ms)}
                </span>
                <span>·</span>
              </>
            )}
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

              <div className="flex flex-col mt-1.5 pt-1.5 border-t border-[#E6EAE3]">
                <div className="flex items-center justify-between py-1 px-3">
                  <span className="font-sans text-[12.5px] font-medium text-[#3E453F]">Total retrieval</span>
                  <span className="font-sans text-[12.5px] font-semibold text-[#176B4F] tabular-nums">
                    {fmtMs(entry.retrieval_ms)}
                  </span>
                </div>
                {entry.ttft_ms != null && (
                  <div className="flex items-center justify-between py-1 px-3">
                    <span className="font-sans text-[12.5px] font-medium text-[#3E453F]">
                      Time to first token
                      <span className="text-[11px] text-[#9AA39D] font-normal ml-1.5">
                        answer starts appearing
                      </span>
                    </span>
                    <span className="font-sans text-[12.5px] font-semibold text-[#176B4F] tabular-nums">
                      {fmtMs(entry.ttft_ms)}
                    </span>
                  </div>
                )}
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

function localToEntry(l: LocalLogEntry): QueryLogEntry {
  return {
    id: l.id, ts: l.ts, query: l.query, transcript: l.transcript,
    answer: l.answer, refused: l.refused, provider: l.provider, model: l.model,
    retrieval_ms: l.retrieval_ms, generation_ms: l.generation_ms,
    guardrail_ms: l.guardrail_ms, end_to_end_ms: l.end_to_end_ms,
    grounding_score: l.grounding_score, ttft_ms: l.ttft_ms, stages: l.stages,
  };
}

export default function InsightsSection() {
  // "My history": this browser's own questions, from localStorage. No token,
  // no server round-trip, visible to nobody else -- see RagContext.tsx for
  // why this is a separate store from the shared server-side log below.
  const [localEntries, setLocalEntries] = useState<QueryLogEntry[]>([]);

  const loadLocal = useCallback(() => {
    setLocalEntries(readLocalLog().map(localToEntry));
  }, []);

  useEffect(() => { loadLocal(); }, [loadLocal]);

  const handleClearLocal = () => {
    if (!window.confirm("Clear your local question history? This only affects this browser and cannot be undone.")) return;
    clearLocalLog();
    loadLocal();
  };

  // "Shared deployment log": every visitor's real queries, server-side.
  // Aggregate `stats` is public; the actual `entries` only come back once a
  // valid admin token is supplied, per the backend's /api/insights contract.
  const [adminToken, setAdminTokenState] = useState<string>(
    () => { try { return localStorage.getItem(ADMIN_TOKEN_KEY) || ""; } catch { return ""; } }
  );
  const [tokenInput, setTokenInput] = useState("");
  const [shared, setShared] = useState<InsightsResponse | null>(null);
  const [sharedError, setSharedError] = useState<string | null>(null);
  const [loadingShared, setLoadingShared] = useState(true);
  const [clearing, setClearing] = useState(false);

  const setAdminToken = (token: string) => {
    setAdminTokenState(token);
    try {
      if (token) localStorage.setItem(ADMIN_TOKEN_KEY, token);
      else localStorage.removeItem(ADMIN_TOKEN_KEY);
    } catch { /* ignore */ }
  };

  const loadShared = useCallback(async (token: string) => {
    setSharedError(null);
    setLoadingShared(true);
    try {
      const res = await fetch(`${resolveApiBaseUrl()}/api/insights?limit=200`, {
        headers: token ? { "X-Admin-Token": token } : {},
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json: InsightsResponse = await res.json();
      setShared(json);
    } catch (e: any) {
      setSharedError(e?.message || "Could not load the shared log.");
    } finally {
      setLoadingShared(false);
    }
  }, []);

  useEffect(() => { loadShared(adminToken); }, [loadShared, adminToken]);

  const handleUnlock = () => {
    if (!tokenInput.trim()) return;
    setAdminToken(tokenInput.trim());
    setTokenInput("");
  };

  const handleLock = () => {
    setAdminToken("");
  };

  const handleClearShared = async () => {
    if (!adminToken) return;
    if (!window.confirm("Clear the ENTIRE shared query log for every visitor? This cannot be undone.")) return;
    setClearing(true);
    try {
      const res = await fetch(`${resolveApiBaseUrl()}/api/insights`, {
        method: "DELETE",
        headers: { "X-Admin-Token": adminToken },
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await loadShared(adminToken);
    } catch (e: any) {
      setSharedError(e?.message || "Could not clear the shared log.");
    } finally {
      setClearing(false);
    }
  };

  const isAdmin = Boolean(shared?.admin);
  const stats = shared?.stats;
  const sharedEntries = shared?.entries || [];

  return (
    <div
      id="insights-section"
      className="flex-1 w-full h-full overflow-y-auto flex flex-col items-center px-6 py-8"
    >
      <div className="w-full max-w-[820px] flex flex-col gap-8">

        <div className="flex flex-col gap-1">
          <span className="font-sans text-[13px] font-semibold text-[#176B4F] tracking-[0.12em] uppercase">
            Insights
          </span>
          <h1 className="font-serif text-[28px] sm:text-[34px] font-semibold text-[#151A17] leading-[1.05] tracking-tight">
            Every question, every answer, every millisecond
          </h1>
          <p className="font-sans text-[14.5px] text-[#686D69] leading-relaxed mt-1 max-w-[600px]">
            Retrieval, generation and end-to-end latency for each request. Expand a row for the
            full per-stage breakdown.
          </p>
        </div>

        {/* Public, aggregate performance -- no privacy risk, visible to anyone */}
        {stats && (
          <div className="flex flex-wrap gap-2.5">
            <StatTile label="Total queries" value={String(stats.total_queries)} />
            <StatTile label="Refused" value={String(stats.total_refused)} />
            <StatTile label="Avg retrieval" value={fmtMs(stats.avg_retrieval_ms)} />
            <StatTile label="Avg first token" value={fmtMs(stats.avg_ttft_ms ?? null)} />
            <StatTile label="Avg generation" value={fmtMs(stats.avg_generation_ms)} />
            <StatTile label="Avg end-to-end" value={fmtMs(stats.avg_end_to_end_ms)} />
          </div>
        )}

        {/* ---- My history (local, this browser only) ---- */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <h2 className="font-sans font-semibold text-[16px] text-[#176B4F]">
              My history
              <span className="font-sans font-normal text-[13px] text-[#9AA39D] ml-2">
                stored in this browser only, visible to nobody else
              </span>
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={loadLocal}
                className="w-[30px] h-[30px] rounded-full flex items-center justify-center border border-[#DADDD7] text-[#59635D] hover:bg-[#F3F2ED] transition-colors duration-150 cursor-pointer outline-none"
                aria-label="Refresh my history"
              >
                <RefreshCw className="w-[13px] h-[13px]" />
              </button>
              {localEntries.length > 0 && (
                <button
                  onClick={handleClearLocal}
                  className="h-[30px] px-3 rounded-[8px] flex items-center gap-1.5 border border-[#F0D0C8] text-[#B8480F] hover:bg-[#FCEEE6] transition-colors duration-150 cursor-pointer outline-none"
                >
                  <Trash2 className="w-[13px] h-[13px]" />
                  <span className="font-sans text-[12.5px] font-medium">Clear my history</span>
                </button>
              )}
            </div>
          </div>

          {localEntries.length === 0 ? (
            <p className="font-sans text-[13.5px] text-[#9AA39D] italic py-2">
              You haven't asked anything in this browser yet.
            </p>
          ) : (
            <div className="flex flex-col gap-2.5">
              {localEntries.map((e) => <EntryRow key={e.id} entry={e} />)}
            </div>
          )}
        </div>

        {/* ---- Shared deployment log (server-side, admin-gated) ---- */}
        <div className="flex flex-col gap-3 pb-6">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h2 className="font-sans font-semibold text-[16px] text-[#176B4F]">
              Shared deployment log
              <span className="font-sans font-normal text-[13px] text-[#9AA39D] ml-2">
                every visitor's real questions — organizer access only
              </span>
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={() => loadShared(adminToken)}
                disabled={loadingShared}
                className="w-[30px] h-[30px] rounded-full flex items-center justify-center border border-[#DADDD7] text-[#59635D] hover:bg-[#F3F2ED] transition-colors duration-150 cursor-pointer outline-none disabled:opacity-50"
                aria-label="Refresh shared log"
              >
                <RefreshCw className={`w-[13px] h-[13px] ${loadingShared ? "animate-spin" : ""}`} />
              </button>
              {isAdmin && (
                <>
                  <button
                    onClick={handleClearShared}
                    disabled={clearing}
                    className="h-[30px] px-3 rounded-[8px] flex items-center gap-1.5 border border-[#F0D0C8] text-[#B8480F] hover:bg-[#FCEEE6] transition-colors duration-150 cursor-pointer outline-none disabled:opacity-50"
                  >
                    <Trash2 className="w-[13px] h-[13px]" />
                    <span className="font-sans text-[12.5px] font-medium">Clear all</span>
                  </button>
                  <button
                    onClick={handleLock}
                    className="h-[30px] px-3 rounded-[8px] flex items-center gap-1.5 border border-[#DADDD7] text-[#59635D] hover:bg-[#F3F2ED] transition-colors duration-150 cursor-pointer outline-none"
                  >
                    <Lock className="w-[13px] h-[13px]" />
                    <span className="font-sans text-[12.5px] font-medium">Lock</span>
                  </button>
                </>
              )}
            </div>
          </div>

          {sharedError && (
            <div className="flex items-center gap-2 rounded-[10px] border border-[#F0D9A8] bg-[#FDF7EA] px-3.5 py-2.5">
              <AlertCircle className="w-[15px] h-[15px] text-[#B8801F] shrink-0" />
              <span className="font-sans text-[13px] text-[#8A5F12]">{sharedError}</span>
            </div>
          )}

          {!isAdmin && !loadingShared && (
            <div className="flex items-center gap-2 rounded-[12px] border border-[#E6EAE3] bg-[#F7F9F6] px-4 py-3">
              <Lock className="w-[15px] h-[15px] text-[#727873] shrink-0" />
              <input
                type="password"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleUnlock(); }}
                placeholder="Admin token"
                className="flex-1 bg-transparent font-sans text-[13.5px] text-[#151A17] outline-none placeholder:text-[#9AA39D]"
              />
              <button
                onClick={handleUnlock}
                className="flex items-center gap-1.5 h-[30px] px-3 rounded-[8px] bg-[#176B4F] text-white font-sans text-[12.5px] font-medium cursor-pointer hover:bg-[#14694F] transition-colors duration-150 outline-none"
              >
                <Unlock className="w-[13px] h-[13px]" />
                Unlock
              </button>
            </div>
          )}

          {isAdmin && (
            sharedEntries.length === 0 ? (
              <p className="font-sans text-[13.5px] text-[#9AA39D] italic py-2">
                No questions have been asked yet on this deployment.
              </p>
            ) : (
              <div className="flex flex-col gap-2.5">
                {sharedEntries.map((e) => <EntryRow key={e.id} entry={e} />)}
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
}
