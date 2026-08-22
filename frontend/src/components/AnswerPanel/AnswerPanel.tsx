/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from "react";
import { Timer, Target, ShieldCheck, AlertTriangle, BarChart2 } from "lucide-react";
import { useRag } from "../../context/RagContext";

/**
 * SourceRowProps - Definition of row data for sources.
 */
interface Source {
  id: string;
  passage: string;
  score: string;
}

/**
 * SourceRow Component - Displays an individual source passage line with a subtle hover effect.
 */
function SourceRow({ source }: { source: Source }) {
  return (
    <div className="flex items-center justify-between h-[44px] px-2 rounded-[8px] hover:bg-[#F7F8F4] transition-all duration-150 group cursor-pointer">
      <div className="flex items-center gap-2">
        {/* Solid rounded square badge (28px x 28px) */}
        <div className="w-[28px] h-[28px] rounded-[6px] bg-[#176B4F] flex items-center justify-center text-white shrink-0">
          <span className="font-sans font-semibold text-[12.5px] leading-none">
            {source.id}
          </span>
        </div>
        {/* Passage Title */}
        <span
          title={source.passage}
          className="font-sans font-normal text-[14px] text-[#252B27] group-hover:text-[#176B4F] transition-colors duration-150 truncate max-w-[180px]"
        >
          {source.passage}
        </span>
      </div>
      {/* Passage Relevance Score */}
      <span className="font-sans font-normal text-[11.5px] text-[#727873] pr-1">
        Score: {source.score}
      </span>
    </div>
  );
}

/**
 * Right panel answer & evidence visualization element.
 */
export default function AnswerPanel({ onOpenInsights }: { onOpenInsights?: () => void }) {
  const {
    query, answer, sources, metrics, groundingWarning, isProcessing,
    pendingTranscript, awaitingSend, isListening, sendPending, discardPending,
    statusStage
  } = useRag();

  // Tokens are appended one delta at a time as they arrive from the provider,
  // but a short answer can complete in a few hundred ms, which reads as a
  // single flash rather than as generation. The caret marks the stream as
  // still open -- it is driven by the real pipeline stage, so it disappears
  // the moment the last token lands. Nothing here delays or paces the text.
  const isStreaming = statusStage === "generating" && Boolean(answer);

  // No placeholder content anywhere below: a fresh page, or a query with no
  // sources/answer yet, shows an honest empty/loading state rather than
  // invented numbers or passages. Real measurements only ever come from an
  // actual response -- previously this component defaulted to a fabricated
  // "renewable energy" answer, three fake passage IDs, "87ms", and "96%"
  // whenever the real values were empty, which is exactly backwards: it
  // looked like a working demo even when nothing had actually run yet, and
  // silently kept showing fake data if a real request came back empty.
  const hasAnswered = sources.length > 0 || Boolean(answer) || Boolean(query);

  const displaySources: Source[] = sources.slice(0, 3).map((s, idx) => {
    const text = (s.child_text || s.parent_text || "").trim();
    const snippet = text.length > 60 ? `${text.slice(0, 60)}…` : text;
    return {
      id: String(idx + 1).padStart(2, "0"),
      passage: snippet || `Passage #${s.parent_id || idx + 1}`,
      score: typeof s.score === "number" ? s.score.toFixed(2) : "—",
    };
  });

  const displayQuery = query || null;
  const displayAnswer = answer || null;

  const retrievalTime = metrics?.retrieval_latency_ms != null
    ? `${Math.round(metrics.retrieval_latency_ms)}ms` : "—";
  // Only ever the real grounding score from the guardrail, never a value
  // derived from latency or anything else standing in for it.
  const groundedPercentage = metrics?.grounding_score != null
    ? `${(metrics.grounding_score * 100).toFixed(0)}%` : "—";
  const passagesCount = sources.length > 0 ? String(sources.length) : "—";

  return (
    <div
      id="right-answer-panel"
      className="w-[calc(100%-32px)] sm:w-full max-w-[380px] lg:w-[380px] h-auto shrink-0 bg-[#FFFFFF] border border-[#EAE8E2] rounded-[20px] p-4 sm:p-5 flex flex-col justify-start z-10 select-none font-sans self-center lg:self-start mt-[16px] lg:mt-[16px] mb-[16px] mx-auto lg:mx-0 lg:mr-[32px] lg:ml-[12px]"
      style={{
        boxShadow: "0 8px 28px rgba(35, 54, 44, 0.055)"
      }}
    >
      <style>{`
        @keyframes stream-caret { 0%, 45% { opacity: 1; } 55%, 100% { opacity: 0; } }
        .animate-stream-caret { animation: stream-caret 1s steps(1, end) infinite; }
        @media (prefers-reduced-motion: reduce) {
          .animate-stream-caret { animation: none; opacity: 1; }
        }
      `}</style>

      {/* UPPER SECTION: Question & Generated Answer */}
      <div className="flex flex-col">
        {/* Row 1: Label and Decorative Wave indicator */}
        <div className="flex items-center justify-between">
          <span className="font-sans font-semibold text-[15px] text-[#176B4F]">
            You asked
          </span>
          {/* Decorative small audio waveform icon (24px x 24px) */}
          <div
            className="w-[24px] h-[24px] flex items-center justify-center gap-[2.5px] opacity-90"
            aria-hidden="true"
          >
            <div className="w-[2px] h-[10px] bg-[#4D8F70] rounded-full" />
            <div className="w-[2px] h-[18px] bg-[#4D8F70] rounded-full animate-pulse" />
            <div className="w-[2px] h-[14px] bg-[#4D8F70] rounded-full" />
            <div className="w-[2px] h-[22px] bg-[#4D8F70] rounded-full animate-pulse" style={{ animationDelay: '0.2s' }} />
            <div className="w-[2px] h-[12px] bg-[#4D8F70] rounded-full" />
            <div className="w-[2px] h-[7px] bg-[#4D8F70] rounded-full" />
          </div>
        </div>

        {/* Row 2: Question text -- OR, while a spoken question is being
            reviewed, the live/pending transcript with Send + Retry. This
            used to live in AskHero above the Best Match dropdown, where its
            appearing/disappearing height pushed that dropdown up and down.
            It belongs next to "You asked" anyway, so it moved here instead. */}
        {(pendingTranscript || awaitingSend) ? (
          <div className="mt-[14px]">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="font-sans text-[11px] font-medium text-[#176B4F] uppercase tracking-wide">
                {isListening ? "Hearing" : "You said"}
              </span>
              {isListening && (
                <span className="w-[6px] h-[6px] rounded-full bg-[#E55353] animate-pulse" />
              )}
            </div>
            <p className="font-sans text-[14.5px] text-[#1B211E] leading-[1.5] bg-[#F7F9F6] border border-[#E4EAE2] rounded-[10px] px-3 py-2 min-h-[40px] break-words">
              {pendingTranscript || (
                <span className="text-[#9AA39D]">सुन रहे हैं…</span>
              )}
            </p>

            {awaitingSend && (
              <div className="flex items-center gap-2 mt-2">
                <button
                  onClick={sendPending}
                  className="flex-1 h-[36px] rounded-[10px] bg-[#176B4F] text-white font-sans text-[13.5px] font-medium flex items-center justify-center gap-1.5 cursor-pointer hover:bg-[#14694F] active:scale-[0.99] transition-all duration-150 outline-none"
                >
                  Send
                  <span className="text-[14px]">→</span>
                </button>
                <button
                  onClick={discardPending}
                  className="h-[36px] px-3 rounded-[10px] border border-[#DADDD7] text-[#59635D] font-sans text-[12.5px] cursor-pointer hover:bg-[#F3F2ED] transition-colors duration-150 outline-none"
                >
                  Retry
                </button>
              </div>
            )}
          </div>
        ) : displayQuery ? (
          <p className="font-sans font-normal text-[14.5px] text-[#1B211E] leading-[1.5] mt-[14px] pr-2">
            “{displayQuery}”
          </p>
        ) : (
          <p className="font-sans font-normal text-[14.5px] text-[#9AA39D] italic leading-[1.5] mt-[14px] pr-2">
            Ask a question to get started.
          </p>
        )}

        {/* First Divider */}
        <hr className="border-0 border-t border-[#E6E6E1] mt-[18px]" />

        {/* Row 3: Answer heading */}
        <span className="font-sans font-semibold text-[15px] text-[#176B4F] mt-[18px]">
          Answer
        </span>

        {/* Row 4: Generated Answer text paragraph.
            Answers stream in live and are validated afterwards, so this text
            can already be on screen when the grounding gate rejects it. It is
            dimmed rather than removed -- silently deleting text the user has
            already read reads as a bug, while the banner below states plainly
            that the system caught it. */}
        <p
          className={`font-sans leading-[1.5] mt-[12px] text-justify transition-opacity duration-300 ${
            displayAnswer
              ? `font-normal text-[14.5px] ${groundingWarning ? "text-[#6B716D] opacity-70" : "text-[#202522]"}`
              : "font-normal text-[14px] text-[#9AA39D] italic"
          }`}
        >
          {displayAnswer
            ? displayAnswer
            : isProcessing
              ? "Generating an answer…"
              : "No answer yet — ask a question to see one here."}
          {isStreaming && (
            <span
              aria-hidden="true"
              className="inline-block w-[2px] h-[1em] ml-[2px] align-text-bottom bg-[#176B4F] animate-stream-caret"
            />
          )}
        </p>

        {groundingWarning && (
          <div className="mt-[12px] flex items-start gap-2 rounded-[10px] border border-[#F0D9A8] bg-[#FDF7EA] px-3 py-2.5">
            <AlertTriangle className="w-[15px] h-[15px] stroke-[2] text-[#B8801F] shrink-0 mt-[2px]" />
            <div className="flex flex-col">
              <span className="font-sans font-medium text-[13px] text-[#8A5F12] leading-[1.4]">
                {groundingWarning.message}
              </span>
              {groundingWarning.score != null && (
                <span className="font-sans font-normal text-[11.5px] text-[#A07B31] mt-[3px]">
                  Grounding overlap {(groundingWarning.score * 100).toFixed(0)}%
                  {groundingWarning.detail ? ` — ${groundingWarning.detail}` : ""}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Second Divider */}
        <hr className="border-0 border-t border-[#E6E6E1] mt-[18px]" />
      </div>

      {/* LOWER SECTION: Sources & Metrics Card */}
      <div className="flex flex-col mt-5 pt-2">
        {/* Section Heading: Sources (Top 3) */}
        <div className="flex items-baseline gap-1.5 mb-[10px]">
          <span className="font-sans font-semibold text-[15px] text-[#176B4F]">
            Sources
          </span>
          <span className="font-sans font-normal text-[14px] text-[#8A928D]">
            (Top 3)
          </span>
        </div>

        {/* Map Source Rows with light dividers in between, or an honest
            empty state -- no fabricated passage IDs when nothing has been
            retrieved yet. */}
        {displaySources.length > 0 ? (
          <div className="flex flex-col gap-1">
            {displaySources.map((source, index) => (
              <React.Fragment key={source.id}>
                <SourceRow source={source} />
                {index < displaySources.length - 1 && (
                  <hr className="border-0 border-t border-[#EEEEEA]" />
                )}
              </React.Fragment>
            ))}
          </div>
        ) : (
          <p className="font-sans text-[13px] text-[#9AA39D] italic py-2">
            No sources yet — ask a question first.
          </p>
        )}

        {/* Retrieval/Grounding statistics metrics card (74px height) */}
        <div className="w-full h-[74px] bg-[#EEF5ED] rounded-[12px] flex items-center justify-between px-3 mt-[16px]">
          
          {/* Column 1: Retrieval Time */}
          <div className="flex-1 flex flex-col items-center justify-center gap-0.5">
            <div className="flex items-center gap-1 text-[#176B4F]">
              <Timer className="w-[14px] h-[14px] stroke-[1.8]" />
              <span className="font-sans font-medium text-[15.5px] leading-none">
                {retrievalTime}
              </span>
            </div>
            <span className="font-sans font-normal text-[10.5px] text-[#303A34] tracking-wide uppercase">
              Retrieval Time
            </span>
          </div>

          {/* Vertical Separator */}
          <div className="h-[30px] w-[1px] bg-[#C9D8CB] opacity-70" />

          {/* Column 2: Grounded */}
          <div className="flex-1 flex flex-col items-center justify-center gap-0.5">
            <div className="flex items-center gap-1 text-[#176B4F]">
              <Target className="w-[14px] h-[14px] stroke-[1.8]" />
              <span className="font-sans font-medium text-[15.5px] leading-none">
                {groundedPercentage}
              </span>
            </div>
            <span className="font-sans font-normal text-[10.5px] text-[#303A34] tracking-wide uppercase">
              Grounded
            </span>
          </div>

          {/* Vertical Separator */}
          <div className="h-[30px] w-[1px] bg-[#C9D8CB] opacity-70" />

          {/* Column 3: Passages */}
          <div className="flex-1 flex flex-col items-center justify-center gap-0.5">
            <div className="flex items-center gap-1 text-[#176B4F]">
              <ShieldCheck className="w-[14px] h-[14px] stroke-[1.8]" />
              <span className="font-sans font-medium text-[15.5px] leading-none">
                {passagesCount}
              </span>
            </div>
            <span className="font-sans font-normal text-[10px] text-[#303A34] tracking-wide uppercase">
              Passages
            </span>
          </div>

        </div>

        {/* Links to the full latency/query log in the sidebar's Insights
            section -- the same numbers above (retrieval/grounded/passages),
            but every query this deployment has ever answered, with the
            per-stage breakdown expandable per row. */}
        <button
          onClick={onOpenInsights}
          className="w-full h-[38px] rounded-[10px] bg-[#EEF5ED] border border-[#D7E1D9] flex items-center justify-center gap-2 mt-[10px] cursor-pointer hover:bg-[#E4EFE1] transition-colors duration-150 select-none outline-none group"
        >
          <BarChart2 className="w-[14px] h-[14px] stroke-[1.8] text-[#176B4F]" />
          <span className="font-sans font-medium text-[13px] text-[#176B4F]">
            Check Insights
          </span>
          <span className="font-sans text-[13px] text-[#176B4F] transition-transform duration-150 group-hover:translate-x-1">
            →
          </span>
        </button>
      </div>
    </div>
  );
}
