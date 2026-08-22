/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Mic, ChevronDown, Loader2, Sparkles } from "lucide-react";
import BottomAskSection from "../BottomAskSection/BottomAskSection";
import { useRag } from "../../context/RagContext";

/**
 * The microphone has a real, unavoidable wait in it: the Sarvam realtime
 * socket has to open before a single word can be captured. The interface has
 * to say so, and the three states below are how it does that.
 *
 * Every colour here is derived from the palette the rest of the page already
 * uses -- the forest green #176B4F and the warm #F2B66C of the headline mark.
 * Stock Tailwind `emerald`/`amber` were doing this job before and they are far
 * more saturated than anything else on the page, so the moment the mic lit up
 * the card stopped looking like it belonged to the same product.
 *
 * Each state also gets ONE animation, not three. Previously "connecting" ran a
 * pulsing banner, a spinning icon, a pulsing halo and fourteen pulsing bars
 * simultaneously; motion that competes with itself reads as busy, not as
 * responsive.
 */
type MicState = "idle" | "connecting" | "listening";

interface MicTheme {
  halo: string;
  ring: string;
  button: string;
  buttonGlow: string;
  cardBorder: string;
  cardShadow: string;
  pill: string;
  pillDot: string;
  barSoft: string;
  barStrong: string;
}

const MIC_THEME: Record<MicState, MicTheme> = {
  idle: {
    halo: "bg-[#F2F6F2]",
    ring: "bg-[#DCEBE2]",
    button: "bg-[#176B4F] hover:bg-[#125B43]",
    buttonGlow: "0 6px 18px rgba(23,107,79,0.22)",
    cardBorder: "border-[#EBE9E2]",
    cardShadow: "0 1px 2px rgba(35,54,44,0.04), 0 14px 34px rgba(35,54,44,0.055)",
    pill: "bg-[#F4F7F3] border-[#DFE7DC] text-[#3F4A43]",
    pillDot: "bg-[#B3C2B7]",
    barSoft: "bg-[#CFDFD5]",
    barStrong: "bg-[#A6C4B3]",
  },
  connecting: {
    halo: "bg-[#FBF3E7]",
    ring: "bg-[#F1DFC0]",
    button: "bg-[#C0863B] hover:bg-[#A87031]",
    buttonGlow: "0 6px 18px rgba(192,134,59,0.26)",
    cardBorder: "border-[#EBD6B2]",
    cardShadow: "0 1px 2px rgba(150,100,30,0.05), 0 14px 34px rgba(150,100,30,0.10)",
    pill: "bg-[#FDF6EC] border-[#EBD6B2] text-[#7C551F]",
    pillDot: "bg-[#D9AE6E]",
    barSoft: "bg-[#EBD3AC]",
    barStrong: "bg-[#D3A05A]",
  },
  listening: {
    halo: "bg-[#E6F4EC]",
    ring: "bg-[#B4DFC9]",
    button: "bg-[#0E7A57] hover:bg-[#0B6748]",
    buttonGlow: "0 8px 24px rgba(14,122,87,0.30)",
    cardBorder: "border-[#9FD6BC]",
    cardShadow: "0 1px 2px rgba(14,122,87,0.06), 0 14px 36px rgba(14,122,87,0.13)",
    pill: "bg-[#E9F6F0] border-[#A6D8C1] text-[#0C5C42]",
    pillDot: "bg-[#16A46F]",
    barSoft: "bg-[#9AD5BA]",
    barStrong: "bg-[#16A46F]",
  },
};

/**
 * AudioWaveform - Horizontal bar visualization flanking the microphone button.
 */
interface AudioWaveformProps {
  side: "left" | "right";
  isAnimating: boolean;
  statusStage: string;
}

export function AudioWaveform({ side, isAnimating, statusStage }: AudioWaveformProps) {
  // Configured heights for left/right side matching reference composition
  const bars =
    side === "left"
      ? [4, 6, 10, 6, 12, 18, 8, 14, 24, 16, 28, 12, 8, 4]
      : [4, 8, 12, 28, 16, 24, 14, 8, 18, 12, 6, 10, 6, 4];

  const state: MicState =
    statusStage === "connecting" ? "connecting"
      : statusStage === "listening" ? "listening"
        : "idle";
  const theme = MIC_THEME[state];

  return (
    <div
      className={`w-[38px] sm:w-[90px] lg:w-[145px] h-[45px] flex items-center ${side === "left" ? "justify-end" : "justify-start"
        } pointer-events-none`}
      aria-hidden="true"
    >
      <div className="flex items-center gap-[4px] h-[36px]">
        {bars.map((height, idx) => {
          // While connecting, a single highlight travels outward from the
          // microphone on both sides. It is indeterminate on purpose -- the
          // socket handshake has no progress to report -- but a travelling
          // pulse reads as "working on it" where a uniform blink reads as a
          // stuck element.
          const outward = side === "left" ? bars.length - 1 - idx : idx;
          const style: React.CSSProperties = { height: `${height}px` };

          if (state === "connecting") {
            style.animation = "wave-scan 1.5s ease-in-out infinite";
            style.animationDelay = `${outward * 0.075}s`;
          } else if (state === "listening") {
            style.animation = "wave-live 0.75s ease-in-out infinite alternate";
            style.animationDelay = `${idx * 0.05}s`;
          } else if (isAnimating) {
            style.animation = "wave-idle 1.9s ease-in-out infinite alternate";
            style.animationDelay = `${idx * 0.09}s`;
          } else {
            style.transition = "height 0.3s ease";
          }

          return (
            <div
              key={idx}
              className={`w-[3px] rounded-full transition-colors duration-500 ${height > 14 ? theme.barStrong : theme.barSoft
                }`}
              style={style}
            />
          );
        })}
      </div>
    </div>
  );
}

/**
 * StatusPill - one row, one voice. The Hindi line leads because the product is
 * Hindi-first; the English follows it as a quieter gloss rather than as a
 * parenthetical, which is what made these read like debug strings.
 */
function StatusPill({
  theme, icon, hi, en,
}: { theme: MicTheme; icon: React.ReactNode; hi: string; en: string }) {
  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border pl-2.5 pr-3.5 py-[5px] transition-colors duration-500 animate-rise ${theme.pill}`}
    >
      <span className="shrink-0 flex items-center">{icon}</span>
      <span className="font-sans text-[13px] font-medium leading-none whitespace-nowrap">{hi}</span>
      <span className={`h-[3px] w-[3px] rounded-full shrink-0 opacity-60 ${theme.pillDot}`} />
      <span className="font-sans text-[12px] font-normal leading-none opacity-65 whitespace-nowrap">
        {en}
      </span>
    </div>
  );
}

/**
 * AskHero - Main center container of the Voice-powered RAG interface.
 */
export default function AskHero() {
  const {
    isListening,
    isProcessing,
    startListening,
    stopListening,
    selectedStrategy,
    setSelectedStrategy,
    awaitingSend,
    statusStage
  } = useRag();

  const [showDropdown, setShowDropdown] = useState(false);

  const isConnecting = isListening && statusStage === "connecting";
  const isLiveListening = isListening && statusStage === "listening";

  const micState: MicState = isConnecting ? "connecting"
    : isLiveListening ? "listening"
      : "idle";
  const theme = MIC_THEME[micState];

  const handleMicClick = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <div
      id="ask-hero-root"
      className="relative flex-1 w-full min-h-full bg-transparent flex flex-col items-center justify-start pt-[16px] pb-[20px] px-6 select-none"
    >
      {/* Motion is deliberately small in amplitude and slow in tempo -- the
          premium read comes from restraint, and from every state owning one
          animation instead of stacking three. All of it is disabled outright
          under prefers-reduced-motion. */}
      <style>{`
        @keyframes wave-idle {
          0%   { transform: scaleY(0.74); }
          100% { transform: scaleY(1.06); }
        }
        @keyframes wave-live {
          0%   { transform: scaleY(0.44); }
          100% { transform: scaleY(1.30); }
        }
        @keyframes wave-scan {
          0%, 68%, 100% { transform: scaleY(0.52); opacity: 0.4; }
          32%           { transform: scaleY(1.12); opacity: 1; }
        }
        @keyframes mic-breathe {
          0%, 100% { transform: scale(1);     opacity: 0.72; }
          50%      { transform: scale(1.045); opacity: 1; }
        }
        @keyframes mic-halo-wait {
          0%, 100% { opacity: 0.55; }
          50%      { opacity: 0.9; }
        }
        .animate-mic-breathe { animation: mic-breathe 3.2s ease-in-out infinite; }
        .animate-mic-wait    { animation: mic-halo-wait 1.6s ease-in-out infinite; }
        @keyframes rise {
          from { opacity: 0; transform: translateY(-3px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .animate-rise { animation: rise 0.26s cubic-bezier(0.16, 1, 0.3, 1) both; }
        .animate-fade-in {
          animation: fadeIn 0.22s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @media (prefers-reduced-motion: reduce) {
          #ask-hero-root *,
          #ask-hero-root *::before,
          #ask-hero-root *::after {
            animation: none !important;
            transition-duration: 0.01ms !important;
          }
        }
      `}</style>

      {/* Center main visual content container */}
      <div className="w-full max-w-[660px] z-10 flex flex-col items-center">

        {/* Headline block */}
        <div className="relative text-center w-full">
          <h1 className="font-serif font-semibold text-[32px] sm:text-[40px] md:text-[48px] lg:text-[52px] text-[#151917] leading-[1.02] tracking-[-0.8px] relative inline-block select-none">
            <span className="relative inline-block">
              Ask anything.
              {/* Elegant orange annotated rays mark */}
              <div
                className="absolute top-[-14px] right-[-32px] w-[30px] h-[30px] pointer-events-none"
                aria-hidden="true"
              >
                <svg
                  viewBox="0 0 24 24"
                  className="w-full h-full stroke-[#F2B66C]"
                  fill="none"
                  strokeLinecap="round"
                >
                  <path d="M 5 19 C 7 14, 9 9, 13 5" strokeWidth="2.8" />
                  <path d="M 6 20 C 10 17, 14 14, 20 11" strokeWidth="3" />
                  <path d="M 7 21 C 12 21, 17 21, 22 20.5" strokeWidth="2.8" />
                </svg>
              </div>
            </span>
            <br />
            Get <span className="text-[#176B4F]">grounded</span> answers.
          </h1>

          {/* Curved underline SVG */}
          <div className="flex justify-center mt-3 h-[20px]">
            <svg
              viewBox="0 0 300 20"
              className="w-[170px] sm:w-[220px] md:w-[250px] lg:w-[280px] fill-none stroke-[#C7DCD2] opacity-75"
              strokeWidth="3.5"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M 15 10 C 50 1, 75 19, 110 10 C 145 1, 170 19, 205 10 C 240 1, 265 19, 285 10" />
            </svg>
          </div>
        </div>

        {/* Subtitle - Inter, 18px */}
        <p className="font-sans text-[18px] text-[#686D69] font-normal tracking-[0px] leading-relaxed text-center mt-[8px]">
          Voice-powered RAG on MSMARCO-XI
        </p>

        {/* Voice Interaction Card */}
        <div
          className={`w-full max-w-[640px] h-auto min-h-[290px] lg:h-[345px] bg-[#FFFFFF] rounded-[20px] border flex flex-col items-center justify-between py-[22px] lg:py-[28px] px-4 sm:px-6 mt-[12px] z-20 ${theme.cardBorder}`}
          style={{
            boxShadow: theme.cardShadow,
            transition: "border-color 500ms ease, box-shadow 500ms ease",
          }}
        >
          {/* Card Top Title & Status Banner */}
          <div className="flex flex-col items-center gap-1.5 text-center select-none w-full min-h-[46px] justify-center">
            {isConnecting ? (
              <StatusPill
                theme={theme}
                icon={<Loader2 className="w-[13px] h-[13px] animate-spin text-[#B7822F]" />}
                hi="वॉइस इंजन कनेक्ट हो रहा है"
                en="Connecting"
              />
            ) : isLiveListening ? (
              <StatusPill
                theme={theme}
                icon={
                  <span className="relative flex h-[9px] w-[9px]">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#16A46F] opacity-60" />
                    <span className="relative inline-flex rounded-full h-[9px] w-[9px] bg-[#0E7A57]" />
                  </span>
                }
                hi="अब बोलिए"
                en="Listening"
              />
            ) : awaitingSend ? (
              <StatusPill
                theme={MIC_THEME.idle}
                icon={<Sparkles className="w-[13px] h-[13px] text-[#176B4F]" />}
                hi="प्रश्न देखें और भेजें"
                en="Review & send"
              />
            ) : isProcessing ? (
              <StatusPill
                theme={MIC_THEME.idle}
                icon={<Loader2 className="w-[13px] h-[13px] animate-spin text-[#176B4F]" />}
                hi="उत्तर तैयार किया जा रहा है"
                en="Generating"
              />
            ) : (
              <div className="flex flex-col items-center animate-rise">
                <h2 className="font-sans text-[18px] font-semibold text-[#176B4F] tracking-[-0.1px]">
                  Tap to speak your question
                </h2>
                <span className="text-[12.5px] text-[#868D88] mt-0.5">
                  माइक्रोफ़ोन दबाकर हिंदी में पूछें
                </span>
              </div>
            )}
          </div>

          {/* Microphone and Symmetrical Waveforms Group */}
          <div className="flex items-center justify-center w-full gap-1 sm:gap-3 lg:gap-4 my-auto">

            {/* Left waveform illustration */}
            <AudioWaveform
              side="left"
              isAnimating={isListening || isProcessing}
              statusStage={statusStage}
            />

            {/* Spherically layered microphone trigger */}
            <div className="relative flex items-center justify-center w-[104px] h-[104px] sm:w-[140px] sm:h-[140px] lg:w-[170px] lg:h-[170px] shrink-0">

              {/* Outer halo. Breathes only while actually listening; during the
                  connect wait it fades gently instead, so the spinner in the
                  button stays the one thing carrying the "waiting" message. */}
              <div
                className={`absolute inset-0 rounded-full transition-colors duration-500 ${theme.halo} ${micState === "listening" ? "animate-mic-breathe"
                  : micState === "connecting" ? "animate-mic-wait"
                    : "opacity-70"
                  }`}
              />

              {/* Middle ring */}
              <div
                className={`absolute w-[86px] h-[86px] sm:w-[115px] sm:h-[115px] lg:w-[140px] lg:h-[140px] rounded-full flex items-center justify-center transition-colors duration-500 ${theme.ring}`}
              >

                {/* Inner button */}
                <button
                  onClick={handleMicClick}
                  aria-label={
                    isLiveListening ? "Stop listening"
                      : isConnecting ? "Connecting to the voice engine"
                        : "Speak your question"
                  }
                  aria-busy={isConnecting}
                  className={`w-[64px] h-[64px] sm:w-[85px] sm:h-[85px] lg:w-[102px] lg:h-[102px] rounded-full flex items-center justify-center text-white cursor-pointer transition-all duration-300 hover:scale-[1.035] active:scale-[0.97] border-none group relative focus:outline-none focus-visible:ring-4 focus-visible:ring-[#176B4F]/25 focus-visible:ring-offset-2 focus-visible:ring-offset-white ${theme.button}`}
                  style={{ boxShadow: theme.buttonGlow }}
                >
                  {isConnecting ? (
                    <Loader2 className="w-[28px] h-[28px] sm:w-[36px] sm:h-[36px] lg:w-[44px] lg:h-[44px] animate-spin" />
                  ) : (
                    <Mic className="w-[28px] h-[28px] sm:w-[36px] sm:h-[36px] lg:w-[44px] lg:h-[44px] stroke-[2] transition-transform duration-300 group-hover:scale-105" />
                  )}
                </button>
              </div>
            </div>

            {/* Right waveform illustration */}
            <AudioWaveform
              side="right"
              isAnimating={isListening || isProcessing}
              statusStage={statusStage}
            />

          </div>

          {/* Best Match Dropdown Component */}
          <div className="relative z-20">
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="w-[205px] h-[44px] bg-[#FFFFFF] border border-[#DFDDD6] rounded-[22px] flex items-center justify-between px-[18px] cursor-pointer hover:border-[#176B4F] hover:bg-[#FAFBF8] transition-all duration-200 select-none focus:outline-none focus-visible:ring-4 focus-visible:ring-[#176B4F]/20"
            >
              <span className="font-sans text-[14px] md:text-[14.5px] font-normal text-[#1B211E]">
                {selectedStrategy}
              </span>
              <ChevronDown
                className={`w-[17px] h-[17px] stroke-[1.8] text-[#1C2520] transition-transform duration-200 ${showDropdown ? "rotate-180" : ""
                  }`}
              />
            </button>

            {/* Dropdown Options List */}
            {showDropdown && (
              <div
                className="absolute top-[50px] left-1/2 -translate-x-1/2 w-[240px] bg-white border border-[#E8E4DB] rounded-[14px] py-1.5 z-30 animate-fade-in"
                style={{ boxShadow: "0 2px 4px rgba(35,54,44,0.04), 0 16px 36px rgba(35,54,44,0.11)" }}
              >
                {["Best Match", "Hierarchical (Parent-Child)", "Sliding Window (Overlap)"].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => {
                      setSelectedStrategy(mode);
                      setShowDropdown(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 text-[13.5px] font-sans transition-colors focus:outline-none focus-visible:bg-[#F0F5F0] ${selectedStrategy === mode
                      ? "text-[#176B4F] font-semibold bg-[#F1F6F0]"
                      : "text-[#59635D] hover:bg-[#F6F8F4] hover:text-[#176B4F]"
                      }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
            )}
          </div>

        </div>

        {/* BottomAskSection */}
        <BottomAskSection />

      </div>
    </div>
  );
}
