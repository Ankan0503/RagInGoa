/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Mic, ChevronDown, Loader2, Sparkles } from "lucide-react";
import BottomAskSection from "../BottomAskSection/BottomAskSection";
import { useRag } from "../../context/RagContext";

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

  const isConnecting = statusStage === "connecting";
  const isListening = statusStage === "listening";

  return (
    <div
      className={`w-[38px] sm:w-[90px] lg:w-[145px] h-[45px] flex items-center ${side === "left" ? "justify-end" : "justify-start"
        } pointer-events-none`}
      aria-hidden="true"
    >
      <div className="flex items-center gap-[4px] h-[36px] opacity-90">
        {bars.map((height, idx) => {
          const delay = idx * (isListening ? 0.06 : 0.1);
          const style: React.CSSProperties = isAnimating
            ? {
              animation: isListening
                ? `pulse-waveform 0.8s ease-in-out infinite alternate`
                : `pulse-waveform 1.8s ease-in-out infinite alternate`,
              animationDelay: `${delay}s`,
              height: `${height}px`,
            }
            : {
              height: `${height}px`,
              transition: "height 0.3s ease",
            };

          let bgClass = "bg-[#C2D8CB]";
          if (isConnecting) {
            bgClass = height > 14 ? "bg-[#F59E0B]" : "bg-[#FCD34D]";
          } else if (isListening) {
            bgClass = height > 16 ? "bg-[#10B981]" : "bg-[#34D399]";
          } else if (height > 18) {
            bgClass = "bg-[#176B4F] opacity-65";
          } else if (height > 8) {
            bgClass = "bg-[#A8C6B6]";
          }

          return (
            <div
              key={idx}
              className={`w-[3px] rounded-full transition-colors duration-300 ${bgClass}`}
              style={style}
            />
          );
        })}
      </div>
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

  const isConnecting = statusStage === "connecting";
  const isLiveListening = statusStage === "listening";
  const isTranscribing = statusStage === "transcribing";

  const handleMicClick = () => {
    if (isListening || isConnecting || isLiveListening) {
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
      {/* Dynamic Keyframe style for pulse animations */}
      <style>{`
        @keyframes pulse-waveform {
          0% { transform: scaleY(0.5); }
          100% { transform: scaleY(1.4); }
        }
        @keyframes pulse-ring {
          0% { transform: scale(0.96); opacity: 0.6; }
          100% { transform: scale(1.05); opacity: 0.95; }
        }
        .animate-pulse-subtle {
          animation: pulse-ring 2s ease-in-out infinite alternate;
        }
        .animate-fade-in {
          animation: fadeIn 0.22s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(-4px); }
          to { opacity: 1; transform: translateY(0); }
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
          className={`w-full max-w-[640px] h-auto min-h-[295px] lg:h-[350px] bg-[#FFFFFF] rounded-[22px] border transition-all duration-300 flex flex-col items-center justify-between py-[22px] lg:py-[28px] px-4 sm:px-6 mt-[12px] z-20 hover:shadow-xl ${
            isConnecting
              ? "border-orange-300 shadow-[0_12px_36px_rgba(249,115,22,0.15)] bg-gradient-to-b from-orange-50/20 to-white"
              : isLiveListening
                ? "border-emerald-400 shadow-[0_12px_36px_rgba(16,185,129,0.18)] bg-gradient-to-b from-emerald-50/20 to-white"
                : isTranscribing
                  ? "border-purple-300 shadow-[0_12px_36px_rgba(147,51,234,0.12)] bg-gradient-to-b from-purple-50/20 to-white"
                  : "border-[#F0EFEA] shadow-[0_10px_30px_rgba(35,54,44,0.05)]"
          }`}
        >
          {/* Card Top Title & Status Banner */}
          <div className="flex flex-col items-center gap-1.5 text-center select-none w-full min-h-[40px] justify-center">
            {isConnecting ? (
              /* Small Cute Connecting Pill (Non-bold, Soft Peach) */
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#FEF6EB] border border-[#FCD8AC] rounded-full shadow-[0_1px_2px_rgba(217,119,6,0.04)] animate-pulse select-none">
                {/* Small halo with delicate spinner */}
                <div className="relative flex items-center justify-center w-[14px] h-[14px] rounded-full bg-[#FDE5C5] shrink-0">
                  <Loader2 className="w-[9px] h-[9px] text-[#C2410C] animate-spin" />
                </div>
                <span className="font-sans text-[13px] font-normal text-[#8C2C07] leading-none tracking-[0px]">
                  प्रतीक्षा करें
                </span>
                <span className="text-[#EA580C] text-[10px] font-normal select-none leading-none opacity-80">
                  •
                </span>
                <span className="font-sans text-[12.5px] font-normal text-[#C2410C] leading-none">
                  Connecting
                </span>
              </div>
            ) : isLiveListening ? (
              /* Small Cute Green Listening Pill (Non-bold, Soft Mint, Matching Image) */
              <div className="inline-flex items-center gap-2 px-3 py-1 bg-[#EBF5EF] border border-[#BCE1CC] rounded-full shadow-[0_1px_2px_rgba(15,118,85,0.04)] animate-fade-in select-none">
                {/* Delicate circular halo with emerald dot */}
                <div className="relative flex items-center justify-center w-[14px] h-[14px] rounded-full bg-[#CCEADB] shrink-0">
                  <span className="w-[6.5px] h-[6.5px] rounded-full bg-[#0F7655] animate-pulse" />
                </div>
                <span className="font-sans text-[13px] font-normal text-[#125E42] leading-none tracking-[0px]">
                  अब बोलिए
                </span>
                <span className="text-[#75A691] text-[10px] font-normal select-none leading-none opacity-80">
                  •
                </span>
                <span className="font-sans text-[12.5px] font-normal text-[#4A7F69] leading-none">
                  Listening
                </span>
              </div>
            ) : isTranscribing ? (
              /* Small Cute Finalizing Pill on Stop */
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-[#F5F3FF] border border-[#DDD6FE] rounded-full shadow-xs animate-pulse select-none">
                <div className="relative flex items-center justify-center w-[14px] h-[14px] rounded-full bg-[#EDE9FE] shrink-0">
                  <Loader2 className="w-[9px] h-[9px] text-[#7C3AED] animate-spin" />
                </div>
                <span className="font-sans text-[12.5px] font-normal text-[#5B21B6] leading-none">
                  आवाज़ पहचानी जा रही है
                </span>
                <span className="text-[#8B5CF6] text-[10px] font-normal select-none leading-none opacity-80">
                  •
                </span>
                <span className="font-sans text-[12px] font-normal text-[#7C3AED] leading-none">
                  Finalizing
                </span>
              </div>
            ) : awaitingSend ? (
              /* Small Cute Teal Review Pill */
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-[#F0F9F6] border border-[#C5E8DC] rounded-full shadow-xs animate-fade-in select-none">
                <Sparkles className="w-[11px] h-[11px] text-[#0D9488] shrink-0" />
                <span className="font-sans text-[12.5px] font-normal text-[#115E59] leading-none">
                  पहचाना गया प्रश्न देखें
                </span>
                <span className="text-[#5EEAD4] text-[10px] font-normal select-none leading-none opacity-80">
                  •
                </span>
                <span className="font-sans text-[12px] font-normal text-[#0F766E] leading-none">
                  Send when ready
                </span>
              </div>
            ) : isProcessing ? (
              /* Small Cute Generating Pill */
              <div className="inline-flex items-center gap-1.5 px-3 py-1 bg-[#F4F9F5] border border-[#CDE5D6] rounded-full shadow-xs animate-fade-in select-none">
                <Loader2 className="w-[11px] h-[11px] animate-spin text-[#176B4F] shrink-0" />
                <span className="font-sans text-[12.5px] font-normal text-[#14532D] leading-none">
                  उत्तर तैयार किया जा रहा है
                </span>
                <span className="text-[#86EFAC] text-[10px] font-normal select-none leading-none opacity-80">
                  •
                </span>
                <span className="font-sans text-[12px] font-normal text-[#15803D] leading-none">
                  Generating
                </span>
              </div>
            ) : (
              /* Idle State */
              <div className="flex flex-col items-center">
                <h2 className="font-sans text-[18px] font-semibold text-[#176B4F]">
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
              isAnimating={isListening || isProcessing || isTranscribing}
              statusStage={statusStage}
            />

            {/* Spherically layered microphone trigger */}
            <div className="relative flex items-center justify-center w-[104px] h-[104px] sm:w-[140px] sm:h-[140px] lg:w-[170px] lg:h-[170px] shrink-0">

              {/* Outer light translucent ring */}
              <div
                className={`absolute inset-0 rounded-full transition-all duration-500 ${
                  isConnecting
                    ? "bg-[#FFEDD5] border border-orange-200 opacity-90 animate-pulse"
                    : isLiveListening
                      ? "bg-[#D1FAE5] border border-emerald-200 opacity-90 animate-pulse-subtle"
                      : isTranscribing
                        ? "bg-[#F3E8FF] border border-purple-200 opacity-90 animate-pulse"
                        : "bg-[#F0F5F0] opacity-75"
                }`}
              />

              {/* Middle ring */}
              <div
                className={`absolute w-[86px] h-[86px] sm:w-[115px] sm:h-[115px] lg:w-[140px] lg:h-[140px] rounded-full flex items-center justify-center transition-all duration-300 ${
                  isConnecting
                    ? "bg-[#FED7AA]"
                    : isLiveListening
                      ? "bg-[#A7F3D0]"
                      : isTranscribing
                        ? "bg-[#E9D5FF]"
                        : "bg-[#DCEBE2]"
                }`}
              >

                {/* Inner button */}
                <button
                  onClick={handleMicClick}
                  aria-label="Speak your question"
                  className={`w-[64px] h-[64px] sm:w-[85px] sm:h-[85px] lg:w-[102px] lg:h-[102px] rounded-full flex items-center justify-center text-white cursor-pointer transition-all duration-300 hover:scale-[1.03] active:scale-[0.97] outline-none border-none group relative shadow-lg ${
                    isConnecting
                      ? "bg-gradient-to-tr from-amber-500 to-orange-600 hover:from-amber-600 hover:to-orange-700 shadow-orange-500/30"
                      : isLiveListening
                        ? "bg-gradient-to-tr from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 shadow-emerald-600/30"
                        : isTranscribing
                          ? "bg-gradient-to-tr from-purple-600 to-indigo-700 shadow-purple-500/30"
                          : "bg-[#176B4F] hover:bg-[#14694F] shadow-emerald-900/20"
                  }`}
                >
                  {isConnecting || isTranscribing ? (
                    <Loader2 className="w-[28px] h-[28px] sm:w-[36px] sm:h-[36px] lg:w-[44px] lg:h-[44px] animate-spin" />
                  ) : (
                    <Mic className="w-[28px] h-[28px] sm:w-[36px] sm:h-[36px] lg:w-[44px] lg:h-[44px] stroke-[2] transition-transform duration-300 group-hover:rotate-2" />
                  )}

                  {/* Ping border animation when listening */}
                  {isLiveListening && (
                    <span className="absolute inset-0 rounded-full border-2 border-emerald-300 animate-ping opacity-75" />
                  )}
                  {isConnecting && (
                    <span className="absolute inset-0 rounded-full border-2 border-orange-300 animate-ping opacity-60" />
                  )}
                </button>
              </div>
            </div>

            {/* Right waveform illustration */}
            <AudioWaveform
              side="right"
              isAnimating={isListening || isProcessing || isTranscribing}
              statusStage={statusStage}
            />

          </div>

          {/* Best Match Dropdown Component */}
          <div className="relative z-20">
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="w-[205px] h-[44px] bg-[#FFFFFF] border border-[#DADDD7] rounded-[22px] flex items-center justify-between px-[18px] cursor-pointer hover:border-[#176B4F] hover:bg-[#FAF9F5] transition-all duration-200 select-none outline-none shadow-xs"
            >
              <span className="font-sans text-[14px] md:text-[14.5px] font-normal text-[#1B211E]">
                {selectedStrategy}
              </span>
              <ChevronDown
                className={`w-[17px] h-[17px] stroke-[1.8] text-[#1C2520] transition-transform duration-200 ${
                  showDropdown ? "rotate-180" : ""
                }`}
              />
            </button>

            {/* Dropdown Options List */}
            {showDropdown && (
              <div className="absolute top-[50px] left-1/2 -translate-x-1/2 w-[210px] bg-white border border-[#E8E4DB] rounded-xl shadow-lg py-1 z-30 animate-fade-in">
                {["Best Match", "Hierarchical (Parent-Child)", "Sliding Window (Overlap)"].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => {
                      setSelectedStrategy(mode);
                      setShowDropdown(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 text-[14px] font-sans hover:bg-[#F0F5F0] hover:text-[#176B4F] transition-colors ${
                      selectedStrategy === mode
                        ? "text-[#176B4F] font-semibold bg-[#EDF3E8]"
                        : "text-[#59635D]"
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
