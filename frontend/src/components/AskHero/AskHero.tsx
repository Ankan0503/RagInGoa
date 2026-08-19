/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from "react";
import { Mic, ChevronDown } from "lucide-react";
import BottomAskSection from "../BottomAskSection/BottomAskSection";
import { useRag } from "../../context/RagContext";

/**
 * AudioWaveform - Horizontal bar visualization flanking the microphone button.
 */
interface AudioWaveformProps {
  side: "left" | "right";
  isAnimating: boolean;
}

export function AudioWaveform({ side, isAnimating }: AudioWaveformProps) {
  // Configured heights for left/right side matching reference composition
  const bars =
    side === "left"
      ? [4, 6, 10, 6, 12, 18, 8, 14, 24, 16, 28, 12, 8, 4]
      : [4, 8, 12, 28, 16, 24, 14, 8, 18, 12, 6, 10, 6, 4];

  return (
    <div
      className={`w-[38px] sm:w-[90px] lg:w-[145px] h-[45px] flex items-center ${side === "left" ? "justify-end" : "justify-start"
        } pointer-events-none`}
      aria-hidden="true"
    >
      <div className="flex items-center gap-[4px] h-[36px] opacity-85">
        {bars.map((height, idx) => {
          // Add a subtle randomized dynamic scaling when active
          const delay = idx * 0.08;
          const style: React.CSSProperties = isAnimating
            ? {
              animation: `pulse-waveform 1.2s ease-in-out infinite alternate`,
              animationDelay: `${delay}s`,
              height: `${height}px`,
            }
            : {
              height: `${height}px`,
              transition: "height 0.3s ease",
            };

          // Color palette mapped directly to match design specifications
          let bgClass = "bg-[#C2D8CB]";
          if (height > 18) {
            bgClass = "bg-[#176B4F] opacity-65";
          } else if (height > 8) {
            bgClass = "bg-[#A8C6B6]";
          }

          return (
            <div
              key={idx}
              className={`w-[3px] rounded-full ${bgClass}`}
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
    setSelectedStrategy
  } = useRag();

  const [showDropdown, setShowDropdown] = useState(false);

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
      {/* Dynamic Keyframe style for pulse animations inserted safely */}
      <style>{`
        @keyframes pulse-waveform {
          0% { transform: scaleY(0.6); }
          100% { transform: scaleY(1.3); }
        }
        @keyframes pulse-ring {
          0% { transform: scale(0.96); opacity: 0.6; }
          100% { transform: scale(1.04); opacity: 0.95; }
        }
        .animate-pulse-subtle {
          animation: pulse-ring 2s ease-in-out infinite alternate;
        }
        .animate-fade-in {
          animation: fadeIn 0.18s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translate(-50%, -8px); }
          to { opacity: 1; transform: translate(-50%, 0); }
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
                  {/* Ray 1: Top-right (steep) */}
                  <path d="M 5 19 C 7 14, 9 9, 13 5" strokeWidth="2.8" />
                  {/* Ray 2: Middle-right (medium) */}
                  <path d="M 6 20 C 10 17, 14 14, 20 11" strokeWidth="3" />
                  {/* Ray 3: Bottom-right (horizontal) */}
                  <path d="M 7 21 C 12 21, 17 21, 22 20.5" strokeWidth="2.8" />
                </svg>
              </div>
            </span>
            <br />
            Get <span className="text-[#176B4F]">grounded</span> answers.
          </h1>

          {/* Low contrast gently irregular curved underline SVG */}
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

        {/* Voice Interaction Card - centered, 640px width, 335px height */}
        <div
          className="w-full max-w-[640px] h-auto min-h-[280px] lg:h-[335px] bg-[#FFFFFF] rounded-[20px] border border-[#F0EFEA] flex flex-col items-center justify-between py-[24px] lg:py-[32px] px-4 sm:px-6 mt-[12px] z-20 transition-all duration-300 hover:shadow-lg"
          style={{
            boxShadow: "0 10px 30px rgba(35, 54, 44, 0.05)",
          }}
        >
          {/* Card title - Tap to speak your question */}
          <h2 className="font-sans text-[18px] font-semibold text-[#176B4F] text-center select-none">
            {isListening ? "Listening to your question..." : isProcessing ? "Finding answer..." : "Tap to speak your question"}
          </h2>

          {/* Microphone and Symmetrical Waveforms Group */}
          <div className="flex items-center justify-center w-full gap-1 sm:gap-3 lg:gap-4 my-auto">

            {/* Left waveform illustration */}
            <AudioWaveform side="left" isAnimating={isListening || isProcessing} />

            {/* Spherically layered microphone trigger */}
            <div className="relative flex items-center justify-center w-[104px] h-[104px] sm:w-[140px] sm:h-[140px] lg:w-[170px] lg:h-[170px] shrink-0">

              {/* Outer light translucent ring (170px at lg+, scaled below) */}
              <div className="absolute inset-0 rounded-full bg-[#F0F5F0] opacity-75 animate-pulse-subtle" />

              {/* Middle ring (140px at lg+, scaled below) */}
              <div className="absolute w-[86px] h-[86px] sm:w-[115px] sm:h-[115px] lg:w-[140px] lg:h-[140px] rounded-full bg-[#DCEBE2] flex items-center justify-center">

                {/* Inner button (102px at lg+, scaled below) */}
                <button
                  onClick={handleMicClick}
                  aria-label="Speak your question"
                  className={`w-[64px] h-[64px] sm:w-[85px] sm:h-[85px] lg:w-[102px] lg:h-[102px] rounded-full flex items-center justify-center text-white cursor-pointer transition-all duration-300 hover:scale-[1.03] active:scale-[0.97] outline-none border-none group relative ${isListening ? "bg-[#14694F]" : "bg-[#176B4F] hover:bg-[#14694F]"
                    }`}
                  style={{
                    boxShadow: "0 5px 12px rgba(23, 107, 79, 0.15)",
                  }}
                >
                  <Mic className="w-[28px] h-[28px] sm:w-[36px] sm:h-[36px] lg:w-[44px] lg:h-[44px] stroke-[2] transition-transform duration-300 group-hover:rotate-2" />

                  {/* Ping border animation when listening */}
                  {isListening && (
                    <span className="absolute inset-0 rounded-full border-2 border-[#176B4F] animate-ping opacity-60" />
                  )}
                </button>
              </div>
            </div>

            {/* Right waveform illustration */}
            <AudioWaveform side="right" isAnimating={isListening || isProcessing} />

          </div>

          {/* Best Match Dropdown Component */}
          <div className="relative z-20">
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="w-[198px] h-[45px] bg-[#FFFFFF] border border-[#DADDD7] rounded-[22px] flex items-center justify-between px-[20px] cursor-pointer hover:border-[#176B4F] hover:bg-[#FAF9F5] transition-all duration-200 select-none outline-none"
            >
              <span className="font-sans text-[14px] md:text-[15px] font-normal text-[#1B211E]">
                {selectedStrategy}
              </span>
              <ChevronDown
                className={`w-[17px] h-[17px] stroke-[1.8] text-[#1C2520] transition-transform duration-200 ${showDropdown ? "rotate-180" : ""
                  }`}
              />
            </button>

            {/* Dropdown Options List */}
            {showDropdown && (
              <div className="absolute top-[50px] left-1/2 -translate-x-1/2 w-[198px] bg-white border border-[#E8E4DB] rounded-xl shadow-md py-1 z-30 animate-fade-in">
                {["Best Match", "Hierarchical (Parent-Child)", "Sliding Window (Overlap)"].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => {
                      setSelectedStrategy(mode);
                      setShowDropdown(false);
                    }}
                    className={`w-full text-left px-4 py-2.5 text-[14px] font-sans hover:bg-[#F0F5F0] hover:text-[#176B4F] transition-colors ${selectedStrategy === mode
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

        {/* New BottomAskSection component (wrapping CapabilityGrid, CoastalDecorations, BuiltInGoa) */}
        <BottomAskSection />

      </div>
    </div>
  );
}
