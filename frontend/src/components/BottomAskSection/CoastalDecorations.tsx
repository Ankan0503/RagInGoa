import React, { useState } from "react";

export function Seashell() {
  return (
    <svg
      viewBox="0 0 50 50"
      width="50"
      height="50"
      className="fill-none select-none pointer-events-none"
      aria-hidden="true"
    >
      {/* Subtle shadow under shell */}
      <ellipse cx="25" cy="44" rx="14" ry="3" fill="#3A4D42" fillOpacity="0.05" />

      {/* Shell main contour fill */}
      <path
        d="M 25 42 C 12 36, 6 22, 11 12 C 13 8, 19 5, 25 8 C 31 5, 37 8, 39 12 C 44 22, 38 36, 25 42 Z"
        fill="#D8C9AE"
        fillOpacity="0.7"
      />

      {/* Ridges and details */}
      <path d="M 25 42 L 25 8" stroke="#E6DCC9" strokeWidth="1.2" strokeLinecap="round" />
      <path d="M 25 42 Q 19 25, 18 10" stroke="#E6DCC9" strokeWidth="1.2" strokeLinecap="round" />
      <path d="M 25 42 Q 31 25, 32 10" stroke="#E6DCC9" strokeWidth="1.2" strokeLinecap="round" />
      <path d="M 25 42 Q 13 30, 12 14" stroke="#E6DCC9" strokeWidth="1" strokeLinecap="round" />
      <path d="M 25 42 Q 37 30, 38 14" stroke="#E6DCC9" strokeWidth="1" strokeLinecap="round" />
      <path d="M 25 42 Q 8 34, 9 20" stroke="#E6DCC9" strokeWidth="0.8" strokeLinecap="round" />
      <path d="M 25 42 Q 42 34, 41 20" stroke="#E6DCC9" strokeWidth="0.8" strokeLinecap="round" />

      {/* Base articulation */}
      <path d="M 20 40 C 22 43, 28 43, 30 40" stroke="#D8C9AE" strokeWidth="2.5" strokeLinecap="round" />

      {/* Tiny sand dots around base */}
      <circle cx="16" cy="44" r="0.7" fill="#D8C9AE" />
      <circle cx="19" cy="46" r="0.5" fill="#D8C9AE" />
      <circle cx="32" cy="45" r="0.6" fill="#D8C9AE" />
      <circle cx="35" cy="43" r="0.4" fill="#D8C9AE" />
    </svg>
  );
}

export function PalmLeaf() {
  return (
    <svg
      viewBox="0 0 220 190"
      width="220"
      height="190"
      className="fill-none select-none pointer-events-none"
      aria-hidden="true"
    >
      {/* Palm Leaf Stem */}
      <path
        d="M 210 170 Q 115 120, 20 30"
        stroke="#AFC8BB"
        strokeWidth="2"
        strokeLinecap="round"
      />

      {/* Leaf Blades - radiating and tapering */}
      {/* Blade 1 */}
      <path d="M 175 145 Q 130 160, 95 170" stroke="#C0D5C9" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M 175 145 Q 140 152, 110 158" stroke="#D5E2DA" strokeWidth="0.8" strokeLinecap="round" />

      {/* Blade 2 */}
      <path d="M 155 132 Q 105 142, 60 148" stroke="#C0D5C9" strokeWidth="1.8" strokeLinecap="round" />

      {/* Blade 3 */}
      <path d="M 135 118 Q 75 120, 35 120" stroke="#AFC8BB" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M 135 118 Q 85 119, 50 118" stroke="#D5E2DA" strokeWidth="0.8" strokeLinecap="round" />

      {/* Blade 4 */}
      <path d="M 115 102 Q 55 98, 15 90" stroke="#C0D5C9" strokeWidth="1.5" strokeLinecap="round" />

      {/* Blade 5 */}
      <path d="M 95 86 Q 40 75, 5 60" stroke="#D5E2DA" strokeWidth="1.5" strokeLinecap="round" />

      {/* Blade 6 */}
      <path d="M 75 70 Q 30 55, 2 35" stroke="#AFC8BB" strokeWidth="1.2" strokeLinecap="round" />

      {/* Blade 7 */}
      <path d="M 55 54 Q 20 35, 1 12" stroke="#C0D5C9" strokeWidth="1.2" strokeLinecap="round" />

      {/* Interleaved layers for depth */}
      <path d="M 190 155 Q 165 178, 140 188" stroke="#AFC8BB" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M 165 138 Q 130 152, 100 158" stroke="#C0D5C9" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M 145 124 Q 100 132, 70 132" stroke="#D5E2DA" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M 125 110 Q 75 110, 45 105" stroke="#AFC8BB" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M 105 94 Q 60 90, 30 80" stroke="#C0D5C9" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

export default function CoastalDecorations() {
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  const showFallback = !imageLoaded || imageError;

  return (
    <div className="coastal-decorations-container">
      {/* Target Image Asset */}
      <img
        src="/assets/rag_bottom.avif"
        alt="Goa coastal background decorations"
        onLoad={() => setImageLoaded(true)}
        onError={() => setImageError(true)}
        // Anchored 120px to the right, taking up exactly 150% of the parent container's width
        className="absolute bottom-[-30px] left-[40px] h-[195px] w-[180%] object-fill pointer-events-none select-none z-10"
        style={{
          display: imageLoaded ? "block" : "none"
        }}
      />


      {/* Fallback Coded SVGs */}
      {showFallback && (
        <>
          {/* Wave lines SVG */}
          <svg
            viewBox="0 0 700 140"
            className="coastal-waves"
            aria-hidden="true"
          >
            {/* Path 1: Primary Greenish */}
            <path d="M 20 60 Q 150 40, 280 80 T 540 50 T 680 70" stroke="#C9DED4" strokeWidth="2" strokeLinecap="round" />
            {/* Path 2: Secondary Greenish */}
            <path d="M 50 85 Q 200 100, 350 70 T 650 90" stroke="#D9E6DD" strokeWidth="2" strokeLinecap="round" />
            {/* Path 3: Warm Accent */}
            <path d="M 10 40 Q 220 20, 400 60 T 690 35" stroke="#E8DDCA" strokeWidth="1.5" strokeLinecap="round" />
            {/* Path 4: Primary Greenish */}
            <path d="M 30 110 Q 180 90, 330 120 T 670 100" stroke="#C9DED4" strokeWidth="1.5" strokeLinecap="round" />
          </svg>

          {/* Seashell Placement */}
          <div className="coastal-seashell">
            <Seashell />
          </div>

          {/* Palm Leaf Placement */}
          <div className="coastal-palm-leaf">
            <PalmLeaf />
          </div>
        </>
      )}
    </div>
  );
}
