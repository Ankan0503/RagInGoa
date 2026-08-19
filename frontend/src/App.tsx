/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from "react";
import { Menu } from "lucide-react";
import LeftSidebar from "./components/LeftSidebar/LeftSidebar";
import AskHero from "./components/AskHero/AskHero";
import AnswerPanel from "./components/AnswerPanel/AnswerPanel";
import { RagProvider } from "./context/RagContext";

export default function App() {
  // Sidebar is a static 250px column at lg+ (unchanged) and an off-canvas
  // drawer below lg (new). This state only matters below lg -- LeftSidebar
  // ignores it at lg+ via its own md:/lg: classes.
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  return (
    <RagProvider>
      <div
        id="app-viewport"
        className="w-full h-screen bg-[#F0EDE6] p-0 flex items-center justify-center overflow-hidden"
      >
        {/* Complete application outer rounded shell/container */}
        <div
          id="app-shell"
          className="w-[calc(100%-30px)] h-[calc(100vh-28px)] mx-[15px] my-[14px] rounded-[20px] border border-[#E8E4DB] shadow-sm bg-[#FAF9F5] flex flex-col lg:flex-row overflow-hidden"
        >
          {/* Mobile-only top bar: hamburger + compact brand mark, replaces
              the always-visible sidebar's brand section below lg. Hidden
              entirely at lg+, so desktop is untouched. */}
          <div
            id="mobile-topbar"
            className="lg:hidden flex items-center justify-between h-[52px] px-4 border-b border-[var(--border)] shrink-0"
          >
            <span className="font-serif text-[20px] font-bold text-[var(--forest-green)] tracking-tight">
              RAG in GOA
            </span>
            <button
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open menu"
              className="w-9 h-9 flex items-center justify-center rounded-[8px] text-[#1C2420] hover:bg-[#F3F2ED] outline-none"
            >
              <Menu size={22} strokeWidth={1.8} />
            </button>
          </div>

          {/* Backdrop, mobile-only, only rendered while the drawer is open */}
          {mobileNavOpen && (
            <div
              onClick={() => setMobileNavOpen(false)}
              aria-hidden="true"
              className="lg:hidden fixed inset-0 bg-black/40 z-40"
            />
          )}

          {/* Left Sidebar navigation panel -- drawer below lg, static at lg+ */}
          <LeftSidebar isOpen={mobileNavOpen} onClose={() => setMobileNavOpen(false)} />

          {/* Scrollable central and rightward workspace column cluster */}
          <div
            id="main-scroll-wrapper"
            className="flex-1 h-full overflow-y-auto flex flex-col lg:flex-row items-start justify-start relative"
          >
            {/* Main interactive content area - AskHero */}
            <AskHero />

            {/* Right Answer/Results Panel */}
            <AnswerPanel />
          </div>
        </div>
      </div>
    </RagProvider>
  );
}
