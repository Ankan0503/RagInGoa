/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import LeftSidebar from "./components/LeftSidebar/LeftSidebar";
import AskHero from "./components/AskHero/AskHero";
import AnswerPanel from "./components/AnswerPanel/AnswerPanel";
import { RagProvider } from "./context/RagContext";

export default function App() {
  return (
    <RagProvider>
      <div 
        id="app-viewport" 
        className="w-full h-screen bg-[#F0EDE6] p-0 flex items-center justify-center overflow-hidden"
      >
        {/* Complete application outer rounded shell/container */}
        <div 
          id="app-shell"
          className="w-[calc(100%-30px)] h-[calc(100vh-28px)] mx-[15px] my-[14px] rounded-[20px] border border-[#E8E4DB] shadow-sm bg-[#FAF9F5] flex overflow-hidden"
        >
          {/* Left Sidebar navigation panel */}
          <LeftSidebar />

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
