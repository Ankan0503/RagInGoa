import React, { useState } from "react";
import { Mic, BarChart2, Info } from "lucide-react";

// Types for Navigation Items
interface NavItem {
  id: string;
  label: string;
  icon: React.ComponentType<{ className?: string; strokeWidth?: number; size?: number }>;
}

// Sidebar Image constant - can be replaced with real AVIF/PNG by user
const SIDEBAR_IMAGE = "/assets/rag_photo.avif";

export default function LeftSidebar() {
  const [activeId, setActiveId] = useState<string>("ask");

  const navItems: NavItem[] = [
    {
      id: "ask",
      label: "Ask",
      icon: Mic,
    },
    {
      id: "insights",
      label: "Insights",
      icon: BarChart2,
    },
    {
      id: "about",
      label: "About",
      icon: Info,
    },
  ];

  return (
    <aside
      id="left-sidebar"
      className="relative flex flex-col w-[250px] min-w-[250px] max-w-[265px] h-full bg-[#fcfbf6] border-r border-[var(--border)] overflow-hidden select-none"
    >
      {/* Brand Section - Section A */}
      <div
        id="sidebar-brand-container"
        className="pt-[28px] pl-[45px] pb-[12px] flex flex-col justify-start bg-[#fcfbf6] relative z-10"
      >
        <div id="sidebar-logo-group" className="flex items-start">
          <div className="flex flex-col">
            <span
              id="logo-eyebrow"
              className="font-serif text-[22px] font-semibold text-[#151A17] tracking-tight leading-[0.95]"
            >
              RAG in
            </span>
            <span
              id="logo-brandname"
              className="font-serif text-[48px] font-bold text-[var(--forest-green)] leading-[0.85] tracking-tight relative"
            >
              GOA
            </span>
            {/* Wave Underline under GOA */}
            <div className="mt-0.5 mb-2">
              <svg
                width="100"
                height="32"
                viewBox="0 0 170 55"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="block overflow-visible"
                aria-hidden="true"
              >
                {/* Main Ocean Wave Line Art matching reference image */}
                <path
                  d="M 5 32 C 18 36, 32 32, 48 24 C 64 16, 78 7, 92 7 C 102 7, 108 11, 102 18 C 96 25, 87 28, 92 33 C 98 38, 115 35, 128 27 C 137 21, 142 19, 145 22 C 141 24, 139 27, 142 29 C 146 31, 152 29, 158 31 C 162 32, 166 31, 170 32"
                  stroke="#6B9142"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Inner barrel / trough curve */}
                <path
                  d="M 40 33 C 50 28, 62 26, 70 29 C 78 32, 73 37, 68 39 C 63 41, 62 36, 68 32"
                  stroke="#6B9142"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Minor trailing ripple line at bottom right */}
                <path
                  d="M 148 35 C 154 33, 160 36, 168 35"
                  stroke="#6B9142"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </div>
          </div>

          {/* Palm Tree Detail - Section A Accent */}
          <div
            id="logo-palm-icon"
            className="ml-3 mt-1 text-[var(--muted-green)]"
            title="Goa Palm"
          >
            <svg
              id="palm-tree-svg"
              width="24"
              height="36"
              viewBox="0 0 24 36"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="opacity-90"
            >
              {/* Palm Trunk */}
              <path
                d="M10 33C10.5 28 11.5 21 13 16.5"
                stroke="#76A893"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <path
                d="M11.5 33C12 28.5 12.8 22.5 14 18"
                stroke="#5D927D"
                strokeWidth="1"
                strokeLinecap="round"
              />
              {/* Palm Fronds / Leaves */}
              {/* Leaf 1 - Top Left */}
              <path
                d="M13 16.5C11 13 7 12.5 3 14C5.5 15.5 9 16 13 16.5Z"
                fill="#76A893"
              />
              {/* Leaf 2 - Mid Left */}
              <path
                d="M13 16.5C10 15 5 16 1.5 19.5C4 19 8.5 18 13 16.5Z"
                fill="#5D927D"
              />
              {/* Leaf 3 - Top Right */}
              <path
                d="M13 16.5C15 13 19 12.5 23 14C20.5 15.5 17 16 13 16.5Z"
                fill="#76A893"
              />
              {/* Leaf 4 - Mid Right */}
              <path
                d="M13 16.5C16 15 21 16 24.5 19.5C22 19 17.5 18 13 16.5Z"
                fill="#5D927D"
              />
              {/* Leaf 5 - Top Center vertical */}
              <path
                d="M13 16.5C12.5 11 14 6 16.5 2C15 6 14.5 11.5 13 16.5Z"
                fill="#76A893"
              />
            </svg>
          </div>
        </div>
      </div>

      {/* Brand Divider */}
      <hr
        id="sidebar-brand-divider"
        className="w-full border-t border-[var(--border)] opacity-[0.7] h-px"
      />

      {/* Navigation Area - Section B */}
      <nav
        id="sidebar-nav-container"
        className="mt-[28px] pl-[44px] pr-[20px] flex flex-col gap-1.5 bg-[#fcfbf6] relative z-10"
      >
        {navItems.map((item) => {
          const IconComponent = item.icon;
          const isActive = activeId === item.id;

          return (
            <button
              key={item.id}
              id={`nav-item-${item.id}`}
              onClick={() => setActiveId(item.id)}
              className={`
                group flex items-center w-[200px] h-[52px] px-4 rounded-[12px] transition-all duration-200 ease-in-out cursor-pointer text-left outline-none
                ${isActive
                  ? "bg-[var(--soft-green)] text-[var(--dark-green)]"
                  : "text-[#1C2420] hover:bg-[#F3F2ED]"
                }
              `}
            >
              <span
                className={`mr-3.5 flex items-center justify-center transition-colors duration-200
                  ${isActive ? "text-[var(--dark-green)]" : "text-[#202723] group-hover:text-[#111]"}
                `}
              >
                <IconComponent size={20} strokeWidth={1.8} />
              </span>
              <span
                className={`font-sans text-[16px] transition-colors duration-200
                  ${isActive ? "font-medium text-[var(--dark-green)]" : "font-normal text-[#1C2420] group-hover:text-[#111]"}
                `}
              >
                {item.label}
              </span>
            </button>
          );
        })}
      </nav>

      {/* Lower Goa Illustration - Section C */}
      <div
        id="goa-illustration-container"
        className="absolute bottom-0 left-0 w-full h-[62%] z-1 overflow-hidden"
      >
        <img
          id="goa-coastal-image"
          src={SIDEBAR_IMAGE}
          alt="Goa coastal illustration"
          className="w-full h-full object-cover object-bottom"
          onError={(e) => {
            const imgElement = e.currentTarget;
            imgElement.style.display = "none";
            const parentElement = imgElement.parentElement;
            if (parentElement) {
              parentElement.classList.add("bg-gradient-to-t", "from-[#114030]", "via-[#5D927D]", "to-[#FAF8F3]");
              const fallbackOverlay = document.createElement("div");
              fallbackOverlay.className = "absolute inset-0 flex flex-col justify-end p-8 pb-32 text-white/40 pointer-events-none";
              fallbackOverlay.innerHTML = `
                <div class="w-16 h-16 rounded-full bg-orange-100/10 mb-2 blur-xs"></div>
                <div class="h-0.5 w-24 bg-white/10 rounded mb-1"></div>
                <div class="h-0.5 w-16 bg-white/10 rounded"></div>
              `;
              parentElement.appendChild(fallbackOverlay);
            }
          }}
        />
        {/* Subtle blending warm gradient overlay */}
        <div
          id="goa-gradient-overlay"
          className="absolute inset-0 bg-gradient-to-b from-transparent to-[rgba(24,57,42,0.08)] pointer-events-none"
        />
      </div>

      {/* System Status Card - Section D */}
      <div
        id="system-status-card"
        className="absolute left-[32.5px] bottom-[20px] w-[185px] h-[92px] bg-[var(--warm-cream)] border border-[#E5E6D9] rounded-[16px] px-[16px] py-[14px] flex flex-col justify-between z-10 transition-shadow duration-300 hover:shadow-md"
        style={{
          boxShadow: "0 8px 25px rgba(38, 53, 44, 0.12)",
        }}
      >
        {/* Status Title Row */}
        <div id="status-title-row" className="flex items-center gap-[8px]">
          {/* Status Dot */}
          <span
            id="status-indicator-dot"
            className="w-[10px] h-[10px] rounded-full bg-[var(--status-green)] animate-pulse"
          />
          <span
            id="status-title-text"
            className="font-sans text-[13px] font-medium text-[#18221C] tracking-wide"
          >
            System Status
          </span>
        </div>

        {/* Status Label Text */}
        <p
          id="status-description-text"
          className="font-sans text-[13px] font-normal text-[#405149] leading-tight"
        >
          All systems operational
        </p>
      </div>
    </aside>
  );
}
