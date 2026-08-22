import React from "react";
import {
  Github, Linkedin, Instagram,
  Mic, Search, Sparkles, ShieldCheck, Gauge, Layers,
} from "lucide-react";

/** The actual X (formerly Twitter) wordmark. Lucide's "X" icon is a generic
 *  close/multiply glyph -- same crossing-lines shape as a dismiss button --
 *  not the brand mark, so it read as wrong rather than as the logo. */
function XLogo({ size = 16 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
    </svg>
  );
}

interface TeamMember {
  name: string;
  github: string;
  linkedin: string;
  instagram: string;
  x: string;
}

const TEAM: TeamMember[] = [
  {
    name: "Ankan Giri",
    github: "https://github.com/Ankan0503",
    linkedin: "https://www.linkedin.com/in/ankan-giri-71a34935a",
    instagram: "https://www.instagram.com/_xquisite_xplorer/",
    x: "https://x.com/Ankan0305",
  },
  {
    name: "Sayan Sinha",
    github: "https://github.com/Sayan260106",
    linkedin: "https://www.linkedin.com/in/sayan-sinha-300a20363",
    instagram: "https://www.instagram.com/_sayansinha_26/",
    x: "https://x.com/Sayan260106",
  },
];

interface Capability {
  icon: React.ComponentType<{ className?: string; strokeWidth?: number; size?: number }>;
  title: string;
  detail: string;
}

const CAPABILITIES: Capability[] = [
  {
    icon: Mic,
    title: "Live voice input",
    detail:
      "Speech streams to Sarvam's saaras:v3-realtime over WebSocket as you talk, so the transcript " +
      "appears live instead of after you stop. Nothing is sent to the model until you review the " +
      "text and press Send — catching a mis-hearing before it wastes a query.",
  },
  {
    icon: Layers,
    title: "Hybrid retrieval",
    detail:
      "Dense embeddings (multilingual-e5-small) and BM25 sparse search run in parallel and are " +
      "fused by Reciprocal Rank Fusion, over a Qdrant HNSW index of the full MSMARCO-XI Hindi " +
      "validation set — 97,941 queries, 3.4M vectors, 953K parent passages.",
  },
  {
    icon: Sparkles,
    title: "Streaming answers",
    detail:
      "The LLM's response streams token-by-token over the same WebSocket the moment generation " +
      "starts, instead of waiting for the full answer before showing anything.",
  },
  {
    icon: ShieldCheck,
    title: "Guardrails that stay visible",
    detail:
      "Input safety and retrieval-relevance gates run before generation, refusing unsafe or " +
      "unsupported questions outright. Because the grounding check needs the complete answer, it " +
      "runs after streaming and flags — rather than silently hides — anything it can't verify " +
      "against the retrieved context.",
  },
  {
    icon: Gauge,
    title: "Measured, not assumed",
    detail:
      "Every stage — embedding, search, fusion, generation, guardrails — is profiled per request, " +
      "with P50/P70/P100 latency tracked against a 200ms retrieval budget.",
  },
  {
    icon: Search,
    title: "Hindi-first, by design",
    detail:
      "The index, prompts, and grounding checks are all built for Devanagari Hindi rather than " +
      "translated from an English-first pipeline.",
  },
];

function TeamCard({ member }: { member: TeamMember; key?: React.Key }) {
  return (
    <div className="flex items-center justify-between rounded-[12px] border border-[#E8E4DB] bg-[#FEFEFC] px-4 py-3">
      <span className="font-sans font-semibold text-[14.5px] text-[#151A17]">
        {member.name}
      </span>
      <div className="flex items-center gap-1.5">
        <a
          href={member.github}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${member.name} on GitHub`}
          className="w-[32px] h-[32px] flex items-center justify-center rounded-[8px] text-[#252B27] hover:bg-[#F3F2ED] hover:text-[#176B4F] transition-colors duration-150"
        >
          <Github size={17} strokeWidth={1.8} />
        </a>
        <a
          href={member.linkedin}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${member.name} on LinkedIn`}
          className="w-[32px] h-[32px] flex items-center justify-center rounded-[8px] text-[#252B27] hover:bg-[#F3F2ED] hover:text-[#176B4F] transition-colors duration-150"
        >
          <Linkedin size={17} strokeWidth={1.8} />
        </a>
        <a
          href={member.instagram}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${member.name} on Instagram`}
          className="w-[32px] h-[32px] flex items-center justify-center rounded-[8px] text-[#252B27] hover:bg-[#F3F2ED] hover:text-[#176B4F] transition-colors duration-150"
        >
          <Instagram size={17} strokeWidth={1.8} />
        </a>
        <a
          href={member.x}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${member.name} on X`}
          className="w-[32px] h-[32px] flex items-center justify-center rounded-[8px] text-[#252B27] hover:bg-[#F3F2ED] hover:text-[#176B4F] transition-colors duration-150"
        >
          <XLogo size={15} />
        </a>
      </div>
    </div>
  );
}

export default function AboutSection() {
  return (
    <div
      id="about-section"
      className="flex-1 w-full h-full overflow-y-auto flex flex-col items-center px-6 py-8"
    >
      <div className="w-full max-w-[720px] flex flex-col gap-8">

        {/* Team header */}
        <div className="flex flex-col items-center text-center gap-1">
          <span className="font-sans text-[13px] font-semibold text-[#176B4F] tracking-[0.12em] uppercase">
            Team: Byte Me
          </span>
          <h1 className="font-serif text-[30px] sm:text-[36px] font-semibold text-[#151A17] leading-[1.05] tracking-tight mt-1">
            About RAG in GOA
          </h1>
          <p className="font-sans text-[15px] text-[#686D69] leading-relaxed mt-2 max-w-[560px]">
            A voice-first Hindi retrieval-augmented generation system, built for the
            HH Goa 2026 hackathon.
          </p>
        </div>

        {/* Team members */}
        <div className="flex flex-col gap-2">
          {TEAM.map((m) => (
            <TeamCard key={m.name} member={m} />
          ))}
        </div>

        <hr className="border-0 border-t border-[#E6E6E1]" />

        {/* What the site does */}
        <div className="flex flex-col gap-3">
          <h2 className="font-sans font-semibold text-[16px] text-[#176B4F]">
            What this website does
          </h2>
          <p className="font-sans text-[14.5px] text-[#252B27] leading-[1.65] text-justify">
            You ask a question out loud in Hindi. The system transcribes it live as you
            speak, retrieves the most relevant passages from a large indexed corpus using
            a combination of dense and keyword-based search, and streams back an answer
            generated only from what was actually retrieved &mdash; refusing to answer, or
            flagging its own answer, when the retrieved context doesn't actually support
            a confident response.
          </p>
        </div>

        <hr className="border-0 border-t border-[#E6E6E1]" />

        {/* Chunking strategy */}
        <div className="flex flex-col gap-3">
          <h2 className="font-sans font-semibold text-[16px] text-[#176B4F]">
            Chunking strategy
          </h2>
          <p className="font-sans text-[14.5px] text-[#252B27] leading-[1.65] text-justify">
            Fixed-size, one-size-fits-all chunking throws away information here: passages in
            MSMARCO&#8209;XI Hindi range from single-sentence fragments to dense multi-sentence
            paragraphs, and a naive fixed window either shreds the short ones or buries the long
            ones' relevant sentence inside noise. So each passage is routed to one of four
            chunking strategies based on its own length, and every strategy that applies to a
            passage runs &mdash; nothing is chunked only one way.
          </p>
          <div className="flex flex-col gap-2 mt-1">
            {[
              {
                tag: "S1 · passage",
                body: "The whole passage as one vector. Runs for every passage, short or long — the baseline strategy nothing else replaces.",
              },
              {
                tag: "S2 · sentence",
                body: "One context-prefixed vector per sentence, for passages of 3–6 sentences. Lets a single precise sentence surface on its own instead of being outweighed by the rest of its passage.",
              },
              {
                tag: "S3 · sliding window",
                body: "Overlapping windows of 3 sentences with a 1-sentence overlap, for passages of 7+ sentences — keeps cross-sentence context intact at a boundary instead of cutting it in half.",
              },
              {
                tag: "S4 · semantic",
                body: "Embedding-similarity boundary detection groups sentences by topic shift rather than by a fixed count, also for 7+ sentence passages — catches a topic change a fixed window would miss.",
              },
            ].map((s) => (
              <div key={s.tag} className="flex gap-3 rounded-[10px] bg-[#F7F9F6] px-3.5 py-2.5">
                <span className="font-sans font-semibold text-[12.5px] text-[#176B4F] shrink-0 w-[132px]">
                  {s.tag}
                </span>
                <span className="font-sans text-[13px] text-[#3E453F] leading-[1.55]">
                  {s.body}
                </span>
              </div>
            ))}
          </div>
          <p className="font-sans text-[14.5px] text-[#252B27] leading-[1.65] text-justify mt-1">
            All 10 retrieved passages per query are indexed &mdash; not only the ~7% marked
            "selected" in the source dataset, since discarding the rest would throw away 93% of
            the corpus and leave over a third of queries with nothing indexed at all. Every chunk
            is a <em>child</em> pointing back to its full passage as a <em>parent</em>, stored once
            in a separate SQLite store. Retrieval searches over the children, for precision, but
            answers are generated from the parent passage they resolve to, for context. At query
            time, the dense (E5) and sparse (BM25) results across all four strategies are merged
            by Reciprocal Rank Fusion and deduplicated on parent ID, so the strategies compete
            for relevance instead of returning four near-duplicate hits from the same passage.
          </p>
        </div>

        {/* Capability grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {CAPABILITIES.map((cap) => {
            const Icon = cap.icon;
            return (
              <div
                key={cap.title}
                className="flex flex-col gap-2 rounded-[14px] border border-[#EAE8E2] bg-[#FFFFFF] p-4"
                style={{ boxShadow: "0 4px 16px rgba(35, 54, 44, 0.04)" }}
              >
                <div className="w-[30px] h-[30px] rounded-[8px] bg-[#EEF5ED] flex items-center justify-center text-[#176B4F]">
                  <Icon size={16} strokeWidth={1.8} />
                </div>
                <span className="font-sans font-semibold text-[14px] text-[#151A17]">
                  {cap.title}
                </span>
                <p className="font-sans text-[13px] text-[#5C635D] leading-[1.55]">
                  {cap.detail}
                </p>
              </div>
            );
          })}
        </div>

        <p className="font-sans text-[12px] text-[#9AA39D] text-center pb-4">
          Built for Goa &mdash; HH Goa 2026, Task 02.
        </p>
      </div>
    </div>
  );
}
