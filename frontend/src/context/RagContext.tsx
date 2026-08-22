import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from "react";

/** Resolve the REST API base URL from .env or window.location. Standalone
 *  (not a closure over component state) so pages outside the voice flow --
 *  Insights, for one -- can hit the same backend without duplicating this
 *  dev-port/prod-port resolution logic. */
export function resolveApiBaseUrl(): string {
  const envUrl = (import.meta as any).env?.VITE_API_BASE_URL;
  if (envUrl) return envUrl;
  const protocol = (typeof window !== "undefined" && window.location.protocol) || "http:";
  const host = (typeof window !== "undefined" && window.location.hostname) || "localhost";
  const port = (typeof window !== "undefined" && window.location.port) || "";
  const isDevPort = ["3000", "3001", "5173", "4173"].includes(port);
  const targetPort = isDevPort ? ":8000" : (port ? `:${port}` : "");
  return `${protocol}//${host}${targetPort}`;
}

/** Per-browser query history for the Insights page. This is intentionally
 *  separate from the server-side query log: the server log is shared across
 *  every visitor (admin-gated, since it holds everyone's real questions),
 *  while this is private to whoever is sitting at this browser -- visible
 *  with no token, cleared with no token, because it can only ever contain
 *  what this browser itself asked. */
const LOCAL_LOG_KEY = "rag_local_query_log";
const LOCAL_LOG_MAX = 200;

export interface LocalLogEntry {
  id: string;
  ts: string;
  query: string;
  transcript: string | null;
  answer: string;
  refused: boolean;
  provider: string | null;
  model: string | null;
  retrieval_ms: number | null;
  generation_ms: number | null;
  guardrail_ms: number | null;
  end_to_end_ms: number | null;
  grounding_score: number | null;
  // Null on the REST fallback path, which cannot measure a first token.
  ttft_ms?: number | null;
  stages: Array<{ stage: string; ms: number; in_budget: boolean; detail?: string }>;
}

export function readLocalLog(): LocalLogEntry[] {
  try {
    const raw = localStorage.getItem(LOCAL_LOG_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function appendLocalLog(entry: Omit<LocalLogEntry, "id">): void {
  try {
    const withId: LocalLogEntry = {
      ...entry,
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    };
    const updated = [withId, ...readLocalLog()].slice(0, LOCAL_LOG_MAX);
    localStorage.setItem(LOCAL_LOG_KEY, JSON.stringify(updated));
  } catch {
    // localStorage can throw (quota exceeded, private browsing) -- a logging
    // failure must never break the actual request that triggered it.
  }
}

export function clearLocalLog(): void {
  try {
    localStorage.removeItem(LOCAL_LOG_KEY);
  } catch {
    // ignore
  }
}

export interface SourceHit {
  score: number;
  strategy: string;
  child_text?: string;
  parent_id: string;
  parent_text: string;
  chunk_index?: number;
  total_chunks?: number;
  query_id?: string;
  language?: string;
}

export interface LatencyMetrics {
  stt_latency_ms?: number | null;
  retrieval_latency_ms?: number;
  embed_latency_ms?: number;
  search_latency_ms?: number;
  ttft_ms?: number | null;
  first_token_latency_ms?: number | null;
  total_generation_time_ms?: number;
  total_pipeline_latency_ms?: number;
  tokens_per_second?: number;
  total_tokens?: number;
  sla_passed?: boolean;
  /** The real values from guardrails on this answer -- not derived from
   *  latency or any other proxy. Only present once a "done" event with a
   *  real grounding check has been received. */
  grounding_score?: number | null;
  retrieval_score?: number | null;
}

export type PipelineStage =
  | "idle"
  | "connecting"
  | "listening"
  | "transcribing"
  | "retrieving"
  | "generating"
  | "done"
  | "error";

/** Result of the grounding gate, which now runs AFTER the answer has streamed
 *  out. `null` means it has not run or the answer was fine. */
export interface GroundingWarning {
  score?: number | null;
  detail?: string | null;
  message: string;
}

interface RagContextType {
  isConnected: boolean;
  isListening: boolean;
  isProcessing: boolean;
  statusStage: PipelineStage;
  query: string;
  answer: string;
  sources: SourceHit[];
  metrics: LatencyMetrics | null;
  selectedStrategy: string;
  error: string | null;
  /** Live transcript from Sarvam, shown while speaking and editable-in-spirit
   *  until the user presses Send. */
  pendingTranscript: string;
  /** True once speech has ended and a transcript is waiting to be sent. */
  awaitingSend: boolean;
  groundingWarning: GroundingWarning | null;
  setSelectedStrategy: (strategy: string) => void;
  setPendingTranscript: (text: string) => void;
  startListening: () => Promise<void>;
  stopListening: () => void;
  sendPending: () => void;
  discardPending: () => void;
  sendTextQuery: (text: string) => void;
  resetSession: () => void;
}

const RagContext = createContext<RagContextType | undefined>(undefined);

export function RagProvider({ children }: { children: ReactNode }) {
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusStage, setStatusStage] = useState<PipelineStage>("idle");
  // Seeded empty. These were previously populated with a hardcoded answer, a
  // fabricated source passage and a full set of plausible latency numbers, so a
  // freshly loaded page displayed "181.4ms" and a citation before anything had
  // run. Measurements must only ever come from a real request.
  const [query, setQuery] = useState<string>("");
  const [answer, setAnswer] = useState<string>("");
  const [sources, setSources] = useState<SourceHit[]>([]);
  const [metrics, setMetrics] = useState<LatencyMetrics | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("Best Match");
  const [error, setError] = useState<string | null>(null);
  const [pendingTranscript, setPendingTranscript] = useState<string>("");
  const [awaitingSend, setAwaitingSend] = useState(false);
  const [groundingWarning, setGroundingWarning] = useState<GroundingWarning | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const isStreamingRef = useRef(false);
  const sttLatencyRef = useRef<number | null>(null);
  // handleServerMessage is set up once and would otherwise see a stale
  // `query` from mount time -- this ref is the live value at send-time,
  // for local-log entries written when "done"/"refused" arrive.
  const currentQueryRef = useRef<string>("");
  const currentTranscriptRef = useRef<string | null>(null);

  // Live audio capture. MediaRecorder is gone: it produces webm/opus in
  // ~250ms containers, but Sarvam's realtime socket wants a continuous raw
  // PCM feed, so we tap the WebAudio graph directly instead.
  const audioCtxRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Resolve WebSocket and REST Base URLs from .env or window.location
  const getWsUrl = (): string => {
    const envUrl = (import.meta as any).env?.VITE_WS_URL;
    if (envUrl) return envUrl;
    const isHttps = typeof window !== "undefined" && window.location.protocol === "https:";
    const protocol = isHttps ? "wss:" : "ws:";
    const host = (typeof window !== "undefined" && window.location.hostname) || "localhost";
    const port = (typeof window !== "undefined" && window.location.port) || "";
    const isDevPort = ["3000", "3001", "5173", "4173"].includes(port);
    const targetPort = isDevPort ? ":8000" : (port ? `:${port}` : "");
    return `${protocol}//${host}${targetPort}/ws/voice-rag`;
  };

  const getApiBaseUrl = resolveApiBaseUrl;

  // Initialize and Maintain WebSocket Connection
  useEffect(() => {
    let reconnectTimer: NodeJS.Timeout;

    function connect() {
      const wsUrl = getWsUrl();
      
      try {
        const socket = new WebSocket(wsUrl);
        socket.binaryType = "arraybuffer";

        socket.onopen = () => {
          setIsConnected(true);
          setError(null);
        };

        socket.onclose = () => {
          setIsConnected(false);
          reconnectTimer = setTimeout(connect, 2500);
        };

        socket.onerror = () => {
          setIsConnected(false);
        };

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            handleServerMessage(data);
          } catch (e) {
            console.error("WS Parse error:", e);
          }
        };

        wsRef.current = socket;
      } catch (err) {
        setIsConnected(false);
        reconnectTimer = setTimeout(connect, 3000);
      }
    }

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  function handleServerMessage(data: any) {
    if (data.type === "status") {
      setStatusStage(data.stage as PipelineStage);
    } else if (data.type === "transcript.partial" || data.type === "transcript.final") {
      // Live feedback while the user is still speaking.
      setPendingTranscript(data.text || "");
    } else if (data.type === "transcript.done") {
      // Speech ended. Hold the transcript for review -- nothing is retrieved
      // or generated until the user presses Send.
      setPendingTranscript(data.text || "");
      setAwaitingSend(Boolean(data.text));
      sttLatencyRef.current = data.stt_latency_ms ?? null;
      setStatusStage("idle");
      if (!data.text) {
        setError("आवाज़ पहचानी नहीं जा सकी। कृपया दोबारा बोलें।");
      }
    } else if (data.type === "transcript") {
      setQuery(data.text);
      if (data.stt_latency_ms) {
        setMetrics((prev) => ({ ...prev, stt_latency_ms: data.stt_latency_ms }));
      }
    } else if (data.type === "grounding_failed") {
      // The answer has already streamed onto the screen by this point. We do
      // not retract it -- we flag it, and the panel dims what is shown.
      setGroundingWarning({
        score: data.score ?? null,
        detail: data.detail ?? null,
        message: data.message || "यह उत्तर संदर्भ द्वारा पूर्ण रूप से समर्थित नहीं है।"
      });
    } else if (data.type === "refused") {
      isStreamingRef.current = false;
      setIsProcessing(false);
      setStatusStage("done");
      setAnswer(data.answer ?? "");
      setSources(data.sources || []);
      if (data.metrics) {
        setMetrics((prev) => ({
          ...prev,
          ...data.metrics,
          grounding_score: data.guardrails?.grounding_score ?? null,
          retrieval_score: data.guardrails?.retrieval_score ?? null,
        }));
      }
      appendLocalLog({
        ts: new Date().toISOString(),
        query: currentQueryRef.current,
        transcript: currentTranscriptRef.current,
        answer: data.answer ?? "",
        refused: true,
        provider: data.metrics?.llm_provider ?? null,
        model: data.metrics?.llm_model ?? null,
        retrieval_ms: data.metrics?.retrieval_ms ?? null,
        generation_ms: data.metrics?.generation_ms ?? null,
        guardrail_ms: data.metrics?.guardrail_ms ?? null,
        end_to_end_ms: data.metrics?.wall_ms ?? null,
        grounding_score: data.guardrails?.grounding_score ?? null,
        ttft_ms: data.metrics?.ttft_ms ?? null,
        stages: data.metrics?.stages ?? [],
      });
    } else if (data.type === "retrieval") {
      setStatusStage("generating");
      setSources(data.sources || []);
      setMetrics((prev) => ({
        ...prev,
        retrieval_latency_ms: data.retrieval_latency_ms,
        embed_latency_ms: data.embed_latency_ms,
        search_latency_ms: data.search_latency_ms
      }));
    } else if (data.type === "token") {
      if (!isStreamingRef.current) {
        setAnswer("");
        isStreamingRef.current = true;
      }
      setAnswer((prev) => prev + data.delta);
    } else if (data.type === "done") {
      isStreamingRef.current = false;
      setIsProcessing(false);
      setStatusStage("done");
      // Always sync to the server's real value, including an empty one --
      // `if (data.full_answer)` used to skip this on an empty string, which
      // meant a failed/empty generation left whatever was on screen before
      // (stale text from a prior question, or nothing) with no indication
      // anything had gone wrong. The backend now turns a 0-token stream into
      // a real Hindi failure message rather than an empty string, but the
      // frontend should not silently swallow "" either way.
      setAnswer(data.full_answer ?? "");
      if (data.metrics) {
        setMetrics((prev) => ({
          ...prev,
          ...data.metrics,
          grounding_score: data.guardrails?.grounding_score ?? prev?.grounding_score ?? null,
          retrieval_score: data.guardrails?.retrieval_score ?? prev?.retrieval_score ?? null,
        }));
      }
      appendLocalLog({
        ts: new Date().toISOString(),
        query: currentQueryRef.current,
        transcript: currentTranscriptRef.current,
        answer: data.full_answer ?? "",
        refused: false,
        provider: data.metrics?.llm_provider ?? null,
        model: data.metrics?.llm_model ?? null,
        retrieval_ms: data.metrics?.retrieval_ms ?? null,
        generation_ms: data.metrics?.generation_ms ?? null,
        guardrail_ms: data.metrics?.guardrail_ms ?? null,
        end_to_end_ms: data.metrics?.wall_ms ?? null,
        grounding_score: data.guardrails?.grounding_score ?? null,
        ttft_ms: data.metrics?.ttft_ms ?? null,
        stages: data.metrics?.stages ?? [],
      });
    } else if (data.type === "error") {
      isStreamingRef.current = false;
      setIsProcessing(false);
      setStatusStage("error");
      setError(data.message || "An error occurred during query execution.");
    }
  }

  /** Linear-interpolation resample to 16kHz mono s16le, the only format the
   *  realtime STT socket accepts. Browsers give us 44.1k or 48k depending on
   *  the device, so this cannot be skipped. */
  const toPcm16k = (input: Float32Array, inRate: number): ArrayBuffer => {
    const ratio = inRate / 16000;
    const outLength = Math.floor(input.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const pos = i * ratio;
      const lo = Math.floor(pos);
      const hi = Math.min(lo + 1, input.length - 1);
      const sample = input[lo] + (input[hi] - input[lo]) * (pos - lo);
      const clamped = Math.max(-1, Math.min(1, sample));
      out[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return out.buffer;
  };

  const teardownAudio = () => {
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  };

  const startListening = async () => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      setError("Backend WebSocket is not connected.");
      return;
    }

    // Temporary timing instrumentation to localize a reported 5-6s delay
    // before the mic UI shows as active, with no permission popup involved
    // (so the stall is somewhere in this function, not a browser dialog).
    // Remove once the slow step is identified.
    const t0 = performance.now();
    const mark = (label: string) =>
      console.log(`[mic-timing] ${label}: +${(performance.now() - t0).toFixed(0)}ms`);

    try {
      setError(null);
      setGroundingWarning(null);
      setPendingTranscript("");
      setAwaitingSend(false);
      setAnswer("");
      setStatusStage("connecting");

      // Sent first, before getUserMedia, so the server's Sarvam handshake
      // (a real network round trip, ~1-3s) runs concurrently with the
      // browser's own mic-permission prompt and audio graph setup instead
      // of strictly after them. The server buffers any audio that arrives
      // before its side is ready, so this is safe even if getUserMedia
      // resolves first.
      mark("start");
      ws.send(JSON.stringify({ type: "stt_start" }));
      mark("ws.send done");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mark("getUserMedia resolved");
      mediaStreamRef.current = stream;

      const AudioCtx =
        (window as any).AudioContext || (window as any).webkitAudioContext;
      const ctx: AudioContext = new AudioCtx();
      audioCtxRef.current = ctx;
      mark("AudioContext created");

      const source = ctx.createMediaStreamSource(stream);
      // ScriptProcessor is deprecated in favour of AudioWorklet, but the
      // worklet path needs a separately served module file, which is awkward
      // inside the single-container deploy. 4096 frames is ~85ms at 48kHz --
      // small enough that partials still feel live.
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const socket = wsRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        socket.send(toPcm16k(e.inputBuffer.getChannelData(0), ctx.sampleRate));
      };

      source.connect(processor);
      // Route to destination with the gain implicitly zero-length; some
      // browsers will not run onaudioprocess for a disconnected node.
      processor.connect(ctx.destination);

      // isListening drives the mic UI/animation and can flip as soon as
      // capture is physically wired up. statusStage stays whatever the
      // server's last "status" message set it to ("connecting" until the
      // Sarvam handshake completes, then "listening") -- overriding it here
      // would show "listening" before the server can actually hear anything.
      mark("audio graph wired, setIsListening(true) now");
      setIsListening(true);
    } catch (err: any) {
      console.error("Microphone error:", err);
      setError("Microphone permission denied or device not found.");
      setIsListening(false);
      teardownAudio();
    }
  };

  const stopListening = () => {
    if (!isListening) return;
    teardownAudio();
    setIsListening(false);
    setStatusStage("transcribing");
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stt_stop" }));
    }
  };

  /** Commit the reviewed transcript. This is the Send button. */
  const sendPending = () => {
    const text = pendingTranscript.trim();
    if (!text) return;
    setAwaitingSend(false);
    sendTextQuery(text, sttLatencyRef.current);
    sttLatencyRef.current = null;
  };

  const discardPending = () => {
    setPendingTranscript("");
    setAwaitingSend(false);
    sttLatencyRef.current = null;
    setStatusStage("idle");
  };

  // Text Query Submission
  const sendTextQuery = (text: string, sttLatencyMs?: number | null) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    setError(null);
    setQuery(trimmed);
    setIsProcessing(true);
    setStatusStage("retrieving");
    isStreamingRef.current = false;
    setAnswer("");
    setGroundingWarning(null);
    currentQueryRef.current = trimmed;
    currentTranscriptRef.current = sttLatencyMs != null ? trimmed : null;

    // Map UI strategy label to backend strategy key
    let strategyKey: string | undefined = undefined;
    if (selectedStrategy.includes("Parent-Child") || selectedStrategy.includes("Hierarchical")) {
      strategyKey = "parent_child";
    } else if (selectedStrategy.includes("Sliding") || selectedStrategy.includes("Window")) {
      strategyKey = "sliding_window";
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const payload = {
        type: "text_query",
        text: trimmed,
        strategy: strategyKey,
        top_k: 3,
        stt_latency_ms: sttLatencyMs ?? undefined,
        transcript: sttLatencyMs != null ? trimmed : undefined
      };
      wsRef.current.send(JSON.stringify(payload));
    } else {
      // Fallback REST call
      fetch(`${getApiBaseUrl()}/api/text-query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: trimmed, strategy: strategyKey, top_k: 3 })
      })
        .then((res) => res.json())
        .then((data) => {
          setAnswer(data.answer);
          setSources(data.sources || []);
          setMetrics(data.metrics || null);
          setStatusStage("done");
          setIsProcessing(false);
          appendLocalLog({
            ts: new Date().toISOString(),
            query: trimmed,
            transcript: null,
            answer: data.answer ?? "",
            refused: Boolean(data.refused),
            provider: data.metrics?.llm_provider ?? null,
            model: data.metrics?.llm_model ?? null,
            retrieval_ms: data.metrics?.retrieval_ms ?? null,
            generation_ms: data.metrics?.generation_ms ?? null,
            guardrail_ms: data.metrics?.guardrail_ms ?? null,
            end_to_end_ms: data.metrics?.wall_ms ?? null,
            grounding_score: data.guardrails?.grounding_score ?? null,
            stages: data.metrics?.stages ?? [],
          });
        })
        .catch((err) => {
          setError(`Request failed: ${err.message}`);
          setStatusStage("error");
          setIsProcessing(false);
        });
    }
  };

  const resetSession = () => {
    setStatusStage("idle");
    setError(null);
    setGroundingWarning(null);
    setPendingTranscript("");
    setAwaitingSend(false);
  };

  // Cloudflare drops idle WebSockets at ~100s. A light client ping keeps the
  // socket alive between questions so the first one after a pause does not
  // silently fail on a dead connection.
  useEffect(() => {
    const timer = setInterval(() => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 30000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => teardownAudio, []);

  return (
    <RagContext.Provider
      value={{
        isConnected,
        isListening,
        isProcessing,
        statusStage,
        query,
        answer,
        sources,
        metrics,
        selectedStrategy,
        error,
        pendingTranscript,
        awaitingSend,
        groundingWarning,
        setSelectedStrategy,
        setPendingTranscript,
        startListening,
        stopListening,
        sendPending,
        discardPending,
        sendTextQuery,
        resetSession
      }}
    >
      {children}
    </RagContext.Provider>
  );
}

export function useRag() {
  const context = useContext(RagContext);
  if (!context) {
    throw new Error("useRag must be used within a RagProvider");
  }
  return context;
}
