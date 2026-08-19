import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from "react";

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
}

export type PipelineStage = 
  | "idle"
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

  const getApiBaseUrl = (): string => {
    const envUrl = (import.meta as any).env?.VITE_API_BASE_URL;
    if (envUrl) return envUrl;
    const protocol = (typeof window !== "undefined" && window.location.protocol) || "http:";
    const host = (typeof window !== "undefined" && window.location.hostname) || "localhost";
    const port = (typeof window !== "undefined" && window.location.port) || "";
    const isDevPort = ["3000", "3001", "5173", "4173"].includes(port);
    const targetPort = isDevPort ? ":8000" : (port ? `:${port}` : "");
    return `${protocol}//${host}${targetPort}`;
  };

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
      setAnswer(data.answer || "");
      setSources(data.sources || []);
      if (data.metrics) setMetrics(data.metrics);
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
      if (data.full_answer) {
        setAnswer(data.full_answer);
      }
      if (data.metrics) {
        setMetrics(data.metrics);
      }
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

    try {
      setError(null);
      setGroundingWarning(null);
      setPendingTranscript("");
      setAwaitingSend(false);
      setAnswer("");

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const AudioCtx =
        (window as any).AudioContext || (window as any).webkitAudioContext;
      const ctx: AudioContext = new AudioCtx();
      audioCtxRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      // ScriptProcessor is deprecated in favour of AudioWorklet, but the
      // worklet path needs a separately served module file, which is awkward
      // inside the single-container deploy. 4096 frames is ~85ms at 48kHz --
      // small enough that partials still feel live.
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      ws.send(JSON.stringify({ type: "stt_start" }));

      processor.onaudioprocess = (e) => {
        const socket = wsRef.current;
        if (!socket || socket.readyState !== WebSocket.OPEN) return;
        socket.send(toPcm16k(e.inputBuffer.getChannelData(0), ctx.sampleRate));
      };

      source.connect(processor);
      // Route to destination with the gain implicitly zero-length; some
      // browsers will not run onaudioprocess for a disconnected node.
      processor.connect(ctx.destination);

      setIsListening(true);
      setStatusStage("listening");
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
