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
  setSelectedStrategy: (strategy: string) => void;
  startListening: () => Promise<void>;
  stopListening: () => void;
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

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const isStreamingRef = useRef(false);

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
    } else if (data.type === "transcript") {
      setQuery(data.text);
      if (data.stt_latency_ms) {
        setMetrics((prev) => ({ ...prev, stt_latency_ms: data.stt_latency_ms }));
      }
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

  // Voice Recording
  const startListening = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      let mimeType = "audio/webm;codecs=opus";
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = "audio/webm";
      }

      const recorder = new MediaRecorder(stream, { mimeType });
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        const arrayBuffer = await audioBlob.arrayBuffer();

        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          setIsProcessing(true);
          setStatusStage("transcribing");
          isStreamingRef.current = false;
          wsRef.current.send(arrayBuffer);
        } else {
          setError("Backend WebSocket is not connected. Ensure server is running on port 8000.");
        }

        stream.getTracks().forEach((track) => track.stop());
      };

      recorder.start(250);
      mediaRecorderRef.current = recorder;
      setIsListening(true);
      setStatusStage("listening");
    } catch (err: any) {
      console.error("Microphone error:", err);
      setError("Microphone permission denied or device not found.");
      setIsListening(false);
    }
  };

  const stopListening = () => {
    if (mediaRecorderRef.current && isListening) {
      mediaRecorderRef.current.stop();
      setIsListening(false);
    }
  };

  // Text Query Submission
  const sendTextQuery = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    setError(null);
    setQuery(trimmed);
    setIsProcessing(true);
    setStatusStage("retrieving");
    isStreamingRef.current = false;
    setAnswer("");

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
        top_k: 3
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
  };

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
        setSelectedStrategy,
        startListening,
        stopListening,
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
