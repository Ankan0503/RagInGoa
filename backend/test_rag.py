#!/usr/bin/env python3
"""
Indic RAG Benchmark & Streaming Generation Harness (test_rag.py)
================================================================
Designed for HH Goa 2026 Shortlisting Task 2: Sub-200ms Voice-Enabled Indic RAG System.

Key Engineering Features:
1. Seamless retrieval-to-generation pipeline with IndicRetriever & Groq LLaMA-3.1-8B-Instant.
2. Microsecond-accurate latency breakdown:
   - Retrieval Latency (Embed + Search + Parent Context Resolution)
   - Time to First Token (TTFT)
   - Total Generation Latency & Tokens Per Second (TPS)
   - End-to-End First-Token Pipeline Latency vs <200ms SLA Target
3. Ground-truth constrained Hindi prompt engineering with strict hallucination guardrails.
4. Interactive zero-buffering terminal streaming loop.
"""

import os
import sys
import io
import time
import logging
from typing import Generator, Dict, Any, Optional
from dotenv import load_dotenv

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load environment variables
load_dotenv()

# Import local retriever
from retriever import IndicRetriever, RetrievalResult, RetrievedHit

try:
    from groq import Groq
except ImportError:
    print("Error: 'groq' package not found. Please run: pip install groq", file=sys.stderr)
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TestRAG")


# ============================================================================
# PROMPT TEMPLATES (HINDI GROUND-TRUTH CONSTRAINED <200ms TARGET)
# ============================================================================

SYSTEM_PROMPT_TEMPLATE = """आप एक अत्यंत तीव्र, सटीक और सहायक AI सहायक हैं।
नीचे दिए गए संदर्भ (Context) के आधार पर ही प्रश्न का उत्तर केवल 1 संक्षिप्त, सीधा और तथ्यपरक वाक्य (अधिकतम 20-30 शब्द) में केवल शुद्ध हिंदी (Devanagari) में दें।
यदि संदर्भ में उत्तर मौजूद नहीं है, तो सीधे कहें "दिए गए संदर्भ में इसकी जानकारी उपलब्ध नहीं है।"

संदर्भ (Context):
{context}"""


# ============================================================================
# STREAMING RAG GENERATOR
# ============================================================================

def generate_rag_stream(
    query: str,
    retriever: IndicRetriever,
    groq_client: Groq,
    model: str = "llama-3.1-8b-instant",
    top_k: int = 3,
    strategy: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 35
) -> Generator[Dict[str, Any], None, None]:
    """
    Executes end-to-end RAG:
    1. Retrieves relevant child chunks & parent contexts via IndicRetriever.
    2. Constructs ground-truth constrained Hindi prompt.
    3. Streams generation tokens live via Groq API with microsecond latency profiling.

    Yields:
        Dict events:
        - {"type": "retrieval_done", "retrieval": RetrievalResult}
        - {"type": "token", "delta": str}
        - {"type": "done", "metrics": Dict[str, Any], "answer": str}
    """
    t_pipeline_start = time.perf_counter()

    # 1. Retrieval Phase
    retrieval_res: RetrievalResult = retriever.retrieve(
        query=query,
        top_k=top_k,
        strategy=strategy
    )
    yield {"type": "retrieval_done", "retrieval": retrieval_res}

    context_text = retrieval_res.combined_parent_context.strip()
    if not context_text:
        context_text = "कोई प्रासंगिक संदर्भ नहीं मिला।"

    # 2. Prompt Construction
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"प्रश्न: {query}"}
    ]

    # 3. LLM Generation Stream with Latency Profiling
    t_llm_start = time.perf_counter()
    ttft_ms: Optional[float] = None
    generated_tokens_count = 0
    full_answer_parts = []

    try:
        completion_stream = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in completion_stream:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if delta:
                if ttft_ms is None:
                    # Record Time to First Token (TTFT)
                    ttft_ms = (time.perf_counter() - t_llm_start) * 1000.0
                
                generated_tokens_count += 1
                full_answer_parts.append(delta)
                yield {"type": "token", "delta": delta}

    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        error_msg = f"\n[त्रुटि: {e}]"
        yield {"type": "token", "delta": error_msg}
        full_answer_parts.append(error_msg)

    t_pipeline_end = time.perf_counter()
    total_generation_time_ms = (t_pipeline_end - t_llm_start) * 1000.0
    total_pipeline_ms = (t_pipeline_end - t_pipeline_start) * 1000.0
    ttft_ms = ttft_ms if ttft_ms is not None else total_generation_time_ms

    # End-to-end First Token Latency = Retrieval Time + TTFT
    first_token_latency_ms = retrieval_res.total_retrieval_latency_ms + ttft_ms
    
    # Calculate generation speed (Tokens Per Second)
    gen_duration_sec = total_generation_time_ms / 1000.0
    tps = (generated_tokens_count / gen_duration_sec) if gen_duration_sec > 0 else 0.0

    metrics = {
        "retrieval_latency_ms": round(retrieval_res.total_retrieval_latency_ms, 2),
        "embed_latency_ms": round(retrieval_res.embed_latency_ms, 2),
        "search_latency_ms": round(retrieval_res.search_latency_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "first_token_latency_ms": round(first_token_latency_ms, 2),
        "total_generation_time_ms": round(total_generation_time_ms, 2),
        "total_pipeline_ms": round(total_pipeline_ms, 2),
        "total_tokens": generated_tokens_count,
        "tokens_per_second": round(tps, 2),
        "sla_passed": first_token_latency_ms < 200.0
    }

    yield {
        "type": "done",
        "metrics": metrics,
        "answer": "".join(full_answer_parts)
    }


# ============================================================================
# INTERACTIVE CLI HARNESS
# ============================================================================

def run_single_query(query: str, retriever: IndicRetriever, groq_client: Groq, top_k: int = 3, strategy: Optional[str] = None):
    print("\n" + "-" * 75)
    print(f"Query: \"{query}\"")
    print("-" * 75)

    final_metrics = None
    retrieval_info = None

    # Stream generation
    print("Answer: ", end="", flush=True)
    for event in generate_rag_stream(query=query, retriever=retriever, groq_client=groq_client, top_k=top_k, strategy=strategy):
        if event["type"] == "retrieval_done":
            retrieval_info = event["retrieval"]
        elif event["type"] == "token":
            sys.stdout.write(event["delta"])
            sys.stdout.flush()
        elif event["type"] == "done":
            final_metrics = event["metrics"]

    print("\n" + "-" * 75)

    # Display Retrieval Evidence
    if retrieval_info and retrieval_info.hits:
        print("\n[Retrieved Evidence - Top Sources]:")
        for rank, hit in enumerate(retrieval_info.hits, 1):
            snip = (hit.parent_text[:110] + "...") if len(hit.parent_text) > 110 else hit.parent_text
            print(f"  #{rank} [Score: {hit.score:.4f} | Strategy: {hit.strategy} | ID: {hit.parent_id}]")
            print(f"     Context: \"{snip}\"")

    # Display Latency Analytics Summary Card
    if final_metrics:
        sla_label = "PASSED (<200ms) [OK]" if final_metrics["sla_passed"] else "OVER BUDGET [WARNING]"
        sla_color = "\033[92m" if final_metrics["sla_passed"] else "\033[91m"
        reset_color = "\033[0m"

        print("\n" + "=" * 45)
        print("      LATENCY & PERFORMANCE ANALYTICS")
        print("=" * 45)
        print(f"  * Retrieval Latency    : {final_metrics['retrieval_latency_ms']} ms")
        print(f"    - Embedding Time     : {final_metrics['embed_latency_ms']} ms")
        print(f"    - Qdrant Search Time : {final_metrics['search_latency_ms']} ms")
        print(f"  * Time to First Token  : {final_metrics['ttft_ms']} ms")
        print(f"  * First-Token Latency  : {final_metrics['first_token_latency_ms']} ms  <-- (Retrieval + TTFT)")
        print(f"  * Total Generation Time: {final_metrics['total_generation_time_ms']} ms")
        print(f"  * Total Pipeline Time  : {final_metrics['total_pipeline_ms']} ms")
        print(f"  * Generation Speed     : {final_metrics['tokens_per_second']} tokens/sec ({final_metrics['total_tokens']} tokens)")
        print(f"  * Target SLA Budget    : {sla_color}{sla_label}{reset_color}")
        print("=" * 45)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Indic RAG Test Harness (Groq LLaMA-3.1-8B-Instant)")
    parser.add_argument("--query", "-q", type=str, default=None, help="Query string to evaluate.")
    parser.add_argument("--stream", action="store_true", default=True, help="Enable streaming response output (default: True).")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run automated multi-query benchmark.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved sources.")
    parser.add_argument("--strategy", type=str, default=None, help="Chunking strategy filter ('parent_child', 'sliding_window', or None).")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  HH GOA 2026: INDIC VOICE RAG - TEXT INFERENCE HARNESS (Groq LLaMA-3.1)")
    print("  Target SLA Budget: First Token < 200ms | Sub-15ms Local Vector Retrieval")
    print("=" * 75)

    # Resolve API Key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("QROQ_API_KEY")
    if not api_key:
        print("\n[ERROR] Groq API Key not found in environment or .env file!", file=sys.stderr)
        print("Please configure 'GROQ_API_KEY' or 'QROQ_API_KEY' in your .env file.", file=sys.stderr)
        sys.exit(1)

    # Initialize Clients
    print("\nInitializing IndicRetriever and Groq client...")
    groq_client = Groq(api_key=api_key)
    try:
        retriever = IndicRetriever(qdrant_path="./qdrant_data")
        print("Ready!\n")
    except RuntimeError as e:
        if "already accessed by another instance" in str(e):
            print("[NOTICE] Server is currently running. Querying running RAG backend service...")
            # Fallback to querying the running FastAPI server
            import httpx
            def run_server_query(q):
                print(f"\n[QUERY]: {q}\nRetrieving from running server...")
                try:
                    r = httpx.post("http://127.0.0.1:8000/api/text-query", json={"query": q, "top_k": args.top_k, "strategy": args.strategy})
                    data = r.json()
                    print("\n[ANSWER]:")
                    print(data["answer"])
                    print("\n" + "=" * 45)
                    print(f"  * Retrieval Latency    : {data['metrics']['retrieval_latency_ms']} ms")
                    print(f"  * Total Generation Time: {data['metrics']['total_generation_time_ms']} ms")
                    print(f"  * Total Pipeline Time  : {data['metrics']['total_pipeline_ms']} ms")
                    print(f"  * Generation Speed     : {data['metrics']['tokens_per_second']} tokens/sec")
                    print("=" * 45)
                except Exception as ex:
                    print(f"Server query error: {ex}")

            if args.query:
                run_server_query(args.query)
            elif args.benchmark:
                for q in ["What is a corporation?", "कॉर्पोरेशन क्या है?"]:
                    run_server_query(q)
            return
        else:
            raise e

    default_queries = [
        "कॉर्पोरेशन क्या है?",
        "कंप्यूटर कैसे काम करता है?",
        "स्वस्थ रहने के लिए क्या खाना चाहिए?"
    ]

    # Non-interactive mode: single query
    if args.query:
        run_single_query(args.query, retriever, groq_client, top_k=args.top_k, strategy=args.strategy)
        return

    # Non-interactive mode: benchmark
    if args.benchmark:
        for q in default_queries:
            run_single_query(q, retriever, groq_client, top_k=args.top_k, strategy=args.strategy)
        return

    # Interactive CLI loop
    query_idx = 0
    while True:
        try:
            prompt_hint = default_queries[query_idx % len(default_queries)]
            user_input = input(f"\nEnter Hindi query (press Enter for '{prompt_hint}', or 'exit'): ").strip()
            
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting RAG harness. Goodbye!\n")
                break

            query = user_input if user_input else prompt_hint
            query_idx += 1

            run_single_query(query, retriever, groq_client, top_k=args.top_k, strategy=args.strategy)

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Exiting.")
            break
        except Exception as e:
            print(f"\n[Unexpected Error]: {e}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
