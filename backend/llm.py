#!/usr/bin/env python3
"""
Pluggable LLM Backend (Groq & Sarvam)
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator, Tuple

import httpx

logger = logging.getLogger("LLM")


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    ttft_ms: Optional[float] = None
    total_ms: float = 0.0
    finish_reason: str = ""

    @property
    def tokens_per_second(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return round(self.completion_tokens / (self.total_ms / 1000.0), 2)


class LLMError(RuntimeError):
    pass


class BaseLLM:
    provider = "base"

    def __init__(self, model: str, api_key: str, base_url: str,
                 timeout_s: float = 8.0):
        if not api_key:
            raise LLMError(f"{self.provider}: no API key configured")
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )

    def headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def payload(self, messages: List[Dict[str, str]], temperature: float,
                max_tokens: int, stream: bool) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    async def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0,
                       max_tokens: int = 48) -> LLMResult:
        t0 = time.perf_counter()
        body = self.payload(messages, temperature, max_tokens, stream=False)
        try:
            r = await self._client.post(self.endpoint, headers=self.headers(), json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"{self.provider}: transport error: {e}") from e

        if r.status_code != 200:
            raise LLMError(f"{self.provider}: HTTP {r.status_code}: {r.text[:300]}")

        data = r.json()
        return self._parse(data, (time.perf_counter() - t0) * 1000.0)

    async def stream(self, messages: List[Dict[str, str]], temperature: float = 0.0,
                     max_tokens: int = 48) -> AsyncIterator[Tuple[str, Optional[LLMResult]]]:
        t0 = time.perf_counter()
        body = self.payload(messages, temperature, max_tokens, stream=True)
        ttft: Optional[float] = None
        parts: List[str] = []
        finish = ""

        try:
            async with self._client.stream("POST", self.endpoint,
                                           headers=self.headers(), json=body) as r:
                if r.status_code != 200:
                    detail = (await r.aread())[:300]
                    raise LLMError(f"{self.provider}: HTTP {r.status_code}: {detail!r}")

                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        obj = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if "error" in obj:
                        err = obj["error"]
                        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                        raise LLMError(f"{self.provider}: mid-stream error: {msg[:300]}")
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    finish = choices[0].get("finish_reason") or finish
                    if delta:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000.0
                        parts.append(delta)
                        yield delta, None
        except httpx.HTTPError as e:
            raise LLMError(f"{self.provider}: transport error: {e}") from e

        text = "".join(parts)
        total = (time.perf_counter() - t0) * 1000.0
        yield "", LLMResult(
            text=text, provider=self.provider, model=self.model,
            completion_tokens=len(parts),
            ttft_ms=round(ttft, 2) if ttft is not None else None,
            total_ms=round(total, 2), finish_reason=finish,
        )

    def _parse(self, data: Dict[str, Any], elapsed_ms: float) -> LLMResult:
        choices = data.get("choices") or [{}]
        msg = choices[0].get("message") or {}
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return LLMResult(
            text=(msg.get("content") or "").strip(),
            provider=self.provider,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
            total_ms=round(elapsed_ms, 2),
            finish_reason=choices[0].get("finish_reason", ""),
        )

    async def aclose(self):
        await self._client.aclose()


class GroqLLM(BaseLLM):
    provider = "groq"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout_s: float = 8.0):
        super().__init__(
            model=model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            api_key=api_key or os.getenv("GROQ_API_KEY") or os.getenv("QROQ_API_KEY") or "",
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout_s=timeout_s,
        )

    def headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def payload(self, messages, temperature, max_tokens, stream) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": stream,
        }
        # gpt-oss models on Groq are reasoning models: hidden reasoning
        # tokens are drawn from the same max_tokens budget as the visible
        # answer. Left unset, Groq's default reasoning effort can consume
        # the entire (deliberately small, latency-budgeted) max_tokens
        # before any content is emitted, yielding finish_reason="length"
        # with zero content -- observed live on openai/gpt-oss-20b. "low"
        # keeps reasoning minimal so the small token budget goes to the
        # actual answer. Other Groq models ignore the unknown field.
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"
        return body


class SarvamLLM(BaseLLM):
    provider = "sarvam"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout_s: float = 8.0):
        super().__init__(
            model=model or os.getenv("SARVAM_LLM_MODEL", "sarvam-105b-conversations"),
            api_key=api_key or os.getenv("SARVAM_API_KEY") or "",
            base_url=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1"),
            timeout_s=timeout_s,
        )
        self.reasoning_effort = (os.getenv("SARVAM_REASONING_EFFORT", "none") or "none").lower()

    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

    def payload(self, messages, temperature, max_tokens, stream) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self.model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens, "stream": stream,
        }
        disabled = self.reasoning_effort in ("none", "null", "off", "false", "")
        body["reasoning_effort"] = None if disabled else self.reasoning_effort
        return body


_PROVIDERS = {"groq": GroqLLM, "sarvam": SarvamLLM}


def build_llm(provider: Optional[str] = None, timeout_s: float = 8.0,
              model: Optional[str] = None) -> BaseLLM:
    name = (provider or os.getenv("LLM_PROVIDER", "groq")).lower().strip()
    if name not in _PROVIDERS:
        raise LLMError(f"Unknown LLM_PROVIDER '{name}'. Options: {', '.join(_PROVIDERS)}")
    llm = _PROVIDERS[name](model=model, timeout_s=timeout_s)
    logger.info(f"LLM backend: {llm.provider} | model={llm.model} | endpoint={llm.endpoint}")
    return llm


def available_providers() -> Dict[str, bool]:
    return {
        "groq": bool(os.getenv("GROQ_API_KEY") or os.getenv("QROQ_API_KEY")),
        "sarvam": bool(os.getenv("SARVAM_API_KEY")),
    }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ok = fail = 0

    def check(label, cond):
        global ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {label}")
        else:
            fail += 1
            print(f"  FAIL  {label}")

    msgs = [{"role": "user", "content": "कॉर्पोरेशन क्या है?"}]

    print("\n-- request shaping --")
    g = GroqLLM(api_key="test-key")
    gp = g.payload(msgs, 0.0, 48, False)
    check("groq endpoint", g.endpoint.endswith("/chat/completions"))
    check("groq bearer auth", g.headers()["Authorization"].startswith("Bearer "))
    check("groq no reasoning field", "reasoning_effort" not in gp)

    s = SarvamLLM(api_key="test-key")
    sp = s.payload(msgs, 0.0, 48, False)
    check("sarvam conversational model", "conversations" in s.model)
    check("sarvam sends both auth headers",
          "api-subscription-key" in s.headers() and "Authorization" in s.headers())
    check("reasoning DISABLED by default", sp["reasoning_effort"] is None)
    check("max_tokens preserved", sp["max_tokens"] == 48)

    os.environ["SARVAM_REASONING_EFFORT"] = "low"
    s2 = SarvamLLM(api_key="test-key")
    check("reasoning can be re-enabled", s2.payload(msgs, 0.0, 48, False)["reasoning_effort"] == "low")
    os.environ.pop("SARVAM_REASONING_EFFORT")

    print("\n-- factory --")
    os.environ["GROQ_API_KEY"] = "k"
    check("factory builds groq", build_llm("groq").provider == "groq")
    os.environ["SARVAM_API_KEY"] = "k"
    check("factory builds sarvam", build_llm("sarvam").provider == "sarvam")
    try:
        build_llm("openai"); check("unknown provider rejected", False)
    except LLMError:
        check("unknown provider rejected", True)

    print(f"\n  {ok} passed, {fail} failed\n")
    sys.exit(1 if fail else 0)
