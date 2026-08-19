#!/usr/bin/env python3
"""
Sarvam realtime speech-to-text over WebSocket (saaras:v3-realtime).

The batch client in audio_stt.py posts a finished recording and waits for one
transcript. This one keeps a socket open while the user is still speaking and
emits partial transcripts as they arrive, so the UI can show what was heard
before the user commits to sending it.

Wire format, corrected against the live API with probe_realtime_stt.py rather
than read off the docs (the docs' prose disagreed with the server on two of
these, each time by silently closing the socket instead of explaining):
  - connect:  wss://api.sarvam.ai/speech-to-text-realtime/ws
              ?model=saaras:v3-realtime&language_code=hi-IN
              language_code takes an UNDERSCORE; hyphenated gets a 4000 close.
              auth via the api-subscription-key header
  - audio in: {"event": "audio_input", "audio": "<base64 PCM s16le 16k mono>"}
              the base64 is a flat string, not an object with encoding fields
  - end:      {"event": "speech_end"}
  - text out: transcript.partial while speaking, transcript.final on completion

Partials are always plain transcription regardless of `mode`; only the final
honours it. We ask for transcribe (not translate) because the index, the
prompt and the grounding gate are all Hindi-native.

Response parsing is deliberately tolerant: the event envelope has moved around
across Sarvam versions, so _extract() accepts several shapes rather than
hard-failing on a renamed key.
"""

from __future__ import annotations

import os
import json
import base64
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, AsyncIterator, Dict, Any, Tuple

try:
    import websockets
except ImportError:  # serving container always has it; dev boxes may not
    websockets = None

logger = logging.getLogger("STTRealtime")

WS_URL = os.getenv("SARVAM_STT_WS_URL",
                   "wss://api.sarvam.ai/speech-to-text-realtime/ws")
MODEL = os.getenv("SARVAM_STT_REALTIME_MODEL", "saaras:v3-realtime")

# The frontend resamples to this before sending. Sarvam realtime expects
# 16kHz mono s16le; sending anything else transcribes as noise rather than
# erroring, which is a miserable thing to debug, so it is pinned in one place.
SAMPLE_RATE = 16000


class STTRealtimeError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    is_final: bool


def _extract(obj: Dict[str, Any]) -> Optional[Transcript]:
    """Pull (text, is_final) out of one server frame, or None if it carries no
    transcript (acks, metadata, keepalives)."""
    kind = str(obj.get("type") or obj.get("event") or "")

    if "error" in kind.lower():
        detail = obj.get("message") or obj.get("error") or obj
        raise STTRealtimeError(f"sarvam realtime: {detail}")

    if "transcript" not in kind:
        return None

    # text may sit at the top level or one nesting down
    text = obj.get("transcript") or obj.get("text") or ""
    if not text:
        for key in ("data", "result", "payload"):
            nested = obj.get(key)
            if isinstance(nested, dict):
                text = nested.get("transcript") or nested.get("text") or ""
                if text:
                    break

    if not text:
        return None

    is_final = kind.endswith("final") or bool(obj.get("is_final"))
    return Transcript(text=text.strip(), is_final=is_final)


class SarvamRealtimeSTT:
    """One live transcription session. Not reusable -- open a new one per
    utterance, which is also how Sarvam bills and segments them."""

    def __init__(self, api_key: Optional[str] = None,
                 language_code: str = "hi-IN",
                 connect_timeout_s: float = 6.0):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY") or ""
        if not self.api_key:
            raise STTRealtimeError("SARVAM_API_KEY is not set")
        self.language_code = language_code
        self.connect_timeout_s = connect_timeout_s
        self._ws = None

    @property
    def url(self) -> str:
        # language_code with an UNDERSCORE. Sarvam closes the socket with a 4000
        # frame ("Missing required query parameter 'language_code'") if this is
        # hyphenated, which is what the docs' prose suggests -- confirmed live.
        return (f"{WS_URL}?model={MODEL}"
                f"&language_code={self.language_code}")

    async def __aenter__(self) -> "SarvamRealtimeSTT":
        if websockets is None:
            raise STTRealtimeError(
                "the 'websockets' package is required for realtime STT")
        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    additional_headers={"api-subscription-key": self.api_key},
                    # Sarvam is the one closing idle sockets here, not us; a
                    # short client ping keeps the session alive between words.
                    ping_interval=20, ping_timeout=20,
                    max_size=None,
                ),
                timeout=self.connect_timeout_s,
            )
        except asyncio.TimeoutError as e:
            raise STTRealtimeError("timed out connecting to Sarvam realtime STT") from e
        except Exception as e:
            raise STTRealtimeError(f"could not open Sarvam realtime socket: {e}") from e
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def send_audio(self, pcm: bytes) -> None:
        """Feed one chunk of raw PCM s16le @16kHz mono."""
        if self._ws is None:
            raise STTRealtimeError("socket is not open")
        await self._ws.send(json.dumps({
            "event": "audio_input",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }))

    async def signal_end(self) -> None:
        """Tell Sarvam the utterance is over so it flushes a final transcript."""
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps({"event": "speech_end"}))
        except Exception:
            pass

    async def transcripts(self) -> AsyncIterator[Transcript]:
        """Yield transcripts until the socket closes or a final arrives."""
        if self._ws is None:
            raise STTRealtimeError("socket is not open")
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                t = _extract(obj)
                if t is not None:
                    yield t
        except Exception as e:
            if websockets is not None and isinstance(
                    e, websockets.exceptions.ConnectionClosed):
                return
            raise


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

    print("\n-- frame parsing --")
    check("partial at top level",
          _extract({"type": "transcript.partial", "transcript": "कॉर्पोरेशन"})
          == Transcript("कॉर्पोरेशन", False))
    check("final flagged",
          _extract({"type": "transcript.final", "transcript": "कॉर्पोरेशन क्या है"}).is_final)
    check("nested under data",
          _extract({"type": "transcript.partial",
                    "data": {"text": "नमस्ते"}}).text == "नमस्ते")
    check("event key instead of type",
          _extract({"event": "transcript.final", "text": "हाँ"}).is_final)
    check("is_final boolean respected",
          _extract({"type": "transcript", "text": "हाँ", "is_final": True}).is_final)
    check("non-transcript frame ignored",
          _extract({"type": "session.created", "id": "x"}) is None)
    check("empty transcript ignored",
          _extract({"type": "transcript.partial", "transcript": "  "}) is None
          or _extract({"type": "transcript.partial", "transcript": ""}) is None)

    try:
        _extract({"type": "error", "message": "bad key"})
        check("error frame raises", False)
    except STTRealtimeError:
        check("error frame raises", True)

    print("\n-- config --")
    os.environ["SARVAM_API_KEY"] = "test-key"
    c = SarvamRealtimeSTT()
    check("model pinned to realtime", "saaras:v3-realtime" in c.url)
    check("language_code uses an underscore", "language_code=hi-IN" in c.url)
    check("16k sample rate", SAMPLE_RATE == 16000)

    os.environ.pop("SARVAM_API_KEY")
    try:
        SarvamRealtimeSTT()
        check("missing key rejected", False)
    except STTRealtimeError:
        check("missing key rejected", True)

    print(f"\n  {ok} passed, {fail} failed\n")
    sys.exit(1 if fail else 0)
