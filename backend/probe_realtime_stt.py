#!/usr/bin/env python3
"""
Throwaway probe: does Sarvam's realtime STT socket accept the frames that
stt_realtime.py sends?

This exists because the send-side event names are the one part of the realtime
integration that could not be verified from the docs alone, and a wrong name
fails SILENTLY (no transcripts ever arrive) rather than loudly. Running this is
much cheaper than discovering it through a container rebuild.

Needs only SARVAM_API_KEY -- no index, no Qdrant, no server.

    pip install websockets
    python probe_realtime_stt.py                # synthetic tone
    python probe_realtime_stt.py speech.wav     # real 16k mono wav, better test

A real recording of Hindi speech is the far stronger test: the tone confirms
only that the envelope is accepted, while speech confirms transcripts actually
come back.
"""

import os
import sys
import wave
import math
import struct
import asyncio
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s | %(message)s")

from stt_realtime import SarvamRealtimeSTT, STTRealtimeError, SAMPLE_RATE


def load_pcm(path: str) -> bytes:
    """Read a wav as raw s16le mono @16k, complaining if it is not."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getframerate() != SAMPLE_RATE or w.getsampwidth() != 2:
            raise SystemExit(
                f"{path}: need mono 16-bit {SAMPLE_RATE}Hz, got "
                f"{w.getnchannels()}ch {w.getsampwidth()*8}-bit {w.getframerate()}Hz.\n"
                f"Convert with:  ffmpeg -i {path} -ac 1 -ar 16000 -sample_fmt s16 out.wav")
        return w.readframes(w.getnframes())


def tone_pcm(seconds: float = 2.0) -> bytes:
    out = bytearray()
    for i in range(int(SAMPLE_RATE * seconds)):
        v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        out += struct.pack("<h", v)
    return bytes(out)


async def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    pcm = load_pcm(path) if path else tone_pcm()
    label = path or "synthetic 440Hz tone (expect no words, just no errors)"

    print(f"\nsource : {label}")
    print(f"bytes  : {len(pcm)} ({len(pcm)/2/SAMPLE_RATE:.1f}s)")

    try:
        stt = await SarvamRealtimeSTT().__aenter__()
    except STTRealtimeError as e:
        print(f"\nCONNECT FAILED: {e}")
        print("If this is a 401/403 the key is wrong. If it is a 404 the URL or "
              "model name is wrong -- check SARVAM_STT_WS_URL.")
        return 1

    print("connected. streaming...\n")
    seen = []

    async def listen():
        try:
            async for t in stt.transcripts():
                tag = "FINAL  " if t.is_final else "partial"
                print(f"  {tag} {t.text!r}")
                seen.append(t)
        except STTRealtimeError as e:
            print(f"\n  SERVER ERROR FRAME: {e}")
            print("  ^ this is the useful failure -- it usually names the field "
                  "or event it did not like.")

    pump = asyncio.create_task(listen())

    # 100ms chunks, paced in real time the way the browser will send them.
    step = SAMPLE_RATE // 10 * 2
    try:
        for i in range(0, len(pcm), step):
            await stt.send_audio(pcm[i:i + step])
            await asyncio.sleep(0.1)
        await stt.signal_end()
    except Exception as e:
        # A server-side reject closes the socket mid-send. The listener task
        # has already printed the reason, which is the useful part.
        print(f"\n  socket closed while sending: {type(e).__name__}")
    try:
        await asyncio.wait_for(pump, timeout=6.0)
    except asyncio.TimeoutError:
        pump.cancel()
    await stt.close()

    print(f"\n{len(seen)} transcript frame(s) received.")
    if seen:
        print("SEND FORMAT IS CORRECT -- stt_realtime.py needs no change.")
    elif path:
        print("NO TRANSCRIPTS from real speech. The send format is very likely "
              "wrong.\nTry swapping the event name in stt_realtime.py "
              "send_audio(): 'audio' -> 'audio_input', and signal_end(): "
              "'stop' -> 'speech_end'.")
    else:
        print("No transcripts, but a pure tone has no words in it, so this is "
              "inconclusive.\nRe-run with a real recording of speech to get a "
              "definitive answer.")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.exit(asyncio.run(main()))
