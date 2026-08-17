#!/usr/bin/env python3
"""
Sarvam AI Speech-to-Text (STT) Module (audio_stt.py)
===================================================
Designed for HH Goa 2026 Shortlisting Task 2: Sub-200ms Voice-Enabled Indic RAG System.

Key Engineering Features:
1. High-speed synchronous & asynchronous audio transcription interfaces.
2. Direct memory-byte processing without disk I/O bottlenecks.
3. Microsecond latency profiling for STT SLA tracking.
4. In-memory PCM 16-bit 16kHz WAV audio synthesis helper for automated testing.
5. Multi-format support (.wav, .mp3, .webm, .ogg) with MIME auto-detection.
"""

import os
import sys
import io
import time
import math
import struct
import wave
import logging
from typing import Optional, Dict, Any, Union
import requests
import httpx
from pydantic import BaseModel, Field
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

# Configure Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AudioSTT")


# ============================================================================
# DATA SCHEMAS
# ============================================================================

class TranscriptionResult(BaseModel):
    """Structured response from the Sarvam AI STT API."""
    transcript: str = Field(..., description="Transcribed Hindi text in Devanagari script")
    language_code: str = Field("hi-IN", description="Detected or requested language code")
    audio_duration_sec: Optional[float] = Field(None, description="Duration of the audio clip in seconds")
    stt_latency_ms: float = Field(0.0, description="Roundtrip API latency in milliseconds")
    status: str = Field("success", description="Status code ('success' or 'error')")
    error_message: Optional[str] = Field(None, description="Error description if status is error")


# ============================================================================
# AUDIO SYNTHESIS & UTILITY HELPERS
# ============================================================================

def create_dummy_wav_bytes(
    duration_sec: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    silence: bool = False
) -> bytes:
    """
    Synthesizes a valid in-memory PCM 16-bit Mono 16kHz WAV audio byte stream.
    Used for automated pipeline verification and latency ping tests.
    """
    num_samples = int(sample_rate * duration_sec)
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)        # 16-bit (2 bytes per sample)
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            if silence:
                sample_val = 0
            else:
                # Generate a 440Hz sine wave
                t = float(i) / sample_rate
                sample_val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * frequency * t))
            frames.extend(struct.pack("<h", sample_val))
            
        wav_file.writeframes(frames)

    return buffer.getvalue()


def infer_mime_type(filename: str) -> str:
    """Infers standard audio MIME type from file extension."""
    ext = os.path.splitext(filename)[1].lower()
    mapping = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/m4a"
    }
    return mapping.get(ext, "audio/wav")


# ============================================================================
# SARVAM AI SPEECH-TO-TEXT CLIENT
# ============================================================================

class SarvamSTTClient:
    """
    Dedicated high-performance client for Sarvam AI Speech-to-Text API.
    Supports both synchronous and asynchronous byte-level transcription.
    """

    API_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "saaras:v3",
        language_code: str = "hi-IN",
        timeout_sec: float = 15.0
    ):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Sarvam API Key missing. Please set 'SARVAM_API_KEY' in your .env file or pass it to SarvamSTTClient."
            )

        self.model = model
        self.language_code = language_code
        self.timeout_sec = timeout_sec

        self.headers = {
            "api-subscription-key": self.api_key
        }

        # Persistent HTTP connection session for connection pooling & low latency
        self._session = requests.Session()
        self._session.headers.update(self.headers)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Synchronously transcribes an in-memory audio byte array.

        Args:
            audio_bytes: Raw binary audio bytes (.wav, .mp3, .webm, etc.)
            filename: Virtual filename used in multipart/form-data upload.
            mime_type: Optional MIME content type. If omitted, inferred from filename.

        Returns:
            TranscriptionResult with Hindi transcript and latency metrics.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=0.0,
                status="error",
                error_message="Audio payload is empty."
            )

        mime = mime_type or infer_mime_type(filename)
        files = {
            "file": (filename, io.BytesIO(audio_bytes), mime)
        }
        data = {
            "model": self.model
        }
        if self.language_code:
            data["language_code"] = self.language_code

        t_start = time.perf_counter()
        try:
            response = self._session.post(
                self.API_URL,
                files=files,
                data=data,
                timeout=self.timeout_sec
            )
            stt_latency_ms = (time.perf_counter() - t_start) * 1000.0

            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Sarvam STT API Error (HTTP {response.status_code}): {error_text}")
                return TranscriptionResult(
                    transcript="",
                    language_code=self.language_code,
                    stt_latency_ms=round(stt_latency_ms, 2),
                    status="error",
                    error_message=f"HTTP {response.status_code}: {error_text}"
                )

            res_json = response.json()
            transcript = res_json.get("transcript", "").strip()
            lang_detected = res_json.get("language_code") or self.language_code

            return TranscriptionResult(
                transcript=transcript,
                language_code=lang_detected,
                stt_latency_ms=round(stt_latency_ms, 2),
                status="success"
            )

        except requests.exceptions.Timeout:
            stt_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=round(stt_latency_ms, 2),
                status="error",
                error_message="Request to Sarvam AI STT timed out."
            )
        except requests.exceptions.RequestException as e:
            stt_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=round(stt_latency_ms, 2),
                status="error",
                error_message=f"Network error connecting to Sarvam AI: {str(e)}"
            )

    async def transcribe_bytes_async(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        mime_type: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Asynchronously transcribes audio bytes using httpx.AsyncClient.
        Ideal for WebSocket servers and non-blocking asynchronous FastAPI endpoints.
        """
        if not audio_bytes or len(audio_bytes) == 0:
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=0.0,
                status="error",
                error_message="Audio payload is empty."
            )

        mime = mime_type or infer_mime_type(filename)
        files = {
            "file": (filename, audio_bytes, mime)
        }
        data = {
            "model": self.model
        }
        if self.language_code:
            data["language_code"] = self.language_code

        t_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout_sec) as client:
                response = await client.post(
                    self.API_URL,
                    headers=self.headers,
                    files=files,
                    data=data
                )
                stt_latency_ms = (time.perf_counter() - t_start) * 1000.0

                if response.status_code != 200:
                    return TranscriptionResult(
                        transcript="",
                        language_code=self.language_code,
                        stt_latency_ms=round(stt_latency_ms, 2),
                        status="error",
                        error_message=f"HTTP {response.status_code}: {response.text}"
                    )

                res_json = response.json()
                transcript = res_json.get("transcript", "").strip()
                lang_detected = res_json.get("language_code") or self.language_code

                return TranscriptionResult(
                    transcript=transcript,
                    language_code=lang_detected,
                    stt_latency_ms=round(stt_latency_ms, 2),
                    status="success"
                )

        except Exception as e:
            stt_latency_ms = (time.perf_counter() - t_start) * 1000.0
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=round(stt_latency_ms, 2),
                status="error",
                error_message=f"Async STT Error: {str(e)}"
            )

    def transcribe_file(self, file_path: str) -> TranscriptionResult:
        """
        Convenience method to load audio from a local file path and transcribe.
        """
        if not os.path.exists(file_path):
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=0.0,
                status="error",
                error_message=f"Audio file '{file_path}' not found."
            )

        try:
            with open(file_path, "rb") as f:
                audio_data = f.read()
            filename = os.path.basename(file_path)
            return self.transcribe_bytes(audio_data, filename=filename)
        except Exception as e:
            return TranscriptionResult(
                transcript="",
                language_code=self.language_code,
                stt_latency_ms=0.0,
                status="error",
                error_message=f"Failed to read file '{file_path}': {str(e)}"
            )


# ============================================================================
# STANDALONE CLI & BENCHMARK HARNESS
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sarvam AI Speech-to-Text (STT) Client & Benchmark Harness")
    parser.add_argument("--file", "-f", type=str, default=None, help="Path to local audio file to transcribe.")
    parser.add_argument("--model", "-m", type=str, default="saaras:v3", help="Sarvam AI STT model (default: 'saaras:v3').")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run latency benchmark using synthetic audio.")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  HH GOA 2026: SARVAM AI SPEECH-TO-TEXT (STT) INFERENCE HARNESS")
    print(f"  Model: {args.model} | Language: hi-IN | Target SLA: Ultra-Low Latency")
    print("=" * 70)

    try:
        client = SarvamSTTClient(model=args.model)
        print("SarvamSTTClient initialized successfully.\n")
    except ValueError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # 1. Test local audio file if provided (or default phase1 audio if exists)
    target_file = args.file
    if not target_file:
        phase1_audio = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "phase1", "test_audio.wav"))
        if os.path.exists(phase1_audio):
            target_file = phase1_audio

    if target_file and os.path.exists(target_file):
        print("-" * 70)
        print(f"Transcribing Local Audio File: '{target_file}'...")
        print("-" * 70)
        result = client.transcribe_file(target_file)
        
        print("\n--- TRANSCRIPTION RESULT ---")
        print(f"Status      : {result.status.upper()}")
        print(f"Language    : {result.language_code}")
        print(f"STT Latency : {result.stt_latency_ms} ms")
        if result.status == "success":
            print(f"Transcript  : \"{result.transcript}\"")
        else:
            print(f"Error       : {result.error_message}")
        print("-" * 70)

    # 2. Benchmark Ping Test with in-memory synthesized WAV
    if args.benchmark or not target_file:
        print("\n" + "-" * 70)
        print("Running Synthetic In-Memory Audio Ping Benchmark (1.0s WAV)...")
        print("-" * 70)
        dummy_wav = create_dummy_wav_bytes(duration_sec=1.0, sample_rate=16000)
        res_dummy = client.transcribe_bytes(dummy_wav, filename="synthetic_test.wav")
        
        print(f"Synthetic Audio Size : {len(dummy_wav)} bytes")
        print(f"API Latency          : {res_dummy.stt_latency_ms} ms")
        print(f"Status               : {res_dummy.status.upper()}")
        if res_dummy.transcript:
            print(f"Transcript           : \"{res_dummy.transcript}\"")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("STT CLIENT VERIFICATION COMPLETED.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
