"""Text-to-speech for the interviewer using edge-tts.

Synthesizes an MP3 per utterance via Microsoft Edge online neural voices, then
plays it locally with ``sounddevice``. No API key required, no ffmpeg required
(MP3 is decoded in-process by ``soundfile``).
"""

from __future__ import annotations

import asyncio
import io

import edge_tts  # type: ignore[import-untyped]
try:
    import sounddevice as sd  # type: ignore[import-untyped]
except (ImportError, OSError):
    sd = None  # server environment has no audio output device; speak() is a no-op
import soundfile as sf  # type: ignore[import-untyped]

_VOICE = "en-US-AriaNeural"


async def synthesize_mp3(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, _VOICE)
    buf = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.extend(chunk["data"])
    return bytes(buf)


async def speak_async(text: str) -> None:
    if not text or not text.strip():
        return
    mp3_bytes = await synthesize_mp3(text)
    if sd is None:
        # Server environment — synthesize only, don't play locally.
        return
    data, sample_rate = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()


def speak(text: str) -> None:
    asyncio.run(speak_async(text))


__all__ = ["speak", "speak_async", "synthesize_mp3"]
