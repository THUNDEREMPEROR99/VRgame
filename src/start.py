"""
Voice-driven interview entrypoint.

Run from repo root (PowerShell):
  $env:PYTHONPATH="src"; python src/start.py

Hotkeys: K = start interview, SPACE = start/stop recording one clip, ESC = quit.

Recordings: recordings/<session_id>_<YYYYMMDD-HHMMSS>/turn_NN.wav
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import keyboard  # noqa: E402
import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from interview_system import EvaluationAgent, InterviewAgent, analyze_voice, transcribe_audio  # noqa: E402
from interview_system.config import ModelConfig, create_chat_model  # noqa: E402
from interview_system.loaders import load_questions  # noqa: E402
from interview_system.tts import speak_async  # noqa: E402

RECORDINGS_ROOT = _REPO_ROOT / "recordings"
STOP_DEBOUNCE_S = 0.35
POLL_S = 0.03


class InterviewQuit(Exception):
    """User pressed ESC (or aborted)."""


def wait_for_key_edge(target_name: str, quit_event: threading.Event) -> None:
    """Block until target key goes from released->pressed, or quit."""
    target = target_name.lower()
    prev = keyboard.is_pressed(target)
    while True:
        if quit_event.is_set():
            raise InterviewQuit
        pressed = keyboard.is_pressed(target)
        if pressed and not prev:
            return
        prev = pressed
        time.sleep(POLL_S)


def record_toggle_space(
    session_dir: Path,
    turn_index: int,
    quit_event: threading.Event,
    samplerate: int = 16000,
) -> Path:
    """Wait SPACE edge (start), record mic, wait SPACE edge (stop). Writes mono int16 WAV."""
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / f"turn_{turn_index:02d}.wav"
    chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    print("Press SPACE to start recording...")
    wait_for_key_edge("space", quit_event)

    print("(recording...) Press SPACE again to stop.")
    stream_start = time.monotonic()

    with sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype="int16",
        callback=callback,
    ):
        prev_space = keyboard.is_pressed("space")
        while True:
            if quit_event.is_set():
                raise InterviewQuit
            pressed = keyboard.is_pressed("space")
            edge = pressed and not prev_space
            prev_space = pressed
            if edge and time.monotonic() - stream_start >= STOP_DEBOUNCE_S:
                break
            time.sleep(POLL_S)

    if not chunks:
        raise RuntimeError("No audio captured")

    audio = np.concatenate(chunks, axis=0)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())

    print(f"Saved: {out_path}")
    return out_path


async def main() -> None:
    load_dotenv()

    print("=" * 48)
    print("Voice interview — K: start | SPACE: record toggle | ESC: quit")
    print("=" * 48)

    quit_event = threading.Event()

    def on_esc() -> None:
        quit_event.set()
        print("\n(quit requested)")

    keyboard.add_hotkey("esc", on_esc, suppress=False)

    try:
        questions_path = _REPO_ROOT / "examples" / "questions.json"
        questions = load_questions(questions_path)

        model = create_chat_model(
            ModelConfig(
                model=os.environ["OPENROUTER_MODEL"],
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            )
        )
        interview = InterviewAgent(questions, model=model)
        evaluation = EvaluationAgent(model=model, session_id=interview.session_id)

        print(f"Session ID: {interview.session_id}")
        print("Press K to start the interview...\n")

        await asyncio.to_thread(wait_for_key_edge, "k", quit_event)

        RECORDINGS_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = RECORDINGS_ROOT / f"{interview.session_id}_{stamp}"

        first = await interview.start()
        if first.intro_text:
            print(f"\nInterviewer: {first.intro_text}\n")
            await speak_async(first.intro_text)
        if first.next_question_text:
            print(f"\nQuestion: {first.next_question_text}\n")
            await speak_async(first.next_question_text)

        record_turn = 0

        while not interview.is_complete():
            if quit_event.is_set():
                raise InterviewQuit

            wav_path = await asyncio.to_thread(
                record_toggle_space,
                session_dir,
                record_turn,
                quit_event,
                16000,
            )
            record_turn += 1

            transcript = await asyncio.to_thread(transcribe_audio, str(wav_path))
            voice = await asyncio.to_thread(analyze_voice, str(wav_path))
            print(f"\nYou said: {transcript}\n")
            print(f"Voice analysis: {voice.model_dump_json(indent=2)}\n")

            result = await interview.submit_answer(transcript, voice_analysis=voice)

            if result.character_response:
                print(f"Interviewer: {result.character_response}\n")
                await speak_async(result.character_response)

            if result.stage_status == "ready_for_next":
                nxt = await interview.get_next_question()
                if nxt.is_complete:
                    print("Interview finished (no more questions).\n")
                    break
                if nxt.intro_text:
                    print(f"\nInterviewer: {nxt.intro_text}\n")
                    await speak_async(nxt.intro_text)
                if nxt.next_question_text:
                    print(f"\nQuestion: {nxt.next_question_text}\n")
                    await speak_async(nxt.next_question_text)

        summaries = await interview.wait_for_summaries()
        report = await evaluation.evaluate(interview.get_session_data(), summaries)
        print(json.dumps(report.model_dump(), indent=2))
    except InterviewQuit:
        print("Stopped by user.")
    finally:
        keyboard.unhook_all()


if __name__ == "__main__":
    asyncio.run(main())
