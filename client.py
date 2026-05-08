"""
Voice client for the FastAPI interview server.

Run from repo root after starting the server:
  python client.py

Optional:
  $env:SERVER_URL="http://127.0.0.1:8000"

Hotkeys: K = start interview, SPACE = start/stop recording one clip, ESC = quit.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

import keyboard
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

RECORDINGS_ROOT = Path(__file__).resolve().parent / "recordings"
SERVER_URL = os.getenv("SERVER_URL", "http://127.0.0.1:8000").rstrip("/")
STOP_DEBOUNCE_S = 0.35
POLL_S = 0.03


def wait_for_key_edge(target_name: str, quit_event: threading.Event) -> bool:
    target = target_name.lower()
    prev = keyboard.is_pressed(target)
    while True:
        if quit_event.is_set():
            return False
        pressed = keyboard.is_pressed(target)
        if pressed and not prev:
            return True
        prev = pressed
        time.sleep(POLL_S)


def record_toggle_space(
    session_dir: Path,
    turn_index: int,
    quit_event: threading.Event,
    samplerate: int = 16000,
) -> Path | None:
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path = session_dir / f"turn_{turn_index:02d}.wav"
    chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(status, file=sys.stderr)
        chunks.append(indata.copy())

    print("Press SPACE to start recording...")
    if not wait_for_key_edge("space", quit_event):
        return None

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
                return None
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


def post_json(path: str) -> dict[str, Any]:
    response = requests.post(f"{SERVER_URL}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def get_json(path: str) -> dict[str, Any]:
    response = requests.get(f"{SERVER_URL}{path}", timeout=120)
    response.raise_for_status()
    return response.json()


def post_audio(session_id: str, wav_path: Path) -> dict[str, Any]:
    with wav_path.open("rb") as audio_file:
        files = {"file": (wav_path.name, audio_file, "audio/wav")}
        response = requests.post(
            f"{SERVER_URL}/sessions/{session_id}/turns/audio",
            files=files,
            timeout=600,
        )
    response.raise_for_status()
    return response.json()


def play_audio(audio_url: str | None) -> None:
    if not audio_url:
        return
    response = requests.get(f"{SERVER_URL}{audio_url}", timeout=120)
    response.raise_for_status()
    data, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
    sd.play(data, sample_rate)
    sd.wait()


def print_server_message(payload: dict[str, Any]) -> None:
    text = payload.get("agent_response_text")
    if text:
        print(f"\nInterviewer: {text}\n")


def main() -> None:
    print("=" * 56)
    print(f"Server voice interview - {SERVER_URL}")
    print("K: start | SPACE: record toggle | ESC: quit")
    print("=" * 56)

    quit_event = threading.Event()

    def on_esc() -> None:
        quit_event.set()
        print("\n(quit requested)")

    keyboard.add_hotkey("esc", on_esc, suppress=False)

    try:
        print("Press K to start the interview...\n")
        if not wait_for_key_edge("k", quit_event):
            print("Stopped by user.")
            return

        first = post_json("/sessions/start")
        session_id = first["session_id"]
        print(f"Session ID: {session_id}")

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = RECORDINGS_ROOT / f"client_{session_id}_{stamp}"

        print_server_message(first)
        play_audio(first.get("agent_audio_url"))

        record_turn = 0
        is_complete = bool(first.get("is_complete"))

        while not is_complete:
            if quit_event.is_set():
                print("Stopped by user.")
                return

            wav_path = record_toggle_space(session_dir, record_turn, quit_event)
            if wav_path is None:
                print("Stopped by user.")
                return
            record_turn += 1

            turn = post_audio(session_id, wav_path)
            print(f"\nYou said: {turn['transcript']}\n")
            print(f"Voice analysis: {json.dumps(turn['voice_analysis'], indent=2)}\n")

            print_server_message(turn)
            play_audio(turn.get("agent_audio_url"))
            is_complete = bool(turn.get("is_complete"))

        print("Interview finished. Requesting final report...\n")
        report = get_json(f"/sessions/{session_id}/report")
        print(json.dumps(report, indent=2))
    finally:
        keyboard.unhook_all()


if __name__ == "__main__":
    main()
