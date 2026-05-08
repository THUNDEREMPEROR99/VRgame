from pathlib import Path

from fastapi import UploadFile

from interview_system.tts import synthesize_mp3

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_ROOT = REPO_ROOT / "recordings"


def session_dir(session_id: str) -> Path:
    path = RECORDINGS_ROOT / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload(session_id: str, index: int, file: UploadFile) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    out_path = session_dir(session_id) / f"user_{index:03d}{suffix or '.wav'}"
    out_path.write_bytes(await file.read())
    return out_path


async def save_agent_audio(session_id: str, index: int, text: str) -> str:
    filename = f"agent_{index:03d}.mp3"
    out_path = session_dir(session_id) / filename
    out_path.write_bytes(await synthesize_mp3(text))
    return f"/media/{session_id}/{filename}"


def media_path(session_id: str, filename: str) -> Path | None:
    if filename != Path(filename).name:
        return None
    path = session_dir(session_id) / filename
    if not path.exists() or not path.is_file():
        return None
    return path
