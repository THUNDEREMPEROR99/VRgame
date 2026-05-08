from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from interview_system import EvaluationAgent, InterviewAgent, process_audio, warm_audio_models
from interview_system.config import create_chat_model_from_env
from interview_system.loaders import load_questions
from interview_system.models.evaluation import FinalEvaluationReport

from server.audio_files import media_path, save_agent_audio, save_upload
from server.schemas import AudioTurnResponse, HealthResponse, StartSessionResponse
from server.sessions import SessionState, add_session, get_session, replace_session

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = REPO_ROOT / "examples" / "questions.json"
LOCAL_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup_task = asyncio.create_task(asyncio.to_thread(warm_audio_models))
    warmup_task.add_done_callback(_log_warmup_failure)
    yield


def _log_warmup_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.warning("Audio model warm-up failed: %s", exc)


app = FastAPI(title="VR Interview Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CLIENT_ORIGINS", LOCAL_ORIGINS).split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/sessions/start", response_model=StartSessionResponse)
async def start_session() -> StartSessionResponse:
    state = _new_session()
    add_session(state)
    return await _start_state(state)


@app.post("/sessions/{session_id}/restart", response_model=StartSessionResponse)
async def restart_session(session_id: str) -> StartSessionResponse:
    _require_session(session_id)
    state = _new_session(session_id=session_id)
    replace_session(session_id, state)
    return await _start_state(state)


async def _start_state(state: SessionState) -> StartSessionResponse:
    interview = state.interview
    output = await interview.start()
    agent_text = _join_text(output.intro_text, output.next_question_text)
    audio_url = await _save_audio_response(state, agent_text)
    return StartSessionResponse(
        session_id=output.session_id,
        stage_status=output.stage_status,
        intro_text=output.intro_text,
        next_question_id=output.next_question_id,
        next_question_text=output.next_question_text,
        agent_response_text=agent_text,
        agent_audio_url=audio_url,
        is_complete=output.is_complete,
    )


@app.post("/sessions/{session_id}/turns/audio", response_model=AudioTurnResponse)
async def submit_audio_turn(session_id: str, file: UploadFile = File(...)) -> AudioTurnResponse:
    state = _require_session(session_id)
    _require_wav(file)
    audio_path = await save_upload(session_id, state.upload_index, file)
    state.upload_index += 1

    transcript, voice = await asyncio.to_thread(process_audio, str(audio_path))
    output = await state.interview.submit_answer(transcript, voice_analysis=voice)

    intro_text = None
    next_question_id = None
    next_question_text = None
    is_complete = output.is_complete
    stage_status = output.stage_status
    response_parts = [output.character_response]

    if output.stage_status == "ready_for_next":
        next_output = await state.interview.get_next_question()
        intro_text = next_output.intro_text
        next_question_id = next_output.next_question_id
        next_question_text = next_output.next_question_text
        is_complete = next_output.is_complete
        stage_status = next_output.stage_status if next_output.is_complete else output.stage_status
        response_parts.extend([next_output.intro_text, next_output.next_question_text, next_output.closing_message])

    agent_text = _join_text(*response_parts)
    audio_url = await _save_audio_response(state, agent_text)
    return AudioTurnResponse(
        session_id=session_id,
        transcript=transcript,
        voice_analysis=voice,
        stage_status=stage_status,
        character_response=output.character_response,
        intro_text=intro_text,
        next_question_id=next_question_id,
        next_question_text=next_question_text,
        agent_response_text=agent_text,
        agent_audio_url=audio_url,
        is_complete=is_complete,
    )


@app.get("/sessions/{session_id}/report", response_model=FinalEvaluationReport)
async def final_report(session_id: str) -> FinalEvaluationReport:
    state = _require_session(session_id)
    if not state.interview.is_complete():
        raise HTTPException(status_code=409, detail="interview is not complete")
    summaries = await state.interview.wait_for_summaries()
    return await state.evaluation.evaluate(state.interview.get_session_data(), summaries)


@app.get("/media/{session_id}/{filename}")
async def media(session_id: str, filename: str) -> FileResponse:
    path = media_path(session_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="media file not found")
    return FileResponse(path, media_type="audio/mpeg")


def _create_model():
    return create_chat_model_from_env()


def _new_session(session_id: str | None = None) -> SessionState:
    model = _create_model()
    questions = load_questions(QUESTIONS_PATH)
    interview = InterviewAgent(questions, session_id=session_id, model=model)
    evaluation = EvaluationAgent(model=model, session_id=interview.session_id)
    return SessionState(interview=interview, evaluation=evaluation)


def _require_session(session_id: str) -> SessionState:
    state = get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    return state


def _require_wav(file: UploadFile) -> None:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".wav" and file.content_type not in {"audio/wav", "audio/x-wav", "audio/wave"}:
        raise HTTPException(status_code=415, detail="only WAV uploads are supported")


async def _save_audio_response(state: SessionState, text: str) -> str:
    url = await save_agent_audio(state.interview.session_id, state.response_index, text)
    state.response_index += 1
    return url


def _join_text(*parts: str | None) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())
