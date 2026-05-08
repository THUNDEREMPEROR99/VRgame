from typing import Literal

from pydantic import BaseModel

from interview_system.models.voice_analysis import VoiceAnalysis


class HealthResponse(BaseModel):
    status: Literal["ok"]


class StartSessionResponse(BaseModel):
    session_id: str
    stage_status: str
    intro_text: str | None = None
    next_question_id: str | None = None
    next_question_text: str | None = None
    agent_response_text: str
    agent_audio_url: str
    is_complete: bool


class AudioTurnResponse(BaseModel):
    session_id: str
    transcript: str
    voice_analysis: VoiceAnalysis
    stage_status: str
    character_response: str | None = None
    intro_text: str | None = None
    next_question_id: str | None = None
    next_question_text: str | None = None
    agent_response_text: str
    agent_audio_url: str
    is_complete: bool
