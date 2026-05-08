from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from interview_system.models.voice_analysis import VoiceAnalysis


class InterviewOutput(BaseModel):
    session_id: str
    character_response: str | None = None
    next_question_text: str | None = None
    next_question_id: str | None = None
    intro_text: str | None = None
    stage_status: Literal["asking", "continue_stage", "ready_for_next", "complete"]
    is_complete: bool = False
    closing_message: str | None = None
    debug_events: list[dict[str, object]] = Field(default_factory=list)


class AnswerRecord(BaseModel):
    turn_index: int
    stage_turn_index: int
    question_id: str
    question_text: str
    answer_text: str
    character_response: str
    timestamp: datetime
    voice_analysis: VoiceAnalysis | None = None


class StageDecision(BaseModel):
    status: Literal["continue_stage", "ready_for_next"]
    clarification_focus: str | None = None
    reason: str | None = None
