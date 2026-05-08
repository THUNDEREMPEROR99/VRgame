from pydantic import BaseModel, Field

from interview_system.models.turn import AnswerRecord


class QuestionSummary(BaseModel):
    question_index: int
    question_id: str
    concise_summary: str
    detected_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    answer_quality_hint: int = Field(ge=0, le=10)


class SummaryInput(BaseModel):
    session_id: str
    question_index: int
    question_id: str
    question_text: str
    expected_signals: list[str] = Field(default_factory=list)
    stage_transcript: list[AnswerRecord] = Field(default_factory=list)
