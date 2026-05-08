from datetime import datetime

from pydantic import BaseModel, Field

from interview_system.models.question import Question
from interview_system.models.turn import AnswerRecord


class InterviewSessionData(BaseModel):
    session_id: str
    questions: list[Question]
    answer_records: list[AnswerRecord] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime | None = None
