from typing import Literal

from pydantic import BaseModel, Field


class PerQuestionOverview(BaseModel):
    question_id: str
    score: int = Field(ge=0, le=100)
    summary: str


class FinalEvaluationReport(BaseModel):
    final_answer_quality_score: int = Field(ge=0, le=100)
    overall_score: int = Field(ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    per_question_overview: list[PerQuestionOverview] = Field(default_factory=list)
    recommendation: Literal["proceed", "hold", "reject", "needs_review"]
    final_summary: str
