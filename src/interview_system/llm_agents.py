from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from interview_system.models.evaluation import FinalEvaluationReport
from interview_system.models.question import Question
from interview_system.models.session import InterviewSessionData
from interview_system.models.summary import QuestionSummary, SummaryInput
from interview_system.models.turn import AnswerRecord, StageDecision
from interview_system.prompts import (
    CHARACTER_RESPONSE_SYSTEM_PROMPT,
    FINAL_EVALUATION_SYSTEM_PROMPT,
    QUESTION_INTRO_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    STAGE_DECISION_SYSTEM_PROMPT,
    character_response_message,
    final_evaluation_message,
    question_intro_message,
    stage_decision_message,
    summary_message,
)


def _extract_message_debug(message: Any) -> dict[str, Any]:
    if message is None:
        return {}
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    return {
        "reasoning_content": additional_kwargs.get("reasoning_content"),
        "reasoning_details": additional_kwargs.get("reasoning_details"),
        "reasoning_summaries": _extract_reasoning_summaries(additional_kwargs.get("reasoning_details")),
        "response_metadata": getattr(message, "response_metadata", {}) or {},
        "usage_metadata": getattr(message, "usage_metadata", None),
    }


def _extract_reasoning_summaries(reasoning_details: Any) -> list[str]:
    if not isinstance(reasoning_details, list):
        return []
    summaries = []
    for item in reasoning_details:
        if isinstance(item, dict) and item.get("type") == "reasoning.summary" and item.get("summary"):
            summaries.append(item["summary"])
    return summaries


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts) if parts else ""
    return str(content) if content is not None else ""


def _parse_tool_response(message: Any, model_cls: type) -> Any:
    """Extract structured output from a tool call or JSON content."""
    # Prefer tool_calls when present
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        return model_cls.model_validate(tool_calls[0]["args"])

    # Fallback: parse content as JSON
    content = _extract_text_content(getattr(message, "content", None))
    content = content.strip()
    if content:
        # Sometimes models wrap JSON in markdown code blocks
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()
        if content:
            return model_cls.model_validate_json(content)

    raise ValueError(f"Model returned empty content and no tool_calls for {model_cls.__name__}")


@dataclass(slots=True)
class StageDecisionRunner:
    _model: Any
    last_debug: dict[str, Any] | None = None

    async def ainvoke(self, question: Question, stage_transcript: list[AnswerRecord]) -> StageDecision:
        messages = [
            SystemMessage(content=STAGE_DECISION_SYSTEM_PROMPT),
            HumanMessage(content=stage_decision_message(question, stage_transcript)),
        ]
        result = await self._model.ainvoke(messages)
        self.last_debug = _extract_message_debug(result)
        return _parse_tool_response(result, StageDecision)


@dataclass(slots=True)
class CharacterResponseRunner:
    _model: Any
    last_debug: dict[str, Any] | None = None

    async def ainvoke(self, question: Question, decision: StageDecision, stage_transcript: list[AnswerRecord]) -> str:
        messages = [
            SystemMessage(content=CHARACTER_RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=character_response_message(question, decision, stage_transcript)),
        ]
        result = await self._model.ainvoke(messages)
        self.last_debug = _extract_message_debug(result)
        return _extract_text_content(result.content)


@dataclass(slots=True)
class IntroRunner:
    _model: Any
    last_debug: dict[str, Any] | None = None

    async def ainvoke(
        self,
        question_text: str,
        is_first: bool,
        prior_question_text: str | None,
    ) -> str:
        messages = [
            SystemMessage(content=QUESTION_INTRO_SYSTEM_PROMPT),
            HumanMessage(content=question_intro_message(question_text, is_first, prior_question_text)),
        ]
        result = await self._model.ainvoke(messages)
        self.last_debug = _extract_message_debug(result)
        return _extract_text_content(result.content).strip()


@dataclass(slots=True)
class SummaryRunner:
    _model: Any
    last_debug: dict[str, Any] | None = None

    async def ainvoke(self, item: SummaryInput) -> QuestionSummary:
        messages = [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=summary_message(item)),
        ]
        result = await self._model.ainvoke(messages)
        self.last_debug = _extract_message_debug(result)
        return _parse_tool_response(result, QuestionSummary)


@dataclass(slots=True)
class FinalEvaluationRunner:
    _model: Any
    last_debug: dict[str, Any] | None = None

    async def ainvoke(self, session_data: InterviewSessionData, summaries: list[QuestionSummary]) -> FinalEvaluationReport:
        messages = [
            SystemMessage(content=FINAL_EVALUATION_SYSTEM_PROMPT),
            HumanMessage(content=final_evaluation_message(session_data, summaries)),
        ]
        result = await self._model.ainvoke(messages)
        self.last_debug = _extract_message_debug(result)
        return _parse_tool_response(result, FinalEvaluationReport)


def create_stage_decision_runner(model: Any) -> StageDecisionRunner:
    return StageDecisionRunner(_model=model.bind_tools([StageDecision], tool_choice="auto"))


def create_character_response_runner(model: Any) -> CharacterResponseRunner:
    return CharacterResponseRunner(_model=model)


def create_intro_runner(model: Any) -> IntroRunner:
    return IntroRunner(_model=model)


def create_summary_runner(model: Any) -> SummaryRunner:
    return SummaryRunner(_model=model.bind_tools([QuestionSummary], tool_choice="auto"))


def create_final_evaluation_runner(model: Any) -> FinalEvaluationRunner:
    return FinalEvaluationRunner(_model=model.bind_tools([FinalEvaluationReport], tool_choice="auto"))


__all__ = [
    "CharacterResponseRunner",
    "FinalEvaluationRunner",
    "IntroRunner",
    "StageDecisionRunner",
    "SummaryRunner",
    "create_character_response_runner",
    "create_final_evaluation_runner",
    "create_intro_runner",
    "create_stage_decision_runner",
    "create_summary_runner",
]
