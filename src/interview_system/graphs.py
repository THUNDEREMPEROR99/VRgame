from __future__ import annotations

from datetime import datetime
from typing import Annotated, TypedDict

from langgraph.channels import EphemeralValue
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from interview_system.agents.summary_worker import SummaryWorker
from interview_system.llm_agents import CharacterResponseRunner, FinalEvaluationRunner, StageDecisionRunner
from interview_system.models.evaluation import FinalEvaluationReport
from interview_system.models.question import Question
from interview_system.models.session import InterviewSessionData
from interview_system.models.summary import QuestionSummary, SummaryInput
from interview_system.models.turn import AnswerRecord, StageDecision
from interview_system.models.voice_analysis import VoiceAnalysis


class InterviewGraphState(TypedDict):
    session_id: str
    questions: list[dict]
    current_index: int
    user_answer: str
    turn_index: int
    stage_turn_index: int
    current_stage_records: list[dict]
    answer_records: list[dict]
    stage_decision: dict | None
    character_response: str | None
    stage_status: str
    next_question_text: str | None
    next_question_id: str | None
    is_complete: bool
    is_skip: bool
    voice_analysis: dict | None
    validation_error: Annotated[str | None, EphemeralValue(str)]


class EvaluationGraphState(TypedDict):
    session_data: dict
    summaries: list[dict]
    final_report: dict | None
    validation_error: Annotated[str | None, EphemeralValue(str)]


def _has_validation_error(state: InterviewGraphState | EvaluationGraphState) -> str:
    if state.get("validation_error"):
        return "invalid"
    return "valid"


def build_interview_graph(
    stage_decision_runner: StageDecisionRunner,
    character_response_runner: CharacterResponseRunner,
    summary_worker: SummaryWorker,
    checkpointer: InMemorySaver | None = None,
    store: InMemoryStore | None = None,
):
    async def validate_answer(state: InterviewGraphState) -> dict:
        if not state["user_answer"].strip():
            return {"validation_error": "answer text is required"}
        return {"validation_error": None}

    async def decide_stage_status(state: InterviewGraphState) -> dict:
        if state.get("is_skip"):
            skip_decision = StageDecision(status="ready_for_next")
            return {"stage_decision": skip_decision.model_dump(mode="json"), "stage_status": "ready_for_next"}

        writer = get_stream_writer()
        writer("Deciding stage status...")
        question = _current_question(state)
        draft_record = _draft_answer_record(state, question)
        transcript = [*_records_from_state(state["current_stage_records"]), draft_record]
        decision = await stage_decision_runner.ainvoke(question, transcript)
        return {"stage_decision": decision.model_dump(mode="json"), "stage_status": decision.status}

    async def generate_character_response(state: InterviewGraphState) -> dict:
        writer = get_stream_writer()
        writer("Generating character response...")
        question = _current_question(state)
        draft_record = _draft_answer_record(state, question)
        transcript = [*_records_from_state(state["current_stage_records"]), draft_record]
        response = await character_response_runner.ainvoke(
            question,
            StageDecision.model_validate(state["stage_decision"]),
            transcript,
        )
        return {"character_response": response}

    async def record_answer(state: InterviewGraphState) -> dict:
        question = _current_question(state)
        voice_analysis = (
            VoiceAnalysis.model_validate(state["voice_analysis"])
            if state.get("voice_analysis")
            else None
        )
        record = AnswerRecord(
            turn_index=state["turn_index"],
            stage_turn_index=state["stage_turn_index"],
            question_id=question.id,
            question_text=question.text,
            answer_text=state["user_answer"],
            character_response=state["character_response"] or "",
            timestamp=datetime.now(),
            voice_analysis=voice_analysis,
        )
        answer_records = [*state["answer_records"], record.model_dump(mode="json")]
        current_stage_records = [*state["current_stage_records"], record.model_dump(mode="json")]
        return {
            "answer_records": answer_records,
            "current_stage_records": current_stage_records,
        }

    async def queue_summary_if_stage_complete(state: InterviewGraphState) -> dict:
        if state["stage_status"] != "ready_for_next":
            return {}

        question = _current_question(state)
        summary_worker.enqueue(
            SummaryInput(
                session_id=state["session_id"],
                question_index=state["current_index"],
                question_id=question.id,
                question_text=question.text,
                expected_signals=question.expected_signals,
                stage_transcript=_records_from_state(state["current_stage_records"]),
            )
        )
        return {}

    builder = StateGraph(InterviewGraphState)
    builder.add_node("validate_answer", validate_answer)
    builder.add_node("decide_stage_status", decide_stage_status)
    builder.add_node("generate_character_response", generate_character_response)
    builder.add_node("record_answer", record_answer)
    builder.add_node("queue_summary_if_stage_complete", queue_summary_if_stage_complete)
    builder.add_edge(START, "validate_answer")
    builder.add_conditional_edges("validate_answer", _has_validation_error, {"valid": "decide_stage_status", "invalid": END})
    builder.add_edge("decide_stage_status", "generate_character_response")
    builder.add_edge("generate_character_response", "record_answer")
    builder.add_edge("record_answer", "queue_summary_if_stage_complete")
    builder.add_edge("queue_summary_if_stage_complete", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(), store=store)


def build_evaluation_graph(
    final_evaluation_runner: FinalEvaluationRunner,
    checkpointer: InMemorySaver | None = None,
    store: InMemoryStore | None = None,
):
    async def validate_inputs(state: EvaluationGraphState) -> dict:
        session_data = InterviewSessionData.model_validate(state["session_data"])
        summaries = _summaries_from_state(state["summaries"])
        answered_question_ids = {
            record.question_id for record in session_data.answer_records
        }
        summary_question_ids = {summary.question_id for summary in summaries}
        if answered_question_ids != summary_question_ids:
            return {"validation_error": "summaries are missing for answered questions"}
        return {"validation_error": None}

    async def analyze_answer_quality(state: EvaluationGraphState) -> dict:
        sorted_summaries = sorted(_summaries_from_state(state["summaries"]), key=lambda item: item.question_index)
        return {"summaries": [summary.model_dump(mode="json") for summary in sorted_summaries]}

    async def generate_final_report(state: EvaluationGraphState) -> dict:
        writer = get_stream_writer()
        writer("Generating final evaluation report...")
        report = await final_evaluation_runner.ainvoke(
            InterviewSessionData.model_validate(state["session_data"]),
            _summaries_from_state(state["summaries"]),
        )
        return {"final_report": report.model_dump(mode="json")}

    builder = StateGraph(EvaluationGraphState)
    builder.add_node("validate_inputs", validate_inputs)
    builder.add_node("analyze_answer_quality", analyze_answer_quality)
    builder.add_node("generate_final_report", generate_final_report)
    builder.add_edge(START, "validate_inputs")
    builder.add_conditional_edges("validate_inputs", _has_validation_error, {"valid": "analyze_answer_quality", "invalid": END})
    builder.add_edge("analyze_answer_quality", "generate_final_report")
    builder.add_edge("generate_final_report", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver(), store=store)


def _current_question(state: InterviewGraphState) -> Question:
    return Question.model_validate(state["questions"][state["current_index"]])


def _records_from_state(records: list[dict]) -> list[AnswerRecord]:
    return [AnswerRecord.model_validate(record) for record in records]


def _summaries_from_state(summaries: list[dict]) -> list[QuestionSummary]:
    return [QuestionSummary.model_validate(summary) for summary in summaries]


def _draft_answer_record(state: InterviewGraphState, question: Question) -> AnswerRecord:
    voice_analysis = (
        VoiceAnalysis.model_validate(state["voice_analysis"])
        if state.get("voice_analysis")
        else None
    )
    return AnswerRecord(
        turn_index=state["turn_index"],
        stage_turn_index=state["stage_turn_index"],
        question_id=question.id,
        question_text=question.text,
        answer_text=state["user_answer"],
        character_response="",
        timestamp=datetime.now(),
        voice_analysis=voice_analysis,
    )


__all__ = [
    "EvaluationGraphState",
    "InterviewGraphState",
    "build_evaluation_graph",
    "build_interview_graph",
]
