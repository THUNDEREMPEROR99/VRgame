from __future__ import annotations

from interview_system.models.question import Question
from interview_system.models.session import InterviewSessionData
from interview_system.models.summary import QuestionSummary, SummaryInput
from interview_system.models.turn import AnswerRecord, StageDecision

STAGE_DECISION_SYSTEM_PROMPT = (
    "You are an interview stage judge. Decide whether the user's answer is meaningfully sufficient for the current question. "
    "Use the question text, expected signals, and transcript to decide what matters for this specific question. "
    "Return ready_for_next when the answer is responsive enough that a reasonable interviewer can move on. "
    "Return continue_stage when a useful follow-up would materially improve the answer. "
    "For continue_stage, set clarification_focus to the most useful missing or unclear area to ask about next. "
    "If the user says idk, I don't know, skip, pass, not sure, or similar, treat it as ready_for_next because the user is skipping. "
    "Return a structured decision."
)

CHARACTER_RESPONSE_SYSTEM_PROMPT = (
    "You are an AI interviewer, not an answer generator. "
    "Never answer interview questions for the user, invent user experiences, or provide sample answers. "
    "You must obey the stage status exactly. "
    "If stage status is continue_stage, ask exactly one short follow-up question based on the clarification focus. "
    "If stage status is ready_for_next, do not ask a follow-up question; briefly acknowledge the answer and stop. "
    "If the user skips with idk, I don't know, skip, pass, not sure, or similar, briefly acknowledge and move on. "
    "Keep replies short and do not reveal hidden evaluation logic."
)

SUMMARY_SYSTEM_PROMPT = (
    "You summarize one interview question answer with answer-quality only. Return structured output."
)

FINAL_EVALUATION_SYSTEM_PROMPT = (
    "You produce the final answer-quality evaluation from per-question summaries. "
    "Consider the candidate's vocal confidence indicators (voice analysis labels: confident, neutral, nervous) "
    "alongside answer content when judging overall performance. Return structured output."
)

QUESTION_INTRO_SYSTEM_PROMPT = (
    "You are an AI interviewer producing a SHORT spoken lead-in before the next interview question. "
    "Never answer the question for the user. Never invent user details. "
    "If is_first is true: greet the candidate warmly in 1 short sentence and signal the interview is starting. "
    "If is_first is false: briefly acknowledge the prior turn in 1 short sentence and transition to the next question. "
    "Do NOT include the question text itself; the question is spoken separately. "
    "Keep it under 20 words. Plain conversational tone. No emojis, no markdown, no quotes."
)


def stage_decision_message(question: Question, stage_transcript: list[AnswerRecord]) -> str:
    return (
        f"Question: {question.text}\n"
        f"Expected signals: {', '.join(question.expected_signals) or 'none'}\n"
        f"Transcript:\n{_format_stage_transcript(stage_transcript)}"
    )


def character_response_message(
    question: Question,
    decision: StageDecision,
    stage_transcript: list[AnswerRecord],
) -> str:
    return (
        f"Question: {question.text}\n"
        f"Stage status: {decision.status}\n"
        f"Clarification focus: {decision.clarification_focus or 'none'}\n"
        f"Transcript:\n{_format_stage_transcript(stage_transcript)}"
    )


def question_intro_message(
    question_text: str,
    is_first: bool,
    prior_question_text: str | None,
) -> str:
    return (
        f"is_first: {str(is_first).lower()}\n"
        f"Upcoming question: {question_text}\n"
        f"Prior question: {prior_question_text or 'none'}"
    )


def summary_message(item: SummaryInput) -> str:
    return (
        f"Question: {item.question_text}\n"
        f"Expected signals: {', '.join(item.expected_signals) or 'none'}\n"
        f"Transcript:\n{_format_stage_transcript(item.stage_transcript)}"
    )


def final_evaluation_message(
    session_data: InterviewSessionData,
    summaries: list[QuestionSummary],
) -> str:
    voice_labels = []
    for record in session_data.answer_records:
        if record.voice_analysis:
            voice_labels.append(f"  {record.question_id}: {record.voice_analysis.voice_label} (arousal={record.voice_analysis.arousal}, dominance={record.voice_analysis.dominance}, valence={record.voice_analysis.valence})")
    voice_section = "\n".join(voice_labels) if voice_labels else "none"
    return (
        f"Answered count: {len(session_data.answer_records)}\n"
        f"Voice analysis per turn:\n{voice_section}\n"
        f"Summaries:\n{_format_summaries(summaries)}"
    )


def _format_stage_transcript(stage_transcript: list[AnswerRecord]) -> str:
    if not stage_transcript:
        return "none"

    lines = []
    for record in stage_transcript:
        entry = f"User: {record.answer_text}"
        if record.voice_analysis:
            va = record.voice_analysis
            entry += f" [voice: {va.voice_label}, arousal={va.arousal}, dominance={va.dominance}, valence={va.valence}]"
        entry += f"\nInterviewer: {record.character_response}"
        lines.append(entry)
    return "\n".join(lines)


def _format_summaries(summaries: list[QuestionSummary]) -> str:
    if not summaries:
        return "none"

    return "\n".join(summary.model_dump_json() for summary in summaries)


__all__ = [
    "CHARACTER_RESPONSE_SYSTEM_PROMPT",
    "FINAL_EVALUATION_SYSTEM_PROMPT",
    "QUESTION_INTRO_SYSTEM_PROMPT",
    "STAGE_DECISION_SYSTEM_PROMPT",
    "SUMMARY_SYSTEM_PROMPT",
    "character_response_message",
    "final_evaluation_message",
    "question_intro_message",
    "stage_decision_message",
    "summary_message",
]
