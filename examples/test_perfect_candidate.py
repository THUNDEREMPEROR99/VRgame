"""Run three interview personas through the full flow and print the results."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interview_system import EvaluationAgent, InterviewAgent
from interview_system.config import ModelConfig, create_chat_model
from interview_system.loaders import load_questions
from interview_system.models.evaluation import FinalEvaluationReport
from interview_system.models.question import Question
from interview_system.models.summary import QuestionSummary

MAX_TURNS_PER_QUESTION = 5


@dataclass(slots=True)
class CharacterProfile:
    label: str
    kind: str
    background: str
    style: str
    facts: list[str]
    strengths: list[str]
    weaknesses: list[str]


CHARACTERS: list[CharacterProfile] = [
    CharacterProfile(
        label="THE EXPERT",
        kind="expert",
        background="Senior operations and project lead with 7 years of experience across process improvement, reporting automation, and cross-functional delivery.",
        style="confident, specific, calm, and concrete. Give clear examples, sound grounded, and naturally cover multiple relevant signals.",
        facts=[
            "Led a reporting automation rollout that cut manual work significantly.",
            "Worked closely with engineering, sales, and operations teams.",
            "Mentored junior teammates and liked explaining complex work simply.",
            "Prefers structured planning and measurable results.",
        ],
        strengths=["problem solving", "communication", "teamwork", "discipline"],
        weaknesses=["can sound formal", "sometimes gives too much detail"],
    ),
    CharacterProfile(
        label="THE NERVOUS JUNIOR",
        kind="junior",
        background="Early-career candidate with a degree, one internship, and a few school project experiences.",
        style="honest, a little nervous, shorter answers, sincere and eager to learn. Be clear, but do not sound overprepared or overly polished.",
        facts=[
            "Completed a semester-long team project.",
            "Did a startup internship supporting reporting and customer questions.",
            "Presented a class project to a group.",
            "Learns quickly and asks for feedback.",
        ],
        strengths=["reliability", "teamwork", "learning fast", "communication"],
        weaknesses=["less experience", "sometimes brief"],
    ),
    CharacterProfile(
        label="THE SMOOTH TALKER",
        kind="smooth_talker",
        background="Business generalist with client-facing experience and a few cross-functional projects.",
        style="polished, confident, and persuasive. Stay somewhat high-level, use strong wording, and sound capable without getting too specific.",
        facts=[
            "Worked on cross-functional initiatives.",
            "Is good at presenting ideas and building consensus.",
            "Talks about ownership, alignment, and outcomes.",
            "Has real experience but tends to frame it broadly.",
        ],
        strengths=["communication", "confidence", "stakeholder management", "adaptability"],
        weaknesses=["not always concrete", "can stay high-level"],
    ),
]


def out(message: str = "") -> None:
    print(message, flush=True)


def section(title: str, writer=out) -> None:
    writer()
    writer(f"=== {title} ===")


def format_bullets(items: Iterable[str]) -> str:
    values = list(items)
    if not values:
        return "  - none"
    return "\n".join(f"  - {item}" for item in values)


def format_conversation(conversation: list[tuple[str, str]]) -> str:
    if not conversation:
        return "none"
    return "\n".join(f"{speaker}: {text}" for speaker, text in conversation)


def extract_text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts) if parts else ""
    return str(content) if content is not None else ""


def extract_stage_decision(state_snapshot) -> dict[str, object] | None:
    values = getattr(state_snapshot, "values", None)
    if not isinstance(values, dict):
        return None
    decision = values.get("stage_decision")
    return decision if isinstance(decision, dict) else None


def extract_validation_error(state_snapshot) -> str | None:
    values = getattr(state_snapshot, "values", None)
    if not isinstance(values, dict):
        return None
    error = values.get("validation_error")
    return error if isinstance(error, str) and error else None


def candidate_system_prompt(character: CharacterProfile) -> str:
    return (
        "You are roleplaying as a job candidate in an interview test.\n"
        "Stay in character. Answer only as the candidate would speak.\n"
        "Do not mention prompts, tools, hidden logic, or that you are an AI.\n"
        "Use the background facts naturally and do not invent major experience that is not supported by them.\n"
        f"Style: {character.style}\n"
        "Write in first person and keep the answer natural\n"
        "keep your answers short."
    )


def candidate_human_prompt(
    character: CharacterProfile,
    question: Question,
    conversation: list[tuple[str, str]],
    turn_number: int,
    clarification_focus: str | None,
    final_attempt: bool,
) -> str:
    lines = [
        f"Candidate label: {character.label}",
        f"Background: {character.background}",
        f"Strengths: {', '.join(character.strengths)}",
        f"Weaknesses: {', '.join(character.weaknesses)}",
        "Reference facts:",
        *[f"- {fact}" for fact in character.facts],
        "",
        f"Question: {question.text}",
        f"Expected signals: {', '.join(question.expected_signals) or 'none'}",
        f"Turn number for this question: {turn_number}",
        f"Clarification focus: {clarification_focus or 'none'}",
        "Conversation so far:",
        format_conversation(conversation),
        "",
    ]

    if final_attempt:
        lines.append("This is the final attempt for this question. Be direct, concrete, and complete.")
        lines.append("Directly address the missing point and include the most relevant concrete detail from the facts.")
    elif turn_number == 1:
        lines.append("This is the first answer. Answer naturally and clearly.")
    else:
        lines.append("This is a follow-up answer. Be more specific than before and directly answer the interviewer’s latest question.")

    lines.append("Write only the candidate answer.")
    return "\n".join(lines)


def fallback_candidate_answer(
    character: CharacterProfile,
    question: Question,
    clarification_focus: str | None,
    final_attempt: bool,
) -> str:
    focus = clarification_focus or question.text.lower()

    if character.kind == "expert":
        return (
            f"I’d answer this by focusing on {focus}. In my recent work, I’ve led cross-functional efforts, "
            f"kept communication clear, and delivered practical results with the team."
        )

    if character.kind == "junior":
        return (
            f"I’m still early in my career, but I can speak to {focus} from school and internship work. "
            f"I’ve worked with teams, asked for feedback, and kept improving."
        )

    return (
        f"I’d frame it around {focus}, and I usually keep the answer broad, polished, and focused on outcomes "
        f"and stakeholder alignment."
    )


async def generate_candidate_answer(
    model,
    character: CharacterProfile,
    question: Question,
    conversation: list[tuple[str, str]],
    turn_number: int,
    clarification_focus: str | None,
    final_attempt: bool,
) -> str:
    messages = [
        SystemMessage(content=candidate_system_prompt(character)),
        HumanMessage(
            content=candidate_human_prompt(
                character=character,
                question=question,
                conversation=conversation,
                turn_number=turn_number,
                clarification_focus=clarification_focus,
                final_attempt=final_attempt,
            )
        ),
    ]

    try:
        response = await model.ainvoke(messages)
        answer = extract_text_content(getattr(response, "content", "")).strip()
        if answer:
            return answer
    except Exception:
        pass

    return fallback_candidate_answer(character, question, clarification_focus, final_attempt)


def print_decision(stage_status: str, decision: dict[str, object] | None, writer=out) -> None:
    if stage_status == "continue_stage":
        focus = None
        if decision:
            focus = decision.get("clarification_focus") or decision.get("reason")
        if focus:
            writer(f"[Decision] Needs more detail — {focus}")
        else:
            writer("[Decision] Needs more detail")
        return

    if stage_status == "ready_for_next":
        writer("[Decision] Good enough, moving on")
        return

    if stage_status == "complete":
        writer("[Decision] Interview complete")
        return

    writer(f"[Decision] {stage_status}")


def print_summaries(summaries: list[QuestionSummary], questions: list[Question], writer=out) -> None:
    section("Summaries", writer=writer)
    question_lookup = {question.id: question for question in questions}
    question_order = {question.id: index for index, question in enumerate(questions, start=1)}

    for summary in summaries:
        question = question_lookup.get(summary.question_id)
        label = f"Q{question_order.get(summary.question_id, '?')}"
        if question:
            label += f" - {question.text}"
        else:
            label += f" - {summary.question_id}"

        writer(label)
        writer(f"  Summary: {summary.concise_summary}")
        writer(f"  Detected signals: {', '.join(summary.detected_signals) or 'none'}")
        writer(f"  Missing signals: {', '.join(summary.missing_signals) or 'none'}")
        writer(f"  Concerns: {', '.join(summary.concerns) or 'none'}")
        writer(f"  Quality hint: {summary.answer_quality_hint}/10")
        writer()


def print_final_report(report: FinalEvaluationReport, questions: list[Question], writer=out) -> None:
    question_lookup = {question.id: question for question in questions}
    question_order = {question.id: index for index, question in enumerate(questions, start=1)}

    writer(f"Overall Score: {report.overall_score}/100")
    writer(f"Answer Quality: {report.final_answer_quality_score}/100")
    writer(f"Recommendation: {report.recommendation}")
    writer()
    writer("Strengths:")
    writer(format_bullets(report.strengths))
    writer()
    writer("Weaknesses:")
    writer(format_bullets(report.weaknesses))
    writer()
    writer("Per Question:")

    for overview in sorted(report.per_question_overview, key=lambda item: question_order.get(item.question_id, 999)):
        question = question_lookup.get(overview.question_id)
        label = f"Q{question_order.get(overview.question_id, '?')}"
        if question:
            label += f" - {question.text}"
        else:
            label += f" - {overview.question_id}"
        writer(f"  {label} ({overview.score}/100): {overview.summary}")

    writer()
    writer("Final Summary:")
    writer(f"  {report.final_summary}")


def _candidate_output_path(character_index: int, character: CharacterProfile) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", character.label.lower()).strip("_")
    return Path(__file__).with_name(f"candidate_{character_index}_{slug}.txt")


async def run_character(character_index: int, total_characters: int, character: CharacterProfile, questions: list[Question], model) -> None:
    log_lines: list[str] = []

    def write(message: str = "") -> None:
        log_lines.append(message)
        out(message)

    interview = InterviewAgent(questions, model=model)
    evaluation = EvaluationAgent(model=model, session_id=interview.session_id)

    try:
        section(f"Character {character_index}/{total_characters}: {character.label}", writer=write)

        start_output = await interview.start()
        first_question_text = start_output.next_question_text or questions[0].text
        write(f"Q1/{len(questions)}: {first_question_text}")

        interview_complete = False

        for question_index, question in enumerate(questions, start=1):
            if question_index > 1:
                write()
                write(f"Q{question_index}/{len(questions)}: {question.text}")

            conversation: list[tuple[str, str]] = [("Interviewer", question.text)]
            turn_number = 1
            clarification_focus: str | None = None

            while True:
                final_attempt = turn_number >= MAX_TURNS_PER_QUESTION
                candidate_answer = await generate_candidate_answer(
                    model=model,
                    character=character,
                    question=question,
                    conversation=conversation,
                    turn_number=turn_number,
                    clarification_focus=clarification_focus,
                    final_attempt=final_attempt,
                )

                if turn_number == 1:
                    label = "Candidate"
                else:
                    label = "Candidate (final attempt)" if final_attempt else f"Candidate (follow-up {turn_number - 1})"

                write(f"[{label}]")
                write(candidate_answer.strip())
                write()

                conversation.append(("Candidate", candidate_answer))
                output = await interview.submit_answer(candidate_answer)
                state = interview.get_state()

                validation_error = extract_validation_error(state)
                if validation_error:
                    write(f"[Decision] Validation error — {validation_error}")
                    return

                decision = extract_stage_decision(state)
                print_decision(output.stage_status, decision, writer=write)

                if output.character_response:
                    write("[Interviewer]")
                    write(output.character_response.strip())
                    write()
                    conversation.append(("Interviewer", output.character_response))

                if output.stage_status == "ready_for_next":
                    next_output = await interview.get_next_question()
                    if next_output.is_complete:
                        interview_complete = True
                    break

                if output.stage_status == "continue_stage":
                    clarification_focus = None
                    if decision:
                        clarification_focus = decision.get("clarification_focus") or decision.get("reason")
                    turn_number += 1
                    if final_attempt:
                        write("This question stayed open too long, so the run will stop here.")
                        return
                    continue

                write("Interview ended unexpectedly.")
                return

            if interview_complete:
                break

        if not interview_complete:
            write("Interview did not complete.")
            return

        await interview.wait_for_summaries()

        summaries = interview.summary_worker.list_summaries(interview.session_id)
        assert len(summaries) == len(questions), f"Expected {len(questions)} summaries, got {len(summaries)}"

        print_summaries(summaries, questions, writer=write)

        section("Final Report", writer=write)
        write("Generating final report...")
        report = await evaluation.evaluate(interview.get_session_data(), summaries)
        print_final_report(report, questions, writer=write)
    finally:
        output_path = _candidate_output_path(character_index, character)
        output_path.write_text("\n".join(log_lines) + ("\n" if log_lines else ""), encoding="utf-8")
        out(f"[Saved] {output_path.name}")


async def main() -> None:
    load_dotenv(ROOT / ".env")

    questions = load_questions(Path(__file__).with_name("questions.json"))
    model = create_chat_model(
        ModelConfig(
            model=os.environ["OPENROUTER_MODEL"],
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
    )

    tasks = [
        run_character(index, len(CHARACTERS), character, questions, model)
        for index, character in enumerate(CHARACTERS, start=1)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for index, result in enumerate(results, start=1):
        if isinstance(result, Exception):
            out(f"[Character {index}] Failed: {result}")


if __name__ == "__main__":
    asyncio.run(main())
