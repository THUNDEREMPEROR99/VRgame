from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import StateSnapshot

from interview_system.agents.summary_worker import SummaryWorker
from interview_system.config import ModelConfig, create_chat_model
from interview_system.graphs import build_interview_graph
from interview_system.llm_agents import (
    CharacterResponseRunner,
    IntroRunner,
    StageDecisionRunner,
    create_character_response_runner,
    create_intro_runner,
    create_stage_decision_runner,
    create_summary_runner,
)
from interview_system.models.question import Question
from interview_system.models.session import InterviewSessionData
from interview_system.models.summary import QuestionSummary, SummaryInput
from interview_system.models.turn import AnswerRecord, InterviewOutput, StageDecision
from interview_system.models.voice_analysis import VoiceAnalysis


SKIP_ANSWERS = {
    "idk",
    "i don't know",
    "i dont know",
    "don't know",
    "dont know",
    "skip",
    "pass",
    "not sure",
    "n/a",
    "na",
}

class InterviewAgent:
    def __init__(
        self,
        questions: list[Question],
        session_id: str | None = None,
        model=None,
        stage_decision_runner: StageDecisionRunner | None = None,
        character_response_runner: CharacterResponseRunner | None = None,
        intro_runner: IntroRunner | None = None,
        summary_worker: SummaryWorker | None = None,
        checkpointer: InMemorySaver | None = None,
        store: InMemoryStore | None = None,
        max_turns_per_question: int = 5,
    ) -> None:
        if not questions:
            raise ValueError("questions are required")

        self.questions = questions
        self.session_id = session_id or str(uuid4())
        self.max_turns_per_question = max_turns_per_question
        self.store = store or InMemoryStore()
        self.current_index = 0
        self.turn_index = 0
        self.stage_turn_index = 0
        self.current_stage_records: list[AnswerRecord] = []
        self.answer_records: list[AnswerRecord] = []
        self._started = False
        self._awaiting_next = False
        self._complete = False
        self._last_question_text: str | None = None
        self.session_data = InterviewSessionData(
            session_id=self.session_id,
            questions=questions,
            started_at=datetime.now(),
        )

        needs_model = (
            stage_decision_runner is None
            or character_response_runner is None
            or intro_runner is None
            or summary_worker is None
            or summary_worker.runner is None
        )
        if needs_model:
            if model is None:
                raise ValueError("model or runners are required")

            chat_model = create_chat_model(model) if isinstance(model, ModelConfig) else model
            stage_decision_runner = stage_decision_runner or create_stage_decision_runner(chat_model)
            character_response_runner = character_response_runner or create_character_response_runner(chat_model)
            intro_runner = intro_runner or create_intro_runner(chat_model)
            summary_worker = summary_worker or SummaryWorker(create_summary_runner(chat_model), store=self.store)

        self.stage_decision_runner = stage_decision_runner
        self.character_response_runner = character_response_runner
        self.intro_runner = intro_runner
        self.summary_worker = summary_worker
        self._checkpointer = checkpointer or InMemorySaver()
        self.graph = build_interview_graph(
            stage_decision_runner,
            character_response_runner,
            self.summary_worker,
            checkpointer=self._checkpointer,
            store=self.store,
        )

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.session_id}}

    def _default_state(self, answer_text: str, voice_analysis: VoiceAnalysis | None = None) -> dict:
        return {
            "session_id": self.session_id,
            "questions": [question.model_dump(mode="json") for question in self.questions],
            "current_index": self.current_index,
            "user_answer": answer_text,
            "turn_index": self.turn_index,
            "stage_turn_index": self.stage_turn_index,
            "current_stage_records": [record.model_dump(mode="json") for record in self.current_stage_records],
            "answer_records": [record.model_dump(mode="json") for record in self.answer_records],
            "stage_decision": None,
            "character_response": None,
            "stage_status": "continue_stage",
            "next_question_text": None,
            "next_question_id": None,
            "is_complete": False,
            "is_skip": self._is_skip_answer(answer_text),
            "voice_analysis": voice_analysis.model_dump(mode="json") if voice_analysis else None,
        }

    async def start(self) -> InterviewOutput:
        self._started = True
        self._save_session()
        question = self.questions[self.current_index]
        intro_text = await self._generate_intro(question, is_first=True)
        self._last_question_text = question.text
        debug_events = [
            self._debug_event(
                "interview_started",
                "Interview started",
                question=self._question_debug(question),
            )
        ]
        if intro_text:
            debug_events.append(
                self._debug_event(
                    "intro_generated",
                    "Intro generated",
                    is_first=True,
                    intro_text=intro_text,
                )
            )
        return InterviewOutput(
            session_id=self.session_id,
            next_question_text=question.text,
            next_question_id=question.id,
            intro_text=intro_text,
            stage_status="asking",
            debug_events=debug_events,
        )

    async def submit_answer(self, answer_text: str, voice_analysis: VoiceAnalysis | None = None) -> InterviewOutput:
        if not self._started:
            await self.start()
        if self._complete:
            raise ValueError("interview is complete")
        if self._awaiting_next:
            raise ValueError("call get_next_question before submitting another answer")

        question = self.questions[self.current_index]
        debug_events = [
            self._debug_event(
                "answer_received",
                "User answer received",
                question=self._question_debug(question),
                answer=answer_text,
                turn_index=self.turn_index,
                stage_turn_index=self.stage_turn_index,
            )
        ]

        is_skip = self._is_skip_answer(answer_text)

        if is_skip:
            debug_events.append(
                self._debug_event(
                    "skip_detected",
                    "Skip detected",
                    matched_answer=answer_text.strip(),
                    forced_stage_status="ready_for_next",
                )
            )
        else:
            transcript = [
                *self.current_stage_records,
                AnswerRecord(
                    turn_index=self.turn_index,
                    stage_turn_index=self.stage_turn_index,
                    question_id=question.id,
                    question_text=question.text,
                    answer_text=answer_text,
                    character_response="",
                    timestamp=datetime.now(),
                    voice_analysis=voice_analysis,
                ),
            ]
            debug_events.append(
                self._debug_event(
                    "stage_decision_input",
                    "Stage decision input",
                    question=self._question_debug(question),
                    transcript=[self._record_debug(record) for record in transcript],
                )
            )
            debug_events.append(
                self._debug_event(
                    "thinking_summary",
                    "App reasoning plan",
                    checking="Whether the answer has enough detail for this question.",
                    expected_signals=question.expected_signals,
                    current_answer=answer_text,
                    possible_actions=["ask a follow-up", "move to next question"],
                )
            )

        result = await self.graph.ainvoke(
            self._default_state(answer_text, voice_analysis=voice_analysis),
            config=self._config(),
        )

        if result.get("validation_error"):
            debug_events.append(
                self._debug_event(
                    "validation_error",
                    "Validation error",
                    error=result["validation_error"],
                )
            )
            return InterviewOutput(
                session_id=self.session_id,
                stage_status="continue_stage",
                character_response=f"Validation error: {result['validation_error']}",
                debug_events=debug_events,
            )

        stage_decision = StageDecision.model_validate(result["stage_decision"])
        if is_skip:
            debug_events.extend(
                [
                    self._debug_event(
                        "llm_stage_decision_skipped",
                        "Stage decision LLM skipped",
                        skipped=["stage_decision"],
                        reason="Skip detection forced ready_for_next before stage judging.",
                    ),
                    self._debug_event(
                        "decision_rationale",
                        "Decision rationale",
                        status="ready_for_next",
                        reason="The user entered a skip phrase, so the app should move on instead of generating an answer.",
                        next_action="Ask the character agent for a brief acknowledgement, then move to the next question.",
                    ),
                ]
            )
        else:
            debug_events.extend(
                [
                    self._debug_event(
                        "stage_decision_output",
                        "Stage decision output",
                        decision=stage_decision.model_dump(),
                        stage_status=result["stage_status"],
                    ),
                    self._debug_event(
                        "stage_model_reasoning",
                        "Stage model reasoning summary",
                        **self._model_reasoning_debug(getattr(self.stage_decision_runner, "last_debug", None)),
                    ),
                    self._debug_event(
                        "decision_rationale",
                        "Decision rationale",
                        status=stage_decision.status,
                        reason=stage_decision.reason or "No reason returned by the model.",
                        clarification_focus=stage_decision.clarification_focus or "None",
                        next_action=self._next_action_for_status(result["stage_status"]),
                    ),
                ]
            )
        debug_events.extend(
            [
                self._debug_event(
                    "character_response_output",
                    "Character response output",
                    response=result["character_response"],
                ),
                self._debug_event(
                    "character_model_reasoning",
                    "Character model reasoning summary",
                    **self._model_reasoning_debug(getattr(self.character_response_runner, "last_debug", None)),
                ),
            ]
        )

        stage_status = self._apply_graph_result(result)
        debug_events.append(
            self._debug_event(
                "record_saved",
                "Answer record saved",
                stage_status=stage_status,
                answer_record_count=len(self.answer_records),
                current_stage_record_count=len(self.current_stage_records),
            )
        )

        return InterviewOutput(
            session_id=self.session_id,
            character_response=result["character_response"],
            stage_status=stage_status,
            debug_events=debug_events,
        )

    async def astream_answer(self, answer_text: str, voice_analysis: VoiceAnalysis | None = None) -> AsyncIterator[dict]:
        if not self._started:
            await self.start()
        if self._complete:
            raise ValueError("interview is complete")
        if self._awaiting_next:
            raise ValueError("call get_next_question before submitting another answer")

        async for event in self.graph.astream(
            self._default_state(answer_text, voice_analysis=voice_analysis),
            config=self._config(),
            stream_mode="updates",
        ):
            yield event

        state = self.get_state()
        if state and state.values and not state.values.get("validation_error"):
            self._apply_graph_result(state.values)

    async def get_next_question(self) -> InterviewOutput:
        if self._complete:
            return self._complete_output()
        previous_index = self.current_index
        if self._awaiting_next:
            self.current_index += 1
            self._awaiting_next = False

        if self.current_index >= len(self.questions):
            self._complete = True
            self.session_data.completed_at = datetime.now()
            self._save_session()
            return self._complete_output(
                [
                    self._debug_event(
                        "interview_complete",
                        "Interview complete",
                        previous_index=previous_index,
                        completed_at=self.session_data.completed_at.isoformat(),
                    )
                ]
            )

        question = self.questions[self.current_index]
        prior_question_text = self._last_question_text
        intro_text = await self._generate_intro(question, is_first=False)
        self._last_question_text = question.text
        debug_events = [
            self._debug_event(
                "next_question",
                "Next question selected",
                previous_index=previous_index,
                current_index=self.current_index,
                question=self._question_debug(question),
            )
        ]
        if intro_text:
            debug_events.append(
                self._debug_event(
                    "intro_generated",
                    "Intro generated",
                    is_first=False,
                    prior_question_text=prior_question_text,
                    intro_text=intro_text,
                )
            )
        return InterviewOutput(
            session_id=self.session_id,
            next_question_text=question.text,
            next_question_id=question.id,
            intro_text=intro_text,
            stage_status="asking",
            debug_events=debug_events,
        )

    def is_complete(self) -> bool:
        return self._complete

    async def wait_for_summaries(self) -> list[QuestionSummary]:
        return await self.summary_worker.wait_until_done(self.session_id)

    def get_session_data(self) -> InterviewSessionData:
        return self.session_data

    def get_graph(self):
        return self.graph.get_graph()

    def get_state(self) -> StateSnapshot:
        return self.graph.get_state(config=self._config())

    def _apply_graph_result(self, result: dict) -> str:
        self.turn_index += 1
        self.stage_turn_index += 1
        self.answer_records = [AnswerRecord.model_validate(record) for record in result["answer_records"]]
        self.current_stage_records = [AnswerRecord.model_validate(record) for record in result["current_stage_records"]]
        self.session_data.answer_records = self.answer_records
        self._save_session()

        stage_status = result["stage_status"]
        if stage_status == "continue_stage" and self.stage_turn_index >= self.max_turns_per_question:
            stage_status = "ready_for_next"

        if stage_status == "ready_for_next":
            if result["stage_status"] != "ready_for_next":
                self._enqueue_stage_summary()
            self._awaiting_next = True
            self.current_stage_records = []
            self.stage_turn_index = 0
        return stage_status

    def _enqueue_stage_summary(self) -> None:
        question = self.questions[self.current_index]
        self.summary_worker.enqueue(
            SummaryInput(
                session_id=self.session_id,
                question_index=self.current_index,
                question_id=question.id,
                question_text=question.text,
                expected_signals=question.expected_signals,
                stage_transcript=self.current_stage_records,
            )
        )

    def _is_skip_answer(self, answer_text: str) -> bool:
        return answer_text.strip().lower().rstrip(".!?") in SKIP_ANSWERS

    async def _generate_intro(self, question: Question, is_first: bool) -> str | None:
        try:
            text = await self.intro_runner.ainvoke(
                question.text,
                is_first=is_first,
                prior_question_text=None if is_first else self._last_question_text,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Intro generation failed: %s", exc)
            return None
        return text or None

    def _debug_event(self, event: str, title: str, **details: object) -> dict[str, object]:
        return {
            "event": event,
            "title": title,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "details": details,
        }

    def _question_debug(self, question: Question) -> dict[str, object]:
        return {
            "id": question.id,
            "text": question.text,
            "expected_signals": question.expected_signals,
        }

    def _record_debug(self, record: AnswerRecord) -> dict[str, object]:
        return {
            "turn_index": record.turn_index,
            "stage_turn_index": record.stage_turn_index,
            "question_id": record.question_id,
            "answer_text": record.answer_text,
            "character_response": record.character_response,
        }

    def _next_action_for_status(self, stage_status: str) -> str:
        if stage_status == "ready_for_next":
            return "Move to the next question."
        return "Ask a follow-up question for more detail."

    def _model_reasoning_debug(self, debug: dict | None) -> dict[str, object]:
        if not debug:
            return {"available": False, "reason": "No provider reasoning metadata returned."}

        summaries = debug.get("reasoning_summaries") or []
        reasoning_details = debug.get("reasoning_details") or []
        reasoning_content = debug.get("reasoning_content")
        usage_metadata = debug.get("usage_metadata")
        return {
            "available": bool(summaries or reasoning_details or reasoning_content),
            "reasoning_summaries": summaries,
            "reasoning_detail_types": self._reasoning_detail_types(reasoning_details),
            "raw_reasoning_returned": bool(reasoning_content),
            "note": "Only provider-returned reasoning summaries are shown. Raw reasoning text is not displayed.",
            "usage_metadata": usage_metadata,
        }

    def _reasoning_detail_types(self, reasoning_details: object) -> list[str]:
        if not isinstance(reasoning_details, list):
            return []
        return [item.get("type", "unknown") for item in reasoning_details if isinstance(item, dict)]

    def _save_session(self) -> None:
        self.store.put(
            namespace=("sessions", self.session_id),
            key="data",
            value=self.session_data.model_dump(mode="json"),
        )

    def _complete_output(self, debug_events: list[dict[str, object]] | None = None) -> InterviewOutput:
        return InterviewOutput(
            session_id=self.session_id,
            stage_status="complete",
            is_complete=True,
            closing_message="Thank you for completing the interview. We'll review your responses shortly.",
            debug_events=debug_events or [],
        )
