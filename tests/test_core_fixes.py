from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interview_system.agents.interview_agent import InterviewAgent
from interview_system.agents.summary_worker import SummaryWorker
from interview_system.models.question import Question
from interview_system.models.summary import QuestionSummary
from interview_system.models.turn import StageDecision
from interview_system.models.voice_analysis import VoiceAnalysis


class FakeStageRunner:
    last_debug = None

    def __init__(self, statuses: list[str]) -> None:
        self.statuses = statuses
        self.calls: list[tuple[str, list[str]]] = []

    async def ainvoke(self, question: Question, stage_transcript: list) -> StageDecision:
        self.calls.append((question.id, [record.answer_text for record in stage_transcript]))
        status = self.statuses.pop(0)
        return StageDecision(status=status, clarification_focus="more detail")


class FakeCharacterRunner:
    last_debug = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    async def ainvoke(self, question: Question, decision: StageDecision, stage_transcript: list) -> str:
        self.calls.append((question.id, decision.status, [record.answer_text for record in stage_transcript]))
        return "Okay, let's move on." if decision.status == "ready_for_next" else "Can you add more detail?"


class FakeIntroRunner:
    last_debug = None

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def ainvoke(self, question_text: str, is_first: bool, prior_question_text: str | None) -> str:
        if self.fail:
            raise RuntimeError("intro failed")
        return ""


class FakeSummaryRunner:
    async def ainvoke(self, item) -> QuestionSummary:
        return QuestionSummary(
            question_index=item.question_index,
            question_id=item.question_id,
            concise_summary="summary",
            answer_quality_hint=5,
        )


def questions() -> list[Question]:
    return [
        Question(id="q1", text="Question 1?", expected_signals=[]),
        Question(id="q2", text="Question 2?", expected_signals=[]),
    ]


def agent(stage_runner: FakeStageRunner, character_runner: FakeCharacterRunner | None = None, **kwargs) -> InterviewAgent:
    return InterviewAgent(
        questions(),
        stage_decision_runner=stage_runner,
        character_response_runner=character_runner or FakeCharacterRunner(),
        intro_runner=kwargs.pop("intro_runner", FakeIntroRunner()),
        summary_worker=SummaryWorker(FakeSummaryRunner()),
        **kwargs,
    )


async def test_stage_transcript_does_not_duplicate_across_turns() -> None:
    stage_runner = FakeStageRunner(["continue_stage", "continue_stage", "ready_for_next"])
    interview = agent(stage_runner)

    await interview.start()
    await interview.submit_answer("a1")
    await interview.submit_answer("a2")
    await interview.submit_answer("a3")

    assert stage_runner.calls == [
        ("q1", ["a1"]),
        ("q1", ["a1", "a2"]),
        ("q1", ["a1", "a2", "a3"]),
    ]
    assert [record.answer_text for record in interview.answer_records] == ["a1", "a2", "a3"]


async def test_next_question_starts_with_empty_stage_transcript() -> None:
    stage_runner = FakeStageRunner(["ready_for_next", "ready_for_next"])
    interview = agent(stage_runner)

    await interview.start()
    await interview.submit_answer("q1 answer")
    await interview.get_next_question()
    await interview.submit_answer("q2 answer")

    assert stage_runner.calls == [
        ("q1", ["q1 answer"]),
        ("q2", ["q2 answer"]),
    ]
    assert [record.answer_text for record in interview.answer_records] == ["q1 answer", "q2 answer"]


async def test_max_turns_returns_ready_for_next() -> None:
    stage_runner = FakeStageRunner(["continue_stage", "continue_stage"])
    interview = agent(stage_runner, max_turns_per_question=2)

    await interview.start()
    first = await interview.submit_answer("a1")
    second = await interview.submit_answer("a2")
    summaries = await interview.wait_for_summaries()

    assert first.stage_status == "continue_stage"
    assert second.stage_status == "ready_for_next"
    assert len(summaries) == 1
    assert summaries[0].question_id == "q1"


async def test_skip_bypasses_stage_decision_but_uses_character_response() -> None:
    stage_runner = FakeStageRunner([])
    character_runner = FakeCharacterRunner()
    interview = agent(stage_runner, character_runner)

    await interview.start()
    output = await interview.submit_answer("idk")

    assert output.stage_status == "ready_for_next"
    assert stage_runner.calls == []
    assert character_runner.calls == [("q1", "ready_for_next", ["idk"])]


async def test_intro_failure_skips_intro_and_continues() -> None:
    stage_runner = FakeStageRunner(["ready_for_next"])
    interview = agent(stage_runner, intro_runner=FakeIntroRunner(fail=True))

    output = await interview.start()

    assert output.stage_status == "asking"
    assert output.intro_text is None
    assert output.next_question_text == "Question 1?"


def test_process_audio_loads_file_once(monkeypatch) -> None:
    from interview_system import audio

    waveform = np.array([0.1, 0.2], dtype=np.float32)
    calls = {"load": 0, "transcribe": None, "voice": None}

    def fake_load(path: str, sr: int, mono: bool):
        calls["load"] += 1
        return waveform, sr

    def fake_transcribe(value):
        calls["transcribe"] = value
        return "hello"

    def fake_analyze(value):
        calls["voice"] = value
        return VoiceAnalysis(arousal=0.1, dominance=0.8, valence=0.5, voice_label="confident")

    monkeypatch.setattr(audio.librosa, "load", fake_load)
    monkeypatch.setattr(audio, "transcribe_waveform", fake_transcribe)
    monkeypatch.setattr(audio, "analyze_voice_waveform", fake_analyze)

    transcript, voice = audio.process_audio("sample.wav")

    assert transcript == "hello"
    assert voice.voice_label == "confident"
    assert calls["load"] == 1
    assert calls["transcribe"] is waveform
    assert calls["voice"] is waveform
