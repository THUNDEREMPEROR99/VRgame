"""Bridge audio files to the interview agent (transcribe then submit as text answer)."""

from __future__ import annotations

from interview_system.agents.interview_agent import InterviewAgent
from interview_system.audio import process_audio
from interview_system.models.turn import InterviewOutput


async def submit_audio_answer(agent: InterviewAgent, audio_path: str) -> InterviewOutput:
    transcript, voice = process_audio(audio_path)
    return await agent.submit_answer(transcript, voice_analysis=voice)


__all__ = ["submit_audio_answer"]
