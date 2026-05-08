"""Bridge audio files to the interview agent (transcribe then submit as text answer)."""

from __future__ import annotations

from interview_system.agents.interview_agent import InterviewAgent
from interview_system.audio import analyze_voice, transcribe_audio
from interview_system.models.turn import InterviewOutput
from interview_system.models.voice_analysis import VoiceAnalysis


async def submit_audio_answer(agent: InterviewAgent, audio_path: str) -> InterviewOutput:
    transcript = transcribe_audio(audio_path)
    voice = analyze_voice(audio_path)
    return await agent.submit_answer(transcript, voice_analysis=voice)


__all__ = ["submit_audio_answer"]
