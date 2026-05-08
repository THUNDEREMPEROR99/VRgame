import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version\..*",
    category=LangChainPendingDeprecationWarning,
)

from interview_system.agents.evaluation_agent import EvaluationAgent
from interview_system.agents.interview_agent import InterviewAgent
from interview_system.agents.summary_worker import SummaryWorker
from interview_system import prompts
from interview_system.audio import analyze_voice, process_audio, transcribe_audio, warm_audio_models
from interview_system.audio_io import submit_audio_answer
from interview_system.models.voice_analysis import VoiceAnalysis
from interview_system.tts import speak, speak_async, synthesize_mp3

__all__ = [
    "EvaluationAgent",
    "InterviewAgent",
    "SummaryWorker",
    "VoiceAnalysis",
    "analyze_voice",
    "process_audio",
    "prompts",
    "speak",
    "speak_async",
    "synthesize_mp3",
    "submit_audio_answer",
    "transcribe_audio",
    "warm_audio_models",
]
