from dataclasses import dataclass

from interview_system import EvaluationAgent, InterviewAgent


@dataclass
class SessionState:
    interview: InterviewAgent
    evaluation: EvaluationAgent
    upload_index: int = 0
    response_index: int = 0


_SESSIONS: dict[str, SessionState] = {}


def add_session(state: SessionState) -> None:
    _SESSIONS[state.interview.session_id] = state


def replace_session(session_id: str, state: SessionState) -> None:
    _SESSIONS[session_id] = state


def get_session(session_id: str) -> SessionState | None:
    return _SESSIONS.get(session_id)
