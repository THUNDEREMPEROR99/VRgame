import asyncio
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import os

import edge_tts
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# ============================================================
# STATIC CONFIG - edit here only
# ============================================================

OPENROUTER_NAME = "OpenRouter"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-oss-120b:free"

CUSTOM_NAME = "Custom"
CUSTOM_BASE_URL = "https://api.your-custom-provider.com/v1"
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
CUSTOM_MODEL = "your-custom-model"

# Change this by hand: "openrouter" or "custom"
ACTIVE_PROVIDER = "openrouter"

# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from interview_system import EvaluationAgent, InterviewAgent
from interview_system.config import ModelConfig, create_chat_model
from interview_system.loaders import load_questions


def get_active_model_settings() -> tuple[str, str, str, str]:
    if ACTIVE_PROVIDER == "openrouter":
        return OPENROUTER_NAME, OPENROUTER_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
    if ACTIVE_PROVIDER == "custom":
        return CUSTOM_NAME, CUSTOM_MODEL, CUSTOM_API_KEY, CUSTOM_BASE_URL
    raise ValueError(f"Unknown ACTIVE_PROVIDER: {ACTIVE_PROVIDER!r}")


def default_state() -> None:
    defaults = {
        "started": False,
        "complete": False,
        "interview": None,
        "evaluation": None,
        "messages": [],
        "pending_answer": None,
        "async_loop": None,
        "report": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_state() -> None:
    for key in ["started", "complete", "interview", "evaluation", "messages", "pending_answer", "report"]:
        if key in st.session_state:
            del st.session_state[key]
    default_state()


def run_async(coro):
    loop = st.session_state.async_loop
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state.async_loop = loop
    return loop.run_until_complete(coro)


async def synthesize_tts_mp3(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def assistant_message(content: str) -> dict[str, str]:
    message: dict[str, str] = {"role": "assistant", "content": content}
    clean = content.strip()
    if not clean:
        return message
    try:
        audio_bytes = run_async(synthesize_tts_mp3(clean))
        if audio_bytes:
            message["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
    except Exception as error:
        append_agent_log(
            "tts_error",
            "TTS generation failed",
            error_class=error.__class__.__name__,
            message=str(error)[:800],
            content_preview=clean[:200],
        )
    return message


def start_interview(questions_path: str, temperature: float, thinking_enabled: bool, thinking_effort: str) -> None:
    _, model_name, api_key, base_url = get_active_model_settings()
    if not api_key:
        raise ValueError("API key is empty. Set it in the top config block.")

    reasoning = {"effort": thinking_effort} if thinking_enabled else None
    questions = load_questions(Path(questions_path))
    model = create_chat_model(
        ModelConfig(model=model_name, api_key=api_key, base_url=base_url, temperature=temperature, reasoning=reasoning)
    )
    interview = InterviewAgent(questions, model=model)
    evaluation = EvaluationAgent(model=model, session_id=interview.session_id)
    output = run_async(interview.start())

    st.session_state.interview = interview
    st.session_state.evaluation = evaluation
    st.session_state.started = True
    st.session_state.complete = False
    st.session_state.report = None
    st.session_state.messages = []
    if output.intro_text:
        st.session_state.messages.append(assistant_message(output.intro_text))
    st.session_state.messages.append(assistant_message(output.next_question_text))
    append_agent_logs(output_debug_events(output))


def finish_interview() -> None:
    interview = st.session_state.interview
    evaluation = st.session_state.evaluation
    try:
        summaries = run_async(interview.wait_for_summaries())
        append_agent_log(
            "summaries_ready",
            "Question summaries ready",
            count=len(summaries),
            summaries=[summary.model_dump() for summary in summaries],
        )
        st.session_state.report = run_async(evaluation.evaluate(interview.get_session_data(), summaries))
        append_agent_log(
            "final_evaluation_output",
            "Final evaluation output",
            report=st.session_state.report.model_dump(),
        )
    except Exception as error:
        add_model_error_message(error)
    st.session_state.complete = True


def submit_answer(answer: str) -> None:
    interview = st.session_state.interview
    append_agent_log("submit_answer", "Submitting answer to agent", answer=answer)

    try:
        output = run_async(interview.submit_answer(answer))
    except Exception as error:
        add_model_error_message(error)
        return

    append_agent_logs(output_debug_events(output))
    if output.character_response:
        st.session_state.messages.append(assistant_message(output.character_response))

    if output.stage_status == "ready_for_next":
        next_output = run_async(interview.get_next_question())
        append_agent_logs(output_debug_events(next_output))
        if next_output.is_complete:
            if next_output.closing_message:
                st.session_state.messages.append(assistant_message(next_output.closing_message))
            finish_interview()
        else:
            if next_output.intro_text:
                st.session_state.messages.append(assistant_message(next_output.intro_text))
            st.session_state.messages.append(assistant_message(next_output.next_question_text))


def append_agent_logs(events: list[dict[str, object]]) -> None:
    for event in events:
        print_agent_log(event)


def output_debug_events(output) -> list[dict[str, object]]:
    return getattr(output, "debug_events", [])


def append_agent_log(event: str, title: str, **details: object) -> None:
    print_agent_log(
        {
            "event": event,
            "title": title,
            "details": details,
        }
    )


def print_agent_log(event: dict[str, object]) -> None:
    event_with_timestamp = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    title = event_with_timestamp.get("title", event_with_timestamp.get("event", "Agent event"))
    pretty_event = json.dumps(event_with_timestamp, indent=2, default=str, ensure_ascii=False)
    json_line = json.dumps(event_with_timestamp, default=str, ensure_ascii=False)

    print(f"\n[Agent Log] {title}", flush=True)
    print(pretty_event, flush=True)

    log_file = ROOT / "logs" / "streamlit_agent_logs.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as file:
        file.write(json_line + "\n")


def add_model_error_message(error: Exception) -> None:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "The AI provider returned an error. Please try again.",
        }
    )
    append_agent_log(
        "provider_error",
        "Provider error",
        error_class=error.__class__.__name__,
        message=str(error)[:1200],
    )


st.set_page_config(layout="wide")
default_state()

st.markdown(
    """
    <style>
    .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Interview")
st.caption("Answer each question in the chat. The interviewer will ask follow-ups when more detail is needed.")

with st.sidebar:
    st.header("Setup")

    questions_path = st.text_input("Questions file", value=str(ROOT / "examples" / "questions.json"))
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    thinking_enabled = st.checkbox("Enable thinking", value=True)
    thinking_effort = st.selectbox("Thinking effort", ["minimal", "low", "medium", "high"], index=1)

    _, _, active_api_key, _ = get_active_model_settings()
    can_start = bool(questions_path and active_api_key)
    if st.button("Start interview", disabled=st.session_state.started or not can_start):
        try:
            start_interview(questions_path, temperature, thinking_enabled, thinking_effort)
        except Exception as error:
            add_model_error_message(error)
        st.rerun()

    if st.button("Reset"):
        reset_state()
        st.rerun()

chat_col = st.container()

with chat_col:
    if not st.session_state.started:
        st.info("Set the config at the top of the file, then start the interview.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("audio_b64"):
                st.markdown(
                    f"""
                    <audio controls autoplay style="width: 100%; margin-top: 0.25rem;">
                      <source src="data:audio/mpeg;base64,{message['audio_b64']}" type="audio/mpeg">
                    </audio>
                    """,
                    unsafe_allow_html=True,
                )

    if st.session_state.report:
        report = st.session_state.report
        st.subheader("Final Evaluation")
        col1, col2, col3 = st.columns(3)
        col1.metric("Overall", report.overall_score)
        col2.metric("Answer Quality", report.final_answer_quality_score)
        col3.metric("Recommendation", report.recommendation)
        st.write(report.final_summary)
        st.json(report.model_dump())

if st.session_state.pending_answer:
    answer = st.session_state.pending_answer
    st.session_state.pending_answer = None
    with st.spinner("Interviewer is thinking..."):
        submit_answer(answer)
    st.rerun()

prompt = st.chat_input(
    "Type your answer",
    disabled=not st.session_state.started or st.session_state.complete or bool(st.session_state.pending_answer),
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_answer = prompt
    st.rerun()
