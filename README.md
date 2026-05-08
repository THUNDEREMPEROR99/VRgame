# 🎤 VR Interview System

An AI-powered, turn-based interview simulation system built with **LangGraph**, **LangChain**, and **Streamlit**. It conducts structured interviews, evaluates answers in real-time using LLMs, and generates a detailed final evaluation report — optionally with voice input/output.

---

## ✨ Features

- **Turn-Based Interview Flow** — Ask questions, receive answers, dynamically follow up or advance based on AI judgment
- **Stage Decision AI** — Determines whether a candidate's answer is complete or needs a follow-up question
- **Character Response AI** — Generates natural, conversational interviewer replies
- **Background Summarization** — Per-question summaries are generated asynchronously while the interview continues
- **Final Evaluation Report** — Structured scoring with overall score, strengths, weaknesses, per-question breakdowns, and hiring recommendation
- **Streaming Support** — Real-time streaming of graph updates during both interview and evaluation
- **Streamlit UI** — Interactive web interface for running interviews
- **CLI Interface** — Terminal-based demo for debugging and testing
- **Voice I/O** — Audio recording via `sounddevice`, transcription via `openai-whisper`, and text-to-speech via `edge-tts`
- **Candidate Simulation** — Stress-test the pipeline with multiple candidate personas

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Entry Points                            │
│  run_interview_streamlit.py  │  run_interview_cli.py            │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   InterviewAgent    │  ← owns live session state
          └──────────┬──────────┘
                     │  invokes
          ┌──────────▼──────────┐
          │  Interview Graph    │  ← LangGraph state machine
          │  (graphs.py)        │
          │                     │
          │  1. validate_answer │
          │  2. decide_stage    │──► StageDecisionRunner (LLM)
          │  3. character_resp  │──► CharacterResponseRunner (LLM)
          │  4. record_answer   │
          │  5. queue_summary   │──► SummaryWorker (async)
          └─────────────────────┘
                     │
          ┌──────────▼──────────┐
          │   EvaluationAgent   │
          └──────────┬──────────┘
                     │  invokes
          ┌──────────▼──────────┐
          │  Evaluation Graph   │
          │  (graphs.py)        │
          │                     │
          │  1. validate_inputs │
          │  2. analyze_quality │
          │  3. final_report    │──► FinalEvaluationRunner (LLM)
          └─────────────────────┘
```

---

## 📁 Project Structure

```
VRgame/
├── src/
│   └── interview_system/
│       ├── __init__.py              # Public API (InterviewAgent, EvaluationAgent)
│       ├── config.py                # LLM model configuration (ModelConfig)
│       ├── loaders.py               # Question JSON loading & validation
│       ├── prompts.py               # Centralized system prompts
│       ├── graphs.py                # LangGraph interview & evaluation state machines
│       ├── llm_agents.py            # LLM adapters with structured Pydantic parsing
│       ├── audio.py                 # Audio recording and transcription (Whisper)
│       ├── audio_io.py              # Audio I/O helpers
│       ├── tts.py                   # Text-to-speech (edge-tts)
│       ├── agents/
│       │   ├── interview_agent.py   # Main interview session coordinator
│       │   ├── summary_worker.py    # Async background summarization
│       │   └── evaluation_agent.py  # Evaluation orchestrator
│       └── models/
│           ├── question.py          # Question input contract
│           ├── turn.py              # InterviewOutput, AnswerRecord, StageDecision
│           ├── summary.py           # SummaryInput, QuestionSummary
│           ├── session.py           # Session snapshot contract
│           └── evaluation.py        # FinalEvaluationReport contract
├── examples/
│   ├── questions.json               # Sample interview questions
│   ├── run_interview_cli.py         # Terminal-based interview demo
│   ├── run_interview_streamlit.py   # Streamlit UI demo
│   ├── test_perfect_candidate.py    # Candidate simulation / stress test
│   ├── candidate_1_the_expert.txt
│   ├── candidate_2_the_nervous_junior.txt
│   └── candidate_3_the_smooth_talker.txt
├── logs/                            # Streamlit debug event logs
├── recordings/                      # Audio recordings
├── hr_vr_project_v3.py              # Legacy prototype (speech + emotion analysis)
├── requirements.txt
├── pytest.ini
├── .env                             # API keys (not committed)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10+**
- An [OpenRouter](https://openrouter.ai/) API key

### 1. Clone and set up the environment

```bash
git clone <repo-url>
cd VRgame

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini   # or any model on OpenRouter
```

### 3. Run the Streamlit UI

```bash
streamlit run examples/run_interview_streamlit.py
```

### 4. Or run the CLI demo

```bash
python examples/run_interview_cli.py
```

---

## 🎯 Interview Flow

```
start()
  └─► Returns first question

submit_answer(text)             ← user answers
  ├─► validate_answer           ← blocks empty answers
  ├─► decide_stage_status       ← LLM: continue_stage | ready_for_next
  ├─► generate_character_response ← LLM: natural interviewer reply
  ├─► record_answer             ← persists AnswerRecord
  └─► queue_summary_if_done     ← async background summary

get_next_question()             ← advance to next question
  └─► Returns next question, or completion message

wait_for_summaries()            ← sync after all questions done

EvaluationAgent.evaluate(session, summaries)
  └─► Returns FinalEvaluationReport
```

---

## 📋 Question Format

Questions are defined in a JSON file (see `examples/questions.json`):

```json
[
  {
    "id": "q1",
    "text": "Tell me about yourself and what led you to choose your field of study.",
    "category": "general",
    "expected_signals": [
      "communication",
      "self awareness",
      "motivation",
      "career direction"
    ]
  }
]
```

| Field              | Type           | Description                                        |
|--------------------|----------------|----------------------------------------------------|
| `id`               | `str`          | Unique question identifier                         |
| `text`             | `str`          | The question shown to the candidate                |
| `category`         | `str` (opt.)   | Topic category (e.g. `general`, `problem_solving`) |
| `expected_signals` | `list[str]`    | Behavioral signals the AI looks for in answers     |

---

## 📊 Evaluation Report

After all questions are complete, the system generates a `FinalEvaluationReport`:

```python
class FinalEvaluationReport(BaseModel):
    final_answer_quality_score: int   # 0–100
    overall_score: int                # 0–100
    strengths: list[str]
    weaknesses: list[str]
    per_question_overview: list[PerQuestionOverview]
    recommendation: Literal["proceed", "hold", "reject", "needs_review"]
    final_summary: str
```

Each per-question overview includes:
- `question_id`
- `score` (0–10)
- `summary`

---

## 🔌 Public API

```python
from interview_system import InterviewAgent, EvaluationAgent

# Setup
agent = InterviewAgent(questions=questions, model=model)
evaluator = EvaluationAgent(model=model)

# Interview lifecycle
output = agent.start()
output = await agent.submit_answer("My answer here")
output = agent.get_next_question()
summaries = await agent.wait_for_summaries()

# Evaluation
report = await evaluator.evaluate(session=agent.session_data, summaries=summaries)
```

---

## 🧪 Testing

Run the candidate simulation stress test:

```bash
python examples/test_perfect_candidate.py
```

This runs multiple candidate personas (expert, nervous junior, smooth talker) through the full pipeline to validate decision quality and end-to-end stability.

Run unit/async tests:

```bash
pytest
```

---

## 🔧 Configuration

| Environment Variable  | Description                          | Example                          |
|-----------------------|--------------------------------------|----------------------------------|
| `OPENROUTER_API_KEY`  | Your OpenRouter API key              | `sk-or-v1-...`                   |
| `OPENROUTER_MODEL`    | Model identifier on OpenRouter       | `openai/gpt-4o-mini`             |

Model temperature and other parameters can be tuned in `config.py` via `ModelConfig`.

---

## ⚠️ Known Limitations

- **In-memory storage only** — Session data, summaries, and checkpoints are lost on process restart. No database integration yet.
- **Single-process async** — Summary tasks use `asyncio.create_task()`, not durable background queues.
- **No web API** — There is no REST or WebSocket API; the system is integrated via Python directly.
- **CLI double-evaluation** — The CLI demo runs evaluation twice when streaming and then calling `evaluate()`. Choose one in production.

---

## 🛠️ Tech Stack

| Component        | Library / Tool                   |
|------------------|----------------------------------|
| LLM Orchestration | [LangChain](https://langchain.com) + [LangGraph](https://langchain-ai.github.io/langgraph/) |
| LLM Provider     | [OpenRouter](https://openrouter.ai/) |
| Data Validation  | [Pydantic v2](https://docs.pydantic.dev/) |
| UI               | [Streamlit](https://streamlit.io/) |
| Speech-to-Text   | [OpenAI Whisper](https://github.com/openai/whisper) |
| Text-to-Speech   | [edge-tts](https://github.com/rany2/edge-tts) |
| Audio I/O        | [sounddevice](https://python-sounddevice.readthedocs.io/) + [librosa](https://librosa.org/) |
| Deep Learning    | [PyTorch](https://pytorch.org/) + [Transformers](https://huggingface.co/docs/transformers/) |
| Testing          | [pytest](https://pytest.org/) + pytest-asyncio |

---

## 📄 License

This project is for educational and research purposes.
