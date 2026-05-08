# Codebase File Summary (Explained)

Scope: app-owned files in `src`, `examples`, and root project docs/config.  
Excluded: generated/vendor/cache folders (`.venv`, `.mypy_cache`, `.pytest_cache`) and empty `server/`.

## Runtime Flow First (How files work together)

1. `examples/*` starts the app (CLI, Streamlit, or stress test).
2. `src/interview_system/config.py` builds the LLM client.
3. `src/interview_system/loaders.py` loads question JSON into `Question` models.
4. `src/interview_system/agents/interview_agent.py` drives interview state and calls the interview graph from `graphs.py`.
5. `src/interview_system/llm_agents.py` executes model calls for stage decision, interviewer response, summary, and final report parsing.
6. `src/interview_system/agents/summary_worker.py` runs summary tasks in background and stores them.
7. `src/interview_system/agents/evaluation_agent.py` runs the evaluation graph from `graphs.py` to produce a `FinalEvaluationReport`.
8. `src/interview_system/models/*` defines all contracts passed between these layers.

## Root Files

- `how_app_works.md`: Human-readable architecture spec. Useful as design intent; code has evolved slightly (for example parser fallback behavior), so treat this as high-level guidance, not strict source of truth.
- `hr_vr_project_v3.py`: Older standalone experiment (speech/emotion + Gemini analysis). It does not participate in current interview-system runtime and can be treated as prototype/legacy research code.
- `pytest.ini`: Minimal test config (`asyncio_mode=auto`) so async tests run without manual event-loop boilerplate.
- `requirements.txt`: Single dependency manifest. Includes LangChain/LangGraph core, Streamlit UI, and test tooling.

## Entry Scripts (`examples`)

- `examples/run_interview_cli.py`: Terminal-based reference flow. Shows the exact lifecycle (`start -> answer loop -> get_next_question -> wait_for_summaries -> evaluate`). Good for debugging graph behavior without UI complexity.
- `examples/run_interview_streamlit.py`: Stateful UI wrapper around the same agents. Handles session persistence in `st.session_state`, logs debug events to `logs/streamlit_agent_logs.jsonl`, and presents final metrics/report. This is effectively the interactive demo app.
- `examples/test_perfect_candidate.py`: Simulation harness. Runs multiple candidate personas through the interview pipeline to stress decision/summarization quality and end-to-end stability. Also patches hallucinated summary indices/IDs before evaluation, so it doubles as a reliability test.

## Core Package (`src/interview_system`)

- `src/interview_system/__init__.py`: Public API surface. Keeps imports simple for callers (`from interview_system import InterviewAgent, EvaluationAgent`).
- `src/interview_system/config.py`: Encapsulates model creation so callers pass a typed `ModelConfig` instead of raw kwargs. Central point for provider settings like base URL, temperature, and reasoning options.
- `src/interview_system/loaders.py`: Enforces input integrity early (non-empty questions, unique IDs). Prevents subtle downstream graph/session issues caused by bad question files.
- `src/interview_system/prompts.py`: Keeps prompt text/message formatting centralized. This separates behavior tuning (prompt edits) from orchestration logic.
- `src/interview_system/graphs.py`: Defines two LangGraph state machines:
  - Interview graph: validate answer, judge stage readiness, generate interviewer reply, persist record, queue summary.
  - Evaluation graph: validate summary coverage, normalize summary ordering, generate final report.
  This file is the process backbone.
- `src/interview_system/llm_agents.py`: LLM adapters and structured parsing. Converts model output into typed Pydantic models, captures reasoning/debug metadata, and now provides deterministic fallback payloads when model output is empty so the flow does not halt.

## Agents Layer (`src/interview_system/agents`)

- `src/interview_system/agents/__init__.py`: Export aggregator only.
- `src/interview_system/agents/interview_agent.py`: Main runtime coordinator. Owns in-memory session state, turn/stage counters, skip-answer shortcuts, graph invocation, transition gating (`_awaiting_next`), and debug event generation. This is the highest-impact file for interview behavior.
- `src/interview_system/agents/summary_worker.py`: Async queue-like component (in-process). It schedules summary tasks via `asyncio.create_task`, stores each summary in namespaced in-memory store, and provides `wait_until_done` to synchronize before final evaluation.
- `src/interview_system/agents/evaluation_agent.py`: Thin evaluation orchestrator. Feeds session + summaries to evaluation graph and returns typed `FinalEvaluationReport`. Supports both one-shot and streamed execution.

## Data Contracts (`src/interview_system/models`)

- `src/interview_system/models/__init__.py`: Central export map for all model types.
- `src/interview_system/models/question.py`: Input question contract. `expected_signals` is important because stage decision + summary quality checks depend on it.
- `src/interview_system/models/turn.py`: Live turn contracts:
  - `InterviewOutput`: what callers consume after each action.
  - `AnswerRecord`: persistent transcript unit.
  - `StageDecision`: binary progression decision (`continue_stage` vs `ready_for_next`).
- `src/interview_system/models/summary.py`: Per-question analysis contracts:
  - `SummaryInput`: payload sent to background summarizer.
  - `QuestionSummary`: normalized summary used later in scoring.
- `src/interview_system/models/session.py`: Session snapshot contract (`questions`, full `answer_records`, start/end timestamps). Used as evaluation input and persistence payload.
- `src/interview_system/models/evaluation.py`: Final output contract (`overall_score`, recommendation, per-question breakdown, narrative summary). This is what downstream dashboards/UI should rely on.

## Practical Notes

- Most important behavior files: `interview_agent.py`, `graphs.py`, `llm_agents.py`, `prompts.py`.
- Most important integration files: `run_interview_streamlit.py` (UI) and `run_interview_cli.py` (debug baseline).
- `hr_vr_project_v3.py` is conceptually related to HR evaluation but architecturally separate from the current system.
