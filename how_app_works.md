# How The App Works

This project is a turn-based AI interview system. It asks interview questions, lets the user answer, uses an AI model to decide whether the answer needs more follow-up, creates a structured summary for each completed question, and finally generates an evaluation report.

The implementation is centered around LangGraph workflows, LangChain agents, Pydantic models, and in-memory LangGraph storage.

## High-Level Flow

1. Questions are loaded from a JSON file into typed `Question` objects.
2. `InterviewAgent` starts a session and returns the first question.
3. The user submits an answer.
4. The interview graph validates the answer is not empty.
5. A stage-decision AI agent decides whether the current question needs a follow-up or is ready to finish.
6. A character-response AI agent generates the interviewer response.
7. The answer and interviewer response are recorded in the session.
8. If the question is finished, a background summary task is queued.
9. The app only advances to the next question when `get_next_question()` is called.
10. After all questions are complete, the app waits for summaries and runs final evaluation.

## Main Files

- `src/interview_system/agents/interview_agent.py`: owns the live interview state and exposes `start()`, `submit_answer()`, `astream_answer()`, `get_next_question()`, and `wait_for_summaries()`.
- `src/interview_system/agents/summary_worker.py`: runs async background summary tasks and stores summaries.
- `src/interview_system/agents/evaluation_agent.py`: runs the final evaluation graph.
- `src/interview_system/graphs.py`: defines the LangGraph interview and evaluation workflows.
- `src/interview_system/llm_agents.py`: wraps LangChain `create_agent()` calls for stage decisions, interviewer responses, summaries, and final reports.
- `src/interview_system/prompts.py`: defines system prompts and prompt-message formatting.
- `src/interview_system/loaders.py`: loads and validates question JSON files.
- `src/interview_system/config.py`: creates the OpenRouter chat model.
- `src/interview_system/models/`: contains all structured Pydantic data models.
- `examples/run_interview_streamlit.py`: demonstrates the app with a simple Streamlit UI.

## Question Loading

Questions are loaded by `load_questions()` in `loaders.py`.

The loader:

- Reads JSON from a file path.
- Converts each item into a `Question` model.
- Rejects an empty question list.
- Rejects duplicate question IDs.

Each question has this shape:

```python
class Question(BaseModel):
    id: str
    text: str
    category: str | None = None
    expected_signals: list[str] = Field(default_factory=list)
```

The demo uses `examples/questions.json`, which currently contains three interview questions with expected signals such as `problem solving`, `teamwork`, and `adaptability`.

## Model Setup

The model is configured in `config.py` with `ModelConfig`:

```python
class ModelConfig(BaseModel):
    model: str
    api_key: str
    temperature: float = 0
```

`create_chat_model()` creates a `ChatOpenRouter` model using:

- `model`
- `api_key`
- `temperature`

The Streamlit demo pre-fills these values from environment variables:

- `OPENROUTER_MODEL`
- `OPENROUTER_API_KEY`

## InterviewAgent Responsibilities

`InterviewAgent` controls the live interview session.

It stores:

- `questions`: the loaded interview questions.
- `session_id`: a provided session ID or a generated UUID.
- `current_index`: the active question index.
- `turn_index`: the global answer turn number.
- `stage_turn_index`: the answer turn number inside the current question.
- `current_stage_records`: answers for the current question only.
- `answer_records`: all answers across the full interview.
- `_started`: whether the interview has started.
- `_awaiting_next`: whether the current question is complete and the app is waiting for `get_next_question()`.
- `_complete`: whether the whole interview is complete.
- `session_data`: structured session data saved to the store.

The agent uses:

- `InMemoryStore` for session data and summaries.
- `InMemorySaver` as the LangGraph checkpointer.
- `SummaryWorker` for background summary jobs.
- `build_interview_graph()` for the interview workflow.

## Starting An Interview

`InterviewAgent.start()` does three things:

1. Marks the interview as started.
2. Saves the initial session data to the store.
3. Returns an `InterviewOutput` containing the first question text and ID.

The returned status is `asking`.

## Submitting An Answer

Answers are submitted through either:

- `submit_answer(answer_text)`: runs the graph and returns one final `InterviewOutput`.
- `astream_answer(answer_text)`: streams graph updates while the graph is running.

Before the graph runs, `InterviewAgent` enforces these rules:

- If the interview has not started, it starts automatically.
- If the interview is complete, it raises `ValueError("interview is complete")`.
- If the current question is already complete, it raises `ValueError("call get_next_question before submitting another answer")`.

This means the app does not automatically skip to the next question immediately after the AI decides a question is complete. The caller must explicitly call `get_next_question()`.

## Interview Graph

The interview graph is built in `build_interview_graph()` inside `graphs.py`.

The graph state is `InterviewGraphState`. Important fields include:

- `session_id`
- `questions`
- `current_index`
- `user_answer`
- `turn_index`
- `stage_turn_index`
- `current_stage_records`
- `answer_records`
- `stage_decision`
- `character_response`
- `stage_status`
- `validation_error`

The graph nodes run in this order:

1. `validate_answer`
2. `decide_stage_status`
3. `generate_character_response`
4. `record_answer`
5. `queue_summary_if_stage_complete`

If `validate_answer` fails, the graph stops early.

## Answer Validation

`validate_answer` only checks this rule:

```python
if not state["user_answer"].strip():
    return {"validation_error": "answer text is required"}
```

So validation does not check quality, correctness, length, or whether expected signals were covered. It only blocks empty or whitespace-only answers.

If validation fails, `submit_answer()` returns an `InterviewOutput` with:

- `stage_status="continue_stage"`
- `character_response="Validation error: answer text is required"`

## Stage Decision AI

After validation, `decide_stage_status` calls `StageDecisionRunner`.

The runner uses a LangChain agent with:

- `STAGE_DECISION_SYSTEM_PROMPT`
- structured response format `StageDecision`

`StageDecision` has this shape:

```python
class StageDecision(BaseModel):
    status: Literal["continue_stage", "ready_for_next"]
    clarification_focus: str | None = None
    reason: str | None = None
```

The AI receives:

- the current question text.
- the question expected signals.
- the transcript for the current question, including the newly submitted answer.

The AI returns one of two statuses:

- `continue_stage`: ask another follow-up for the same question.
- `ready_for_next`: the current question is complete.

## Character Response AI

After the stage decision, `generate_character_response` calls `CharacterResponseRunner`.

This runner uses:

- `CHARACTER_RESPONSE_SYSTEM_PROMPT`
- no structured response format.

It receives:

- the current question.
- the stage decision.
- the current question transcript.

It returns the final interviewer text for that turn. The prompt tells the model to respond naturally, keep replies short, and not reveal hidden evaluation logic.

## Answer Recording

After the interviewer response is generated, `record_answer` creates an `AnswerRecord`:

```python
class AnswerRecord(BaseModel):
    turn_index: int
    stage_turn_index: int
    question_id: str
    question_text: str
    answer_text: str
    character_response: str
    timestamp: datetime
```

This record is added to:

- `answer_records`: the full interview transcript.
- `current_stage_records`: only the transcript for the current question.

`InterviewAgent._apply_graph_result()` then updates the agent's local state, saves session data, increments turn counters, and handles the `ready_for_next` state.

## Completing A Question

If the graph result has `stage_status == "ready_for_next"`, `InterviewAgent`:

- Sets `_awaiting_next = True`.
- Clears `current_stage_records`.
- Resets `stage_turn_index` to `0`.

The question index is not incremented at this point. The next question is shown only after `get_next_question()` is called.

## Background Summaries

When a question is complete, `queue_summary_if_stage_complete` sends a `SummaryInput` to `SummaryWorker.enqueue()`.

`SummaryInput` contains:

```python
class SummaryInput(BaseModel):
    session_id: str
    question_index: int
    question_id: str
    question_text: str
    expected_signals: list[str] = Field(default_factory=list)
    stage_transcript: list[AnswerRecord] = Field(default_factory=list)
```

`SummaryWorker.enqueue()` creates an async task with `asyncio.create_task()`. This means the summary can be generated in the background while the app continues.

The summary AI returns a structured `QuestionSummary`:

```python
class QuestionSummary(BaseModel):
    question_index: int
    question_id: str
    concise_summary: str
    detected_signals: list[str] = Field(default_factory=list)
    missing_signals: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    answer_quality_hint: int = Field(ge=0, le=10)
```

Summaries are stored in `InMemoryStore` under:

```python
namespace=("summaries", session_id)
key=summary.question_id
```

`wait_for_summaries()` waits for all queued summary tasks for the session and returns summaries sorted by `question_index`.

## Moving To The Next Question

`get_next_question()` controls stage advancement.

If `_awaiting_next` is true, it increments `current_index` and clears `_awaiting_next`.

If `current_index` is now past the last question, the interview becomes complete. The agent:

- Sets `_complete = True`.
- Sets `session_data.completed_at`.
- Saves the session.
- Returns a completion output with the closing message.

Otherwise, it returns the next question with status `asking`.

## Final Evaluation

Final evaluation is handled by `EvaluationAgent` and `build_evaluation_graph()`.

The evaluation graph state contains:

- `session_data`
- `summaries`
- `final_report`
- `validation_error`

The evaluation graph nodes run in this order:

1. `validate_inputs`
2. `analyze_answer_quality`
3. `generate_final_report`

`validate_inputs` checks that every answered question has a summary:

```python
answered_question_ids = {record.question_id for record in state["session_data"].answer_records}
summary_question_ids = {summary.question_id for summary in state["summaries"]}
if answered_question_ids != summary_question_ids:
    return {"validation_error": "summaries are missing for answered questions"}
```

If summaries are missing, `EvaluationAgent.evaluate()` raises a `ValueError`.

`analyze_answer_quality` currently sorts summaries by `question_index`.

`generate_final_report` calls `FinalEvaluationRunner`, which uses a structured response format of `FinalEvaluationReport`.

## Final Report Shape

The final report is a Pydantic model:

```python
class FinalEvaluationReport(BaseModel):
    final_answer_quality_score: int = Field(ge=0, le=100)
    overall_score: int = Field(ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    per_question_overview: list[PerQuestionOverview] = Field(default_factory=list)
    recommendation: Literal["proceed", "hold", "reject", "needs_review"]
    final_summary: str
```

Each per-question overview contains:

- `question_id`
- `score` from `0` to `10`
- `summary`

## Storage And Checkpointing

The current implementation uses in-memory storage:

- `InMemoryStore` stores session data and summaries.
- `InMemorySaver` stores graph checkpoints.

Session data is saved under:

```python
namespace=("sessions", session_id)
key="data"
```

Summaries are saved under:

```python
namespace=("summaries", session_id)
key=question_id
```

Because both storage and checkpointing are in memory, data is lost if the process restarts.

## CLI Demo Flow

The demo is in `examples/run_interview_cli.py`.

It does the following:

1. Loads `examples/questions.json`.
2. Creates a model from `OPENROUTER_MODEL` and `OPENROUTER_API_KEY`.
3. Creates `InterviewAgent` and `EvaluationAgent`.
4. Starts the interview and prints the first question.
5. Repeatedly reads user input from the terminal.
6. Uses `interview.astream_answer(answer)` to stream graph updates.
7. Prints streamed stage status and character response updates.
8. Calls `get_next_question()` when the stage status is `ready_for_next`.
9. Waits for summaries after the interview completes.
10. Streams evaluation updates with `evaluation.astream_evaluate(...)`.
11. Calls `evaluation.evaluate(...)` and prints the final report as JSON.

Important implementation detail: the CLI currently streams evaluation once and then calls `evaluate()` afterward, so final evaluation can run twice in the demo. A production caller should usually choose either streaming evaluation or non-streaming evaluation, not both, unless it intentionally wants two runs.

## Streaming Behavior

Interview streaming uses LangGraph `stream_mode="updates"`.

During interview streaming, the graph can emit updates from nodes such as:

- `decide_stage_status`
- `generate_character_response`

The graph also uses `get_stream_writer()` to emit text messages:

- `Deciding stage status...`
- `Generating character response...`

During evaluation streaming, the graph can emit updates from evaluation nodes and writes:

- `Generating final evaluation report...`

## Public Package API

The package exports these objects from `src/interview_system/__init__.py`:

- `EvaluationAgent`
- `InterviewAgent`
- `SummaryWorker`
- `prompts`

This lets callers import the main components with:

```python
from interview_system import EvaluationAgent, InterviewAgent
```

## Current Limitations

- Session data, graph checkpoints, and summaries are stored in memory, so they are lost on process restart.
- Summary tasks run as `asyncio` tasks inside the same process, so they are not durable background jobs.
- Empty answers are blocked, but answer quality is judged by the AI stage decision and summary/evaluation agents rather than deterministic validation.
- The CLI demo currently runs evaluation twice when it streams first and then calls `evaluate()` to print the report.
- There is no external database, queue, web API, or UI layer in the current codebase.

## Design Rationale

- LangGraph keeps the interview and evaluation flows explicit and step-based.
- Pydantic models make AI inputs and outputs structured and validated.
- Separate AI runners keep stage decisions, character responses, summaries, and final reports independent.
- Background summaries prevent summary generation from blocking the immediate interview response after a question is completed.
- Keeping final evaluation separate from the live interview makes it easier to validate that all required summaries exist before scoring.
