from __future__ import annotations

from collections.abc import AsyncIterator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from interview_system.graphs import build_evaluation_graph
from interview_system.llm_agents import FinalEvaluationRunner, create_final_evaluation_runner
from interview_system.models.evaluation import FinalEvaluationReport
from interview_system.models.session import InterviewSessionData
from interview_system.models.summary import QuestionSummary


class EvaluationAgent:
    def __init__(
        self,
        model=None,
        final_evaluation_runner: FinalEvaluationRunner | None = None,
        session_id: str | None = None,
        checkpointer: InMemorySaver | None = None,
        store: InMemoryStore | None = None,
    ) -> None:
        self.session_id = session_id or "evaluation"
        self.store = store or InMemoryStore()

        if final_evaluation_runner is None:
            if model is None:
                raise ValueError("model or final_evaluation_runner is required")
            final_evaluation_runner = create_final_evaluation_runner(model)

        self._checkpointer = checkpointer or InMemorySaver()
        self.graph = build_evaluation_graph(
            final_evaluation_runner,
            checkpointer=self._checkpointer,
            store=self.store,
        )

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.session_id}}

    async def evaluate(
        self,
        session_data: InterviewSessionData,
        summaries: list[QuestionSummary],
    ) -> FinalEvaluationReport:
        result = await self.graph.ainvoke(
            {
                "session_data": session_data.model_dump(mode="json"),
                "summaries": [summary.model_dump(mode="json") for summary in summaries],
            },
            config=self._config(),
        )

        if result.get("validation_error"):
            raise ValueError(result["validation_error"])

        return FinalEvaluationReport.model_validate(result["final_report"])

    async def astream_evaluate(
        self,
        session_data: InterviewSessionData,
        summaries: list[QuestionSummary],
    ) -> AsyncIterator[dict]:
        async for event in self.graph.astream(
            {
                "session_data": session_data.model_dump(mode="json"),
                "summaries": [summary.model_dump(mode="json") for summary in summaries],
            },
            config=self._config(),
            stream_mode="updates",
        ):
            yield event