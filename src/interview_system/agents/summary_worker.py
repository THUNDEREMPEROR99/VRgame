from __future__ import annotations

import asyncio

from langgraph.store.memory import InMemoryStore

from interview_system.llm_agents import SummaryRunner
from interview_system.models.summary import QuestionSummary, SummaryInput


class SummaryWorker:
    def __init__(
        self,
        runner: SummaryRunner | None = None,
        store: InMemoryStore | None = None,
    ) -> None:
        self.runner = runner
        self.store = store or InMemoryStore()
        self._tasks: dict[str, list[asyncio.Task[QuestionSummary]]] = {}

    def enqueue(self, item: SummaryInput) -> None:
        task = asyncio.create_task(self._summarize_and_store(item))
        self._tasks.setdefault(item.session_id, []).append(task)

    async def summarize(self, item: SummaryInput) -> QuestionSummary:
        if self.runner is None:
            raise ValueError("summary runner is required")
        return await self.runner.ainvoke(item)

    async def wait_until_done(self, session_id: str) -> list[QuestionSummary]:
        tasks = self._tasks.get(session_id, [])
        if tasks:
            await asyncio.gather(*tasks)
        return self.list_summaries(session_id)

    def list_summaries(self, session_id: str) -> list[QuestionSummary]:
        items = self.store.search(("summaries", session_id), limit=1000)
        summaries = []
        for item in items:
            summaries.append(QuestionSummary.model_validate(item.value))
        return sorted(summaries, key=lambda s: s.question_index)

    def save_summary(self, session_id: str, summary: QuestionSummary) -> None:
        self.store.put(
            namespace=("summaries", session_id),
            key=summary.question_id,
            value=summary.model_dump(mode="json"),
        )

    async def _summarize_and_store(self, item: SummaryInput) -> QuestionSummary:
        summary = await self.summarize(item)
        # Patch hallucinated IDs with the ground-truth values from the input
        summary.question_id = item.question_id
        summary.question_index = item.question_index
        self.save_summary(item.session_id, summary)
        return summary


__all__ = ["SummaryWorker"]