import json
from pathlib import Path

from interview_system.models.question import Question


def load_questions(path: str | Path) -> list[Question]:
    data = json.loads(Path(path).read_text())
    questions = [Question.model_validate(item) for item in data]

    if not questions:
        raise ValueError("questions sheet is empty")

    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("questions sheet contains duplicate question ids")

    return questions


__all__ = ["load_questions"]
