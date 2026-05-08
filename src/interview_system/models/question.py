from pydantic import BaseModel, Field


class Question(BaseModel):
    id: str
    text: str
    category: str | None = None
    expected_signals: list[str] = Field(default_factory=list)
