from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0
    reasoning: dict | None = None


def create_chat_model(config: ModelConfig) -> ChatOpenAI:
    params = {
        "model": config.model,
        "api_key": config.api_key,
        "temperature": config.temperature,
    }
    if config.base_url:
        params["base_url"] = config.base_url
    if config.reasoning:
        params["reasoning"] = config.reasoning
    return ChatOpenAI(**params)


__all__ = ["ModelConfig", "create_chat_model"]
