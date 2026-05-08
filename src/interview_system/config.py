from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_MODEL = "openai/gpt-oss-120b"


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


def create_chat_model_from_env() -> ChatOpenAI:
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider == "groq":
        return create_chat_model(
            ModelConfig(
                model=os.getenv("GROQ_CHAT_MODEL", GROQ_CHAT_MODEL),
                api_key=_required_env("GROQ_API_KEY"),
                base_url=os.getenv("GROQ_BASE_URL", GROQ_BASE_URL),
            )
        )

    return create_chat_model(
        ModelConfig(
            model=_required_env("OPENROUTER_MODEL"),
            api_key=_required_env("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        )
    )


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


__all__ = ["ModelConfig", "create_chat_model", "create_chat_model_from_env"]
