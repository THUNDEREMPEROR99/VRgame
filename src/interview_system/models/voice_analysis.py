from typing import Literal

from pydantic import BaseModel, Field


class VoiceAnalysis(BaseModel):
    arousal: float = Field(description="MSP-DIM arousal dimension")
    dominance: float = Field(description="MSP-DIM dominance dimension")
    valence: float = Field(description="MSP-DIM valence dimension")
    voice_label: Literal["confident", "neutral", "nervous"]
