"""Audio transcription (Whisper) and voice analysis (Wav2Vec2 MSP-DIM).

Models load lazily on first use so importing ``interview_system`` stays light.
"""

from __future__ import annotations

import threading
from typing import Any, Literal, Mapping

import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from interview_system.models.voice_analysis import VoiceAnalysis

_WHISPER_MODEL: Any = None
_EMOTION_PROCESSOR: Wav2Vec2Processor | None = None
_EMOTION_MODEL: EmotionModel | None = None
_LOAD_LOCK = threading.Lock()

_EMOTION_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_WHISPER_SIZE = "base.en"


class RegressionHead(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: Any, **kwargs: Any) -> Any:
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    @property
    def _tied_weights_keys(self) -> list[str]:
        # This regression model doesn't use weight tying
        return []

    @property
    def all_tied_weights_keys(self) -> dict[str, list[str]]:
        # Compatibility with transformers >= 5.x which expects this attribute
        return {}

    def forward(self, input_values: Any) -> Any:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return logits


def _get_whisper_model() -> Any:
    global _WHISPER_MODEL
    with _LOAD_LOCK:
        if _WHISPER_MODEL is None:
            import whisper  # type: ignore[import-untyped]

            _WHISPER_MODEL = whisper.load_model(_WHISPER_SIZE)
    return _WHISPER_MODEL


def _get_emotion_stack() -> tuple[Wav2Vec2Processor, EmotionModel]:
    global _EMOTION_PROCESSOR, _EMOTION_MODEL
    with _LOAD_LOCK:
        if _EMOTION_PROCESSOR is None or _EMOTION_MODEL is None:
            _EMOTION_PROCESSOR = Wav2Vec2Processor.from_pretrained(_EMOTION_MODEL_ID)
            _EMOTION_MODEL = EmotionModel.from_pretrained(_EMOTION_MODEL_ID)
    return _EMOTION_PROCESSOR, _EMOTION_MODEL


def transcribe_audio(audio_path: str) -> str:
    # Load audio with librosa (no ffmpeg required), same as analyze_voice.
    # Whisper accepts a float32 numpy array directly, bypassing its internal
    # ffmpeg call that would otherwise be needed when given a file path.
    whisper_model = _get_whisper_model()
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = whisper_model.transcribe(waveform, language="en")
    return result["text"].strip()


def interpret_voice_label(
    scores: Mapping[str, float],
) -> Literal["confident", "neutral", "nervous"]:
    """60% dominance, 40% calmness (1 - arousal). Same thresholds as legacy script."""
    arousal = float(scores["arousal"])
    dominance = float(scores["dominance"])
    confidence = (dominance * 0.6) + ((1 - arousal) * 0.4)

    if confidence > 0.65:
        return "confident"
    if confidence > 0.55:
        return "neutral"
    return "nervous"


def analyze_voice(audio_path: str) -> VoiceAnalysis:
    processor, model = _get_emotion_stack()
    signal, _sample_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(signal, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values)

    scores_vec = logits[0].detach().cpu().numpy()
    arousal = round(float(scores_vec[0]), 3)
    dominance = round(float(scores_vec[1]), 3)
    valence = round(float(scores_vec[2]), 3)
    score_map = {"arousal": arousal, "dominance": dominance, "valence": valence}
    label = interpret_voice_label(score_map)
    return VoiceAnalysis(
        arousal=arousal,
        dominance=dominance,
        valence=valence,
        voice_label=label,
    )


__all__ = [
    "EmotionModel",
    "RegressionHead",
    "VoiceAnalysis",
    "analyze_voice",
    "interpret_voice_label",
    "transcribe_audio",
]
