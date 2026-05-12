"""Audio transcription (Groq) and voice analysis (Wav2Vec2 MSP-DIM).

Heavy ML deps (torch, transformers, whisper) load lazily on first use so
importing ``interview_system`` stays light enough for 512MB-class servers.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Literal, Mapping, TYPE_CHECKING

import librosa
import numpy as np
import requests

if TYPE_CHECKING:
    from transformers import Wav2Vec2Processor
    from interview_system.models.voice_analysis import VoiceAnalysis

_WHISPER_MODEL: Any = None
_EMOTION_PROCESSOR: Any = None
_EMOTION_MODEL: Any = None
_LOAD_LOCK = threading.Lock()

_EMOTION_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
_WHISPER_SIZE = "base.en"
_GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_GROQ_STT_MODEL = "whisper-large-v3-turbo"

# When set to "1", skip all local-model loading. Use on memory-constrained
# servers (Render free tier, etc.) — only Groq-backed transcription will work.
_SKIP_LOCAL_MODELS = os.getenv("SKIP_LOCAL_AUDIO_MODELS", "0") == "1"


def _build_emotion_classes() -> tuple[type, type]:
    """Build EmotionModel/RegressionHead lazily so torch isn't imported at module load."""
    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

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
            return []

        @property
        def all_tied_weights_keys(self) -> dict[str, list[str]]:
            return {}

        def forward(self, input_values: Any) -> Any:
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)
            return logits

    return EmotionModel, RegressionHead


def _get_whisper_model() -> Any:
    global _WHISPER_MODEL
    if _SKIP_LOCAL_MODELS:
        raise RuntimeError("Local whisper disabled (SKIP_LOCAL_AUDIO_MODELS=1). Use Groq instead.")
    with _LOAD_LOCK:
        if _WHISPER_MODEL is None:
            import whisper  # type: ignore[import-untyped]
            _WHISPER_MODEL = whisper.load_model(_WHISPER_SIZE)
    return _WHISPER_MODEL


def _get_emotion_stack() -> tuple[Any, Any]:
    global _EMOTION_PROCESSOR, _EMOTION_MODEL
    if _SKIP_LOCAL_MODELS:
        raise RuntimeError("Voice emotion analysis disabled on this deployment (SKIP_LOCAL_AUDIO_MODELS=1).")
    with _LOAD_LOCK:
        if _EMOTION_PROCESSOR is None or _EMOTION_MODEL is None:
            from transformers import Wav2Vec2Processor
            EmotionModel, _ = _build_emotion_classes()
            _EMOTION_PROCESSOR = Wav2Vec2Processor.from_pretrained(_EMOTION_MODEL_ID)
            _EMOTION_MODEL = EmotionModel.from_pretrained(_EMOTION_MODEL_ID)
    return _EMOTION_PROCESSOR, _EMOTION_MODEL


def warm_audio_models() -> None:
    """Pre-load models. No-op on memory-constrained deployments."""
    if _SKIP_LOCAL_MODELS:
        return
    if not os.getenv("GROQ_API_KEY"):
        _get_whisper_model()
    _get_emotion_stack()


def load_audio_16k(audio_path: str) -> np.ndarray:
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    return waveform


def transcribe_waveform(waveform: np.ndarray) -> str:
    whisper_model = _get_whisper_model()
    result = whisper_model.transcribe(waveform, language="en")
    return result["text"].strip()


def transcribe_audio_with_groq(audio_path: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")

    with open(audio_path, "rb") as file:
        response = requests.post(
            _GROQ_TRANSCRIPTION_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            data={
                "model": os.getenv("GROQ_STT_MODEL", _GROQ_STT_MODEL),
                "language": "en",
                "response_format": "json",
                "temperature": "0",
            },
            files={"file": (os.path.basename(audio_path), file)},
            timeout=30,
        )

    if response.status_code >= 400:
        raise RuntimeError(f"Groq transcription failed with HTTP {response.status_code}: {response.text}")

    text = response.json().get("text", "")
    return str(text).strip()


def transcribe_audio(audio_path: str) -> str:
    if os.getenv("GROQ_API_KEY"):
        try:
            return transcribe_audio_with_groq(audio_path)
        except Exception:
            if _SKIP_LOCAL_MODELS:
                raise
            pass
    return transcribe_waveform(load_audio_16k(audio_path))


def interpret_voice_label(
    scores: Mapping[str, float],
) -> Literal["confident", "neutral", "nervous"]:
    arousal = float(scores["arousal"])
    dominance = float(scores["dominance"])
    confidence = (dominance * 0.6) + ((1 - arousal) * 0.4)

    if confidence > 0.65:
        return "confident"
    if confidence > 0.55:
        return "neutral"
    return "nervous"


def analyze_voice_waveform(signal: np.ndarray) -> "VoiceAnalysis":
    from interview_system.models.voice_analysis import VoiceAnalysis

    if _SKIP_LOCAL_MODELS:
        # Return a neutral stub when running without local models
        return VoiceAnalysis(arousal=0.5, dominance=0.5, valence=0.5, voice_label="neutral")

    import torch
    processor, model = _get_emotion_stack()
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


def analyze_voice(audio_path: str) -> "VoiceAnalysis":
    return analyze_voice_waveform(load_audio_16k(audio_path))


def process_audio(audio_path: str) -> tuple[str, "VoiceAnalysis"]:
    waveform = load_audio_16k(audio_path)
    if os.getenv("GROQ_API_KEY"):
        try:
            transcript = transcribe_audio_with_groq(audio_path)
        except Exception:
            transcript = transcribe_waveform(waveform)
    else:
        transcript = transcribe_waveform(waveform)
    return transcript, analyze_voice_waveform(waveform)


__all__ = [
    "analyze_voice",
    "analyze_voice_waveform",
    "interpret_voice_label",
    "load_audio_16k",
    "process_audio",
    "transcribe_audio",
    "transcribe_audio_with_groq",
    "transcribe_waveform",
    "warm_audio_models",
]