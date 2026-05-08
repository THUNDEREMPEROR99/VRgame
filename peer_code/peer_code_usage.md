# Peer Code Usage in the VR Interview System

## Overview

The peer code (`peer_code/hr_vr_project_v3.py`) is a standalone audio-processing demo originally exported from a Google Colab notebook. It served as the **foundational reference** for the audio analysis subsystem of our VR Interview project. This document explains exactly what was taken from the peer code, what was changed, and where each piece lives in the production codebase.

---

## The Peer Code at a Glance

**File:** `peer_code\hr_vr_project_Sence1.py`  
**Original source:** [Google Colab notebook](https://colab.research.google.com/drive/1RNd_u8EOX2m0eCck5ti-N5lC3hzZEyJO)

The peer script contained four main ideas:

| # | Concept | Peer code location |
|---|---------|-------------------|
| 1 | Speech-to-text with **OpenAI Whisper** | `transcribe_audio()` function |
| 2 | **Wav2Vec2** emotion model wrapper (`RegressionHead` + `EmotionModel`) | Lines 30–64 |
| 3 | Voice dimension scoring (arousal / dominance / valence) | `analyze_voice()` function |
| 4 | Confidence label formula | `interpret_voice_scene1()` function |

---

## How We Used It — File by File

### 1. `src/interview_system/audio.py` — Primary Integration Point

This is where **almost all peer code concepts landed**, refactored into production-quality Python.

#### A. `RegressionHead` class (Lines 30–44 in peer → Lines 30–44 in `audio.py`)

The class was copied **exactly** from the peer code, with only type annotations added:

```python
# Peer code (original)
class RegressionHead(nn.Module):
    def __init__(self, config):
        ...

# Our version (audio.py) — same logic, added type hints
class RegressionHead(nn.Module):
    def __init__(self, config: Any) -> None:
        ...
```

#### B. `EmotionModel` class (Lines 46–64 in peer → Lines 47–70 in `audio.py`)

Taken directly from the peer code. We added **two improvements**:
- Return type annotations on `_tied_weights_keys` and `forward`.
- An extra `all_tied_weights_keys` property for compatibility with **Transformers ≥ 5.x**, which expects that attribute and would otherwise crash at model load time.

```python
# Added in our version — not in peer code:
@property
def all_tied_weights_keys(self) -> dict[str, list[str]]:
    # Compatibility with transformers >= 5.x which expects this attribute
    return {}
```

#### C. Model ID and loading (Lines 67–69 in peer → lazy-load helpers in `audio.py`)

The peer code loaded the Wav2Vec2 processor and model **eagerly at import time**:

```python
# Peer code — runs immediately when the script is imported
model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = EmotionModel.from_pretrained(model_id)
```

We replaced this with **thread-safe lazy loading** so that `import interview_system` is fast and models only download when first needed:

```python
# Our version (audio.py)
_EMOTION_MODEL: EmotionModel | None = None
_LOAD_LOCK = threading.Lock()

def _get_emotion_stack() -> tuple[Wav2Vec2Processor, EmotionModel]:
    global _EMOTION_PROCESSOR, _EMOTION_MODEL
    with _LOAD_LOCK:
        if _EMOTION_PROCESSOR is None or _EMOTION_MODEL is None:
            _EMOTION_PROCESSOR = Wav2Vec2Processor.from_pretrained(_EMOTION_MODEL_ID)
            _EMOTION_MODEL = EmotionModel.from_pretrained(_EMOTION_MODEL_ID)
    return _EMOTION_PROCESSOR, _EMOTION_MODEL
```

The same pattern was applied to Whisper via `_get_whisper_model()`.

#### D. `transcribe_audio()` (Line 24–26 in peer → Lines 92–98 in `audio.py`)

The peer version passes a file path directly to Whisper, which internally shells out to `ffmpeg` to decode the audio:

```python
# Peer code — requires ffmpeg to be installed
def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language="en")
    return result["text"].strip()
```

Our version **removes the ffmpeg dependency entirely**. Instead of passing a file path to Whisper, we pre-load the audio with `librosa` (which uses `soundfile` internally — no ffmpeg required) and pass the raw float32 numpy array directly. Whisper accepts numpy arrays just as well as file paths, and skips its internal ffmpeg call when given one:

```python
# Our version — no ffmpeg needed
def transcribe_audio(audio_path: str) -> str:
    whisper_model = _get_whisper_model()
    waveform, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = whisper_model.transcribe(waveform, language="en")
    return result["text"].strip()
```

This is the same loading strategy already used by `analyze_voice()`, so both functions now share a consistent, ffmpeg-free approach.

#### E. `analyze_voice()` (Lines 71–84 in peer → Lines 148–167 in `audio.py`)

The peer code returns a plain `dict`:

```python
# Peer code
def analyze_voice(audio_path):
    ...
    return {
        "arousal": round(float(scores[0]), 3),
        "dominance": round(float(scores[1]), 3),
        "valence": round(float(scores[2]), 3),
    }
```

Our version is **identical in math**, but returns a typed `VoiceAnalysis` Pydantic model (see below) instead of a raw dict. It also calls `detach().cpu()` on the tensor before converting to numpy, making it safe on both CPU and GPU.

#### F. `interpret_voice_scene1()` (Lines 86–96 in peer → `interpret_voice_label()` in `audio.py`)

The confidence formula — `60% dominance + 40% calmness (1 − arousal)` — and the thresholds (`0.65`, `0.55`) were kept **unchanged**. Only the function name was normalised to match our naming conventions.

```python
# Peer code
confidence = (voice_scores["dominance"] * 0.6) + ((1 - voice_scores["arousal"]) * 0.4)

# Our version — exactly the same formula
confidence = (dominance * 0.6) + ((1 - arousal) * 0.4)
```

---

### 2. `src/interview_system/models/voice_analysis.py` — Typed Data Model

The peer code passes voice scores around as plain Python dicts. To prevent key-name typos and enable IDE autocompletion, we wrapped the three dimensions in a **Pydantic model**:

```python
class VoiceAnalysis(BaseModel):
    arousal: float     # MSP-DIM arousal dimension
    dominance: float   # MSP-DIM dominance dimension
    valence: float     # MSP-DIM valence dimension
    voice_label: Literal["confident", "neutral", "nervous"]
```

This model is created at the end of `analyze_voice()` and flows into the graph state, answer records, and the final evaluation report.

---

### 3. `src/interview_system/audio_io.py` — Integration Bridge

The peer code's `process_turn()` function combined transcription + voice analysis into one call. We split those concerns and created a bridge function `submit_audio_answer()` that:

1. Calls `transcribe_audio(audio_path)` → gets text.
2. Calls `analyze_voice(audio_path)` → gets `VoiceAnalysis`.
3. Forwards both to `agent.submit_answer()`.

This mirrors the peer's `process_turn()` concept but integrates it cleanly into the agent/graph pipeline.

---

### 4. `src/interview_system/graphs.py` — Graph State Integration

The `InterviewGraphState` TypedDict holds a `voice_analysis: dict | None` field. When the graph's `record_answer` node runs, it deserializes that dict back into a `VoiceAnalysis` object (using `VoiceAnalysis.model_validate()`), which is then stored on the `AnswerRecord`. This ensures voice scores are preserved across conversation turns and available to the final evaluator.

---

## Summary Table

| Peer code element | Where it appears in our project | Key change |
|---|---|---|
| `RegressionHead` class | `audio.py` | Type annotations added |
| `EmotionModel` class | `audio.py` | `all_tied_weights_keys` compat property added |
| Eager model loading | `audio.py` | Replaced with thread-safe lazy loading |
| `transcribe_audio()` | `audio.py` | Loads audio via `librosa` (no ffmpeg); passes numpy array to Whisper |
| `analyze_voice()` | `audio.py` | Returns `VoiceAnalysis` model; GPU-safe tensor detach |
| `interpret_voice_scene1()` | `audio.py` as `interpret_voice_label()` | Same formula, renamed |
| `process_turn()` combined flow | `audio_io.py` as `submit_audio_answer()` | Split into agent pipeline |
| Plain dict return values | `models/voice_analysis.py` | Replaced with Pydantic `VoiceAnalysis` |
| Hardcoded model load at import | `audio.py` global constants | Extracted to named constants |

---

## What Was NOT Taken from the Peer Code

- The peer code's **global script-level test call** (`result = process_turn("test_B_ElevenLabs_Audio_project.wav")`) was not included — our tests live in the `tests/` directory.
- The peer code had **no graph orchestration, LLM agents, TTS, or session management** — those are entirely our own work built with LangGraph and LangChain.
