# -*- coding: utf-8 -*-
"""Legacy standalone audio demo (Colab export).

Use the package API for integration:

- ``interview_system.transcribe_audio``
- ``interview_system.analyze_voice``
- ``interview_system.submit_audio_answer``

Original notebook:
https://colab.research.google.com/drive/1RNd_u8EOX2m0eCck5ti-N5lC3hzZEyJO
"""

import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2PreTrainedModel, Wav2Vec2Model

import whisper

whisper_model = whisper.load_model("base.en")

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language="en")
    return result["text"].strip()

# building the voice model

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    @property
    def _tied_weights_keys(self):
        # This regression model doesn't use weight tying
        return []

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return logits


model_id = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = EmotionModel.from_pretrained(model_id)

def analyze_voice(audio_path):
    signal, sample_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(signal, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values)

    scores = logits[0].numpy()

    return {
        "arousal": round(float(scores[0]), 3),
        "dominance": round(float(scores[1]), 3),
        "valence": round(float(scores[2]), 3)
    }

def interpret_voice_scene1(voice_scores):
    """Is the user confident or nervous in the interview?"""
    # weighted formula: 60% Dominance, 40% calmness (1-arousal)
    confidence = (voice_scores["dominance"] * 0.6) + ((1 - voice_scores["arousal"]) * 0.4)

    if confidence > 0.65:
        return "confident"
    elif confidence > 0.55:
        return "neutral"
    else:
        return "nervous"

def process_turn(audio_path):

    print("Processing interview turn...")
    try:
      voice_scores = analyze_voice(audio_path)
      print(f"RAW VOICE SCORES: {voice_scores}")
      transcript = transcribe_audio(audio_path)
      voice_label = interpret_voice_scene1(voice_scores)

      # final output
      final_report = f"""
      Transcript: "{transcript}"

      Vocal Tone Analysis:
      Based on the acoustic features, the user sounded {voice_label}.
      Raw Voice Scores:
      {voice_scores}"""

      return final_report
    except Exception as e:
        return f"SYSTEM ERROR: Failed to process audio. Details: {str(e)}"

result = process_turn("test_B_ElevenLabs_Audio_project.wav") # test
print(result)

