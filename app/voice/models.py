from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.qa.models import QAResult


@dataclass(frozen=True)
class SpeechTranscript:
    text: str
    language: str
    confidence: float


@dataclass(frozen=True)
class SpeechSynthesisResult:
    audio_path: Path | None
    sample_rate: int
    num_channels: int
    provider: str


@dataclass(frozen=True)
class VoiceTurnResult:
    transcript: SpeechTranscript
    qa_result: QAResult
    spoken_text: str
    synthesis: SpeechSynthesisResult | None = None
