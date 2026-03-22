from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel
from livekit.agents import stt as lk_stt
from livekit.agents import utils as lk_utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN

from app.config.settings import AppSettings
from app.voice.models import SpeechTranscript


class VoiceSTTError(RuntimeError):
    """Raised when Armenian STT cannot produce a transcript."""


class FasterWhisperArmenianSTT:
    def __init__(self, settings: AppSettings) -> None:
        self._model_name = settings.voice_stt_model
        self._language = settings.voice_stt_language
        self._device = settings.voice_stt_device
        self._compute_type = settings.voice_stt_compute_type
        self._beam_size = settings.voice_stt_beam_size
        self._model: WhisperModel | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def prewarm(self) -> None:
        self._load_model()

    def transcribe_file(self, audio_path: Path) -> SpeechTranscript:
        segments, info = self._load_model().transcribe(
            str(audio_path),
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return self._segments_to_transcript(segments, info.language_probability or 0.0)

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> SpeechTranscript:
        audio, _sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)

        segments, info = self._load_model().transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return self._segments_to_transcript(segments, info.language_probability or 0.0)

    def _segments_to_transcript(self, segments, confidence: float) -> SpeechTranscript:
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        if not text:
            raise VoiceSTTError("faster-whisper returned an empty transcript.")
        return SpeechTranscript(
            text=text,
            language=self._language,
            confidence=float(confidence),
        )

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self._model_name,
                device=self._device,
                compute_type=self._compute_type,
            )
        return self._model


class LiveKitFasterWhisperSTT(lk_stt.STT):
    def __init__(self, provider: FasterWhisperArmenianSTT) -> None:
        super().__init__(
            capabilities=lk_stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self._provider = provider

    @property
    def model(self) -> str:
        return self._provider.model_name

    @property
    def provider(self) -> str:
        return "faster_whisper"

    def prewarm(self) -> None:
        self._provider.prewarm()

    async def _recognize_impl(
        self,
        buffer: lk_utils.AudioBuffer,
        *,
        language=NOT_GIVEN,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    ) -> lk_stt.SpeechEvent:
        del language, conn_options
        merged_frame = lk_utils.merge_frames(buffer)
        transcript = await asyncio.to_thread(
            self._provider.transcribe_audio_bytes,
            merged_frame.to_wav_bytes(),
        )
        return lk_stt.SpeechEvent(
            type=lk_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                lk_stt.SpeechData(
                    language=transcript.language,
                    text=transcript.text,
                    confidence=transcript.confidence,
                )
            ],
        )

    async def aclose(self) -> None:
        return None


def build_livekit_stt(settings: AppSettings) -> LiveKitFasterWhisperSTT:
    if settings.voice_stt_provider != "faster_whisper":
        raise VoiceSTTError(f"Unsupported STT provider: {settings.voice_stt_provider}")
    return LiveKitFasterWhisperSTT(FasterWhisperArmenianSTT(settings))


def build_local_stt(settings: AppSettings) -> FasterWhisperArmenianSTT:
    if settings.voice_stt_provider != "faster_whisper":
        raise VoiceSTTError(f"Unsupported STT provider: {settings.voice_stt_provider}")
    return FasterWhisperArmenianSTT(settings)
