from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from app.qa.pipeline import GroundedQAService
from app.voice.models import SpeechSynthesisResult, SpeechTranscript, VoiceTurnResult


class STTProvider(Protocol):
    def transcribe_file(self, audio_path: Path) -> SpeechTranscript:
        ...


class TTSProvider(Protocol):
    def synthesize_to_file(self, text: str, output_path: Path) -> SpeechSynthesisResult:
        ...


class VoicePipelineService:
    def __init__(
        self,
        *,
        qa_service: GroundedQAService,
        stt_provider: STTProvider | None = None,
        tts_provider: TTSProvider | None = None,
    ) -> None:
        self._qa_service = qa_service
        self._stt_provider = stt_provider
        self._tts_provider = tts_provider

    async def process_text(self, transcript_text: str) -> VoiceTurnResult:
        transcript = SpeechTranscript(
            text=transcript_text.strip(),
            language="hy",
            confidence=1.0,
        )
        qa_result = await asyncio.to_thread(self._qa_service.answer_query, transcript.text)
        return VoiceTurnResult(
            transcript=transcript,
            qa_result=qa_result,
            spoken_text=qa_result.answer_armenian,
        )

    async def process_audio_file(
        self,
        audio_path: Path,
        *,
        output_audio_path: Path | None = None,
    ) -> VoiceTurnResult:
        if self._stt_provider is None:
            raise RuntimeError("Audio-file testing requires an STT provider.")

        transcript = await asyncio.to_thread(self._stt_provider.transcribe_file, audio_path)
        qa_result = await asyncio.to_thread(self._qa_service.answer_query, transcript.text)

        synthesis: SpeechSynthesisResult | None = None
        if output_audio_path is not None:
            if self._tts_provider is None:
                raise RuntimeError("Audio-file synthesis requires a TTS provider.")
            synthesis = await asyncio.to_thread(
                self._tts_provider.synthesize_to_file,
                qa_result.answer_armenian,
                output_audio_path,
            )

        return VoiceTurnResult(
            transcript=transcript,
            qa_result=qa_result,
            spoken_text=qa_result.answer_armenian,
            synthesis=synthesis,
        )
