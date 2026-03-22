from __future__ import annotations

import asyncio
from io import BytesIO
import logging
import os
from pathlib import Path
import wave

from livekit.agents import tts as lk_tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.utils import shortuuid

from app.config.settings import AppSettings
from app.voice.models import SpeechSynthesisResult


LOGGER = logging.getLogger(__name__)

# Official Gemini API TTS docs list supported languageCode values, and hy-AM is
# not currently among them. We keep the env setting for transparency, but fall
# back to prompt-only steering to avoid a hard runtime failure.
_GEMINI_TTS_SUPPORTED_LANGUAGE_CODES = {
    "ar-XA",
    "bn-IN",
    "cmn-CN",
    "de-DE",
    "en-AU",
    "en-GB",
    "en-IN",
    "en-US",
    "es-ES",
    "es-US",
    "fr-CA",
    "fr-FR",
    "gu-IN",
    "hi-IN",
    "id-ID",
    "it-IT",
    "ja-JP",
    "kn-IN",
    "ko-KR",
    "ml-IN",
    "mr-IN",
    "nl-NL",
    "pl-PL",
    "pt-BR",
    "ru-RU",
    "ta-IN",
    "te-IN",
    "th-TH",
    "tr-TR",
    "vi-VN",
}


class VoiceTTSError(RuntimeError):
    """Raised when Armenian TTS synthesis fails."""


class GeminiAPITTS:
    def __init__(self, settings: AppSettings) -> None:
        self._model_name = settings.voice_tts_model
        self._language = settings.voice_tts_language
        self._voice_name = settings.voice_tts_voice_name or "Kore"
        self._prompt = settings.voice_tts_prompt
        self._sample_rate = settings.voice_tts_sample_rate
        self._num_channels = settings.voice_tts_num_channels
        self._client = None
        self._warned_about_language = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def prewarm(self) -> None:
        self._get_client()

    def synthesize_to_file(self, text: str, output_path: Path) -> SpeechSynthesisResult:
        wav_bytes = self.synthesize_wav_bytes(text)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(wav_bytes)
        return SpeechSynthesisResult(
            audio_path=output_path,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
            provider="gemini_api_tts",
        )

    def synthesize_pcm_bytes(self, text: str) -> bytes:
        response = self._generate_audio(text)
        return _extract_pcm_audio(response)

    def synthesize_wav_bytes(self, text: str) -> bytes:
        pcm_bytes = self.synthesize_pcm_bytes(text)
        return _wrap_pcm_as_wav(
            pcm_bytes,
            sample_rate=self._sample_rate,
            num_channels=self._num_channels,
        )

    def _generate_audio(self, text: str):
        try:
            from google.genai import types
        except ImportError as exc:
            raise VoiceTTSError("The google-genai package is not installed.") from exc

        contents = self._build_prompt(text)
        config_kwargs: dict[str, object] = {
            "response_modalities": ["AUDIO"],
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self._voice_name,
                    )
                ),
            ),
        }
        language_code = self._speech_language_code()
        if language_code:
            speech_config = config_kwargs["speech_config"]
            assert isinstance(speech_config, types.SpeechConfig)
            speech_config.language_code = language_code

        response = self._get_client().models.generate_content(
            model=self._model_name,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response

    def _speech_language_code(self) -> str | None:
        if self._language in _GEMINI_TTS_SUPPORTED_LANGUAGE_CODES:
            return self._language
        if not self._warned_about_language:
            LOGGER.warning(
                "Gemini API TTS languageCode %s is not currently listed as supported; "
                "falling back to prompt-only Armenian steering.",
                self._language,
            )
            self._warned_about_language = True
        return None

    def _build_prompt(self, text: str) -> str:
        instructions = [self._prompt.strip(), "Speak the following reply in Armenian."]
        if self._language:
            instructions.append(f"Preferred locale: {self._language}.")
        instructions.append("Read the text exactly and do not add extra words.")
        instructions.append(f"Text: {text.strip()}")
        return "\n".join(part for part in instructions if part)

    def _get_client(self):
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise VoiceTTSError("GEMINI_API_KEY must be set for Gemini API TTS.")
            try:
                from google import genai
            except ImportError as exc:
                raise VoiceTTSError("The google-genai package is not installed.") from exc
            self._client = genai.Client(api_key=api_key)
        return self._client


class LiveKitGeminiAPITTS(lk_tts.TTS):
    def __init__(self, provider: GeminiAPITTS) -> None:
        super().__init__(
            capabilities=lk_tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=provider.sample_rate,
            num_channels=provider.num_channels,
        )
        self._provider = provider

    @property
    def model(self) -> str:
        return self._provider.model_name

    @property
    def provider(self) -> str:
        return "gemini_api_tts"

    def prewarm(self) -> None:
        self._provider.prewarm()

    def synthesize(
        self,
        text: str,
        *,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_GeminiAPIChunkedStream":
        return _GeminiAPIChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            provider=self._provider,
        )

    async def aclose(self) -> None:
        return None


class _GeminiAPIChunkedStream(lk_tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: LiveKitGeminiAPITTS,
        input_text: str,
        conn_options,
        provider: GeminiAPITTS,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._provider = provider

    async def _run(self, output_emitter: lk_tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
            stream=False,
        )
        pcm_bytes = await asyncio.to_thread(self._provider.synthesize_pcm_bytes, self._input_text)
        output_emitter.push(pcm_bytes)
        output_emitter.flush()


def build_livekit_tts(settings: AppSettings) -> LiveKitGeminiAPITTS:
    if settings.voice_tts_provider != "gemini_api_tts":
        raise VoiceTTSError(f"Unsupported TTS provider: {settings.voice_tts_provider}")
    return LiveKitGeminiAPITTS(GeminiAPITTS(settings))


def build_local_tts(settings: AppSettings) -> GeminiAPITTS:
    if settings.voice_tts_provider != "gemini_api_tts":
        raise VoiceTTSError(f"Unsupported TTS provider: {settings.voice_tts_provider}")
    return GeminiAPITTS(settings)


def _extract_pcm_audio(response) -> bytes:
    try:
        parts = response.candidates[0].content.parts
    except (AttributeError, IndexError, TypeError) as exc:
        raise VoiceTTSError("Gemini API TTS returned an unexpected response structure.") from exc

    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        data = getattr(inline_data, "data", None)
        if data:
            return bytes(data)
    raise VoiceTTSError("Gemini API TTS returned empty audio.")


def _wrap_pcm_as_wav(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    num_channels: int,
    sample_width: int = 2,
) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(num_channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_bytes)
    return buffer.getvalue()
