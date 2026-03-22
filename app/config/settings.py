from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


_load_dotenv(ENV_FILE)


def _resolve_path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@dataclass(frozen=True)
class AppSettings:
    project_root: Path
    banks_config_path: Path
    raw_data_dir: Path
    processed_data_dir: Path
    chunks_data_dir: Path
    indexes_data_dir: Path
    qa_gemini_model: str
    qa_generator: str
    qa_retrieval_top_k: int
    qa_context_chunks: int
    qa_min_top_score: float
    qa_min_total_score: float
    voice_index_dir: Path | None
    voice_agent_name: str
    voice_greeting_text: str
    livekit_url: str | None
    livekit_api_key: str | None
    livekit_api_secret: str | None
    voice_stt_provider: str
    voice_stt_model: str
    voice_stt_language: str
    voice_stt_device: str
    voice_stt_compute_type: str
    voice_stt_beam_size: int
    voice_tts_provider: str
    voice_tts_model: str
    voice_tts_language: str
    voice_tts_voice_name: str | None
    voice_tts_prompt: str
    voice_tts_sample_rate: int
    voice_tts_num_channels: int
    voice_vad_min_silence_duration: float
    voice_vad_activation_threshold: float
    request_timeout_seconds: int
    request_retries: int
    request_backoff_seconds: float
    user_agent: str
    log_level: str


def get_settings() -> AppSettings:
    return AppSettings(
        project_root=PROJECT_ROOT,
        banks_config_path=_resolve_path(
            os.getenv("BANKS_CONFIG_PATH", "app/config/banks.yaml"),
            PROJECT_ROOT / "app/config/banks.yaml",
        ),
        raw_data_dir=_resolve_path(
            os.getenv("RAW_DATA_DIR", "data/raw"),
            PROJECT_ROOT / "data/raw",
        ),
        processed_data_dir=_resolve_path(
            os.getenv("PROCESSED_DATA_DIR", "data/processed"),
            PROJECT_ROOT / "data/processed",
        ),
        chunks_data_dir=_resolve_path(
            os.getenv("CHUNKS_DATA_DIR", "data/chunks"),
            PROJECT_ROOT / "data/chunks",
        ),
        indexes_data_dir=_resolve_path(
            os.getenv("INDEXES_DATA_DIR", "data/indexes"),
            PROJECT_ROOT / "data/indexes",
        ),
        qa_gemini_model=os.getenv("QA_GEMINI_MODEL", "gemini-2.5-flash"),
        qa_generator=os.getenv("QA_GENERATOR", "gemini"),
        qa_retrieval_top_k=int(os.getenv("QA_RETRIEVAL_TOP_K", "6")),
        qa_context_chunks=int(os.getenv("QA_CONTEXT_CHUNKS", "4")),
        qa_min_top_score=float(os.getenv("QA_MIN_TOP_SCORE", "0.095")),
        qa_min_total_score=float(os.getenv("QA_MIN_TOTAL_SCORE", "0.22")),
        voice_index_dir=(
            _resolve_path(os.getenv("VOICE_INDEX_DIR", ""), PROJECT_ROOT / "data/indexes")
            if os.getenv("VOICE_INDEX_DIR")
            else None
        ),
        voice_agent_name=os.getenv("VOICE_AGENT_NAME", "armenian-bank-agent"),
        voice_greeting_text=os.getenv(
            "VOICE_GREETING_TEXT",
            "Բարև։ Կարող եք հարցնել վարկերի, ավանդների կամ մասնաճյուղերի մասին։",
        ),
        livekit_url=os.getenv("LIVEKIT_URL"),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
        voice_stt_provider=os.getenv("VOICE_STT_PROVIDER", "faster_whisper"),
        voice_stt_model=os.getenv("VOICE_STT_MODEL", "large-v3-turbo"),
        voice_stt_language=os.getenv("VOICE_STT_LANGUAGE", "hy"),
        voice_stt_device=os.getenv("VOICE_STT_DEVICE", "auto"),
        voice_stt_compute_type=os.getenv("VOICE_STT_COMPUTE_TYPE", "int8"),
        voice_stt_beam_size=int(os.getenv("VOICE_STT_BEAM_SIZE", "5")),
        voice_tts_provider=os.getenv("VOICE_TTS_PROVIDER", "gemini_api_tts"),
        voice_tts_model=os.getenv("VOICE_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
        voice_tts_language=os.getenv("VOICE_TTS_LANGUAGE", "hy-AM"),
        voice_tts_voice_name=os.getenv("VOICE_TTS_VOICE_NAME") or None,
        voice_tts_prompt=os.getenv(
            "VOICE_TTS_PROMPT",
            "Speak clearly in Armenian for a bank support assistant. Keep a calm, professional tone.",
        ),
        voice_tts_sample_rate=int(os.getenv("VOICE_TTS_SAMPLE_RATE", "24000")),
        voice_tts_num_channels=int(os.getenv("VOICE_TTS_NUM_CHANNELS", "1")),
        voice_vad_min_silence_duration=float(
            os.getenv("VOICE_VAD_MIN_SILENCE_DURATION", "0.55")
        ),
        voice_vad_activation_threshold=float(
            os.getenv("VOICE_VAD_ACTIVATION_THRESHOLD", "0.5")
        ),
        request_timeout_seconds=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "20")),
        request_retries=int(os.getenv("REQUEST_RETRIES", "2")),
        request_backoff_seconds=float(os.getenv("REQUEST_BACKOFF_SECONDS", "0.5")),
        user_agent=os.getenv(
            "USER_AGENT",
            "voice-ai-bank-agent-metric/0.1 (+https://example.local)",
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
