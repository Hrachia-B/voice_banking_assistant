from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.voice.tts import build_local_tts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Gemini API TTS.")
    parser.add_argument(
        "--text",
        default="Բարև։ Սա հայերեն ձայնային թեստ է բանկային օգնականի համար։",
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        default="tmp/gemini_tts_smoke.wav",
        help="Output WAV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    tts = build_local_tts(settings)
    output_path = Path(args.output)
    result = tts.synthesize_to_file(args.text, output_path)

    print(f"provider: {result.provider}")
    print(f"model: {tts.model_name}")
    print(f"output: {result.audio_path}")
    print(f"sample_rate: {result.sample_rate}")
    print(f"num_channels: {result.num_channels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
