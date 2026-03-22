from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.qa.factory import build_grounded_qa_service
from app.voice.service import VoicePipelineService
from app.voice.stt import build_local_stt
from app.voice.tts import build_local_tts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local voice pipeline without LiveKit.")
    parser.add_argument("--text", help="Skip STT and use this Armenian text directly.")
    parser.add_argument("--audio-in", help="Path to an input WAV/MP3/OGG file for STT.")
    parser.add_argument(
        "--audio-out",
        help="Optional path to save the Armenian TTS output WAV.",
    )
    parser.add_argument(
        "--index-dir",
        help="Path to a retrieval index directory. Defaults to latest configured index.",
    )
    parser.add_argument(
        "--generator",
        choices=("gemini", "extractive"),
        help="Grounded QA answer generator override.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON record.")
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    index_dir = Path(args.index_dir) if args.index_dir else None
    qa_service = build_grounded_qa_service(settings, index_dir=index_dir, generator_mode=args.generator)
    stt_provider = build_local_stt(settings) if args.audio_in else None
    tts_provider = build_local_tts(settings) if args.audio_out else None
    service = VoicePipelineService(
        qa_service=qa_service,
        stt_provider=stt_provider,
        tts_provider=tts_provider,
    )

    output_audio_path = Path(args.audio_out) if args.audio_out else None
    if args.text:
        result = await service.process_text(args.text)
        if output_audio_path is not None:
            synthesis = await asyncio.to_thread(
                tts_provider.synthesize_to_file,
                result.spoken_text,
                output_audio_path,
            )
            result = replace(result, synthesis=synthesis)
    elif args.audio_in:
        result = await service.process_audio_file(
            Path(args.audio_in),
            output_audio_path=output_audio_path,
        )
    else:
        raise SystemExit("Provide either --text or --audio-in.")

    if args.json:
        payload = {
            "transcript": result.transcript.text,
            "spoken_text": result.spoken_text,
            "qa_result": result.qa_result.to_record(),
            "synthesis_path": str(result.synthesis.audio_path) if result.synthesis else None,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"transcript: {result.transcript.text}")
    print(f"qa_status: {result.qa_result.status}")
    print(f"scope: {result.qa_result.scope_decision.scope}")
    print(f"answer_provider: {result.qa_result.generator_name}")
    print()
    print("spoken_text:")
    print(result.spoken_text)
    if result.synthesis:
        print()
        print(f"audio_out: {result.synthesis.audio_path}")
    return 0


def main() -> int:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
