from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.loader import load_bank_registry
from app.config.settings import get_settings
from app.ingestion.models import SUPPORTED_TOPICS, TopicName
from app.ingestion.pipeline import ScrapePipeline, build_run_id
from app.ingestion.storage import DocumentStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape configured Armenian banks.")
    parser.add_argument(
        "--config",
        help="Optional path to a YAML bank config file. Defaults to BANKS_CONFIG_PATH.",
    )
    parser.add_argument(
        "--bank",
        action="append",
        help="Bank code to scrape. Can be passed multiple times.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        choices=SUPPORTED_TOPICS,
        help="Topic to scrape. Can be passed multiple times.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional explicit run id. Defaults to a UTC timestamp.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    config_path = settings.banks_config_path
    if args.config:
        config_override = Path(args.config)
        config_path = (
            config_override
            if config_override.is_absolute()
            else settings.project_root / config_override
        )
    registry = load_bank_registry(config_path)
    run_id = args.run_id or build_run_id()
    store = DocumentStore(
        raw_root=settings.raw_data_dir,
        processed_root=settings.processed_data_dir,
        run_id=run_id,
    )
    pipeline = ScrapePipeline(settings=settings, store=store)

    selected_banks = set(args.bank or [])
    selected_topics = set(args.topic or [])
    totals = {"documents": 0, "failures": 0}
    bank_summaries: dict[str, dict[str, dict[str, int]]] = {}

    for bank in registry.banks:
        if selected_banks and bank.bank_code not in selected_banks:
            continue

        stats = pipeline.scrape_bank(
            bank=bank,
            selected_topics=cast_topics(selected_topics),
        )
        bank_summaries[bank.bank_code] = {}
        for topic, topic_stats in stats.items():
            bank_summaries[bank.bank_code][topic] = {
                "documents": topic_stats.documents,
                "failures": topic_stats.failures,
            }
            totals["documents"] += topic_stats.documents
            totals["failures"] += topic_stats.failures

    summary = {
        "run_id": run_id,
        "config_path": str(config_path),
        "totals": totals,
        "banks": bank_summaries,
    }
    store.write_summary(summary)
    logging.getLogger(__name__).info("Finished run %s: %s", run_id, summary)
    return 0


def cast_topics(values: Iterable[str]) -> set[TopicName] | None:
    topics = {value for value in values if value in SUPPORTED_TOPICS}
    if not topics:
        return None
    return topics  # type: ignore[return-value]


if __name__ == "__main__":
    raise SystemExit(main())
