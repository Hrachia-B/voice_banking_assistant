from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.qa.factory import build_grounded_qa_service
from app.retrieval.storage import find_latest_index_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grounded QA over the bank retrieval index.")
    parser.add_argument("--query", required=True, help="User question in Armenian.")
    parser.add_argument("--index-dir", help="Path to an index directory. Defaults to latest index.")
    parser.add_argument(
        "--generator",
        choices=("gemini", "extractive"),
        help="Answer generation backend. Defaults to QA_GENERATOR.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full QA result as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    index_dir = Path(args.index_dir) if args.index_dir else find_latest_index_dir(settings.indexes_data_dir)
    if not index_dir.is_absolute():
        index_dir = settings.project_root / index_dir

    requested_generator = (args.generator or settings.qa_generator).casefold()
    if requested_generator == "gemini" and not os.getenv("GEMINI_API_KEY"):
        logging.getLogger(__name__).info(
            "GEMINI_API_KEY is not set; falling back to extractive grounded QA mode."
        )

    service = build_grounded_qa_service(
        settings,
        index_dir=index_dir,
        generator_mode=args.generator,
    )
    qa_result = service.answer_query(args.query)

    if args.json:
        print(json.dumps(qa_result.to_record(), ensure_ascii=False, indent=2))
        return 0

    print(f"scope: {qa_result.scope_decision.scope}")
    print(f"scope_reason: {qa_result.scope_decision.reason}")
    print(f"detected_bank: {qa_result.scope_decision.bank_code or 'not_specified'}")
    print(f"generator: {qa_result.generator_name}")
    print(f"model: {qa_result.model_name or 'n/a'}")
    print(f"retrieved_results: {len(qa_result.retrieved_results)}")
    print(f"supporting_results: {len(qa_result.supporting_results)}")
    if qa_result.evidence_assessment:
        print(f"evidence_status: {qa_result.evidence_assessment.status}")
        print(f"evidence_reason: {qa_result.evidence_assessment.reason}")
        print(f"top_score: {qa_result.evidence_assessment.top_score:.4f}")
        print(f"total_score: {qa_result.evidence_assessment.total_score:.4f}")
        if qa_result.evidence_assessment.matched_query_terms:
            print("matched_query_terms: " + ", ".join(qa_result.evidence_assessment.matched_query_terms))
    print()
    print("final_answer:")
    print(qa_result.answer_armenian)
    print()
    print("supporting_chunk_ids:")
    for result in qa_result.supporting_results:
        print(f"- {result.chunk_id}")
    print()
    print("supporting_sources:")
    for result in qa_result.supporting_results:
        print(f"- [{result.bank_code}/{result.topic}] {result.page_title} | {result.source_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
