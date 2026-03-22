from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import get_settings
from app.retrieval.indexer import load_hybrid_index
from app.retrieval.retriever import ChunkRetriever, RetrievalFilters
from app.retrieval.storage import find_latest_index_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual retrieval query.")
    parser.add_argument("--query", required=True, help="Search query text.")
    parser.add_argument("--index-dir", help="Path to an index directory. Defaults to latest index.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    parser.add_argument("--bank", help="Optional bank name or bank code filter.")
    parser.add_argument("--topic", help="Optional topic filter.")
    parser.add_argument("--source-url", help="Optional exact source URL filter.")
    parser.add_argument("--allow-duplicates", action="store_true", help="Disable top-k dedup suppression.")
    parser.add_argument("--json", action="store_true", help="Print results as JSON.")
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

    index = load_hybrid_index(index_dir)
    retriever = ChunkRetriever(index)
    results = retriever.search(
        query=args.query,
        top_k=args.top_k,
        filters=RetrievalFilters(
            bank_name=args.bank,
            topic=args.topic,
            source_url=args.source_url,
        ),
        deduplicate=not args.allow_duplicates,
    )

    if args.json:
        print(json.dumps([result.to_record() for result in results], ensure_ascii=False, indent=2))
        return 0

    if not results:
        print("No retrieval results matched the query and filters.")
        return 0

    for result in results:
        print(f"[{result.rank}] score={result.score:.4f} bank={result.bank_code} topic={result.topic}")
        print(f"source: {result.source_url}")
        print(f"title: {result.page_title}")
        print(result.chunk_text[:800].strip())
        print("-" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
