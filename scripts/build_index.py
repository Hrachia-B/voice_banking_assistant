from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking.cleaner import BoilerplateCleaner, BoilerplateConfig
from app.chunking.chunker import ChunkingConfig, DocumentChunker
from app.config.settings import get_settings
from app.retrieval.indexer import build_hybrid_index, save_hybrid_index
from app.retrieval.storage import (
    find_latest_documents_path,
    load_source_documents,
    write_chunks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build retrieval chunks and a local index.")
    parser.add_argument("--documents", help="Path to processed documents.jsonl. Defaults to latest run.")
    parser.add_argument("--run-id", required=True, help="Output run id for chunks and index artifacts.")
    parser.add_argument("--chunk-size", type=int, default=900, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=180, help="Chunk overlap in characters.")
    parser.add_argument("--min-chars", type=int, default=120, help="Minimum chunk size in characters.")
    parser.add_argument(
        "--edge-window-lines",
        type=int,
        default=18,
        help="Number of top/bottom lines per document to inspect for repeated boilerplate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    documents_path = (
        Path(args.documents)
        if args.documents
        else find_latest_documents_path(settings.processed_data_dir)
    )
    if not documents_path.is_absolute():
        documents_path = settings.project_root / documents_path

    documents = load_source_documents(documents_path)
    boilerplate_cleaner = BoilerplateCleaner(
        BoilerplateConfig(edge_window_lines=args.edge_window_lines)
    )
    cleaned_documents, boilerplate_stats = boilerplate_cleaner.fit_transform(documents)
    chunking_config = ChunkingConfig(
        chunk_size_chars=args.chunk_size,
        chunk_overlap_chars=args.chunk_overlap,
        min_chunk_chars=args.min_chars,
    )
    chunker = DocumentChunker(chunking_config)
    chunks = chunker.chunk_documents(cleaned_documents)

    build_metadata = {
        "documents_path": str(documents_path),
        "document_count": len(documents),
        "cleaned_document_count": len(cleaned_documents),
        "chunk_size_chars": chunking_config.chunk_size_chars,
        "chunk_overlap_chars": chunking_config.chunk_overlap_chars,
        "min_chunk_chars": chunking_config.min_chunk_chars,
        "boilerplate_cleanup": {
            "edge_window_lines": args.edge_window_lines,
            "removed_lines_total": boilerplate_stats.removed_lines_total,
            "removed_pattern_lines": boilerplate_stats.removed_pattern_lines,
            "removed_repeated_lines": boilerplate_stats.removed_repeated_lines,
        },
    }

    chunks_dir = settings.chunks_data_dir / args.run_id
    chunks_path = write_chunks(chunks, chunks_dir, build_metadata)
    index = build_hybrid_index(chunks, build_metadata)
    index_dir = settings.indexes_data_dir / args.run_id
    save_hybrid_index(index, index_dir, chunks_path)

    logging.getLogger(__name__).info(
        "Built retrieval artifacts for %s: %s documents -> %s chunks",
        args.run_id,
        len(documents),
        len(chunks),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
