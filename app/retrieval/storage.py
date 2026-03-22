from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.retrieval.models import ChunkRecord, SourceDocument


def load_source_documents(documents_path: Path) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    with documents_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            documents.append(SourceDocument.from_record(json.loads(line)))
    return documents


def write_chunks(
    chunks: list[ChunkRecord],
    output_dir: Path,
    build_metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_record(), ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    summary = {
        "chunk_count": len(chunks),
        "build_metadata": build_metadata,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks_path


def load_chunks(chunks_path: Path) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            chunks.append(ChunkRecord.from_record(json.loads(line)))
    return chunks


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_latest_documents_path(processed_root: Path) -> Path:
    candidates = [
        path
        for path in processed_root.glob("*/documents.jsonl")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError("No processed documents.jsonl files were found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def find_latest_index_dir(indexes_root: Path) -> Path:
    candidates = [
        path
        for path in indexes_root.iterdir()
        if path.is_dir() and (path / "hybrid_tfidf.joblib").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No retrieval indexes were found.")
    return max(candidates, key=lambda path: path.stat().st_mtime)
