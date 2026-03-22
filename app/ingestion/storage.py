from __future__ import annotations

import json
from pathlib import Path

from app.ingestion.models import ScrapeFailure, ScrapedDocument


class DocumentStore:
    def __init__(self, raw_root: Path, processed_root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.raw_run_dir = raw_root / run_id
        self.processed_run_dir = processed_root / run_id
        self.documents_dir = self.processed_run_dir / "documents"
        self.raw_run_dir.mkdir(parents=True, exist_ok=True)
        self.processed_run_dir.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self._documents_jsonl = self.processed_run_dir / "documents.jsonl"
        self._failures_jsonl = self.processed_run_dir / "failures.jsonl"

    def save_document(self, document: ScrapedDocument, raw_content: bytes) -> Path:
        source_format = document.metadata["source_format"]
        extension = ".pdf" if source_format == "pdf" else ".html"
        raw_artifact_path = (
            self.raw_run_dir
            / document.bank_code
            / document.topic
            / f"{document.document_id}{extension}"
        )
        raw_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        raw_artifact_path.write_bytes(raw_content)

        record = document.to_record()
        record["metadata"]["raw_artifact_path"] = str(raw_artifact_path)

        with self._documents_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        document_json_path = self.documents_dir / f"{document.document_id}.json"
        document_json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return raw_artifact_path

    def save_failure(self, failure: ScrapeFailure) -> None:
        with self._failures_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(failure.to_record(), ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, object]) -> None:
        summary_path = self.processed_run_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
