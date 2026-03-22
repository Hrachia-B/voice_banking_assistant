from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    bank_name: str
    bank_code: str
    topic: str
    source_url: str
    page_title: str
    scraped_at: str
    cleaned_text: str
    metadata: dict[str, Any]

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "SourceDocument":
        return cls(
            document_id=str(record["document_id"]),
            bank_name=str(record["bank_name"]),
            bank_code=str(record["bank_code"]),
            topic=str(record["topic"]),
            source_url=str(record["source_url"]),
            page_title=str(record["page_title"]),
            scraped_at=str(record["scraped_at"]),
            cleaned_text=str(record["cleaned_text"]),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    parent_document_id: str
    bank_name: str
    bank_code: str
    topic: str
    source_url: str
    page_title: str
    scraped_at: str
    chunk_index: int
    chunk_text: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "ChunkRecord":
        return cls(
            chunk_id=str(record["chunk_id"]),
            parent_document_id=str(record["parent_document_id"]),
            bank_name=str(record["bank_name"]),
            bank_code=str(record["bank_code"]),
            topic=str(record["topic"]),
            source_url=str(record["source_url"]),
            page_title=str(record["page_title"]),
            scraped_at=str(record["scraped_at"]),
            chunk_index=int(record["chunk_index"]),
            chunk_text=str(record["chunk_text"]),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    score: float
    word_score: float
    char_score: float
    chunk_id: str
    parent_document_id: str
    bank_name: str
    bank_code: str
    topic: str
    source_url: str
    page_title: str
    chunk_text: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
