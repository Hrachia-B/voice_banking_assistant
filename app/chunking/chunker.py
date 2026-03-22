from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.retrieval.models import ChunkRecord, SourceDocument
from app.retrieval.text import normalize_text, split_paragraphs, split_sentences


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 180
    min_chunk_chars: int = 120


class DocumentChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self._config = config

    def chunk_documents(self, documents: list[SourceDocument]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks

    def chunk_document(self, document: SourceDocument) -> list[ChunkRecord]:
        text = document.cleaned_text.strip()
        if not text:
            return []

        units = self._build_units(text)
        if not units:
            return []

        chunk_texts = self._assemble_chunks(units)
        chunk_records: list[ChunkRecord] = []
        for chunk_index, chunk_text in enumerate(chunk_texts):
            stable_id_source = f"{document.document_id}|{chunk_index}|{chunk_text}"
            chunk_hash = hashlib.sha256(stable_id_source.encode("utf-8")).hexdigest()[:12]
            metadata = {
                "chunk_char_count": len(chunk_text),
                "source_format": document.metadata.get("source_format"),
                "content_type": document.metadata.get("content_type"),
            }
            chunk_records.append(
                ChunkRecord(
                    chunk_id=f"{document.document_id}-chunk-{chunk_hash}",
                    parent_document_id=document.document_id,
                    bank_name=document.bank_name,
                    bank_code=document.bank_code,
                    topic=document.topic,
                    source_url=document.source_url,
                    page_title=document.page_title,
                    scraped_at=document.scraped_at,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )
        return chunk_records

    def _build_units(self, text: str) -> list[str]:
        units: list[str] = []
        paragraphs = split_paragraphs(text)
        if len(paragraphs) <= 1:
            paragraphs = _split_line_units(text)
        paragraphs = paragraphs or [normalize_text(text)]
        for paragraph in paragraphs:
            if len(paragraph) <= self._config.chunk_size_chars:
                units.append(paragraph)
                continue

            sentences = split_sentences(paragraph)
            if len(sentences) <= 1:
                units.extend(self._split_long_text(paragraph))
                continue

            for sentence in sentences:
                if len(sentence) <= self._config.chunk_size_chars:
                    units.append(sentence)
                else:
                    units.extend(self._split_long_text(sentence))
        return [unit for unit in units if unit]

    def _split_long_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        parts: list[str] = []
        current_words: list[str] = []
        current_len = 0
        for word in words:
            projected = current_len + len(word) + (1 if current_words else 0)
            if current_words and projected > self._config.chunk_size_chars:
                parts.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len = projected

        if current_words:
            parts.append(" ".join(current_words))
        return parts

    def _assemble_chunks(self, units: list[str]) -> list[str]:
        chunks: list[str] = []
        current_units: list[str] = []
        current_len = 0

        for unit in units:
            projected = current_len + len(unit) + (2 if current_units else 0)
            if current_units and projected > self._config.chunk_size_chars:
                chunk_text = "\n\n".join(current_units).strip()
                chunks.append(chunk_text)
                current_units = self._overlap_seed(current_units)
                current_len = self._joined_length(current_units)
                projected = current_len + len(unit) + (2 if current_units else 0)
                while current_units and projected > self._config.chunk_size_chars:
                    current_units.pop(0)
                    current_len = self._joined_length(current_units)
                    projected = current_len + len(unit) + (2 if current_units else 0)

            current_units.append(unit)
            current_len = self._joined_length(current_units)

        if current_units:
            chunk_text = "\n\n".join(current_units).strip()
            if not chunks or len(chunk_text) >= self._config.min_chunk_chars:
                chunks.append(chunk_text)
            elif chunks:
                chunks[-1] = f"{chunks[-1]}\n\n{chunk_text}".strip()

        return chunks

    def _overlap_seed(self, units: list[str]) -> list[str]:
        if self._config.chunk_overlap_chars <= 0:
            return []

        overlap_units: list[str] = []
        current = 0
        for unit in reversed(units):
            overlap_units.insert(0, unit)
            current += len(unit) + (2 if current else 0)
            if current >= self._config.chunk_overlap_chars:
                break
        return overlap_units

    @staticmethod
    def _joined_length(units: list[str]) -> int:
        if not units:
            return 0
        return len("\n\n".join(units))


def _split_line_units(text: str) -> list[str]:
    units = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    return units
