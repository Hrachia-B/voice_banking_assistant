from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from app.retrieval.indexer import CHAR_WEIGHT, WORD_WEIGHT, HybridIndex
from app.retrieval.models import ChunkRecord, RetrievalResult
from app.retrieval.text import normalize_text


@dataclass(frozen=True)
class RetrievalFilters:
    bank_name: str | None = None
    topic: str | None = None
    source_url: str | None = None


class ChunkRetriever:
    def __init__(self, index: HybridIndex) -> None:
        self._index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
        deduplicate: bool = True,
        near_duplicate_threshold: float = 0.92,
    ) -> list[RetrievalResult]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return []

        filters = filters or RetrievalFilters()
        word_query = self._index.word_vectorizer.transform([normalized_query])
        char_query = self._index.char_vectorizer.transform([normalized_query])

        word_scores = (self._index.word_matrix @ word_query.T).toarray().ravel()
        char_scores = (self._index.char_matrix @ char_query.T).toarray().ravel()
        combined_scores = (WORD_WEIGHT * word_scores) + (CHAR_WEIGHT * char_scores)

        candidates: list[tuple[int, float, float, float]] = []
        for idx, chunk in enumerate(self._index.chunks):
            if not _matches_filters(chunk, filters):
                continue
            score = float(combined_scores[idx])
            if score <= 0:
                continue
            candidates.append((idx, score, float(word_scores[idx]), float(char_scores[idx])))

        candidates.sort(key=lambda item: (-item[1], item[0]))
        results: list[RetrievalResult] = []
        kept_normalized_texts: list[str] = []
        for idx, score, word_score, char_score in candidates:
            chunk = self._index.chunks[idx]
            normalized_chunk_text = normalize_text(chunk.chunk_text).casefold()
            if deduplicate and _is_duplicate_candidate(
                normalized_chunk_text,
                kept_normalized_texts,
                near_duplicate_threshold,
            ):
                continue
            kept_normalized_texts.append(normalized_chunk_text)
            results.append(
                RetrievalResult(
                    rank=len(results) + 1,
                    score=score,
                    word_score=word_score,
                    char_score=char_score,
                    chunk_id=chunk.chunk_id,
                    parent_document_id=chunk.parent_document_id,
                    bank_name=chunk.bank_name,
                    bank_code=chunk.bank_code,
                    topic=chunk.topic,
                    source_url=chunk.source_url,
                    page_title=chunk.page_title,
                    chunk_text=chunk.chunk_text,
                    metadata=chunk.metadata,
                )
            )
            if len(results) >= top_k:
                break
        return results


def _matches_filters(chunk: ChunkRecord, filters: RetrievalFilters) -> bool:
    if filters.bank_name:
        bank_filter = filters.bank_name.casefold()
        if bank_filter not in {chunk.bank_name.casefold(), chunk.bank_code.casefold()}:
            return False
    if filters.topic and chunk.topic.casefold() != filters.topic.casefold():
        return False
    if filters.source_url and chunk.source_url != filters.source_url:
        return False
    return True


def _is_duplicate_candidate(
    candidate_text: str,
    kept_texts: list[str],
    near_duplicate_threshold: float,
) -> bool:
    if not candidate_text:
        return True
    for kept_text in kept_texts:
        if candidate_text == kept_text:
            return True
        if SequenceMatcher(None, candidate_text, kept_text).ratio() >= near_duplicate_threshold:
            return True
    return False
