from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval.models import ChunkRecord
from app.retrieval.storage import write_json
from app.retrieval.text import normalize_text


WORD_WEIGHT = 0.45
CHAR_WEIGHT = 0.55


@dataclass
class HybridIndex:
    chunks: list[ChunkRecord]
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    word_matrix: Any
    char_matrix: Any
    build_metadata: dict[str, Any]


def build_hybrid_index(
    chunks: list[ChunkRecord],
    build_metadata: dict[str, Any],
) -> HybridIndex:
    if not chunks:
        raise ValueError("Cannot build an index with zero chunks.")

    corpus = [chunk.chunk_text for chunk in chunks]
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        preprocessor=normalize_text,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        preprocessor=normalize_text,
        sublinear_tf=True,
    )
    word_matrix = word_vectorizer.fit_transform(corpus)
    char_matrix = char_vectorizer.fit_transform(corpus)

    return HybridIndex(
        chunks=chunks,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        word_matrix=word_matrix,
        char_matrix=char_matrix,
        build_metadata=build_metadata,
    )


def save_hybrid_index(index: HybridIndex, index_dir: Path, chunks_path: Path) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "hybrid_tfidf.joblib"
    joblib.dump(index, index_path)

    manifest = {
        "index_type": "hybrid_tfidf",
        "chunk_count": len(index.chunks),
        "word_weight": WORD_WEIGHT,
        "char_weight": CHAR_WEIGHT,
        "chunks_path": str(chunks_path),
        "build_metadata": index.build_metadata,
    }
    write_json(index_dir / "manifest.json", manifest)
    return index_path


def load_hybrid_index(index_dir: Path) -> HybridIndex:
    index_path = index_dir / "hybrid_tfidf.joblib"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    return joblib.load(index_path)
