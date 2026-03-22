from __future__ import annotations

from pathlib import Path

from app.config.settings import AppSettings
from app.qa.evidence import EvidenceThresholds
from app.qa.generator import build_answer_generator
from app.qa.pipeline import GroundedQAService
from app.retrieval.indexer import load_hybrid_index
from app.retrieval.retriever import ChunkRetriever
from app.retrieval.storage import find_latest_index_dir


def build_grounded_qa_service(
    settings: AppSettings,
    *,
    index_dir: Path | None = None,
    generator_mode: str | None = None,
) -> GroundedQAService:
    resolved_index_dir = index_dir or settings.voice_index_dir
    if resolved_index_dir is None:
        resolved_index_dir = find_latest_index_dir(settings.indexes_data_dir)
    elif not resolved_index_dir.is_absolute():
        resolved_index_dir = settings.project_root / resolved_index_dir

    retriever = ChunkRetriever(load_hybrid_index(resolved_index_dir))
    return GroundedQAService(
        retriever=retriever,
        answer_generator=build_answer_generator(settings, mode=generator_mode),
        evidence_thresholds=EvidenceThresholds(
            min_top_score=settings.qa_min_top_score,
            min_total_score=settings.qa_min_total_score,
        ),
        retrieval_top_k=settings.qa_retrieval_top_k,
        context_chunks=settings.qa_context_chunks,
    )
