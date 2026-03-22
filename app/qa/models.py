from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from app.guardrails.scope import ScopeDecision
from app.retrieval.models import RetrievalResult

QAStatus = Literal["answered", "refused"]
RefusalReason = Literal["out_of_scope", "insufficient_evidence", "ambiguous_request", "generation_error"]


@dataclass(frozen=True)
class EvidenceAssessment:
    status: Literal["supported", "insufficient", "ambiguous"]
    reason: str
    top_score: float
    total_score: float
    matched_query_terms: list[str]
    dominant_bank_codes: list[str]


@dataclass(frozen=True)
class GeneratedAnswer:
    decision: Literal["answer", "refuse"]
    answer_armenian: str
    used_chunk_ids: list[str]
    refusal_reason: str | None
    generator_name: str
    model_name: str | None = None


@dataclass(frozen=True)
class QAResult:
    query: str
    status: QAStatus
    answer_armenian: str
    scope_decision: ScopeDecision
    refusal_reason: RefusalReason | None
    evidence_assessment: EvidenceAssessment | None
    retrieved_results: list[RetrievalResult]
    supporting_results: list[RetrievalResult]
    generator_name: str
    model_name: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["retrieved_results"] = [result.to_record() for result in self.retrieved_results]
        record["supporting_results"] = [result.to_record() for result in self.supporting_results]
        return record
