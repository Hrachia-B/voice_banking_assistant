from __future__ import annotations

from dataclasses import dataclass

from app.guardrails.scope import ScopeDecision
from app.qa.query_normalization import normalized_query_tokens, tokens_match
from app.qa.models import EvidenceAssessment
from app.retrieval.models import RetrievalResult
from app.retrieval.text import normalize_text

STOPWORDS = {
    "ինչ",
    "ինչպես",
    "որ",
    "է",
    "եմ",
    "ես",
    "մասին",
    "կարո՞ղ",
    "կարող",
    "պատմիր",
    "ասա",
    "տեղեկություն",
    "տեղեկություններ",
    "որտեղ",
}

COMPARATIVE_HINTS = ("որ բանկ", "որոնք", "բոլոր բանկ", "համեմատ", "տարբեր բանկ", "քանի բանկ")


@dataclass(frozen=True)
class EvidenceThresholds:
    min_top_score: float = 0.095
    min_total_score: float = 0.22
    branch_min_top_score: float = 0.045
    branch_min_total_score: float = 0.14
    bank_ambiguity_margin: float = 0.025
    bank_ambiguity_relative: float = 0.85


def assess_retrieval_evidence(
    query: str,
    scope_decision: ScopeDecision,
    results: list[RetrievalResult],
    thresholds: EvidenceThresholds,
) -> EvidenceAssessment:
    if not results:
        return EvidenceAssessment(
            status="insufficient",
            reason="no_retrieval_results",
            top_score=0.0,
            total_score=0.0,
            matched_query_terms=[],
            dominant_bank_codes=[],
        )

    top_score = results[0].score
    total_score = sum(result.score for result in results[:3])
    matched_query_terms = _matched_query_terms(query, results[:3], scope_decision)
    dominant_bank_codes = _dominant_bank_codes(results, thresholds)
    min_top_score, min_total_score = _topic_thresholds(scope_decision, thresholds)

    if _should_request_bank_clarification(scope_decision, results, top_score, min_top_score):
        return EvidenceAssessment(
            status="ambiguous",
            reason="bank_clarification_needed",
            top_score=top_score,
            total_score=total_score,
            matched_query_terms=matched_query_terms,
            dominant_bank_codes=dominant_bank_codes,
        )

    if (top_score < min_top_score or total_score < min_total_score) and not _has_single_result_support(
        scope_decision,
        results,
        top_score,
        min_top_score,
    ):
        return EvidenceAssessment(
            status="insufficient",
            reason="retrieval_score_too_low",
            top_score=top_score,
            total_score=total_score,
            matched_query_terms=matched_query_terms,
            dominant_bank_codes=dominant_bank_codes,
        )

    if not matched_query_terms and not _has_structural_support(scope_decision, results, top_score, min_top_score):
        return EvidenceAssessment(
            status="insufficient",
            reason="query_terms_not_supported_in_context",
            top_score=top_score,
            total_score=total_score,
            matched_query_terms=matched_query_terms,
            dominant_bank_codes=dominant_bank_codes,
        )

    if scope_decision.bank_code is None and len(dominant_bank_codes) > 1 and not _is_comparative_query(query):
        return EvidenceAssessment(
            status="ambiguous",
            reason="multiple_banks_match_query",
            top_score=top_score,
            total_score=total_score,
            matched_query_terms=matched_query_terms,
            dominant_bank_codes=dominant_bank_codes,
        )

    return EvidenceAssessment(
        status="supported",
        reason="evidence_thresholds_met",
        top_score=top_score,
        total_score=total_score,
        matched_query_terms=matched_query_terms,
        dominant_bank_codes=dominant_bank_codes,
    )


def _matched_query_terms(
    query: str,
    results: list[RetrievalResult],
    scope_decision: ScopeDecision,
) -> list[str]:
    query_terms = {
        token for token in normalized_query_tokens(query) if len(token) > 2 and token not in STOPWORDS
    }
    context_terms: set[str] = set()
    for result in results:
        context_terms.update(_normalized_context_terms(result))

    matched_terms = sorted(
        term
        for term in query_terms
        if any(tokens_match(term, context_term) for context_term in context_terms)
    )
    if matched_terms:
        return matched_terms

    if _has_structural_support(scope_decision, results, results[0].score if results else 0.0, 0.0):
        return scope_decision.matched_topic_terms[:1]
    return []


def _dominant_bank_codes(
    results: list[RetrievalResult],
    thresholds: EvidenceThresholds,
) -> list[str]:
    top_score = results[0].score
    cutoff = max(top_score - thresholds.bank_ambiguity_margin, top_score * thresholds.bank_ambiguity_relative)
    banks = []
    seen: set[str] = set()
    for result in results[:3]:
        if result.score < cutoff:
            continue
        if result.bank_code in seen:
            continue
        seen.add(result.bank_code)
        banks.append(result.bank_code)
    return banks


def _is_comparative_query(query: str) -> bool:
    normalized = normalize_text(query).casefold()
    return any(hint in normalized for hint in COMPARATIVE_HINTS)


def _topic_thresholds(
    scope_decision: ScopeDecision,
    thresholds: EvidenceThresholds,
) -> tuple[float, float]:
    if scope_decision.scope == "branches":
        return thresholds.branch_min_top_score, thresholds.branch_min_total_score
    return thresholds.min_top_score, thresholds.min_total_score


def _normalized_context_terms(result: RetrievalResult) -> set[str]:
    terms = set(normalized_query_tokens(result.chunk_text))
    terms.update(normalized_query_tokens(result.page_title))
    return terms


def _has_structural_support(
    scope_decision: ScopeDecision,
    results: list[RetrievalResult],
    top_score: float,
    min_top_score: float,
) -> bool:
    if not results:
        return False
    top_result = results[0]
    if top_result.topic != scope_decision.scope:
        return False
    if scope_decision.bank_code and top_result.bank_code != scope_decision.bank_code:
        return False
    if scope_decision.matched_topic_terms and top_score >= max(min_top_score, 0.08):
        return True
    return False


def _should_request_bank_clarification(
    scope_decision: ScopeDecision,
    results: list[RetrievalResult],
    top_score: float,
    min_top_score: float,
) -> bool:
    if scope_decision.bank_code is not None or not results:
        return False
    unique_banks = {result.bank_code for result in results[:3]}
    if len(unique_banks) < 2:
        return False
    if top_score < (min_top_score * 0.75):
        return False
    return True


def _has_single_result_support(
    scope_decision: ScopeDecision,
    results: list[RetrievalResult],
    top_score: float,
    min_top_score: float,
) -> bool:
    if len(results) != 1:
        return False
    if top_score < max(min_top_score * 1.5, 0.14):
        return False
    return _has_structural_support(scope_decision, results, top_score, min_top_score)
