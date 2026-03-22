from __future__ import annotations

from app.guardrails.scope import QueryScopeGuard
from app.qa.query_normalization import normalize_user_query
from app.qa.evidence import EvidenceThresholds, assess_retrieval_evidence
from app.qa.generator import AnswerGenerator, AnswerGeneratorError
from app.qa.models import QAResult
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import ChunkRetriever, RetrievalFilters


class GroundedQAService:
    def __init__(
        self,
        retriever: ChunkRetriever,
        answer_generator: AnswerGenerator,
        scope_guard: QueryScopeGuard | None = None,
        evidence_thresholds: EvidenceThresholds | None = None,
        retrieval_top_k: int = 6,
        context_chunks: int = 4,
    ) -> None:
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._scope_guard = scope_guard or QueryScopeGuard()
        self._evidence_thresholds = evidence_thresholds or EvidenceThresholds()
        self._retrieval_top_k = retrieval_top_k
        self._context_chunks = context_chunks

    def answer_query(self, query: str) -> QAResult:
        normalized_query = normalize_user_query(query)
        scope_decision = self._scope_guard.classify(normalized_query)
        if not scope_decision.is_allowed:
            return QAResult(
                query=query,
                status="refused",
                answer_armenian=_out_of_scope_message(scope_decision.reason),
                scope_decision=scope_decision,
                refusal_reason="out_of_scope",
                evidence_assessment=None,
                retrieved_results=[],
                supporting_results=[],
                generator_name="none",
                model_name=None,
            )

        retrieval_results = self._retriever.search(
            query=normalized_query,
            top_k=self._retrieval_top_k,
            filters=RetrievalFilters(
                bank_name=scope_decision.bank_code,
                topic=scope_decision.scope,
            ),
        )
        evidence_assessment = assess_retrieval_evidence(
            query=normalized_query,
            scope_decision=scope_decision,
            results=retrieval_results,
            thresholds=self._evidence_thresholds,
        )
        if evidence_assessment.status == "insufficient":
            return QAResult(
                query=query,
                status="refused",
                answer_armenian=(
                    "Այս հարցին վստահ պատասխան տալու համար բավարար հաստատված տվյալ չգտա "
                    "սկրեյպված պաշտոնական աղբյուրներում։"
                ),
                scope_decision=scope_decision,
                refusal_reason="insufficient_evidence",
                evidence_assessment=evidence_assessment,
                retrieved_results=retrieval_results,
                supporting_results=[],
                generator_name="none",
                model_name=None,
            )

        if evidence_assessment.status == "ambiguous":
            return QAResult(
                query=query,
                status="refused",
                answer_armenian=_clarification_message(scope_decision.scope, evidence_assessment),
                scope_decision=scope_decision,
                refusal_reason="ambiguous_request",
                evidence_assessment=evidence_assessment,
                retrieved_results=retrieval_results,
                supporting_results=[],
                generator_name="none",
                model_name=None,
            )

        context_results = retrieval_results[: self._context_chunks]
        try:
            generated = self._answer_generator.generate(
                query=query,
                scope_decision=scope_decision,
                retrieval_results=context_results,
            )
        except AnswerGeneratorError as exc:
            return QAResult(
                query=query,
                status="refused",
                answer_armenian=(
                    "Պատասխանի կազմման փուլում խնդիր առաջացավ։ Խնդրում եմ կրկին փորձել կամ "
                    "ստուգել QA մոդելի կարգավորումները։"
                ),
                scope_decision=scope_decision,
                refusal_reason="generation_error",
                evidence_assessment=evidence_assessment,
                retrieved_results=retrieval_results,
                supporting_results=[],
                generator_name="generation_error",
                model_name=None,
            )

        supporting_results = _select_supporting_results(context_results, generated.used_chunk_ids)
        if generated.decision == "refuse":
            refusal_reason = (
                "ambiguous_request"
                if generated.refusal_reason == "ambiguous_request"
                else "insufficient_evidence"
            )
            return QAResult(
                query=query,
                status="refused",
                answer_armenian=generated.answer_armenian,
                scope_decision=scope_decision,
                refusal_reason=refusal_reason,
                evidence_assessment=evidence_assessment,
                retrieved_results=retrieval_results,
                supporting_results=supporting_results,
                generator_name=generated.generator_name,
                model_name=generated.model_name,
            )

        return QAResult(
            query=query,
            status="answered",
            answer_armenian=generated.answer_armenian,
            scope_decision=scope_decision,
            refusal_reason=None,
            evidence_assessment=evidence_assessment,
            retrieved_results=retrieval_results,
            supporting_results=supporting_results,
            generator_name=generated.generator_name,
            model_name=generated.model_name,
        )


def _select_supporting_results(
    context_results: list[RetrievalResult],
    used_chunk_ids: list[str],
) -> list[RetrievalResult]:
    if not used_chunk_ids:
        return context_results[:2]
    by_chunk_id = {result.chunk_id: result for result in context_results}
    supporting_results = [by_chunk_id[chunk_id] for chunk_id in used_chunk_ids if chunk_id in by_chunk_id]
    return supporting_results or context_results[:2]


def _out_of_scope_message(reason: str) -> str:
    if reason == "comparative_request":
        return (
            "Չեմ կարող համեմատել կամ խորհուրդ տալ, թե որ բանկն է ավելի լավ, "
            "շահավետ կամ նախընտրելի։ Կարող եմ պատասխանել միայն պաշտոնական տվյալներով՝ "
            "վարկերի, ավանդների և մասնաճյուղերի մասին։"
        )
    return (
        "Կարող եմ պատասխանել միայն վարկերի, ավանդների և մասնաճյուղերի մասին "
        "հարցերի սահմաններում։"
    )


def _clarification_message(scope: str, evidence_assessment) -> str:
    if evidence_assessment.reason in {"bank_clarification_needed", "multiple_banks_match_query"}:
        bank_names = ", ".join(_bank_display_name(bank_code) for bank_code in evidence_assessment.dominant_bank_codes)
        if bank_names:
            return (
                f"Հստակեցրեք՝ որ բանկի { _topic_label(scope) } մասին եք հարցնում։ "
                f"Օրինակ՝ {bank_names}։"
            )
        return (
            f"Հստակեցրեք՝ որ բանկի { _topic_label(scope) } մասին եք հարցնում, "
            "որպեսզի պատասխանեմ միայն հաստատված տվյալների հիման վրա։"
        )
    return (
        "Հարցը հստակեցրեք՝ որ բանկի կամ որ կոնկրետ ծառայության մասին եք հարցնում, "
        "որպեսզի պատասխանեմ միայն հաստատված տվյալների հիման վրա։"
    )


def _topic_label(scope: str) -> str:
    if scope == "credits":
        return "վարկի"
    if scope == "deposits":
        return "ավանդի"
    if scope == "branches":
        return "մասնաճյուղի"
    return "ծառայության"


def _bank_display_name(bank_code: str) -> str:
    mapping = {
        "unibank": "Unibank",
        "aeb": "AEB",
        "vtb": "VTB Bank Armenia",
    }
    return mapping.get(bank_code, bank_code)
