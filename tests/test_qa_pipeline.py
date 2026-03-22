from app.config.settings import get_settings
from app.guardrails.scope import ScopeDecision
from app.qa.evidence import EvidenceThresholds
from app.qa.generator import ExtractiveGroundedAnswerGenerator, build_answer_generator
from app.qa.models import GeneratedAnswer
from app.qa.pipeline import GroundedQAService
from app.retrieval.models import RetrievalResult


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 5, filters=None, deduplicate: bool = True):  # type: ignore[no-untyped-def]
        return self._results[:top_k]


class FakeAnswerGenerator:
    def generate(self, query: str, scope_decision: ScopeDecision, retrieval_results: list[RetrievalResult]) -> GeneratedAnswer:
        return GeneratedAnswer(
            decision="answer",
            answer_armenian="Սա grounded պատասխան է։",
            used_chunk_ids=[retrieval_results[0].chunk_id],
            refusal_reason=None,
            generator_name="fake",
            model_name=None,
        )


def test_qa_pipeline_refuses_out_of_scope_query() -> None:
    service = GroundedQAService(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        answer_generator=FakeAnswerGenerator(),
    )

    result = service.answer_query("Քարտի սպասարկման վճար")

    assert result.status == "refused"
    assert result.refusal_reason == "out_of_scope"


def test_qa_pipeline_refuses_comparative_query() -> None:
    service = GroundedQAService(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        answer_generator=FakeAnswerGenerator(),
    )

    result = service.answer_query("Ո՞ր բանկի վարկն է ավելի լավ")

    assert result.status == "refused"
    assert result.refusal_reason == "out_of_scope"
    assert "Չեմ կարող համեմատել" in result.answer_armenian


def test_qa_pipeline_answers_supported_query() -> None:
    results = [
        RetrievalResult(
            rank=1,
            score=0.3,
            word_score=0.2,
            char_score=0.4,
            chunk_id="chunk-1",
            parent_document_id="doc-1",
            bank_name="Unibank",
            bank_code="unibank",
            topic="credits",
            source_url="https://example.com/loan",
            page_title="Loan",
            chunk_text="Սպառողական վարկի տոկոսադրույքը 21.5% է",
            metadata={},
        )
    ]
    service = GroundedQAService(
        retriever=FakeRetriever(results),  # type: ignore[arg-type]
        answer_generator=FakeAnswerGenerator(),
        evidence_thresholds=EvidenceThresholds(min_top_score=0.1, min_total_score=0.1),
    )

    result = service.answer_query("Յունիբանկի սպառողական վարկի տոկոսադրույքը")

    assert result.status == "answered"
    assert result.refusal_reason is None
    assert result.supporting_results[0].chunk_id == "chunk-1"


def test_gemini_generator_falls_back_without_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    generator = build_answer_generator(get_settings(), mode="gemini")
    result = generator.generate(
        query="Ավանդի տոկոսադրույքը",
        scope_decision=ScopeDecision(
            scope="deposits",
            confidence=1.0,
            bank_code="aeb",
            matched_topic_terms=["ավանդ"],
            matched_bank_terms=["aeb"],
            reason="keyword_match",
        ),
        retrieval_results=[
            RetrievalResult(
                rank=1,
                score=0.3,
                word_score=0.2,
                char_score=0.4,
                chunk_id="chunk-1",
                parent_document_id="doc-1",
                bank_name="Armeconombank",
                bank_code="aeb",
                topic="deposits",
                source_url="https://example.com/deposit",
                page_title="Deposit",
                chunk_text="Ավանդի տոկոսադրույքը մինչեւ 9.5% է",
                metadata={},
            )
        ],
    )

    assert result.generator_name == "extractive"


def test_qa_pipeline_accepts_supported_branch_query_with_branch_thresholds() -> None:
    results = [
        RetrievalResult(
            rank=1,
            score=0.089,
            word_score=0.05,
            char_score=0.11,
            chunk_id="branch-1",
            parent_document_id="doc-1",
            bank_name="VTB Bank Armenia",
            bank_code="vtb",
            topic="branches",
            source_url="https://example.com/map",
            page_title="Branches",
            chunk_text="ՎՏԲ մասնաճյուղ ք. Երևան, Սեբաստիայի փ. 14/55",
            metadata={},
        ),
        RetrievalResult(
            rank=2,
            score=0.088,
            word_score=0.04,
            char_score=0.1,
            chunk_id="branch-2",
            parent_document_id="doc-2",
            bank_name="VTB Bank Armenia",
            bank_code="vtb",
            topic="branches",
            source_url="https://example.com/map",
            page_title="Branches",
            chunk_text="ք. Երևան, Կոմիտաս փ. 41/74",
            metadata={},
        ),
        RetrievalResult(
            rank=3,
            score=0.087,
            word_score=0.04,
            char_score=0.1,
            chunk_id="branch-3",
            parent_document_id="doc-3",
            bank_name="VTB Bank Armenia",
            bank_code="vtb",
            topic="branches",
            source_url="https://example.com/map",
            page_title="Branches",
            chunk_text="ք. Երևան, Ազատության պող. 20/37",
            metadata={},
        ),
    ]
    service = GroundedQAService(
        retriever=FakeRetriever(results),  # type: ignore[arg-type]
        answer_generator=FakeAnswerGenerator(),
    )

    result = service.answer_query("ՎՏԲ-ի մասնաճյուղը որտեղ է Երևանում")

    assert result.status == "answered"


def test_extractive_generator_builds_branch_answer_from_address_lines() -> None:
    generator = ExtractiveGroundedAnswerGenerator()

    result = generator.generate(
        query="ՎՏԲ-ի մասնաճյուղը որտեղ է Երևանում",
        scope_decision=ScopeDecision(
            scope="branches",
            confidence=1.0,
            bank_code="vtb",
            matched_topic_terms=["մասնաճյուղ"],
            matched_bank_terms=["վտբ"],
            reason="keyword_match",
        ),
        retrieval_results=[
            RetrievalResult(
                rank=1,
                score=0.089,
                word_score=0.05,
                char_score=0.11,
                chunk_id="branch-1",
                parent_document_id="doc-1",
                bank_name="VTB Bank Armenia",
                bank_code="vtb",
                topic="branches",
                source_url="https://example.com/map",
                page_title="Branches",
                chunk_text=(
                    "«Բարեկամություն» մասնաճյուղ\n"
                    "ք. Երեւան, Կիեւյան 19 փ., Բարեկամություն մ/ճ\n"
                    "24/7\n"
                    "«Քանաքեռ» մասնաճյուղ\n"
                    "ք. Երեւան, Սարկավագի փ. 123, Քանաքեռ մ/ճ\n"
                ),
                metadata={},
            )
        ],
    )

    assert result.decision == "answer"
    assert "Բարեկամություն" in result.answer_armenian
    assert "Քանաքեռ" in result.answer_armenian


def test_qa_pipeline_asks_for_bank_clarification_instead_of_insufficient_refusal() -> None:
    results = [
        RetrievalResult(
            rank=1,
            score=0.18,
            word_score=0.10,
            char_score=0.22,
            chunk_id="dep-1",
            parent_document_id="doc-1",
            bank_name="Unibank",
            bank_code="unibank",
            topic="deposits",
            source_url="https://example.com/unibank-deposit",
            page_title="Deposits",
            chunk_text="Ավանդի տոկոսադրույքը մինչեւ 9.5% է",
            metadata={},
        ),
        RetrievalResult(
            rank=2,
            score=0.17,
            word_score=0.09,
            char_score=0.21,
            chunk_id="dep-2",
            parent_document_id="doc-2",
            bank_name="AEB",
            bank_code="aeb",
            topic="deposits",
            source_url="https://example.com/aeb-deposit",
            page_title="Deposits",
            chunk_text="Ավանդի տոկոսադրույքը մինչեւ 10% է",
            metadata={},
        ),
    ]
    service = GroundedQAService(
        retriever=FakeRetriever(results),  # type: ignore[arg-type]
        answer_generator=FakeAnswerGenerator(),
    )

    result = service.answer_query("Ավանդի տոկոսադրույքը ինչքա՞ն է")

    assert result.status == "refused"
    assert result.refusal_reason == "ambiguous_request"
    assert "որ բանկի" in result.answer_armenian


def test_qa_pipeline_accepts_bank_and_topic_aligned_result_even_with_weak_token_overlap() -> None:
    results = [
        RetrievalResult(
            rank=1,
            score=0.19,
            word_score=0.08,
            char_score=0.23,
            chunk_id="loan-1",
            parent_document_id="doc-1",
            bank_name="Unibank",
            bank_code="unibank",
            topic="credits",
            source_url="https://example.com/loan",
            page_title="Consumer Product",
            chunk_text="Տարեկան փաստացի տոկոսադրույքը 21.5% է, գումարը մինչեւ 2500000 դրամ",
            metadata={},
        )
    ]
    service = GroundedQAService(
        retriever=FakeRetriever(results),  # type: ignore[arg-type]
        answer_generator=FakeAnswerGenerator(),
    )

    result = service.answer_query("Յունիբանկի վարկի պայմաններն ասեք")

    assert result.status == "answered"
