from app.retrieval.indexer import build_hybrid_index
from app.retrieval.models import ChunkRecord
from app.retrieval.retriever import ChunkRetriever, RetrievalFilters


def test_retrieval_respects_topic_and_bank_filters() -> None:
    chunks = [
        ChunkRecord(
            chunk_id="c1",
            parent_document_id="d1",
            bank_name="Unibank",
            bank_code="unibank",
            topic="credits",
            source_url="https://example.com/loan",
            page_title="Loan",
            scraped_at="2026-03-22T00:00:00Z",
            chunk_index=0,
            chunk_text="Սպառողական վարկ տոկոսադրույք և պայմաններ",
            metadata={},
        ),
        ChunkRecord(
            chunk_id="c2",
            parent_document_id="d2",
            bank_name="VTB Bank Armenia",
            bank_code="vtb",
            topic="deposits",
            source_url="https://example.com/deposit",
            page_title="Deposit",
            scraped_at="2026-03-22T00:00:00Z",
            chunk_index=0,
            chunk_text="Ավանդ տոկոսադրույք և ժամկետ",
            metadata={},
        ),
    ]
    index = build_hybrid_index(chunks, build_metadata={"documents_path": "test"})
    retriever = ChunkRetriever(index)

    results = retriever.search(
        query="ավանդ",
        top_k=5,
        filters=RetrievalFilters(bank_name="vtb", topic="deposits"),
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c2"


def test_retrieval_suppresses_near_duplicate_chunks() -> None:
    chunks = [
        ChunkRecord(
            chunk_id="c1",
            parent_document_id="d1",
            bank_name="Unibank",
            bank_code="unibank",
            topic="credits",
            source_url="https://example.com/loan-a",
            page_title="Loan A",
            scraped_at="2026-03-22T00:00:00Z",
            chunk_index=0,
            chunk_text="Սպառողական վարկ տոկոսադրույք 12 տոկոս ժամկետ 36 ամիս",
            metadata={},
        ),
        ChunkRecord(
            chunk_id="c2",
            parent_document_id="d2",
            bank_name="Unibank",
            bank_code="unibank",
            topic="credits",
            source_url="https://example.com/loan-b",
            page_title="Loan B",
            scraped_at="2026-03-22T00:00:00Z",
            chunk_index=0,
            chunk_text="Սպառողական վարկ տոկոսադրույք 12 տոկոս ժամկետ 36 ամիս",
            metadata={},
        ),
        ChunkRecord(
            chunk_id="c3",
            parent_document_id="d3",
            bank_name="AEB",
            bank_code="aeb",
            topic="credits",
            source_url="https://example.com/loan-c",
            page_title="Loan C",
            scraped_at="2026-03-22T00:00:00Z",
            chunk_index=0,
            chunk_text="Ուսանողական վարկ պետական ծրագրով",
            metadata={},
        ),
    ]
    index = build_hybrid_index(chunks, build_metadata={"documents_path": "test"})
    retriever = ChunkRetriever(index)

    results = retriever.search(query="սպառողական վարկ", top_k=3)

    assert results[0].chunk_id in {"c1", "c2"}
    assert len([result for result in results if result.chunk_id in {"c1", "c2"}]) == 1
