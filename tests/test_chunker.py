from app.chunking.chunker import ChunkingConfig, DocumentChunker
from app.retrieval.models import SourceDocument


def test_chunk_document_preserves_metadata_and_creates_overlap() -> None:
    document = SourceDocument(
        document_id="doc-1",
        bank_name="Unibank",
        bank_code="unibank",
        topic="credits",
        source_url="https://example.com/credits",
        page_title="Credits",
        scraped_at="2026-03-22T00:00:00Z",
        cleaned_text=(
            "Առաջին պարբերություն " * 20
            + "\n\n"
            + "Երկրորդ պարբերություն " * 20
            + "\n\n"
            + "Երրորդ պարբերություն " * 20
        ),
        metadata={"source_format": "html", "content_type": "text/html"},
    )
    chunker = DocumentChunker(ChunkingConfig(chunk_size_chars=220, chunk_overlap_chars=80))

    chunks = chunker.chunk_document(document)

    assert len(chunks) >= 2
    assert all(chunk.parent_document_id == "doc-1" for chunk in chunks)
    assert all(chunk.bank_code == "unibank" for chunk in chunks)
    assert all(len(chunk.chunk_text) <= 320 for chunk in chunks)
    assert chunks[0].chunk_text.split()[-4:] == chunks[1].chunk_text.split()[:4]
