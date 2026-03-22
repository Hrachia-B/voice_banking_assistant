from app.chunking.cleaner import BoilerplateCleaner, BoilerplateConfig
from app.retrieval.models import SourceDocument


def test_boilerplate_cleaner_removes_repeated_edge_lines_but_keeps_content() -> None:
    documents = [
        SourceDocument(
            document_id=f"doc-{idx}",
            bank_name="Unibank",
            bank_code="unibank",
            topic="credits",
            source_url=f"https://example.com/{idx}",
            page_title="Loan",
            scraped_at="2026-03-22T00:00:00Z",
            cleaned_text=(
                "Մեր մասին\nԾառայություններ\nՆորություններ\n"
                f"Յուրահատուկ բովանդակություն {idx}\n"
                "Բոլոր իրավունքները պաշտպանված են\n"
            ),
            metadata={},
        )
        for idx in range(1, 5)
    ]

    cleaner = BoilerplateCleaner(
        BoilerplateConfig(edge_window_lines=3, repeated_line_min_docs=3, repeated_line_min_ratio=0.5)
    )
    cleaned_documents, stats = cleaner.fit_transform(documents)

    assert stats.removed_lines_total > 0
    assert "Մեր մասին" not in cleaned_documents[0].cleaned_text
    assert "Ծառայություններ" not in cleaned_documents[0].cleaned_text
    assert "Յուրահատուկ բովանդակություն 1" in cleaned_documents[0].cleaned_text
