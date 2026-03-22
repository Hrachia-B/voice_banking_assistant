from app.ingestion.parser import parse_html


def test_parse_html_cleans_text_and_extracts_links() -> None:
    html = """
    <html>
      <head><title>Deposits Page</title><style>.x { color: red; }</style></head>
      <body>
        <main>
          <h1>Deposits</h1>
          <p>  High   interest   rates </p>
          <a href="/deposits/detail-1">Details</a>
          <script>console.log("ignore")</script>
        </main>
      </body>
    </html>
    """

    parsed = parse_html(html, "https://bank.example/deposits/")

    assert parsed.page_title == "Deposits Page"
    assert "High interest rates" in parsed.cleaned_text
    assert "console.log" not in parsed.cleaned_text
    assert parsed.discovered_links == ["https://bank.example/deposits/detail-1"]
