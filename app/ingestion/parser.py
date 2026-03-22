from __future__ import annotations

import io
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.ingestion.models import FetchResult, ParsedPage

WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def parse_content(fetch_result: FetchResult) -> ParsedPage:
    if fetch_result.content_format == "pdf":
        return parse_pdf(fetch_result.raw_content, fetch_result.final_url)
    html = fetch_result.raw_content.decode("utf-8", errors="ignore")
    return parse_html(html, fetch_result.final_url)


def parse_html(html: str, page_url: str) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    page_title = extract_page_title(soup)
    cleaned_text = extract_clean_text(soup)
    discovered_links = extract_links(soup, page_url)
    return ParsedPage(
        page_title=page_title,
        cleaned_text=cleaned_text,
        discovered_links=discovered_links,
    )


def parse_pdf(raw_content: bytes, page_url: str) -> ParsedPage:
    reader = PdfReader(io.BytesIO(raw_content))
    chunks: list[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")

    cleaned_text = clean_pdf_text("\n".join(chunks))
    page_title = extract_pdf_title(reader, page_url)
    return ParsedPage(page_title=page_title, cleaned_text=cleaned_text, discovered_links=[])


def extract_page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)

    header = soup.find(["h1", "h2"])
    if header:
        return normalize_line(header.get_text(" ", strip=True))
    return "Untitled Page"


def extract_clean_text(soup: BeautifulSoup) -> str:
    text_root = soup.find("main") or soup.body or soup

    # Remove typical non-content tags before text extraction.
    for tag in text_root.find_all(
        ["script", "style", "noscript", "iframe", "svg", "form", "button"]
    ):
        tag.decompose()

    lines: list[str] = []
    for raw_line in text_root.get_text(separator="\n").splitlines():
        normalized = normalize_line(raw_line)
        if normalized:
            lines.append(normalized)

    return MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()


def extract_links(soup: BeautifulSoup, page_url: str) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        absolute_url = urljoin(page_url, href)
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        links.append(absolute_url)

    return links


def normalize_line(value: str) -> str:
    value = value.replace("\xa0", " ")
    return WHITESPACE_RE.sub(" ", value).strip()


def clean_pdf_text(value: str) -> str:
    lines = [normalize_line(line) for line in value.splitlines()]
    lines = [line for line in lines if line]
    return MULTI_NEWLINE_RE.sub("\n\n", "\n".join(lines)).strip()


def extract_pdf_title(reader: PdfReader, page_url: str) -> str:
    metadata = reader.metadata or {}
    title = metadata.get("/Title") or metadata.get("Title")
    if isinstance(title, str) and title.strip():
        return normalize_line(title)
    return Path(urlparse(page_url).path).name or "Untitled PDF"
