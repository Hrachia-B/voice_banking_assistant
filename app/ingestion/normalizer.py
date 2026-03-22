from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.ingestion.discovery import canonicalize_url
from app.ingestion.models import BankDefinition, CrawlTask, FetchResult, ParsedPage, ScrapedDocument


@dataclass(frozen=True)
class NormalizedPage:
    document: ScrapedDocument
    raw_content: bytes


def build_document(
    bank: BankDefinition,
    crawl_task: CrawlTask,
    fetch_result: FetchResult,
    parsed_page: ParsedPage,
) -> NormalizedPage:
    canonical_url = canonicalize_url(fetch_result.final_url)
    content_hash = hashlib.sha256(parsed_page.cleaned_text.encode("utf-8")).hexdigest()
    stable_id_source = f"{bank.bank_code}|{crawl_task.topic}|{canonical_url}|{content_hash}"
    document_id = hashlib.sha256(stable_id_source.encode("utf-8")).hexdigest()[:12]
    metadata = {
        "parent_url": crawl_task.parent_url,
        "depth": crawl_task.depth,
        "requested_url": crawl_task.url,
        "final_url": canonical_url,
        "fetch_method": fetch_result.fetch_method,
        "status_code": fetch_result.status_code,
        "source_format": fetch_result.content_format,
        "content_type": fetch_result.content_type,
    }

    document = ScrapedDocument(
        document_id=f"{bank.bank_code}-{crawl_task.topic}-{document_id}",
        content_hash=content_hash,
        bank_name=bank.bank_name,
        bank_code=bank.bank_code,
        topic=crawl_task.topic,
        source_url=canonical_url,
        page_title=parsed_page.page_title,
        scraped_at=fetch_result.fetched_at,
        cleaned_text=parsed_page.cleaned_text,
        metadata=metadata,
    )
    return NormalizedPage(document=document, raw_content=fetch_result.raw_content)
