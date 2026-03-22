from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TopicName = Literal["credits", "deposits", "branches"]
SUPPORTED_TOPICS: tuple[TopicName, ...] = ("credits", "deposits", "branches")
ContentFormat = Literal["html", "pdf"]


@dataclass(frozen=True)
class TopicConfig:
    document_urls: list[str]
    include_url_patterns: list[str] = field(default_factory=list)
    exclude_url_patterns: list[str] = field(default_factory=list)
    follow_links: bool = False
    max_depth: int = 1
    max_pages: int | None = None
    renderer: str = "requests"


@dataclass(frozen=True)
class BankDefinition:
    bank_name: str
    bank_code: str
    base_url: str
    allowed_domains: list[str]
    topics: dict[TopicName, TopicConfig]


@dataclass(frozen=True)
class BankRegistry:
    version: int
    banks: list[BankDefinition]

    def get_bank(self, bank_code: str) -> BankDefinition | None:
        for bank in self.banks:
            if bank.bank_code == bank_code:
                return bank
        return None


@dataclass(frozen=True)
class FetchResult:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str
    content_format: ContentFormat
    raw_content: bytes
    fetched_at: str
    fetch_method: str


@dataclass(frozen=True)
class CrawlTask:
    url: str
    topic: TopicName
    depth: int = 0
    parent_url: str | None = None


@dataclass(frozen=True)
class ParsedPage:
    page_title: str
    cleaned_text: str
    discovered_links: list[str]


@dataclass(frozen=True)
class ScrapedDocument:
    document_id: str
    content_hash: str
    bank_name: str
    bank_code: str
    topic: TopicName
    source_url: str
    page_title: str
    scraped_at: str
    cleaned_text: str
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScrapeFailure:
    bank_code: str
    topic: TopicName
    url: str
    reason: str
    parent_url: str | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
