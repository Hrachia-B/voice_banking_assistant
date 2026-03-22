from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

from app.ingestion.models import BankDefinition, TopicConfig


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    cleaned_path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc.lower(),
            cleaned_path,
            parsed.params,
            parsed.query,
            "",
        )
    )


def is_allowed_url(url: str, bank: BankDefinition, topic_config: TopicConfig) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False

    domain = parsed.netloc.lower()
    if domain not in {item.lower() for item in bank.allowed_domains}:
        return False

    normalized = canonicalize_url(url)

    for pattern in topic_config.exclude_url_patterns:
        if re.search(pattern, normalized):
            return False

    if not topic_config.include_url_patterns:
        return True

    return any(re.search(pattern, normalized) for pattern in topic_config.include_url_patterns)


def filter_discovered_links(
    urls: list[str],
    bank: BankDefinition,
    topic_config: TopicConfig,
) -> list[str]:
    filtered: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = canonicalize_url(url)
        if normalized in seen:
            continue
        if is_allowed_url(normalized, bank, topic_config):
            seen.add(normalized)
            filtered.append(normalized)
    return filtered
