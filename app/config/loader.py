from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from app.ingestion.models import (
    BankDefinition,
    BankRegistry,
    SUPPORTED_TOPICS,
    TopicConfig,
    TopicName,
)


class ConfigError(ValueError):
    """Raised when the bank config file is invalid."""


def load_bank_registry(config_path: Path) -> BankRegistry:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ConfigError("Config root must be a mapping.")

    version = payload.get("version", 1)
    banks_payload = payload.get("banks")
    if not isinstance(banks_payload, list) or not banks_payload:
        raise ConfigError("Config must define a non-empty 'banks' list.")

    banks = [parse_bank_definition(item) for item in banks_payload]
    return BankRegistry(version=int(version), banks=banks)


def parse_bank_definition(payload: dict[str, Any]) -> BankDefinition:
    if not isinstance(payload, dict):
        raise ConfigError("Each bank entry must be a mapping.")

    required_fields = ("bank_name", "bank_code", "base_url", "allowed_domains", "topics")
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ConfigError(f"Bank definition missing fields: {', '.join(missing)}")

    allowed_domains = payload["allowed_domains"]
    if not isinstance(allowed_domains, list) or not all(
        isinstance(item, str) for item in allowed_domains
    ):
        raise ConfigError("'allowed_domains' must be a list of strings.")

    topics_payload = payload["topics"]
    if not isinstance(topics_payload, dict):
        raise ConfigError("'topics' must be a mapping of topic names to topic config.")

    topics: dict[TopicName, TopicConfig] = {}
    for topic_name, topic_payload in topics_payload.items():
        if topic_name not in SUPPORTED_TOPICS:
            supported = ", ".join(SUPPORTED_TOPICS)
            raise ConfigError(f"Unsupported topic '{topic_name}'. Expected one of: {supported}")
        if not isinstance(topic_payload, dict):
            raise ConfigError(f"Topic '{topic_name}' must map to a dictionary.")
        topics[cast(TopicName, topic_name)] = parse_topic_config(topic_payload)

    return BankDefinition(
        bank_name=str(payload["bank_name"]),
        bank_code=str(payload["bank_code"]),
        base_url=str(payload["base_url"]),
        allowed_domains=[str(domain) for domain in allowed_domains],
        topics=topics,
    )


def parse_topic_config(payload: dict[str, Any]) -> TopicConfig:
    document_urls = payload.get("document_urls", payload.get("seed_urls"))
    if not isinstance(document_urls, list) or not document_urls:
        raise ConfigError("Each topic must define a non-empty 'document_urls' list.")
    if not all(isinstance(item, str) for item in document_urls):
        raise ConfigError("'document_urls' must contain only strings.")

    def _string_list(key: str) -> list[str]:
        value = payload.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ConfigError(f"'{key}' must be a list of strings.")
        return [str(item) for item in value]

    return TopicConfig(
        document_urls=_dedupe_strings(str(url) for url in document_urls),
        include_url_patterns=_string_list("include_url_patterns"),
        exclude_url_patterns=_string_list("exclude_url_patterns"),
        follow_links=bool(payload.get("follow_links", False)),
        max_depth=int(payload.get("max_depth", 1)),
        max_pages=int(payload["max_pages"]) if payload.get("max_pages") is not None else None,
        renderer=str(payload.get("renderer", "requests")),
    )


def _dedupe_strings(values: Any) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered
