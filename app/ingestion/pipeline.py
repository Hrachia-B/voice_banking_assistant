from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config.settings import AppSettings
from app.ingestion.discovery import canonicalize_url, filter_discovered_links, is_allowed_url
from app.ingestion.fetcher import FetchError, build_fetcher
from app.ingestion.models import (
    BankDefinition,
    CrawlTask,
    ScrapeFailure,
    TopicName,
)
from app.ingestion.normalizer import build_document
from app.ingestion.parser import parse_content
from app.ingestion.storage import DocumentStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TopicRunStats:
    documents: int = 0
    failures: int = 0


class ScrapePipeline:
    def __init__(self, settings: AppSettings, store: DocumentStore) -> None:
        self._settings = settings
        self._store = store

    def scrape_bank(
        self,
        bank: BankDefinition,
        selected_topics: set[TopicName] | None = None,
    ) -> dict[str, TopicRunStats]:
        topic_stats: dict[str, TopicRunStats] = {}
        for topic_name, topic_config in bank.topics.items():
            if selected_topics and topic_name not in selected_topics:
                continue
            topic_stats[topic_name] = self._scrape_topic(bank, topic_name)
        return topic_stats

    def _scrape_topic(self, bank: BankDefinition, topic_name: TopicName) -> TopicRunStats:
        topic_config = bank.topics[topic_name]
        fetcher = build_fetcher(topic_config.renderer, self._settings)
        explicit_tasks = [
            CrawlTask(url=canonicalize_url(url), topic=topic_name)
            for url in topic_config.document_urls
        ]
        max_pages = topic_config.max_pages or len(explicit_tasks)
        queue = deque(explicit_tasks)
        visited: set[str] = set()
        documents = 0
        failures = 0

        while queue and documents < max_pages:
            task = queue.popleft()
            if task.url in visited:
                continue
            visited.add(task.url)

            if not is_allowed_url(task.url, bank, topic_config):
                logger.info("Skipping disallowed URL for %s/%s: %s", bank.bank_code, topic_name, task.url)
                continue

            logger.info("Fetching %s [%s/%s]", task.url, bank.bank_code, topic_name)

            try:
                fetch_result = fetcher.fetch(task.url)
                final_url = canonicalize_url(fetch_result.final_url)
                if not is_allowed_url(final_url, bank, topic_config):
                    raise FetchError(f"Redirected to disallowed URL: {final_url}")
                parsed_page = parse_content(fetch_result)
                normalized = build_document(bank, task, fetch_result, parsed_page)
                self._store.save_document(normalized.document, normalized.raw_content)
                documents += 1
            except (FetchError, ValueError) as exc:
                failures += 1
                logger.warning(
                    "Failed to scrape %s [%s/%s]: %s",
                    task.url,
                    bank.bank_code,
                    topic_name,
                    exc,
                )
                self._store.save_failure(
                    ScrapeFailure(
                        bank_code=bank.bank_code,
                        topic=topic_name,
                        url=task.url,
                        reason=str(exc),
                        parent_url=task.parent_url,
                    )
                )
                continue

            if not topic_config.follow_links or task.depth >= topic_config.max_depth:
                continue

            child_links = filter_discovered_links(parsed_page.discovered_links, bank, topic_config)
            for child_url in child_links:
                if child_url in visited:
                    continue
                queue.append(
                    CrawlTask(
                        url=child_url,
                        topic=topic_name,
                        depth=task.depth + 1,
                        parent_url=task.url,
                    )
                )

        return TopicRunStats(documents=documents, failures=failures)


def build_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
