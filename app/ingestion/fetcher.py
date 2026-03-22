from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config.settings import AppSettings
from app.ingestion.models import FetchResult


class FetchError(RuntimeError):
    """Raised when a page cannot be fetched."""


class Fetcher(Protocol):
    def fetch(self, url: str) -> FetchResult:
        ...


class RequestsFetcher:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._session = requests.Session()
        retry = Retry(
            total=settings.request_retries,
            backoff_factor=settings.request_backoff_seconds,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({"User-Agent": settings.user_agent})

    def fetch(self, url: str) -> FetchResult:
        try:
            response = self._session.get(
                url,
                timeout=self._settings.request_timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FetchError(str(exc)) from exc

        content_type = response.headers.get("Content-Type", "")
        content_format = determine_content_format(response.url, content_type)
        if content_format is None:
            raise FetchError(f"Unsupported content type '{content_type}' for {url}")

        return FetchResult(
            requested_url=url,
            final_url=response.url,
            status_code=response.status_code,
            content_type=content_type,
            content_format=content_format,
            raw_content=response.content,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            fetch_method="requests",
        )


class PlaywrightFetcher:
    """Reserved for future JS-only pages so the main path stays lightweight."""

    def fetch(self, url: str) -> FetchResult:
        raise FetchError(
            "Renderer 'playwright' is configured but no JS fetcher is implemented yet. "
            f"Add an isolated Playwright handler before using it for {url}."
        )


def build_fetcher(renderer: str, settings: AppSettings) -> Fetcher:
    if renderer == "requests":
        return RequestsFetcher(settings)
    if renderer == "playwright":
        return PlaywrightFetcher()
    raise FetchError(f"Unsupported renderer '{renderer}'")


def determine_content_format(url: str, content_type: str) -> str | None:
    lowered = content_type.lower()
    path = urlparse(url).path.lower()
    if "application/pdf" in lowered or path.endswith(".pdf"):
        return "pdf"
    if "text/html" in lowered or "application/xhtml+xml" in lowered:
        return "html"
    return None
