from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

from app.retrieval.models import SourceDocument
from app.retrieval.text import normalize_text

COOKIE_OR_PRIVACY_PATTERNS = (
    re.compile(r"cookie", re.IGNORECASE),
    re.compile(r"privacy", re.IGNORECASE),
    re.compile(r"գաղտնիության", re.IGNORECASE),
    re.compile(r"քուքի", re.IGNORECASE),
)

LEGAL_OR_CHROME_PATTERNS = (
    re.compile(r"բոլոր իրավունքները պաշտպանված են", re.IGNORECASE),
    re.compile(r"բանկը պատասխանատվություն չի կրում", re.IGNORECASE),
    re.compile(r"վերահսկվում է .* կենտրոնական բանկ", re.IGNORECASE),
    re.compile(r"download app", re.IGNORECASE),
    re.compile(r"ներբեռնել հավելված", re.IGNORECASE),
    re.compile(r"contact center", re.IGNORECASE),
    re.compile(r"կոնտակտային կենտրոն", re.IGNORECASE),
)


@dataclass(frozen=True)
class BoilerplateConfig:
    edge_window_lines: int = 18
    repeated_line_min_docs: int = 3
    repeated_line_min_ratio: float = 0.3
    repeated_line_max_chars: int = 90
    repeated_line_max_tokens: int = 8


@dataclass(frozen=True)
class BoilerplateStats:
    removed_lines_total: int
    removed_pattern_lines: int
    removed_repeated_lines: int


class BoilerplateCleaner:
    def __init__(self, config: BoilerplateConfig) -> None:
        self._config = config
        self._repeated_bank_lines: dict[str, set[str]] = {}

    def fit(self, documents: list[SourceDocument]) -> "BoilerplateCleaner":
        bank_doc_counts: Counter[str] = Counter()
        bank_line_doc_counts: dict[str, Counter[str]] = defaultdict(Counter)

        for document in documents:
            lines = _split_lines(document.cleaned_text)
            if not lines:
                continue
            bank_doc_counts[document.bank_code] += 1
            edge_lines = {
                line
                for line in _edge_lines(lines, self._config.edge_window_lines)
                if _is_repeated_line_candidate(
                    line=line,
                    max_chars=self._config.repeated_line_max_chars,
                    max_tokens=self._config.repeated_line_max_tokens,
                )
            }
            for line in edge_lines:
                bank_line_doc_counts[document.bank_code][line] += 1

        repeated_bank_lines: dict[str, set[str]] = {}
        for bank_code, counter in bank_line_doc_counts.items():
            doc_count = bank_doc_counts[bank_code]
            repeated_bank_lines[bank_code] = {
                line
                for line, count in counter.items()
                if count >= self._config.repeated_line_min_docs
                and (count / max(doc_count, 1)) >= self._config.repeated_line_min_ratio
            }

        self._repeated_bank_lines = repeated_bank_lines
        return self

    def transform(self, documents: list[SourceDocument]) -> tuple[list[SourceDocument], BoilerplateStats]:
        cleaned_documents: list[SourceDocument] = []
        removed_lines_total = 0
        removed_pattern_lines = 0
        removed_repeated_lines = 0

        for document in documents:
            cleaned_document, stats = self.clean_document(document)
            cleaned_documents.append(cleaned_document)
            removed_lines_total += stats.removed_lines_total
            removed_pattern_lines += stats.removed_pattern_lines
            removed_repeated_lines += stats.removed_repeated_lines

        return (
            cleaned_documents,
            BoilerplateStats(
                removed_lines_total=removed_lines_total,
                removed_pattern_lines=removed_pattern_lines,
                removed_repeated_lines=removed_repeated_lines,
            ),
        )

    def fit_transform(self, documents: list[SourceDocument]) -> tuple[list[SourceDocument], BoilerplateStats]:
        self.fit(documents)
        return self.transform(documents)

    def clean_document(self, document: SourceDocument) -> tuple[SourceDocument, BoilerplateStats]:
        lines = _split_lines(document.cleaned_text)
        repeated_lines = self._repeated_bank_lines.get(document.bank_code, set())
        kept_lines: list[str] = []
        removed_pattern_lines = 0
        removed_repeated_lines = 0
        seen_recent: set[str] = set()

        for line in lines:
            if _matches_obvious_boilerplate(line):
                removed_pattern_lines += 1
                continue

            if line in repeated_lines:
                removed_repeated_lines += 1
                continue

            if line in seen_recent:
                # Remove immediate duplicate lines caused by page chrome repetition.
                removed_repeated_lines += 1
                continue

            kept_lines.append(line)
            seen_recent = {line} if len(line) > 2 else set()

        cleaned_text = "\n".join(kept_lines).strip()
        metadata = dict(document.metadata)
        metadata["boilerplate_cleanup"] = {
            "removed_lines_total": removed_pattern_lines + removed_repeated_lines,
            "removed_pattern_lines": removed_pattern_lines,
            "removed_repeated_lines": removed_repeated_lines,
        }

        cleaned_document = SourceDocument(
            document_id=document.document_id,
            bank_name=document.bank_name,
            bank_code=document.bank_code,
            topic=document.topic,
            source_url=document.source_url,
            page_title=document.page_title,
            scraped_at=document.scraped_at,
            cleaned_text=cleaned_text,
            metadata=metadata,
        )
        return (
            cleaned_document,
            BoilerplateStats(
                removed_lines_total=removed_pattern_lines + removed_repeated_lines,
                removed_pattern_lines=removed_pattern_lines,
                removed_repeated_lines=removed_repeated_lines,
            ),
        )


def _split_lines(text: str) -> list[str]:
    return [normalize_text(line) for line in text.splitlines() if normalize_text(line)]


def _edge_lines(lines: list[str], window: int) -> list[str]:
    if len(lines) <= window * 2:
        return lines
    return lines[:window] + lines[-window:]


def _matches_obvious_boilerplate(line: str) -> bool:
    lowered = line.casefold()
    if any(pattern.search(lowered) for pattern in COOKIE_OR_PRIVACY_PATTERNS):
        return True
    if any(pattern.search(lowered) for pattern in LEGAL_OR_CHROME_PATTERNS):
        return True
    return False


def _is_repeated_line_candidate(line: str, max_chars: int, max_tokens: int) -> bool:
    if not line or len(line) > max_chars:
        return False
    token_count = len(line.split())
    if token_count == 0 or token_count > max_tokens:
        return False
    if any(char.isdigit() for char in line) and token_count > 4:
        return False
    return True
