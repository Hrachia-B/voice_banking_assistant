from __future__ import annotations

import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")
PARAGRAPH_BREAK_RE = re.compile(r"\n{2,}")
SENTENCE_BREAK_RE = re.compile(r"(?<=[\.\!\?։:;])\s+")


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\xa0", " ")
    return WHITESPACE_RE.sub(" ", normalized).strip()


def split_paragraphs(value: str) -> list[str]:
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").strip()
    paragraphs = [normalize_text(part) for part in PARAGRAPH_BREAK_RE.split(value)]
    return [part for part in paragraphs if part]


def split_sentences(value: str) -> list[str]:
    sentences = [normalize_text(part) for part in SENTENCE_BREAK_RE.split(value)]
    return [part for part in sentences if part]
