from __future__ import annotations

import re
from difflib import SequenceMatcher

from app.retrieval.text import normalize_text


TOKEN_STRIP_RE = re.compile(r"^[^\wա-ֆԱ-Ֆ]+|[^\wա-ֆԱ-Ֆ]+$")
ARMENIAN_SUFFIXES = ("ներում", "երում", "ությամբ", "ություն", "ական", "ային", "ում", "ին", "ի", "ը", "ն")

VOICE_NOISE_REPLACEMENTS: dict[str, str] = {
    "վտբի": "վտբ",
    "վտբ-ի": "վտբ",
    "յունիբանկի": "յունիբանկ",
    "աեբի": "աեբ",
    "մասնաճուղ": "մասնաճյուղ",
    "մասնաճույ": "մասնաճյուղ",
    "մասնաճույգ": "մասնաճյուղ",
    "ավանդը": "ավանդ",
    "վարկը": "վարկ",
    "բանկայի": "բանկային",
}

FILLER_WORDS = {
    "խնդրումեմ",
    "խնդրում",
    "ասա",
    "ասեք",
    "կասեք",
    "կարաս",
    "կարողես",
    "կարողեք",
    "ուզումեմ",
    "ուզում",
    "ինձ",
    "պատմիր",
    "պատասխանիր",
}


def normalize_user_query(query: str) -> str:
    normalized = normalize_text(query).casefold()
    for source, target in VOICE_NOISE_REPLACEMENTS.items():
        normalized = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, normalized)

    cleaned_tokens = []
    for token in normalized.split():
        stripped = TOKEN_STRIP_RE.sub("", token)
        if not stripped or stripped in FILLER_WORDS:
            continue
        cleaned_tokens.append(stripped)
    return " ".join(cleaned_tokens).strip()


def normalized_query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in normalize_user_query(query).split():
        token = raw_token.replace("-", "").replace("—", "").replace("–", "")
        token = TOKEN_STRIP_RE.sub("", token)
        if not token:
            continue
        stem = strip_token_suffix(token)
        if stem:
            tokens.append(stem)
    return tokens


def strip_token_suffix(token: str) -> str:
    for suffix in ARMENIAN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def tokens_match(left: str, right: str, *, fuzzy_threshold: float = 0.84) -> bool:
    if left == right:
        return True
    if len(left) >= 4 and len(right) >= 4:
        if left.startswith(right) or right.startswith(left):
            return True
    if len(left) >= 5 and len(right) >= 5:
        if SequenceMatcher(None, left, right).ratio() >= fuzzy_threshold:
            return True
    return False
