from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from app.qa.query_normalization import normalize_user_query, normalized_query_tokens

ScopeLabel = Literal["credits", "deposits", "branches", "out_of_scope"]


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "credits": (
        "վարկ",
        "վարկեր",
        "սպառողական",
        "հիփոթեք",
        "հիպոթեք",
        "օվերդրաֆտ",
        "վերաֆինանս",
        "ապառիկ",
        "loan",
        "loans",
        "credit",
        "credits",
        "mortgage",
        "overdraft",
        "refinance",
    ),
    "deposits": (
        "ավանդ",
        "ավանդներ",
        "ժամկետային ավանդ",
        "deposit",
        "deposits",
        "տարեկան տոկոսադրույք",
    ),
    "branches": (
        "մասնաճյուղ",
        "մասնաճյուղեր",
        "բանկոմատ",
        "atm",
        "branch",
        "branches",
        "հասցե",
        "քարտեզ",
        "որտեղ",
        "աշխատանքային ժամ",
        "location",
        "address",
    ),
}

COMPARATIVE_OR_PREFERENCE_KEYWORDS: tuple[str, ...] = (
    "ավելի լավ",
    "լավագույն",
    "ամենալավ",
    "ավելի շահավետ",
    "ավելի էժան",
    "էժան",
    "նախընտրելի",
    "խորհուրդ կտաս",
    "կխորհուրդ տաս",
    "որ բանկն է",
    "ո՞ր բանկն է",
    "որ բանկի վարկն է",
    "ո՞ր բանկի վարկն է",
    "recommend",
    "recommended",
    "best",
    "better",
    "cheaper",
    "preferable",
)

OUT_OF_SCOPE_KEYWORDS: tuple[str, ...] = (
    "քարտ",
    "քարտեր",
    "card",
    "cards",
    "փոխանց",
    "transfer",
    "swift",
    "փոխարժեք",
    "exchange rate",
    "արժույթ",
    "mobile banking",
    "հաշիվ բացել",
    "account opening",
)

BANK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "unibank": (
        "unibank",
        "uni bank",
        "յունիբանկ",
        "յունի բանկ",
    ),
    "aeb": (
        "aeb",
        "աեբ",
        "armeconombank",
        "armeconombank",
        "հայէկոնոմբանկ",
        "հայէկոնոմ բանկ",
        "հայ էկոնոմ բանկ",
        "armeconombank",
    ),
    "vtb": (
        "vtb",
        "վտբ",
        "վտբ-հայաստան",
        "վտբ հայաստան",
        "vtb armenia",
        "vtb bank armenia",
    ),
}


@dataclass(frozen=True)
class ScopeDecision:
    scope: ScopeLabel
    confidence: float
    bank_code: str | None
    matched_topic_terms: list[str]
    matched_bank_terms: list[str]
    reason: str

    @property
    def is_allowed(self) -> bool:
        return self.scope != "out_of_scope"


class QueryScopeGuard:
    def classify(self, query: str) -> ScopeDecision:
        normalized = normalize_user_query(query)
        if not normalized:
            return ScopeDecision(
                scope="out_of_scope",
                confidence=0.0,
                bank_code=None,
                matched_topic_terms=[],
                matched_bank_terms=[],
                reason="empty_query",
            )

        matched_bank_terms, bank_code = _detect_bank(normalized)
        if any(keyword in normalized for keyword in COMPARATIVE_OR_PREFERENCE_KEYWORDS):
            return ScopeDecision(
                scope="out_of_scope",
                confidence=0.95,
                bank_code=bank_code,
                matched_topic_terms=[],
                matched_bank_terms=matched_bank_terms,
                reason="comparative_request",
            )

        if any(keyword in normalized for keyword in OUT_OF_SCOPE_KEYWORDS):
            # Allow explicit in-scope branch terms to win over generic card/account keywords.
            if not any(keyword in normalized for keyword in TOPIC_KEYWORDS["branches"]):
                return ScopeDecision(
                    scope="out_of_scope",
                    confidence=0.9,
                    bank_code=bank_code,
                    matched_topic_terms=[],
                    matched_bank_terms=matched_bank_terms,
                    reason="explicit_out_of_scope_keyword",
                )

        topic_scores: dict[str, int] = {}
        topic_matches: dict[str, list[str]] = {}
        for topic, keywords in TOPIC_KEYWORDS.items():
            matches = [keyword for keyword in keywords if _query_matches_keyword(normalized, keyword)]
            topic_matches[topic] = matches
            topic_scores[topic] = len(matches)

        best_topic = max(topic_scores, key=topic_scores.get)
        best_score = topic_scores[best_topic]
        second_score = max(score for topic, score in topic_scores.items() if topic != best_topic)

        if best_score <= 0:
            return ScopeDecision(
                scope="out_of_scope",
                confidence=0.0,
                bank_code=bank_code,
                matched_topic_terms=[],
                matched_bank_terms=matched_bank_terms,
                reason="no_supported_topic_keywords",
            )

        if second_score == best_score:
            return ScopeDecision(
                scope="out_of_scope",
                confidence=0.2,
                bank_code=bank_code,
                matched_topic_terms=topic_matches[best_topic],
                matched_bank_terms=matched_bank_terms,
                reason="ambiguous_topic_keywords",
            )

        confidence = min(1.0, 0.4 + (0.2 * best_score))
        return ScopeDecision(
            scope=best_topic,  # type: ignore[arg-type]
            confidence=confidence,
            bank_code=bank_code,
            matched_topic_terms=topic_matches[best_topic],
            matched_bank_terms=matched_bank_terms,
            reason="keyword_match",
        )


def _detect_bank(normalized_query: str) -> tuple[list[str], str | None]:
    matched_terms: list[str] = []
    matched_bank_code: str | None = None
    for bank_code, keywords in BANK_KEYWORDS.items():
        bank_matches = [keyword for keyword in keywords if _query_matches_keyword(normalized_query, keyword)]
        if bank_matches:
            matched_terms.extend(bank_matches)
            matched_bank_code = bank_code
            break
    return matched_terms, matched_bank_code


def _query_matches_keyword(normalized_query: str, keyword: str) -> bool:
    if keyword in normalized_query:
        return True

    keyword_tokens = keyword.split()
    query_tokens = normalized_query_tokens(normalized_query)
    if not keyword_tokens or not query_tokens:
        return False

    if len(keyword_tokens) > 1:
        return False

    keyword_token = keyword_tokens[0]
    for query_token in query_tokens:
        if query_token == keyword_token:
            return True
        if len(query_token) >= 4 and len(keyword_token) >= 4:
            if query_token.startswith(keyword_token) or keyword_token.startswith(query_token):
                return True
        if len(query_token) >= 5 and len(keyword_token) >= 5:
            if SequenceMatcher(None, query_token, keyword_token).ratio() >= 0.84:
                return True
    return False
