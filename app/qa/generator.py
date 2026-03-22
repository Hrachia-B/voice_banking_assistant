from __future__ import annotations

import json
import os
import re
from typing import Protocol

from app.config.settings import AppSettings
from app.guardrails.scope import ScopeDecision
from app.prompts.grounded_qa import build_grounded_qa_input, build_grounded_qa_instructions
from app.qa.models import GeneratedAnswer
from app.retrieval.models import RetrievalResult
from app.retrieval.text import normalize_text, split_sentences


class AnswerGeneratorError(RuntimeError):
    """Raised when answer generation fails."""


class AnswerGenerator(Protocol):
    def generate(
        self,
        query: str,
        scope_decision: ScopeDecision,
        retrieval_results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        ...


class GeminiGroundedAnswerGenerator:
    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._model = settings.qa_gemini_model

    def generate(
        self,
        query: str,
        scope_decision: ScopeDecision,
        retrieval_results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise AnswerGeneratorError("GEMINI_API_KEY is not set.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise AnswerGeneratorError("The google-genai package is not installed.") from exc

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=self._model,
            contents=build_grounded_qa_input(query, scope_decision, retrieval_results),
            config=types.GenerateContentConfig(
                system_instruction=build_grounded_qa_instructions(),
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        raw_output = (response.text or "").strip()
        payload = _parse_json_object(raw_output)

        used_chunk_ids = [
            chunk_id
            for chunk_id in payload.get("used_chunk_ids", [])
            if isinstance(chunk_id, str)
        ]
        available_ids = {result.chunk_id for result in retrieval_results}
        used_chunk_ids = [chunk_id for chunk_id in used_chunk_ids if chunk_id in available_ids]

        decision = payload.get("decision")
        if decision not in {"answer", "refuse"}:
            raise AnswerGeneratorError(f"Unexpected QA decision: {decision!r}")

        answer_raw = payload.get("answer_armenian")
        answer_armenian = "" if answer_raw is None else str(answer_raw).strip()
        refusal_reason = payload.get("refusal_reason")
        if refusal_reason is not None:
            refusal_reason = str(refusal_reason)

        if decision == "answer" and not answer_armenian:
            raise AnswerGeneratorError("Gemini returned an answer decision without answer text.")
        if decision == "refuse" and not answer_armenian:
            answer_armenian = (
                "Այս հարցին տրամադրված ապացույցների հիման վրա վստահ պատասխան տալ հնարավոր չէ։"
            )

        return GeneratedAnswer(
            decision=decision,
            answer_armenian=answer_armenian,
            used_chunk_ids=used_chunk_ids,
            refusal_reason=refusal_reason,
            generator_name="gemini",
            model_name=self._model,
        )


class ExtractiveGroundedAnswerGenerator:
    def generate(
        self,
        query: str,
        scope_decision: ScopeDecision,
        retrieval_results: list[RetrievalResult],
    ) -> GeneratedAnswer:
        if scope_decision.scope == "branches":
            branch_answer = _build_branch_answer(query, retrieval_results)
            if branch_answer is not None:
                return branch_answer

        candidate_sentences = _rank_candidate_sentences(query, retrieval_results)
        if not candidate_sentences:
            return GeneratedAnswer(
                decision="refuse",
                answer_armenian="Այս հարցին ապացույցների հիման վրա հստակ պատասխան կազմել չկարողացա։",
                used_chunk_ids=[],
                refusal_reason="insufficient_evidence",
                generator_name="extractive",
                model_name=None,
            )

        used_chunk_ids: list[str] = []
        fragments: list[str] = []
        for chunk_id, sentence in candidate_sentences[:3]:
            if chunk_id not in used_chunk_ids:
                used_chunk_ids.append(chunk_id)
            fragments.append(sentence)

        lead_bank = retrieval_results[0].bank_name
        answer = f"{lead_bank} բանկի պաշտոնական կայքից վերցված տվյալների հիման վրա՝ " + " ".join(fragments)
        return GeneratedAnswer(
            decision="answer",
            answer_armenian=answer.strip(),
            used_chunk_ids=used_chunk_ids,
            refusal_reason=None,
            generator_name="extractive",
            model_name=None,
        )


def build_answer_generator(settings: AppSettings, mode: str | None = None) -> AnswerGenerator:
    selected_mode = (mode or settings.qa_generator).casefold()
    if selected_mode == "gemini":
        if os.getenv("GEMINI_API_KEY"):
            return GeminiGroundedAnswerGenerator(settings)
        return ExtractiveGroundedAnswerGenerator()
    if selected_mode == "extractive":
        return ExtractiveGroundedAnswerGenerator()
    raise AnswerGeneratorError(f"Unsupported QA generator mode: {selected_mode}")


def _parse_json_object(raw_output: str) -> dict[str, object]:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if not match:
            raise AnswerGeneratorError("Could not parse JSON from model output.") from None
        return json.loads(match.group(0))
def _rank_candidate_sentences(
    query: str,
    retrieval_results: list[RetrievalResult],
) -> list[tuple[str, str]]:
    query_terms = {
        term
        for term in normalize_text(query).casefold().split()
        if len(term) > 2
    }
    ranked: list[tuple[int, str, str]] = []
    seen_sentences: set[str] = set()
    for result in retrieval_results:
        for sentence in split_sentences(result.chunk_text):
            normalized_sentence = normalize_text(sentence)
            lowered = normalized_sentence.casefold()
            overlap = sum(1 for term in query_terms if term in lowered)
            if overlap <= 0:
                continue
            if lowered in seen_sentences:
                continue
            seen_sentences.add(lowered)
            ranked.append((overlap, result.chunk_id, normalized_sentence))
    ranked.sort(key=lambda item: (-item[0], len(item[2])))
    return [(chunk_id, sentence) for _, chunk_id, sentence in ranked]


def _build_branch_answer(
    query: str,
    retrieval_results: list[RetrievalResult],
) -> GeneratedAnswer | None:
    snippets: list[tuple[str, str]] = []
    seen_snippets: set[str] = set()
    query_markers = _branch_query_markers(query)

    for result in retrieval_results:
        lines = [normalize_text(line) for line in result.chunk_text.splitlines()]
        lines = [line for line in lines if line]
        for index, line in enumerate(lines):
            if not _looks_like_branch_address(line):
                continue
            if query_markers and not any(marker in line.casefold() for marker in query_markers):
                continue

            branch_name = ""
            if index > 0:
                previous = lines[index - 1]
                if _looks_like_branch_name(previous):
                    branch_name = previous

            snippet = f"{branch_name}՝ {line}" if branch_name else line
            normalized_snippet = normalize_text(snippet).casefold()
            if normalized_snippet in seen_snippets:
                continue
            seen_snippets.add(normalized_snippet)
            snippets.append((result.chunk_id, snippet))
            if len(snippets) >= 3:
                break
        if len(snippets) >= 3:
            break

    if not snippets:
        return None

    lead_bank = retrieval_results[0].bank_name
    used_chunk_ids: list[str] = []
    snippet_texts: list[str] = []
    for chunk_id, snippet in snippets[:2]:
        if chunk_id not in used_chunk_ids:
            used_chunk_ids.append(chunk_id)
        snippet_texts.append(snippet)

    answer = (
        f"{lead_bank} բանկի պաշտոնական կայքի տվյալներով՝ "
        + "։ ".join(snippet_texts)
        + "։"
    )
    return GeneratedAnswer(
        decision="answer",
        answer_armenian=answer,
        used_chunk_ids=used_chunk_ids,
        refusal_reason=None,
        generator_name="extractive",
        model_name=None,
    )


def _branch_query_markers(query: str) -> set[str]:
    markers: set[str] = set()
    normalized_query = normalize_text(query).casefold()
    for token in normalized_query.split():
        compact = token.strip(".,:;!?\"'()[]{}")
        compact = compact.removeprefix("վտբ-")
        compact = compact.removesuffix("-ի")
        compact = compact.removesuffix("ում")
        compact = compact.removesuffix("ին")
        if len(compact) >= 4:
            markers.add(compact)
    return {
        marker
        for marker in markers
        if marker not in {"մասնաճյուղ", "որտեղ", "բանկ", "հասցե"}
    }


def _looks_like_branch_address(line: str) -> bool:
    lowered = line.casefold()
    return any(
        marker in lowered
        for marker in (
            "ք. երևան",
            "ք. երեւան",
            "փ.",
            "պող.",
            "մ/ճ",
            "հասցե",
        )
    )


def _looks_like_branch_name(line: str) -> bool:
    lowered = line.casefold()
    return "մասնաճյուղ" in lowered or ("«" in line and "»" in line)
