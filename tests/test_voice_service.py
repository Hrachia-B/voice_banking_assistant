from __future__ import annotations

from pathlib import Path

from app.guardrails.scope import ScopeDecision
from app.qa.models import QAResult
from app.voice.models import SpeechSynthesisResult, SpeechTranscript
from app.voice.service import VoicePipelineService


class FakeQAService:
    def __init__(self, qa_result: QAResult) -> None:
        self._qa_result = qa_result
        self.last_query: str | None = None

    def answer_query(self, query: str) -> QAResult:
        self.last_query = query
        return self._qa_result


class FakeSTTProvider:
    def transcribe_file(self, audio_path: Path) -> SpeechTranscript:
        return SpeechTranscript(text="Յունիբանկի սպառողական վարկ", language="hy", confidence=0.95)


class FakeTTSProvider:
    def __init__(self) -> None:
        self.spoken_text: str | None = None

    def synthesize_to_file(self, text: str, output_path: Path) -> SpeechSynthesisResult:
        self.spoken_text = text
        output_path.write_bytes(b"fake-wav")
        return SpeechSynthesisResult(
            audio_path=output_path,
            sample_rate=22050,
            num_channels=1,
            provider="fake",
        )


def test_voice_pipeline_uses_grounded_qa_answer_for_spoken_text(tmp_path: Path) -> None:
    qa_result = _make_qa_result(
        query="Յունիբանկի սպառողական վարկ",
        answer="Սպառողական վարկի տոկոսադրույքը 21.5% է։",
        status="answered",
    )
    qa_service = FakeQAService(qa_result)
    tts_provider = FakeTTSProvider()
    service = VoicePipelineService(
        qa_service=qa_service,  # type: ignore[arg-type]
        stt_provider=FakeSTTProvider(),  # type: ignore[arg-type]
        tts_provider=tts_provider,  # type: ignore[arg-type]
    )

    result = _run(service.process_audio_file(tmp_path / "input.wav", output_audio_path=tmp_path / "reply.wav"))

    assert qa_service.last_query == "Յունիբանկի սպառողական վարկ"
    assert result.spoken_text == "Սպառողական վարկի տոկոսադրույքը 21.5% է։"
    assert tts_provider.spoken_text == "Սպառողական վարկի տոկոսադրույքը 21.5% է։"
    assert result.synthesis is not None


def test_voice_pipeline_speaks_refusal_text_from_qa_result() -> None:
    qa_result = _make_qa_result(
        query="Քարտի վճար",
        answer="Կարող եմ պատասխանել միայն վարկերի, ավանդների և մասնաճյուղերի մասին հարցերի սահմաններում։",
        status="refused",
    )
    service = VoicePipelineService(qa_service=FakeQAService(qa_result))  # type: ignore[arg-type]

    result = _run(service.process_text("Քարտի վճար"))

    assert result.qa_result.status == "refused"
    assert result.spoken_text == qa_result.answer_armenian


def _make_qa_result(*, query: str, answer: str, status: str) -> QAResult:
    return QAResult(
        query=query,
        status=status,  # type: ignore[arg-type]
        answer_armenian=answer,
        scope_decision=ScopeDecision(
            scope="credits" if status == "answered" else "out_of_scope",
            confidence=1.0,
            bank_code="unibank" if status == "answered" else None,
            matched_topic_terms=["վարկ"] if status == "answered" else [],
            matched_bank_terms=["յունիբանկ"] if status == "answered" else [],
            reason="keyword_match" if status == "answered" else "explicit_out_of_scope_keyword",
        ),
        refusal_reason=None if status == "answered" else "out_of_scope",
        evidence_assessment=None,
        retrieved_results=[],
        supporting_results=[],
        generator_name="gemini" if status == "answered" else "none",
        model_name="gemini-2.5-flash" if status == "answered" else None,
    )


def _run(awaitable):
    import asyncio

    return asyncio.run(awaitable)
