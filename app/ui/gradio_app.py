from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr

from app.config.settings import AppSettings
from app.qa.factory import build_grounded_qa_service
from app.voice.service import VoicePipelineService
from app.voice.stt import build_local_stt
from app.voice.tts import build_local_tts


ChatHistory = list[dict[str, str]]


class PipelineUIController:
    def __init__(self, settings: AppSettings, *, index_dir: Path | None = None) -> None:
        self._settings = settings
        self._index_dir = index_dir

    def send_text(
        self,
        query: str,
        generator: str,
        history: ChatHistory | None,
    ) -> tuple[ChatHistory, ChatHistory, str, str | None, str]:
        query = (query or "").strip()
        if not query:
            raise gr.Error("Մուտքագրեք հարցը։")

        qa_service = build_grounded_qa_service(
            self._settings,
            index_dir=self._index_dir,
            generator_mode=generator,
        )
        tts_provider = build_local_tts(self._settings)
        voice_service = VoicePipelineService(qa_service=qa_service, tts_provider=tts_provider)
        result = _run_async(voice_service.process_text(query))
        audio_path = _synthesize_reply(tts_provider, result.spoken_text, prefix="text_reply")

        updated_history = list(history or [])
        updated_history.append({"role": "user", "content": query})
        updated_history.append({"role": "assistant", "content": result.spoken_text})

        return (
            updated_history,
            updated_history,
            _format_sources(result.qa_result.supporting_results),
            audio_path,
            "",
        )

    def send_audio(
        self,
        audio_path: str | None,
        generator: str,
        history: ChatHistory | None,
    ) -> tuple[ChatHistory, ChatHistory, str, str | None, str, str | None]:
        if not audio_path:
            raise gr.Error("Ձայնագրեք կամ վերբեռնեք աուդիոն։")

        qa_service = build_grounded_qa_service(
            self._settings,
            index_dir=self._index_dir,
            generator_mode=generator,
        )
        stt_provider = build_local_stt(self._settings)
        tts_provider = build_local_tts(self._settings)
        voice_service = VoicePipelineService(
            qa_service=qa_service,
            stt_provider=stt_provider,
            tts_provider=tts_provider,
        )
        output_audio_path = _make_output_path("voice_reply")
        result = _run_async(
            voice_service.process_audio_file(
                Path(audio_path),
                output_audio_path=output_audio_path,
            )
        )

        updated_history = list(history or [])
        updated_history.append({"role": "user", "content": result.transcript.text})
        updated_history.append({"role": "assistant", "content": result.spoken_text})

        return (
            updated_history,
            updated_history,
            _format_sources(result.qa_result.supporting_results),
            str(result.synthesis.audio_path) if result.synthesis and result.synthesis.audio_path else None,
            result.transcript.text,
            None,
        )

    def clear_chat(self) -> tuple[ChatHistory, ChatHistory, str, str | None, str, str]:
        empty_history: ChatHistory = []
        return empty_history, empty_history, "", None, "", ""


def create_demo(settings: AppSettings, *, index_dir: Path | None = None) -> gr.Blocks:
    controller = PipelineUIController(settings, index_dir=index_dir)
    css = """
    .app-shell {max-width: 980px; margin: 0 auto;}
    .title-block {padding: 12px 0 4px 0;}
    .gradio-container {background: linear-gradient(180deg, #f7f2e7 0%, #ffffff 100%);}
    """

    with gr.Blocks(
        title="Armenian Bank Voice Agent",
        css=css,
        theme=gr.themes.Soft(
            primary_hue="amber",
            secondary_hue="stone",
            neutral_hue="slate",
        ),
    ) as demo:
        history_state = gr.State([])

        gr.Markdown(
            """
            <div class="app-shell title-block">
            <h1>Armenian Bank Support Agent</h1>
            <p>Ask about credits, deposits, and branches. Answers stay grounded in official bank data.</p>
            </div>
            """
        )

        with gr.Row():
            generator = gr.Dropdown(
                choices=["gemini", "extractive"],
                value=settings.qa_generator,
                label="QA Generator",
                scale=1,
            )
            clear_btn = gr.Button("Clear Chat", variant="secondary", scale=0)

        chatbot = gr.Chatbot(
            label="Conversation",
            type="messages",
            height=520,
            show_copy_button=True,
        )

        with gr.Row():
            text_query = gr.Textbox(
                label="Type a question",
                placeholder="Օրինակ՝ ՎՏԲ-ի մասնաճյուղը որտեղ է Երևանում",
                lines=2,
                scale=5,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            audio_input = gr.Audio(
                label="Voice input",
                type="filepath",
                sources=["microphone", "upload"],
                scale=5,
            )
            voice_btn = gr.Button("Send Voice", variant="primary", scale=1)

        last_transcript = gr.Textbox(
            label="Last voice transcript",
            interactive=False,
            lines=2,
        )
        latest_audio = gr.Audio(
            label="Latest spoken reply",
            interactive=False,
            type="filepath",
        )
        gr.Markdown("### Latest Supporting Sources")
        latest_sources = gr.Markdown()

        send_btn.click(
            controller.send_text,
            inputs=[text_query, generator, history_state],
            outputs=[chatbot, history_state, latest_sources, latest_audio, text_query],
        )
        text_query.submit(
            controller.send_text,
            inputs=[text_query, generator, history_state],
            outputs=[chatbot, history_state, latest_sources, latest_audio, text_query],
        )
        voice_btn.click(
            controller.send_audio,
            inputs=[audio_input, generator, history_state],
            outputs=[
                chatbot,
                history_state,
                latest_sources,
                latest_audio,
                last_transcript,
                audio_input,
            ],
        )
        clear_btn.click(
            controller.clear_chat,
            outputs=[chatbot, history_state, latest_sources, latest_audio, last_transcript, text_query],
        )

    return demo


def _format_sources(results: list) -> str:
    if not results:
        return "No supporting sources."

    lines = []
    for result in results:
        lines.append(
            f"- `{result.bank_code}/{result.topic}` | [{result.page_title}]({result.source_url})"
        )
    return "\n".join(lines)


def _make_output_path(prefix: str) -> Path:
    tmp_dir = Path("tmp/ui_outputs")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f"{prefix}_", suffix=".wav", dir=tmp_dir, delete=False) as handle:
        return Path(handle.name)


def _synthesize_reply(tts_provider, text: str, *, prefix: str) -> str | None:
    output_audio_path = _make_output_path(prefix)
    synthesis = tts_provider.synthesize_to_file(text, output_audio_path)
    return str(synthesis.audio_path) if synthesis.audio_path else None


def _run_async(awaitable):
    import asyncio

    return asyncio.run(awaitable)
