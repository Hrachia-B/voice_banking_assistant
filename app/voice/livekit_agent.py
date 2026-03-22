from __future__ import annotations

import asyncio
import logging

from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, room_io
from livekit.agents.voice.events import UserInputTranscribedEvent
from livekit.plugins import silero

from app.config.settings import AppSettings
from app.qa.factory import build_grounded_qa_service
from app.voice.service import VoicePipelineService
from app.voice.stt import build_livekit_stt
from app.voice.tts import build_livekit_tts


LOGGER = logging.getLogger(__name__)


class GroundedBankVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You do not generate answers yourself. User speech is handled by an external "
                "grounded QA pipeline and only the approved Armenian text is spoken back."
            )
        )


def create_voice_server(settings: AppSettings) -> AgentServer:
    server = AgentServer(
        ws_url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
        log_level=settings.log_level,
    )

    @server.rtc_session(agent_name=settings.voice_agent_name)
    async def bank_voice_agent(ctx: agents.JobContext) -> None:
        qa_service = build_grounded_qa_service(settings, generator_mode=settings.qa_generator)
        voice_service = VoicePipelineService(qa_service=qa_service)
        vad = silero.VAD.load(
            sample_rate=16000,
            min_silence_duration=settings.voice_vad_min_silence_duration,
            activation_threshold=settings.voice_vad_activation_threshold,
        )
        stt_provider = build_livekit_stt(settings)
        tts_provider = build_livekit_tts(settings)
        stt_provider.prewarm()
        tts_provider.prewarm()

        session = AgentSession(
            stt=stt_provider,
            vad=vad,
            tts=tts_provider,
        )
        agent = GroundedBankVoiceAgent()
        reply_lock = asyncio.Lock()

        async def _reply_to_user(transcript: str) -> None:
            async with reply_lock:
                result = await voice_service.process_text(transcript)
                LOGGER.info(
                    "Voice QA result",
                    extra={
                        "query": result.transcript.text,
                        "status": result.qa_result.status,
                        "scope": result.qa_result.scope_decision.scope,
                        "generator": result.qa_result.generator_name,
                    },
                )
                handle = session.say(
                    result.spoken_text,
                    allow_interruptions=True,
                    add_to_chat_ctx=False,
                )
                await handle.wait_for_playout()

        def _on_user_input(ev: UserInputTranscribedEvent) -> None:
            if not ev.is_final or not ev.transcript.strip():
                return
            asyncio.create_task(_reply_to_user(ev.transcript))

        session.on("user_input_transcribed", _on_user_input)
        await session.start(
            room=ctx.room,
            agent=agent,
            room_options=room_io.RoomOptions(),
        )

        if settings.voice_greeting_text.strip():
            greeting = session.say(
                settings.voice_greeting_text,
                allow_interruptions=False,
                add_to_chat_ctx=False,
            )
            await greeting.wait_for_playout()

    return server
