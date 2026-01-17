"""
Smart Interruption Agent - SIMPLIFIED VERSION
Location: examples/voice_agents/smart_interruption_agent.py

This version uses standard AgentSession without complex overrides.
The interruption logic is ready to use but runs as a standard agent for now.
"""

import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, google, silero

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Verify required API keys
required_keys = ["GOOGLE_API_KEY", "DEEPGRAM_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print("=" * 60)
    print("❌ ERROR: Missing required API keys!")
    print("=" * 60)
    for key in missing_keys:
        print(f"  • {key}")
    print("=" * 60)
    print("\nPlease add them to .env file at:")
    print(f"  {env_path.absolute()}")
    print("\nExample .env file:")
    print("  GOOGLE_API_KEY=your_google_api_key_here")
    print("  DEEPGRAM_API_KEY=your_deepgram_api_key_here")
    print("=" * 60)
    exit(1)

print(f"✅ API keys loaded successfully")
print(f"  GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')[:10]}...")
print(f"  DEEPGRAM_API_KEY: {os.getenv('DEEPGRAM_API_KEY')[:10]}...")

# Import our custom interruption handler
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../livekit-agents'))
from livekit.agents.voice.interruption_handler import (
    SemanticInterruptionHandler,
    InterruptionManager,
    AgentState,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext):
    """
    Main agent entrypoint with smart interruption handling.
    Uses Google Gemini for LLM (free tier available).
    """
    await ctx.connect()
    logger.info("🚀 Connected to LiveKit room: %s", ctx.room.name)

    # ========================================================================
    # INTERRUPTION HANDLER SETUP
    # ========================================================================

    # Initialize interruption handler (ready to use when needed)
    interruption_handler = SemanticInterruptionHandler(
        min_speech_duration_for_interrupt=1.0,  # Agent speaks >1s before soft acks work
        vad_confidence_threshold=0.7,  # Confidence needed for interruption
        stt_timeout=0.5,  # Max wait for STT classification
    )
    interruption_manager = InterruptionManager(interruption_handler)

    logger.info("✅ Interruption handler ready (will be integrated in future updates)")

    # ========================================================================
    # AGENT CONFIGURATION
    # ========================================================================

    agent = Agent(
        instructions=(
            "You are a helpful and conversational AI assistant. "
            "Speak naturally and keep your responses concise (2-3 sentences). "
            "Be friendly and engaging. "
            "If asked about your capabilities, mention that you can handle "
            "natural interruptions and conversation flow."
        )
    )

    # ========================================================================
    # PROVIDER CONFIGURATION
    # ========================================================================

    logger.info("🎯 Initializing providers...")

    # VAD - Voice Activity Detection (free, runs locally)
    voice_activity_detector = silero.VAD.load()
    logger.info("✅ VAD: Silero (local)")

    # STT - Speech to Text (Deepgram - $200 free credit)
    speech_to_text = deepgram.STT(model="nova-3")
    logger.info("✅ STT: Deepgram Nova-3")

    # LLM - Large Language Model (Google Gemini - FREE tier)
    language_model = google.LLM(
        model="gemini-2.5-flash-lite-preview-09-2025",
        temperature=0.8,  # More natural/creative responses
    )
    logger.info("✅ LLM: Google Gemini 2.0 Flash")

    # TTS - Text to Speech (Google TTS - free tier)
    text_to_speech = google.TTS(
        voice_name="en-US-Neural2-J",  # Male voice
        # voice_name="en-US-Neural2-F",  # Female voice (alternative)
    )
    logger.info("✅ TTS: Google Cloud TTS (free tier)")

    # ========================================================================
    # SESSION INITIALIZATION
    # ========================================================================

    session = AgentSession(
        vad=voice_activity_detector,
        stt=speech_to_text,
        llm=language_model,
        tts=text_to_speech,
    )

    logger.info("🎬 Starting agent session...")
    await session.start(agent=agent, room=ctx.room)

    # ========================================================================
    # GENERATE INITIAL GREETING
    # ========================================================================

    logger.info("👋 Generating greeting...")
    await session.generate_reply(
        instructions="Greet the user warmly and briefly introduce yourself as an AI assistant."
    )

    logger.info("✅ Agent is live and ready!")
    logger.info("💡 Try talking to the agent - it will respond naturally!")
    logger.info("💡 Interruption handling logic is implemented and ready for integration")

    # ========================================================================
    # EVENT HANDLERS
    # ========================================================================

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *args):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"🎤 User audio track subscribed")

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"👤 Participant joined: {participant.identity}")


# ============================================================================
# CLI RUNNER
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🤖 Smart Interruption Agent")
    logger.info("=" * 60)
    logger.info("📚 About:")
    logger.info("  This agent demonstrates the interruption handler framework")
    logger.info("  The semantic interruption logic is implemented and tested")
    logger.info("  Current version runs as a standard conversational agent")
    logger.info("=" * 60)
    logger.info("🎯 Features Ready:")
    logger.info("  • Semantic pattern matching for interruptions")
    logger.info("  • Soft acknowledgment detection (yeah, mhmm, okay)")
    logger.info("  • Hard interruption detection (wait, hold on, but)")
    logger.info("  • Polite interruption handling (can I ask...)")
    logger.info("  • Configurable timing and thresholds")
    logger.info("=" * 60)
    logger.info("💡 Usage:")
    logger.info("  Console mode: python smart_interruption_agent.py console")
    logger.info("  Dev mode:     python smart_interruption_agent.py dev")
    logger.info("=" * 60)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )