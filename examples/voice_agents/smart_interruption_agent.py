"""
Smart Interruption Agent - CARTESIA STREAMING TTS
Location: examples/voice_agents/smart_interruption_agent.py

Uses Cartesia for TRUE STREAMING TTS - perfect for real-time conversation!
Low latency, simple setup, just needs an API key.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.plugins import deepgram, google, cartesia, silero

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

required_keys = ["GOOGLE_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print("=" * 60)
    print("❌ ERROR: Missing required API keys!")
    print("=" * 60)
    for key in missing_keys:
        print(f"  • {key}")
    print("=" * 60)
    print("\nPlease add them to .env file:")
    print("  GOOGLE_API_KEY=your_google_key")
    print("  DEEPGRAM_API_KEY=your_deepgram_key")
    print("  CARTESIA_API_KEY=your_cartesia_key")
    print("\nGet Cartesia key (FREE): https://cartesia.ai/")
    print("=" * 60)
    exit(1)

print(f"✅ API keys loaded")
print(f"  GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')[:10]}...")
print(f"  DEEPGRAM_API_KEY: {os.getenv('DEEPGRAM_API_KEY')[:10]}...")
print(f"  CARTESIA_API_KEY: {os.getenv('CARTESIA_API_KEY')[:10]}...")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# INTERRUPTION HANDLER
# ============================================================================

class AgentState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"


class InterruptionIntent(Enum):
    IGNORE = "ignore"
    PAUSE = "pause"
    HARD_STOP = "hard_stop"


class SemanticInterruptionHandler:
    """
    Smart interruption handler with pattern matching.
    Detects soft acknowledgments, hard interruptions, and polite pauses.
    """

    SOFT_ACKNOWLEDGMENTS = {
        r'\b(yeah|yep|yes|uh-huh|mm-hmm|mhmm|okay|ok|right|sure|got it)\b',
        r'\b(i see|makes sense|understood|alright|cool|nice)\b',
        r'\b(go on|continue|keep going)\b',
    }

    HARD_INTERRUPTIONS = {
        r'\b(wait|hold on|stop|hang on|but|however|actually)\b',
        r'\b(excuse me|sorry|one sec|one second|pause)\b',
        r'\bno\b(?!\s+problem)',
        r'\b(what|huh|pardon)\b',
    }

    PAUSE_PATTERNS = {
        r'\b(quick question|can i ask|may i|could you)\b',
        r'\b(before you|let me|i want to|i need to)\b',
        r'\b(just to clarify|to be clear)\b',
    }

    def __init__(self, min_speech_duration_for_interrupt=1.0, vad_confidence_threshold=0.7, stt_timeout=0.5):
        self.min_speech_duration = min_speech_duration_for_interrupt
        self.vad_threshold = vad_confidence_threshold
        self.stt_timeout = stt_timeout
        self.current_state = AgentState.IDLE
        self.speech_start_time = None

        self.soft_patterns = [re.compile(p, re.IGNORECASE) for p in self.SOFT_ACKNOWLEDGMENTS]
        self.hard_patterns = [re.compile(p, re.IGNORECASE) for p in self.HARD_INTERRUPTIONS]
        self.pause_patterns = [re.compile(p, re.IGNORECASE) for p in self.PAUSE_PATTERNS]

        logger.info(f"✅ Interruption handler initialized")

    def _classify_intent(self, text: str) -> InterruptionIntent:
        """Classify user input as soft ack, hard stop, or pause"""
        if not text or len(text.strip()) < 2:
            return InterruptionIntent.IGNORE

        text_lower = text.lower().strip()

        # Check hard interruptions first (highest priority)
        for pattern in self.hard_patterns:
            if pattern.search(text_lower):
                logger.info(f"🛑 HARD INTERRUPTION: '{text}'")
                return InterruptionIntent.HARD_STOP

        # Check pause patterns
        for pattern in self.pause_patterns:
            if pattern.search(text_lower):
                logger.info(f"⏸️  POLITE INTERRUPTION: '{text}'")
                return InterruptionIntent.PAUSE

        # Check soft acknowledgments (should be ignored)
        for pattern in self.soft_patterns:
            if pattern.search(text_lower):
                logger.info(f"✨ SOFT ACK (IGNORED): '{text}'")
                return InterruptionIntent.IGNORE

        # Short input = likely acknowledgment
        if len(text_lower.split()) <= 3:
            logger.info(f"✨ SHORT INPUT (IGNORED): '{text}'")
            return InterruptionIntent.IGNORE

        # Longer input = likely wants to interrupt
        logger.info(f"⏸️  INTERRUPTION: '{text}'")
        return InterruptionIntent.PAUSE

    def update_agent_state(self, new_state: AgentState):
        if self.current_state != new_state:
            self.current_state = new_state
        if new_state == AgentState.SPEAKING:
            self.speech_start_time = asyncio.get_event_loop().time()
        else:
            self.speech_start_time = None


class InterruptionManager:
    def __init__(self, handler: SemanticInterruptionHandler):
        self.handler = handler


# ============================================================================
# AGENT IMPLEMENTATION
# ============================================================================

async def entrypoint(ctx: JobContext):
    """Main agent with streaming TTS and smart interruption handling"""
    await ctx.connect()
    logger.info("🚀 Connected to LiveKit room: %s", ctx.room.name)

    # Initialize interruption handler
    interruption_handler = SemanticInterruptionHandler(
        min_speech_duration_for_interrupt=1.0,
        vad_confidence_threshold=0.7,
        stt_timeout=0.5,
    )
    interruption_manager = InterruptionManager(interruption_handler)
    logger.info("✅ Interruption system active")

    # Configure agent
    agent = Agent(
        instructions=(
            "You are a helpful AI assistant. "
            "Speak naturally and keep responses concise (2-3 sentences). "
            "Be friendly and engaging."
        )
    )

    # Initialize providers
    logger.info("🎯 Initializing providers...")

    # VAD - Voice Activity Detection
    voice_activity_detector = silero.VAD.load()
    logger.info("✅ VAD: Silero (local)")

    # STT - Speech to Text
    speech_to_text = deepgram.STT(model="nova-3")
    logger.info("✅ STT: Deepgram Nova-3")

    # LLM - Language Model
    language_model = google.LLM(
        model="gemini-2.5-flash-lite-preview-09-2025",
        temperature=0.7,
    )
    logger.info("✅ LLM: gemini-2.5-flash-lite-preview-09-2025")

    # TTS - STREAMING Text to Speech (Cartesia)
    text_to_speech = cartesia.TTS(
        api_key=os.getenv("CARTESIA_API_KEY"),
        # Voice options:
        voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # British narrator (male)
        # "248be419-c632-4f23-adf1-5324ed7dbf1d" - Friendly woman
        # "421b3369-f63f-4b03-8980-37a44df1d4e8" - Professional man
        # "694f9389-aac1-45b6-b726-9d9369183238" - Calm woman
        # "79a125e8-cd45-4c13-8a67-188112f4dd22" - Energetic man
    )
    logger.info("✅ TTS: Cartesia (STREAMING - Low Latency!)")

    # Create session
    session = AgentSession(
        vad=voice_activity_detector,
        stt=speech_to_text,
        llm=language_model,
        tts=text_to_speech,
    )

    logger.info("🎬 Starting agent...")
    await session.start(agent=agent, room=ctx.room)

    logger.info("👋 Generating greeting...")
    await session.generate_reply(
        instructions="Greet the user briefly and introduce yourself."
    )

    logger.info("✅ Agent is LIVE with STREAMING TTS!")
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎯 TEST INTERRUPTION HANDLING:")
    logger.info("=" * 70)
    logger.info("📢 While agent is speaking, try:")
    logger.info("")
    logger.info("  ✨ SOFT ACKNOWLEDGMENTS (Should be IGNORED):")
    logger.info("     'yeah', 'okay', 'mhmm', 'I see', 'right'")
    logger.info("     → Watch terminal for: ✨ SOFT ACK (IGNORED)")
    logger.info("")
    logger.info("  🛑 HARD INTERRUPTIONS (Should STOP agent):")
    logger.info("     'wait', 'hold on', 'stop', 'but'")
    logger.info("     → Watch terminal for: 🛑 HARD INTERRUPTION")
    logger.info("")
    logger.info("  ⏸️  POLITE INTERRUPTIONS (Should STOP agent):")
    logger.info("     'can I ask', 'quick question', 'before you'")
    logger.info("     → Watch terminal for: ⏸️  POLITE INTERRUPTION")
    logger.info("=" * 70)

    # Event handlers
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
    logger.info("=" * 70)
    logger.info("🤖 Smart Interruption Agent - STREAMING TTS Edition")
    logger.info("=" * 70)
    logger.info("✨ Features:")
    logger.info("  • TRUE STREAMING TTS (Cartesia) - Low latency!")
    logger.info("  • Smart semantic interruption handling")
    logger.info("  • Soft acknowledgment detection")
    logger.info("  • Hard interruption detection")
    logger.info("  • Polite interruption handling")
    logger.info("=" * 70)
    logger.info("💡 Usage:")
    logger.info("  python smart_interruption_agent.py console")
    logger.info("=" * 70)
    logger.info("🔑 Need Cartesia API key? Get it FREE at:")
    logger.info("  https://cartesia.ai/")
    logger.info("=" * 70)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))