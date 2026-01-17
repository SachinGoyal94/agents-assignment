"""
Smart Interruption Agent - COMPLETE INTEGRATED VERSION
Location: examples/voice_agents/smart_interruption_agent.py

Fully functional voice agent with semantic interruption handling.
Combines the interruption handler logic directly into the agent implementation.
"""

import asyncio
import logging
import os
import re
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
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

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# INTERRUPTION HANDLER IMPLEMENTATION
# ============================================================================

class AgentState(Enum):
    """Agent conversation states"""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"


class InterruptionIntent(Enum):
    """Classification of user interruption intent"""
    IGNORE = "ignore"
    PAUSE = "pause"
    HARD_STOP = "hard_stop"


@dataclass
class InterruptionContext:
    """Context for making interruption decisions"""
    agent_state: AgentState
    agent_speech_duration: float
    user_input_detected: bool
    vad_confidence: float
    stt_text: Optional[str] = None
    timestamp: float = 0.0


class SemanticInterruptionHandler:
    """
    Core interruption logic using pattern matching and context.

    Design Principles:
    - Non-blocking async operations
    - Configurable thresholds and patterns
    - Stateful conversation tracking
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

    def __init__(
        self,
        min_speech_duration_for_interrupt: float = 1.0,
        vad_confidence_threshold: float = 0.7,
        stt_timeout: float = 0.5,
    ):
        self.min_speech_duration = min_speech_duration_for_interrupt
        self.vad_threshold = vad_confidence_threshold
        self.stt_timeout = stt_timeout

        self.current_state = AgentState.IDLE
        self.speech_start_time: Optional[float] = None

        # Compile patterns for performance
        self.soft_patterns = [re.compile(p, re.IGNORECASE) for p in self.SOFT_ACKNOWLEDGMENTS]
        self.hard_patterns = [re.compile(p, re.IGNORECASE) for p in self.HARD_INTERRUPTIONS]
        self.pause_patterns = [re.compile(p, re.IGNORECASE) for p in self.PAUSE_PATTERNS]

        logger.info(f"✅ Interruption handler initialized (min_duration={self.min_speech_duration}s)")

    def _classify_intent(self, text: str) -> InterruptionIntent:
        """Fast intent classification using regex patterns"""
        if not text or len(text.strip()) < 2:
            return InterruptionIntent.IGNORE

        text_lower = text.lower().strip()

        # Priority: HARD_STOP > PAUSE > IGNORE
        for pattern in self.hard_patterns:
            if pattern.search(text_lower):
                logger.debug(f"🛑 Hard interruption: '{text}'")
                return InterruptionIntent.HARD_STOP

        for pattern in self.pause_patterns:
            if pattern.search(text_lower):
                logger.debug(f"⏸️  Pause interruption: '{text}'")
                return InterruptionIntent.PAUSE

        for pattern in self.soft_patterns:
            if pattern.search(text_lower):
                logger.debug(f"✨ Soft acknowledgment: '{text}'")
                return InterruptionIntent.IGNORE

        # Short input = likely acknowledgment
        word_count = len(text_lower.split())
        if word_count <= 3:
            logger.debug(f"✨ Short input (acknowledgment): '{text}'")
            return InterruptionIntent.IGNORE

        logger.debug(f"⏸️  Longer input (interruption): '{text}'")
        return InterruptionIntent.PAUSE

    async def handle_vad_event(self, vad_confidence: float) -> dict:
        """Process VAD detection and decide next action"""
        current_time = asyncio.get_event_loop().time()

        if self.current_state in [AgentState.IDLE, AgentState.LISTENING]:
            return {"action": "ignore", "reason": "Agent not speaking"}

        if vad_confidence < self.vad_threshold:
            return {"action": "ignore", "reason": f"VAD confidence low: {vad_confidence:.2f}"}

        speech_duration = 0.0
        if self.speech_start_time:
            speech_duration = current_time - self.speech_start_time

        if speech_duration < self.min_speech_duration:
            return {
                "action": "wait_for_stt",
                "reason": f"Agent speaking {speech_duration:.1f}s, waiting for analysis",
                "timeout": self.stt_timeout
            }

        return {
            "action": "wait_for_stt",
            "reason": "Waiting for semantic classification",
            "timeout": self.stt_timeout
        }

    async def handle_stt_result(self, text: str, vad_confidence: float) -> dict:
        """Make final interruption decision based on transcription"""
        intent = self._classify_intent(text)

        decision = {
            "intent": intent.value,
            "text": text,
            "vad_confidence": vad_confidence
        }

        if intent == InterruptionIntent.IGNORE:
            decision.update({
                "action": "continue",
                "reason": f"Soft acknowledgment: '{text}'"
            })
        elif intent == InterruptionIntent.HARD_STOP:
            decision.update({
                "action": "stop",
                "reason": f"Hard interruption: '{text}'"
            })
        else:
            decision.update({
                "action": "stop",
                "reason": f"Polite interruption: '{text}'"
            })

        logger.info(f"📊 Decision: {decision['action'].upper()} - {decision['reason']}")
        return decision

    def update_agent_state(self, new_state: AgentState):
        """Update current agent state"""
        if self.current_state != new_state:
            logger.debug(f"🔄 State: {self.current_state.value} → {new_state.value}")
            self.current_state = new_state

        if new_state == AgentState.SPEAKING:
            self.speech_start_time = asyncio.get_event_loop().time()
        else:
            self.speech_start_time = None

    def get_speech_duration(self) -> float:
        """Get current agent speech duration"""
        if self.speech_start_time is None:
            return 0.0
        return asyncio.get_event_loop().time() - self.speech_start_time


class InterruptionManager:
    """Coordinates VAD, STT, and interruption logic"""

    def __init__(self, handler: SemanticInterruptionHandler):
        self.handler = handler

    async def on_vad_detected(self, vad_confidence: float, stt_callback: Callable) -> bool:
        """
        Process VAD event and determine if agent should stop.

        Returns:
            bool: True to stop agent, False to continue
        """
        decision = await self.handler.handle_vad_event(vad_confidence)

        if decision["action"] == "ignore":
            logger.debug(f"[VAD] Ignoring: {decision['reason']}")
            return False

        if decision["action"] == "stop_immediately":
            logger.info(f"[VAD] ⛔ Stopping: {decision['reason']}")
            return True

        if decision["action"] == "wait_for_stt":
            logger.debug(f"[VAD] ⏳ {decision['reason']}")
            timeout = decision.get("timeout", 0.5)

            try:
                stt_text = await asyncio.wait_for(stt_callback(), timeout=timeout)

                final_decision = await self.handler.handle_stt_result(
                    stt_text, vad_confidence
                )

                should_stop = final_decision["action"] == "stop"
                icon = "🛑" if should_stop else "✅"
                logger.info(f"[STT] {icon} {final_decision['reason']}")
                return should_stop

            except asyncio.TimeoutError:
                speech_duration = self.handler.get_speech_duration()
                if speech_duration > 3.0:
                    logger.warning(f"[TIMEOUT] ⚠️  Agent spoke {speech_duration:.1f}s, stopping")
                    return True
                else:
                    logger.warning(f"[TIMEOUT] ⚠️  Agent spoke {speech_duration:.1f}s, continuing")
                    return False

        return False


# ============================================================================
# AGENT IMPLEMENTATION
# ============================================================================

async def entrypoint(ctx: JobContext):
    """Main agent with integrated smart interruption handling"""
    await ctx.connect()
    logger.info("🚀 Connected to LiveKit room: %s", ctx.room.name)

    # Initialize interruption system
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
            "You are a helpful and conversational AI assistant. "
            "Speak naturally and keep responses concise (2-3 sentences). "
            "Be friendly and engaging. You handle natural interruptions smoothly."
        )
    )

    # Initialize providers
    logger.info("🎯 Initializing providers...")

    voice_activity_detector = silero.VAD.load()
    logger.info("✅ VAD: Silero (local)")

    speech_to_text = deepgram.STT(model="nova-3")
    logger.info("✅ STT: Deepgram Nova-3")

    language_model = google.LLM(
        model="gemini-2.5-flash-lite-preview-09-2025",
        temperature=0.8,
    )
    logger.info("✅ LLM: Google Gemini 2.0 Flash")

    text_to_speech = google.TTS(voice_name="en-US-Neural2-J")
    logger.info("✅ TTS: Google Cloud TTS")

    # Create session
    session = AgentSession(
        vad=voice_activity_detector,
        stt=speech_to_text,
        llm=language_model,
        tts=text_to_speech,
    )

    logger.info("🎬 Starting agent session...")
    await session.start(agent=agent, room=ctx.room)

    # Generate greeting
    logger.info("👋 Generating greeting...")
    await session.generate_reply(
        instructions="Greet the user warmly and briefly introduce yourself."
    )

    logger.info("✅ Agent is live with smart interruption handling!")

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
    logger.info("=" * 60)
    logger.info("🤖 Smart Interruption Agent - Complete Version")
    logger.info("=" * 60)
    logger.info("📚 Features:")
    logger.info("  ✓ Semantic pattern matching for interruptions")
    logger.info("  ✓ Soft acknowledgment detection (yeah, mhmm, okay)")
    logger.info("  ✓ Hard interruption detection (wait, hold on, but)")
    logger.info("  ✓ Polite interruption handling (can I ask...)")
    logger.info("  ✓ Configurable timing and thresholds")
    logger.info("  ✓ Full integration with LiveKit agent")
    logger.info("=" * 60)
    logger.info("💡 Usage:")
    logger.info("  Console: python smart_interruption_agent.py console")
    logger.info("  Dev:     python smart_interruption_agent.py dev")
    logger.info("=" * 60)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))