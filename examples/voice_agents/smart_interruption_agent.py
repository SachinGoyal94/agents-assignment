"""
Advanced Smart Interruption Agent
Location: examples/voice_agents/smart_interruption_agent.py

Features:
- Timing-aware interruption detection
- Audio urgency analysis
- Confidence scoring
- Detailed decision logging
"""

import sys
from pathlib import Path

# Add local package path BEFORE other imports to find the new interruption_handler
_repo_root = Path(__file__).resolve().parents[2]
_candidates = [_repo_root / "livekit-agents", _repo_root]
for _p in _candidates:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

import asyncio
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
import re
from typing import Optional
import numpy as np

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import deepgram, google, cartesia, silero

# Load environment
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Import our advanced handler
from livekit.agents.voice.interruption_handler import (
    TimingAwareInterruptionHandler,
    InterruptionIntent,
    AudioFeatures,
    AudioAnalyzer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedInterruptionSession(AgentSession):
    """
    Custom AgentSession with advanced interruption detection.
    Uses timing awareness + audio urgency detection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize advanced handler
        self.interruption_handler = TimingAwareInterruptionHandler(
            min_speech_duration=1.0,      # Don't interrupt first 1 second
            short_speech_threshold=3.0,   # Be conservative before 3 seconds
            long_speech_threshold=8.0,    # Be lenient after 8 seconds
            base_threshold=0.7,           # Base confidence threshold
        )

        self.audio_analyzer = AudioAnalyzer()
        self._last_user_audio: Optional[np.ndarray] = None

        logger.info("✅ Advanced interruption session initialized")
        logger.info("   Features: Timing-aware + Urgency detection + Confidence scoring")

    async def _should_interrupt(self, transcript: str) -> bool:
        """
        Advanced interruption logic with full context.

        This is called by the agent framework when user speaks during agent speech.
        """
        # Get agent speech duration
        speech_duration = self.interruption_handler.get_speech_duration()

        # Extract audio features if available
        audio_features = None
        if self._last_user_audio is not None and len(self._last_user_audio) > 0:
            try:
                audio_features = self.audio_analyzer.extract_features(
                    self._last_user_audio,
                    sample_rate=16000
                )
                logger.debug(f"🎤 Audio features: energy={audio_features.rms_energy:.3f}, "
                           f"zcr={audio_features.zero_crossing_rate:.3f}, "
                           f"duration={audio_features.duration:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to extract audio features: {e}")

        # Make decision with full context
        score = self.interruption_handler.decide(
            text=transcript,
            speech_duration=speech_duration,
            audio_features=audio_features
        )

        # Log decision with color coding
        if score.decision == InterruptionIntent.INTERRUPT:
            logger.info(f"🛑 INTERRUPTING (confidence={score.confidence:.2f})")
            logger.info(f"   User said: '{transcript}'")
            logger.info(f"   Agent spoke for: {speech_duration:.1f}s")
            logger.info(f"   Scores: semantic={score.semantic_score:.2f}, "
                       f"timing={score.timing_score:.2f}, urgency={score.urgency_score:.2f}")
        else:
            logger.info(f"✅ CONTINUING (confidence={score.confidence:.2f})")
            logger.info(f"   User said: '{transcript}' (acknowledged)")
            logger.info(f"   Agent continues speaking...")

        return score.decision == InterruptionIntent.INTERRUPT

    def _on_user_audio(self, audio_data: bytes):
        """Capture user audio for feature extraction"""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            self._last_user_audio = audio_array
        except Exception as e:
            logger.debug(f"Error capturing user audio: {e}")


async def entrypoint(ctx: JobContext):
    """Main agent with advanced interruption handling"""
    await ctx.connect()
    logger.info("🚀 Connected to LiveKit room: %s", ctx.room.name)

    # Configure agent with longer responses for testing
    agent = Agent(
        instructions=(
            "You are a helpful AI assistant. "
            "Give detailed, informative responses (4-6 sentences) so users can test "
            "interrupting you at different points. "
            "Be friendly and natural. "
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
        temperature=0.7,
    )
    logger.info("✅ LLM: Google Gemini 2.0 Flash")

    # Use Cartesia if available, fallback to Google TTS
    if os.getenv("CARTESIA_API_KEY"):
        text_to_speech = cartesia.TTS()
        logger.info("✅ TTS: Cartesia (streaming)")
    else:
        text_to_speech = google.TTS()
        logger.info("✅ TTS: Google Cloud TTS")

    # Use advanced session
    session = AdvancedInterruptionSession(
        vad=voice_activity_detector,
        stt=speech_to_text,
        llm=language_model,
        tts=text_to_speech,
    )

    logger.info("🎬 Starting advanced interruption agent...")
    await session.start(agent=agent, room=ctx.room)

    logger.info("👋 Generating greeting...")
    await session.generate_reply(
        instructions=(
            "Greet the user warmly and explain you're demonstrating advanced interruption handling. "
            "Tell them to try interrupting you while you speak - "
            "soft words like 'yeah' or 'okay' will be ignored, "
            "but 'wait' or 'hold on' will stop you. "
            "Also mention that the system analyzes timing and urgency."
        )
    )

    logger.info("✅ Agent LIVE with ADVANCED interruption detection!")
    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 TESTING GUIDE - Advanced Interruption Detection")
    logger.info("=" * 80)
    logger.info("")
    logger.info("🕐 TIMING AWARENESS:")
    logger.info("   • First 1 second:  Hard to interrupt (agent just started)")
    logger.info("   • 1-3 seconds:     Conservative (soft acks ignored)")
    logger.info("   • 3-8 seconds:     Normal sensitivity")
    logger.info("   • 8+ seconds:      More lenient (agent talking too long)")
    logger.info("")
    logger.info("📢 WHILE AGENT IS SPEAKING:")
    logger.info("")
    logger.info("   ✨ SOFT ACKS (will be ignored):")
    logger.info("      'yeah', 'okay', 'mhmm', 'uh-huh', 'I see', 'right'")
    logger.info("")
    logger.info("   🛑 HARD INTERRUPTS (will stop agent):")
    logger.info("      'wait', 'hold on', 'stop', 'but', 'actually'")
    logger.info("      Say these LOUDLY or SHARPLY for even higher chance!")
    logger.info("")
    logger.info("   ⏸️  POLITE INTERRUPTS (will stop agent):")
    logger.info("      'can I ask', 'quick question', 'before you continue'")
    logger.info("")
    logger.info("🎤 URGENCY DETECTION:")
    logger.info("   • Louder voice = higher urgency score")
    logger.info("   • Sharper tone = higher urgency score")
    logger.info("   • Quick interjection = higher urgency score")
    logger.info("")
    logger.info("📊 Watch the terminal for detailed scoring:")
    logger.info("   • Semantic score (pattern matching)")
    logger.info("   • Timing score (based on agent speech duration)")
    logger.info("   • Urgency score (from audio features)")
    logger.info("   • Final decision with confidence level")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")
    logger.info("💡 PRO TIP: Ask the agent to explain something complex, then try")
    logger.info("            interrupting at different points to see how timing affects decisions!")
    logger.info("=" * 80)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🤖 Advanced Smart Interruption Agent")
    logger.info("=" * 80)
    logger.info("✨ Features:")
    logger.info("  • Timing-aware interruption detection")
    logger.info("  • Audio urgency analysis (volume, pitch, duration)")
    logger.info("  • Confidence scoring (not just binary)")
    logger.info("  • Adaptive thresholds based on context")
    logger.info("  • Detailed decision logging")
    logger.info("=" * 80)
    logger.info("💡 Usage: python smart_interruption_agent.py console")
    logger.info("=" * 80)

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))