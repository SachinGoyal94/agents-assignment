"""
Intelligent Interruption Handler for Voice Agents
Location: livekit-agents/livekit/agents/voice/interruption_handler.py

Implements context-aware interruption logic without modifying VAD/STT core
"""

import asyncio
import re
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent conversation states"""
    IDLE = "idle"
    LISTENING = "listening"
    SPEAKING = "speaking"
    PROCESSING = "processing"


class InterruptionIntent(Enum):
    """Classification of user interruption intent"""
    IGNORE = "ignore"  # Soft acknowledgments like "yeah", "mhmm"
    PAUSE = "pause"  # User wants to interrupt but politely
    HARD_STOP = "hard_stop"  # Strong interruption signals


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
    Core interruption logic layer that sits between VAD/STT and agent control.

    Key Design Principles:
    1. Non-blocking: All operations are async to minimize latency
    2. Configurable: Easy to tune thresholds and patterns
    3. Stateful: Tracks conversation context for smart decisions
    4. Layered: Doesn't modify existing VAD/STT implementations
    """

    # Soft acknowledgment patterns (ignore these)
    SOFT_ACKNOWLEDGMENTS = {
        r'\b(yeah|yep|yes|uh-huh|mm-hmm|mhmm|okay|ok|right|sure|got it)\b',
        r'\b(i see|makes sense|understood|alright|cool|nice)\b',
        r'\b(go on|continue|keep going)\b',
    }

    # Hard interruption patterns (stop immediately)
    HARD_INTERRUPTIONS = {
        r'\b(wait|hold on|stop|hang on|but|however|actually)\b',
        r'\b(excuse me|sorry|one sec|one second|pause)\b',
        r'\bno\b(?!\s+problem)',  # "no" but not "no problem"
        r'\b(what|huh|pardon)\b',  # confusion signals
    }

    # Pause/polite interruption patterns
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

        # Compile regex patterns for speed
        self.soft_patterns = [re.compile(p, re.IGNORECASE) for p in self.SOFT_ACKNOWLEDGMENTS]
        self.hard_patterns = [re.compile(p, re.IGNORECASE) for p in self.HARD_INTERRUPTIONS]
        self.pause_patterns = [re.compile(p, re.IGNORECASE) for p in self.PAUSE_PATTERNS]

        logger.info(f"✅ Interruption handler initialized (min_duration={self.min_speech_duration}s, timeout={self.stt_timeout}s)")

    def _classify_intent(self, text: str) -> InterruptionIntent:
        """
        Fast intent classification using regex patterns.
        Priority: HARD_STOP > PAUSE > IGNORE
        """
        if not text or len(text.strip()) < 2:
            return InterruptionIntent.IGNORE

        text_lower = text.lower().strip()

        # Check hard interruptions first (highest priority)
        for pattern in self.hard_patterns:
            if pattern.search(text_lower):
                logger.debug(f"🛑 Hard interruption detected: '{text}'")
                return InterruptionIntent.HARD_STOP

        # Check pause patterns
        for pattern in self.pause_patterns:
            if pattern.search(text_lower):
                logger.debug(f"⏸️  Pause interruption detected: '{text}'")
                return InterruptionIntent.PAUSE

        # Check soft acknowledgments (ignore these)
        for pattern in self.soft_patterns:
            if pattern.search(text_lower):
                logger.debug(f"✨ Soft acknowledgment detected: '{text}'")
                return InterruptionIntent.IGNORE

        # Default: if text is short (< 5 words), likely acknowledgment
        word_count = len(text_lower.split())
        if word_count <= 3:
            logger.debug(f"✨ Short input treated as acknowledgment: '{text}'")
            return InterruptionIntent.IGNORE

        # Longer input during speech = likely interruption
        logger.debug(f"⏸️  Longer input treated as interruption: '{text}'")
        return InterruptionIntent.PAUSE

    async def handle_vad_event(self, vad_confidence: float) -> dict:
        """
        Called when VAD detects voice activity.
        Returns decision on whether to wait for STT or act immediately.
        """
        current_time = asyncio.get_event_loop().time()

        # If agent is idle or listening, no interruption possible
        if self.current_state in [AgentState.IDLE, AgentState.LISTENING]:
            return {"action": "ignore", "reason": "Agent not speaking"}

        # Check VAD confidence
        if vad_confidence < self.vad_threshold:
            return {"action": "ignore", "reason": f"VAD confidence too low: {vad_confidence:.2f}"}

        # Calculate agent speech duration
        speech_duration = 0.0
        if self.speech_start_time:
            speech_duration = current_time - self.speech_start_time

        # If agent just started speaking, wait for STT to classify
        if speech_duration < self.min_speech_duration:
            return {
                "action": "wait_for_stt",
                "reason": f"Agent speaking for {speech_duration:.1f}s, waiting for semantic analysis",
                "timeout": self.stt_timeout
            }

        # Agent has been speaking for a while, wait for STT to determine intent
        return {
            "action": "wait_for_stt",
            "reason": "Waiting for semantic classification",
            "timeout": self.stt_timeout
        }

    async def handle_stt_result(self, text: str, vad_confidence: float) -> dict:
        """
        Called when STT provides transcription.
        Makes final decision on interruption.
        """
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
        else:  # PAUSE
            decision.update({
                "action": "stop",
                "reason": f"Polite interruption: '{text}'"
            })

        logger.info(f"📊 Decision: {decision['action'].upper()} - {decision['reason']}")
        return decision

    def update_agent_state(self, new_state: AgentState):
        """Update current agent state for context tracking"""
        if self.current_state != new_state:
            logger.debug(f"🔄 State change: {self.current_state.value} → {new_state.value}")
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
    """
    High-level manager that coordinates VAD, STT, and interruption logic.
    This is what you'll integrate into your agent pipeline.
    """

    def __init__(self, handler: SemanticInterruptionHandler):
        self.handler = handler

    async def on_vad_detected(self, vad_confidence: float, stt_callback: Callable) -> bool:
        """
        Called when VAD detects voice.

        Args:
            vad_confidence: VAD confidence score
            stt_callback: Async function to get STT result

        Returns:
            bool: True if should stop agent immediately, False otherwise
        """
        decision = await self.handler.handle_vad_event(vad_confidence)

        if decision["action"] == "ignore":
            logger.debug(f"[VAD] Ignoring: {decision['reason']}")
            return False

        if decision["action"] == "stop_immediately":
            logger.info(f"[VAD] ⛔ Stopping immediately: {decision['reason']}")
            return True

        # Wait for STT with timeout
        if decision["action"] == "wait_for_stt":
            logger.debug(f"[VAD] ⏳ Waiting for STT: {decision['reason']}")
            timeout = decision.get("timeout", 0.5)

            try:
                # Race between STT and timeout
                stt_text = await asyncio.wait_for(
                    stt_callback(),
                    timeout=timeout
                )

                # Got STT result, make semantic decision
                final_decision = await self.handler.handle_stt_result(
                    stt_text, vad_confidence
                )

                should_stop = final_decision["action"] == "stop"
                action_icon = "🛑" if should_stop else "✅"
                logger.info(f"[STT] {action_icon} {final_decision['reason']}")
                return should_stop

            except asyncio.TimeoutError:
                # STT took too long, make conservative decision
                speech_duration = self.handler.get_speech_duration()
                if speech_duration > 3.0:
                    logger.warning(f"[STT TIMEOUT] ⚠️  Agent spoke {speech_duration:.1f}s, stopping")
                    return True
                else:
                    logger.warning(f"[STT TIMEOUT] ⚠️  Agent spoke {speech_duration:.1f}s, continuing")
                    return False

        return False