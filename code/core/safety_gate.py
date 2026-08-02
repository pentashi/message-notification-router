"""
Deterministic Safety Gate — Pre-model guardrail (checked BEFORE any LLM call).
Enforces safety rules that no LLM can override.
Pure Python rules — no LLM dependency for critical safety decisions.
"""

from typing import Tuple
import re

# Injection patterns — shape-based regex catches reworded variants
_INJECTION_PATTERNS = [
    r"routing\s+override",
    r"system\s+note\s+for",
    r"internal[^\w].*metadata",
    r"assistant\s+instruction",
    r"action\s*=\s*\w+",
    r"mark\s+as\s+\w+",
    r"classify\s+as\s+\w+",
    r"ignore\s+previous\s+instructions",
    r"admin\s+override",
    r"escalate\s+priority",
]

# Scam patterns — OTP/account-threat combos
_SCAM_PATTERNS = [
    r"\botp\b",
    r"one\s+time\s+password",
    r"verification\s+code",
    r"account\s+block",
    r"profile\s+block",
    r"temporary\s+block",
    r"guaranteed\s+profit",
    r"urgent\s+transfer",
    r"immediate\s+payment",
    r"password.*confirm|confirm.*password",
    r"security\s+alert|support\s+alert",
    r"verify\s+now|verify\s+immediately|urgent\s+verification",
    r"account.*expire|access.*expire",
    r"click.*link|tap.*below.*link",
    r"payment\s+details|bank\s+details",
]

# Chain-forward spam patterns
_SPAM_CHAIN_PATTERNS = [
    r"forward.*blessing|share.*blessing",
    r"send\s+(this\s+to\s+)?(ten|10)\s+people",
    r"fwd\s+as\s+received",
    r"forwarding\s+because",
    r"good\s+luck.*forward|forward.*good\s+luck",
]

_URGENCY_INDICATORS = ["immediately", "now", "urgent", "hurry", "expire soon", "2 hours"]
_VERIFICATION_INDICATORS = ["verify", "confirm", "otp", "password", "login"]


class SafetyGate:
    """
    Deterministic pre-model safety gate.
    Call `early_gate()` BEFORE any LLM agent to short-circuit on dangerous messages.
    Call `apply_safety_gate()` post-model to enforce conservative floor on final decision.
    """

    INJECTION_PATTERNS = _INJECTION_PATTERNS
    SCAM_PATTERNS = _SCAM_PATTERNS
    SPAM_PATTERNS = _SPAM_CHAIN_PATTERNS

    @staticmethod
    def early_gate(message_text: str, forwarded_count: int = 0) -> Tuple[bool, str, str, float, str]:
        """
        Pre-model safety check. Returns immediately on dangerous messages.

        Returns:
            (blocked, action, message_type, confidence, reason)
            blocked=True means caller should skip LLM and use returned values directly.
        """
        text = message_text or ""
        text_lower = text.lower()

        # 1. Prompt injection — highest priority
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True, "mute", "scam", 0.98, f"prompt injection blocked: {pattern}"

        # 2. Scam patterns
        for pattern in _SCAM_PATTERNS:
            if re.search(pattern, text_lower):
                if SafetyGate._is_urgency_with_verification(text_lower):
                    return True, "mute", "scam", 0.95, f"scam pattern with urgency/verification: {pattern}"

        # 3. Chain-forward spam (forwarded_count >= 5 AND chain language)
        if forwarded_count >= 5:
            for pattern in _SPAM_CHAIN_PATTERNS:
                if re.search(pattern, text_lower):
                    return True, "mute", "spam", 0.90, f"chain-forward spam: forwarded {forwarded_count}x, pattern '{pattern}'"

        return False, "", "", 0.0, ""

    @staticmethod
    def check_prompt_injection(text: str) -> Tuple[bool, str]:
        """Check for prompt injection attempts."""
        if not text:
            return False, ""
        text_lower = text.lower()
        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, text_lower):
                return True, pattern
        return False, ""

    @staticmethod
    def check_scam_patterns(text: str) -> Tuple[bool, str]:
        """Check for known scam patterns."""
        if not text:
            return False, ""
        text_lower = text.lower()
        for pattern in _SCAM_PATTERNS:
            if re.search(pattern, text_lower):
                return True, pattern
        return False, ""

    @staticmethod
    def apply_safety_gate(
        message_text: str,
        action: str,
        message_type: str,
        confidence: float,
        reason: str
    ) -> Tuple[str, str, float, str]:
        """
        Post-model safety floor — can only make decisions MORE conservative, never less safe.
        Returns (action, message_type, confidence, reason).
        """
        # Check 1: Prompt injection — hard block
        is_injection, pattern = SafetyGate.check_prompt_injection(message_text)
        if is_injection:
            return "mute", "scam", 0.98, f"prompt injection blocked: {pattern}"

        # Check 2: Scam patterns — hard block
        is_scam, pattern = SafetyGate.check_scam_patterns(message_text)
        if is_scam:
            return "mute", "scam", 0.95, f"scam pattern detected: {pattern}"

        # Check 3: Low-confidence notify → downgrade to digest
        if action == "notify" and confidence < 0.7:
            return "digest", message_type, confidence, f"downgraded from notify: confidence {confidence:.2f} below threshold"

        # Check 4: Normalize type via schema
        from core.schema import normalize_message_type
        normalized_type = normalize_message_type(message_type)

        return action, normalized_type, confidence, reason

    @staticmethod
    def _is_urgency_with_verification(text: str) -> bool:
        """Check if message combines urgency with verification requests."""
        has_urgency = any(ind in text for ind in _URGENCY_INDICATORS)
        has_verification = any(ind in text for ind in _VERIFICATION_INDICATORS)
        return has_urgency and has_verification

    @staticmethod
    def validate_context_integrity(message: dict) -> Tuple[bool, str]:
        """Validate that message context hasn't been tampered with."""
        if not message.get("message_id"):
            return False, "Missing message_id"
        if not message.get("message_text") and message.get("media_type") == "":
            return False, "Empty message with no media"
        if message.get("message_text") and len(message["message_text"]) > 5000:
            return False, "Suspiciously long message text"
        return True, "Valid"