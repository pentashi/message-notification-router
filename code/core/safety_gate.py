"""
Deterministic Safety Gate - Post-Model Validation
Enforces safety rules that LLM cannot override per Shloka's guardrails guidance.
Pure Python rules - no LLM dependency for critical safety decisions.
"""

from typing import Tuple
import re

class SafetyGate:
    """
    Deterministic safety validation that runs AFTER model decisions.
    Cannot be overridden by prompt injection or model hallucination.
    """
    
    # Known prompt injection patterns
    INJECTION_PATTERNS = [
        "routing override",
        "system note for the notification router",
        "ignore previous instructions",
        "mark as urgent",
        "set action=",
        "admin override",
        "escalate priority"
    ]
    
    # Known scam patterns requiring hard block
    SCAM_PATTERNS = [
        "otp",
        "one time password",
        "verification code",
        "account block",
        "profile block",
        "temporary block",
        "guaranteed profit",
        "urgent transfer",
        "immediate payment"
    ]
    
    @staticmethod
    def check_prompt_injection(text: str) -> Tuple[bool, str]:
        """
        Check for prompt injection attempts.
        Returns (is_injection, pattern_found).
        """
        if not text:
            return False, ""
        
        text_lower = text.lower()
        for pattern in SafetyGate.INJECTION_PATTERNS:
            if pattern in text_lower:
                return True, pattern
        
        return False, ""
    
    @staticmethod
    def check_scam_patterns(text: str) -> Tuple[bool, str]:
        """
        Check for known scam patterns.
        Returns (is_scam, pattern_found).
        """
        if not text:
            return False, ""
        
        text_lower = text.lower()
        for pattern in SafetyGate.SCAM_PATTERNS:
            if pattern in text_lower:
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
        Apply deterministic safety rules to model decision.
        Can only make decision MORE conservative, never less safe.
        
        Returns (action, message_type, confidence, reason).
        """
        # Check 1: Prompt injection - hard block
        is_injection, pattern = SafetyGate.check_prompt_injection(message_text)
        if is_injection:
            return "mute", "scam", 0.98, f"prompt injection blocked: {pattern}"
        
        # Check 2: Scam patterns - hard block
        is_scam, pattern = SafetyGate.check_scam_patterns(message_text)
        if is_scam:
            return "mute", "scam", 0.98, f"scam pattern detected: {pattern}"
        
        # Check 3: If action is notify, require high confidence and valid reason
        if action == "notify":
            if confidence < 0.7:
                # Downgrade to digest for low-confidence notify
                return "digest", message_type, confidence, f"downgraded from notify: confidence {confidence} below threshold"
        
        # Check 4: Normalize types via schema
        from core.schema import normalize_message_type
        normalized_type = normalize_message_type(message_type)
        
        return action, normalized_type, confidence, reason
    
    @staticmethod
    def validate_context_integrity(message: dict) -> Tuple[bool, str]:
        """
        Validate that message context hasn't been tampered with.
        Returns (is_valid, error_message).
        """
        # Check for missing critical fields
        if not message.get("message_id"):
            return False, "Missing message_id"
        
        if not message.get("message_text") and message.get("media_type") == "":
            return False, "Empty message with no media"
        
        # Check for suspiciously long messages (potential injection)
        if message.get("message_text") and len(message["message_text"]) > 5000:
            return False, "Suspiciously long message text"
        
        return True, "Valid"