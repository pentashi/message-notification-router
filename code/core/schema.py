"""
Central Schema Validation - Single Source of Truth
Ensures zero enum drift and validates all inputs/outputs per Shloka's guardrails guidance.
"""

from typing import List, Optional
from enum import Enum

# Centralized allowed values - prevents schema drift
ALLOWED_ACTIONS = ["notify", "digest", "mute"]
# Exact 11-type set from problem_statement.md — no additions, no omissions
ALLOWED_TYPES = [
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
]

# Type mapping for normalization — maps invented/legacy names to official set
TYPE_MAP = {
    "operational": "urgent",
    "informational": "business_update",
    "business": "business_update",
    "question": "unknown",
    "reminder": "unknown",
}

# Confidence ladder — single source of truth, referenced everywhere
CONFIDENCE_TIERS = {
    "injection": 0.98,
    "scam": 0.95,
    "spam": 0.90,
    "urgent": 0.88,
    "strong_match": 0.82,
    "moderate": 0.75,
    "weak": 0.65,
    "voice_fallback": 0.55,
}

def validate_action(action: str) -> bool:
    """Validate action is in allowed set."""
    return action.lower() in ALLOWED_ACTIONS

def validate_message_type(msg_type: str) -> bool:
    """Validate message type is in allowed set."""
    return msg_type.lower() in ALLOWED_TYPES

def normalize_message_type(msg_type: str) -> str:
    """Normalize message type to allowed enum value."""
    if isinstance(msg_type, str):
        return TYPE_MAP.get(msg_type.lower(), msg_type.lower())
    return msg_type

def validate_confidence(confidence: float) -> bool:
    """Validate confidence is in valid range."""
    return 0.0 <= confidence <= 1.0

def validate_evidence_format(evidence: str) -> bool:
    """Validate evidence format is semicolon-separated or 'none'."""
    if not evidence or evidence.lower() == "none":
        return True
    for part in evidence.split(";"):
        if not part.strip():
            return False
    return True

def validate_routing_decision(decision: dict) -> tuple[bool, List[str]]:
    """
    Complete validation of routing decision against schema.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    
    # Validate required fields
    required_fields = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    for field in required_fields:
        if field not in decision:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return False, errors
    
    # Validate action
    if not validate_action(decision["action"]):
        errors.append(f"Invalid action: {decision['action']}")
    
    # Validate and normalize message type
    if not validate_message_type(decision["message_type"]):
        errors.append(f"Invalid message_type: {decision['message_type']}")
    
    # Validate confidence
    if not validate_confidence(decision["confidence"]):
        errors.append(f"Invalid confidence: {decision['confidence']}")
    
    # Validate evidence format
    if not validate_evidence_format(decision["evidence_message_ids"]):
        errors.append(f"Invalid evidence format: {decision['evidence_message_ids']}")
    
    return len(errors) == 0, errors