"""
Core package - Central validation, safety, and configuration.
Ensures zero schema drift and deterministic safety enforcement per Shloka's guardrails guidance.
"""

from core.schema import (
    ALLOWED_ACTIONS,
    ALLOWED_TYPES,
    TYPE_MAP,
    validate_action,
    validate_message_type,
    normalize_message_type,
    validate_confidence,
    validate_evidence_format,
    validate_routing_decision
)

from core.safety_gate import SafetyGate

from core.validation import DataValidator

__all__ = [
    # Schema
    'ALLOWED_ACTIONS',
    'ALLOWED_TYPES',
    'TYPE_MAP',
    'validate_action',
    'validate_message_type',
    'normalize_message_type',
    'validate_confidence',
    'validate_evidence_format',
    'validate_routing_decision',
    # Safety
    'SafetyGate',
    # Validation
    'DataValidator'
]