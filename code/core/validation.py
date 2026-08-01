"""
Input/Output Validation - Pre and Post Model Checks
Validates data integrity before processing and after decisions per Shloka's reliability guidance.
"""

import pandas as pd
import math
from typing import Tuple, List, Dict, Any
from core.schema import validate_routing_decision, normalize_message_type

class DataValidator:
    """
    Validates input data and output decisions for reliability.
    Ensures system handles edge cases: empty rows, missing fields, duplicates, ambiguous evidence.
    """
    
    @staticmethod
    def validate_message_row(message: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate individual message row before processing.
        Returns (is_valid, error_message).
        """
        # Check required fields
        required_fields = ["message_id", "user_id", "conversation_type", "created_at"]
        for field in required_fields:
            if field not in message or pd.isna(message[field]):
                return False, f"Missing required field: {field}"
        
        # Check for valid message_id format
        message_id = message.get("message_id", "")
        if not message_id or not isinstance(message_id, str):
            return False, f"Invalid message_id: {message_id}"
        
        # Check for empty text without media
        message_text = message.get("message_text", "")
        media_type = message.get("media_type", "")
        
        if not message_text or (isinstance(message_text, float) and math.isnan(message_text)):
            message_text = ""
        
        if not message_text and not media_type:
            return False, "Empty message with no media content"
        
        # Check for suspicious injection attempts
        if message_text and len(message_text) > 5000:
            return False, "Suspiciously long message text (potential injection)"
        
        return True, "Valid"
    
    @staticmethod
    def validate_message_batch(messages: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """
        Validate entire batch of messages.
        Returns (valid_count, list_of_errors).
        """
        valid_count = 0
        errors = []
        seen_ids = set()
        
        for i, message in enumerate(messages):
            # Check for duplicate message_ids
            message_id = message.get("message_id", "")
            if message_id in seen_ids:
                errors.append(f"Row {i}: Duplicate message_id: {message_id}")
                continue
            seen_ids.add(message_id)
            
            # Validate individual row
            is_valid, error = DataValidator.validate_message_row(message)
            if is_valid:
                valid_count += 1
            else:
                errors.append(f"Row {i}: {error}")
        
        return valid_count, errors
    
    @staticmethod
    def validate_evidence_quality(evidence_ids: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that evidence is relevant and high quality.
        Returns (is_valid, reason).
        """
        if not evidence_ids or evidence_ids.lower() == "none":
            return True, "No evidence required"
        
        # Check evidence format
        from core.schema import validate_evidence_format
        if not validate_evidence_format(evidence_ids):
            return False, "Invalid evidence format"
        
        # Check that evidence IDs actually exist in context
        available_ids = context.get("available_message_ids", [])
        if available_ids:
            for evidence_id in evidence_ids.split(";"):
                evidence_id = evidence_id.strip()
                if evidence_id and evidence_id not in available_ids:
                    return False, f"Evidence ID not found in available messages: {evidence_id}"
        
        return True, "Valid evidence"
    
    @staticmethod
    def check_consistency(decision1: Dict[str, Any], decision2: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if two similar decisions are consistent.
        Returns (is_consistent, reason).
        """
        # Similar content should have similar actions
        if decision1.get("action") != decision2.get("action"):
            # Different actions are acceptable if confidence differs significantly
            conf_diff = abs(dec1.get("confidence", 0) - decision2.get("confidence", 0))
            if conf_diff < 0.1:
                return False, f"Inconsistent actions with similar confidence: {decision1.get('action')} vs {decision2.get('action')}"
        
        return True, "Consistent"
    
    @staticmethod
    def detect_anomalies(decisions: List[Dict[str, Any]]) -> List[str]:
        """
        Detect anomalous patterns in decisions.
        Returns list of anomaly descriptions.
        """
        anomalies = []
        
        # Check for unexpected message types
        from core.schema import ALLOWED_TYPES
        for decision in decisions:
            msg_type = decision.get("message_type", "")
            if msg_type not in ALLOWED_TYPES:
                anomalies.append(f"Unknown message type detected: {msg_type}")
        
        # Check for unusual confidence distributions
        confidences = [d.get("confidence", 0) for d in decisions]
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            if avg_conf < 0.3:
                anomalies.append(f"Unusually low average confidence: {avg_conf:.2f}")
            elif avg_conf > 0.95:
                anomalies.append(f"Unusually high average confidence: {avg_conf:.2f}")
        
        # Check for too many "mute" decisions (potential over-blocking)
        mute_count = sum(1 for d in decisions if d.get("action") == "mute")
        if mute_count / len(decisions) > 0.5:
            anomalies.append(f"High mute rate: {mute_count}/{len(decisions)} ({mute_count/len(decisions)*100:.1f}%)")
        
        return anomalies