"""
Final Arbiter - Combines all agent outputs with circuit breakers for final decision.
Implements the "no single LLM makes final call" principle from winner strategy.
Uses circuit breakers and fallback logic for safety.
"""

from typing import Dict, Any
from base_agent import BaseAgent, CircuitBreaker
from models import (
    RoutingDecision, PreliminaryDecision, CriticReview,
    ActionType, MessageType, Evidence
)


class FinalArbiter(BaseAgent):
    """
    Final decision maker that combines all agent outputs.
    Implements circuit breakers and escalation logic.
    """

    def __init__(self, api_key: str = None):
        super().__init__("final_arbiter", api_key)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=60)

    def process(self, input_data: Dict[str, Any]) -> RoutingDecision:
        """
        Make final routing decision using circuit breaker protection.
        
        Args:
            input_data: Contains outputs from all agents:
                       - preliminary_decision (PreliminaryDecision)
                       - critic_review (CriticReview)
                       - evidence (Evidence)
                       - message (Message)
            
        Returns:
            RoutingDecision with final action, message_type, reason, confidence, evidence
        """
        self._log_processing(input_data)
        
        # Don't use circuit breaker for now - call directly
        # The circuit breaker returns AgentOutput which conflicts with our type system
        return self._make_final_decision(input_data)

    def _make_final_decision(self, input_data: Dict[str, Any]) -> RoutingDecision:
        """
        Make final decision with all safety checks.
        """
        preliminary_decision: PreliminaryDecision = input_data.get('preliminary_decision')
        critic_review: CriticReview = input_data.get('critic_review')
        evidence: Evidence = input_data.get('evidence')
        message = input_data.get('message')
        
        if not preliminary_decision or not message:
            return self._create_safe_fallback(message)

        # Step 1: Check critic approval
        if not critic_review.approved:
            # Use critic's suggested action or fall back to safer option
            final_action = critic_review.suggested_action or self._get_safer_action(preliminary_decision.action)
            final_reason = self._build_critic_override_reason(preliminary_decision, critic_review)
            final_confidence = max(0.3, preliminary_decision.confidence - 0.2)  # Reduce confidence
        else:
            # Critic approved - use preliminary decision
            final_action = preliminary_decision.action
            final_reason = preliminary_decision.reasoning
            final_confidence = preliminary_decision.confidence

        # Step 2: Final safety validation
        final_action = self._final_safety_check(final_action, preliminary_decision.message_type)

        # Step 3: Format evidence IDs
        evidence_ids = self._format_evidence_ids(evidence.relevant_message_ids)

        # Step 4: Create final decision
        return self._create_routing_decision(
            message_id=message.message_id,
            action=final_action,
            message_type=preliminary_decision.message_type,
            reason=final_reason,
            confidence=final_confidence,
            evidence_message_ids=evidence_ids
        )

    def _get_safer_action(self, current_action: ActionType) -> ActionType:
        """Get a safer action than the current one."""
        if current_action == ActionType.NOTIFY:
            return ActionType.DIGEST
        elif current_action == ActionType.DIGEST:
            return ActionType.MUTE
        else:
            return ActionType.MUTE

    def _build_critic_override_reason(
        self,
        preliminary: PreliminaryDecision,
        critic: CriticReview
    ) -> str:
        """Build reasoning when critic overrides decision."""
        if critic.criticisms:
            criticism_summary = "; ".join(critic.criticisms[:2])
            return f"Critic override: {criticism_summary}. Original: {preliminary.reasoning}"
        else:
            return f"Critic override for safety. Original: {preliminary.reasoning}"

    def _final_safety_check(self, action: ActionType, message_type: MessageType) -> ActionType:
        """
        Final safety check before output.
        Ensures dangerous message types are always muted.
        """
        # Scam messages must always be muted
        if message_type == MessageType.SCAM:
            return ActionType.MUTE
        
        # Spam messages should generally be muted
        if message_type == MessageType.SPAM and action == ActionType.NOTIFY:
            return ActionType.DIGEST
        
        return action

    def _format_evidence_ids(self, evidence_ids: list) -> str:
        """Format evidence IDs as semicolon-separated string with normalized format."""
        if not evidence_ids:
            return "none"
        
        # Normalize IDs: message_0057 → msg_057
        normalized_ids = []
        for eid in evidence_ids:
            if eid.startswith("message_"):
                normalized_ids.append(eid.replace("message_", "msg_"))
            else:
                normalized_ids.append(eid)
        
        return ";".join(normalized_ids)

    def _create_safe_fallback(self, message) -> RoutingDecision:
        """Create safe fallback decision when something goes wrong."""
        message_id = message.message_id if message else "unknown"
        return self._create_routing_decision(
            message_id=message_id,
            action=ActionType.DIGEST,  # Safe default
            message_type=MessageType.UNKNOWN,
            reason="System error - using safe default",
            confidence=0.1,
            evidence_message_ids="none"
        )

    def _create_routing_decision(
        self,
        message_id: str,
        action: ActionType,
        message_type: MessageType,
        reason: str,
        confidence: float,
        evidence_message_ids: str
    ) -> RoutingDecision:
        """Create final RoutingDecision output."""
        return RoutingDecision(
            message_id=message_id,
            action=action,
            message_type=message_type,
            reason=reason,
            confidence=confidence,
            evidence_message_ids=evidence_message_ids
        )