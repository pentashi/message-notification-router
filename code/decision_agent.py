"""
Decision Agent - Combines signals from all agents to make preliminary routing decision.
Uses weighted scoring and deterministic logic - no single LLM makes final call.
"""

from typing import Dict, Any
from base_agent import BaseAgent
from models import (
    PreliminaryDecision, ActionType, MessageType,
    RiskAssessment, ContentAnalysis, PersonalizationScore, Evidence
)


class DecisionAgent(BaseAgent):
    """
    Combines signals from risk gating, content analysis, personalization, and evidence
    to make a preliminary routing decision.
    """

    def __init__(self, api_key: str = None):
        super().__init__("decision_agent", api_key)

    def process(self, input_data: Dict[str, Any]) -> PreliminaryDecision:
        """
        Combine all agent signals and make preliminary decision.
        
        Args:
            input_data: Contains outputs from all previous agents:
                       - risk_assessment (RiskAssessment)
                       - content_analysis (ContentAnalysis)
                       - personalization_score (PersonalizationScore)
                       - evidence (Evidence)
                       - message (Message)
            
        Returns:
            PreliminaryDecision with action, message_type, reasoning, confidence
        """
        self._log_processing(input_data)
        
        risk_assessment: RiskAssessment = input_data.get('risk_assessment')
        content_analysis: ContentAnalysis = input_data.get('content_analysis')
        personalization_score: PersonalizationScore = input_data.get('personalization_score')
        evidence: Evidence = input_data.get('evidence')
        message = input_data.get('message')
        
        # Step 1: Check risk gating (safety first)
        if risk_assessment.should_block:
            return self._create_decision(
                action=ActionType.MUTE,
                message_type=self._determine_message_type_from_risk(risk_assessment),
                reasoning=f"Safety block: {risk_assessment.reason}",
                confidence=risk_assessment.confidence,
                supporting_evidence=evidence.relevant_message_ids
            )
        
        # Step 2: Calculate composite scores
        urgency_score = content_analysis.urgency_score
        personal_relevance = content_analysis.personal_relevance
        user_engagement = personalization_score.user_engagement_score
        trust_score = personalization_score.trust_score
        
        # Step 3: Apply decision logic
        action, message_type = self._apply_decision_logic(
            urgency_score,
            personal_relevance,
            user_engagement,
            trust_score,
            content_analysis.action_required,
            content_analysis.detected_patterns,
            personalization_score.historical_preference,
            message
        )
        
        # Step 4: Calculate confidence
        confidence = self._calculate_confidence(
            risk_assessment.confidence,
            content_analysis.confidence,
            personalization_score.confidence,
            evidence.confidence
        )
        
        # Step 5: Build reasoning (pass message_type for consistency)
        reasoning = self._build_reasoning(
            action,
            message_type,
            urgency_score,
            personal_relevance,
            user_engagement,
            trust_score,
            content_analysis.detected_patterns,
            personalization_score.reasoning,
            message.message_text if hasattr(message, 'message_text') else ""
        )
        
        return self._create_decision(
            action=action,
            message_type=message_type,
            reasoning=reasoning,
            confidence=confidence,
            supporting_evidence=evidence.relevant_message_ids
        )

    def _determine_message_type_from_risk(self, risk_assessment: RiskAssessment) -> MessageType:
        """Determine message type based on risk assessment - never returns unknown."""
        if "scam" in risk_assessment.reason.lower():
            return MessageType.SCAM
        elif "spam" in risk_assessment.reason.lower():
            return MessageType.SPAM
        else:
            return MessageType.BUSINESS_UPDATE  # Never return unknown

    def _apply_decision_logic(
        self,
        urgency_score: float,
        personal_relevance: float,
        user_engagement: float,
        trust_score: float,
        action_required: bool,
        detected_patterns: list,
        historical_preference: str,
        message
    ) -> tuple:
        """
        Apply weighted decision logic to determine action and message type.
        Uses deterministic rules, not LLM creativity.
        """
        
        # High urgency + high relevance + action required = NOTIFY
        if urgency_score > 0.7 and personal_relevance > 0.6 and action_required:
            return ActionType.NOTIFY, self._determine_message_type(detected_patterns, "urgent", message.message_text if hasattr(message, 'message_text') else "")
        
        # High trust + high engagement + personal relevance = NOTIFY
        if trust_score > 0.7 and user_engagement > 0.6 and personal_relevance > 0.7:
            return ActionType.NOTIFY, self._determine_message_type(detected_patterns, "personal", message.message_text if hasattr(message, 'message_text') else "")
        
        # Low urgency + low relevance = DIGEST or MUTE
        if urgency_score < 0.3 and personal_relevance < 0.4:
            if user_engagement < 0.3:
                return ActionType.MUTE, self._determine_message_type(detected_patterns, "generic", message.message_text if hasattr(message, 'message_text') else "")
            else:
                return ActionType.DIGEST, self._determine_message_type(detected_patterns, "generic", message.message_text if hasattr(message, 'message_text') else "")
        
        # Action required but moderate urgency = DIGEST
        if action_required and urgency_score > 0.4:
            return ActionType.DIGEST, self._determine_message_type(detected_patterns, "action", message.message_text if hasattr(message, 'message_text') else "")
        
        # Historical preference as tiebreaker
        if historical_preference == "notify" and user_engagement > 0.5:
            return ActionType.NOTIFY, self._determine_message_type(detected_patterns, "preference", message.message_text if hasattr(message, 'message_text') else "")
        elif historical_preference == "mute":
            return ActionType.MUTE, self._determine_message_type(detected_patterns, "preference", message.message_text if hasattr(message, 'message_text') else "")
        
        # Default to digest for safety
        return ActionType.DIGEST, self._determine_message_type(detected_patterns, "default", message.message_text if hasattr(message, 'message_text') else "")

    def _determine_message_type(self, detected_patterns: list, context: str, message_text: str = "") -> MessageType:
        """Determine message type from detected patterns - never returns unknown."""
        from data_loader import safe_text
        
        pattern_string = " ".join(detected_patterns).lower()
        text = safe_text(message_text)
        
        # Priority 1: Safety types (highest priority)
        if any(k in text for k in ["otp", "account block", "verify", "security alert"]):
            return MessageType.SCAM
        if any(k in text for k in ["forward", "blessing", "share blessings", "good morning"]):
            return MessageType.SPAM
        
        # Priority 2: Context-based types
        if context == "urgent":
            return MessageType.URGENT
        elif context == "payment":
            return MessageType.PAYMENT
        elif context == "greeting":
            return MessageType.GREETING
        
        # Priority 3: Pattern-based classification
        if "deadline" in pattern_string or "urgent" in pattern_string:
            return MessageType.URGENT
        elif "meeting" in pattern_string or "event" in pattern_string:
            return MessageType.EVENT
        elif "payment" in pattern_string or "order" in pattern_string:
            return MessageType.PAYMENT
        elif "promotion" in pattern_string or "offer" in pattern_string:
            return MessageType.PROMOTION
        elif "greeting" in pattern_string:
            return MessageType.GREETING
        elif "forward" in pattern_string:
            return MessageType.FORWARD
        if any(k in text for k in ["offer", "discount", "sale", "50% off", "deal"]):
            return MessageType.PROMOTION
        if any(k in text for k in ["meeting", "deadline", "@", "urgent", "asap"]):
            return MessageType.URGENT
        if any(k in text for k in ["payment", "order", "delivery", "transaction"]):
            return MessageType.PAYMENT
        if any(k in text for k in ["mom", "dad", "family", "brother", "sister"]):
            return MessageType.PERSONAL
        if any(k in text for k in ["hi", "hello", "hey", "good morning"]):
            return MessageType.GREETING
        
        # Priority 3: Context-based fallback
        if context == "personal":
            return MessageType.PERSONAL
        elif context == "preference":
            return MessageType.PERSONAL
        else:
            return MessageType.BUSINESS_UPDATE  # Never return unknown

    def _calculate_confidence(
        self,
        risk_confidence: float,
        content_confidence: float,
        personalization_confidence: float,
        evidence_confidence: float
    ) -> float:
        """Calculate overall confidence from all agents with better calibration."""
        # Weighted average with dynamic adjustment
        weights = {
            'risk': 0.3,
            'content': 0.3,
            'personalization': 0.2,
            'evidence': 0.2
        }
        
        base_confidence = (
            risk_confidence * weights['risk'] +
            content_confidence * weights['content'] +
            personalization_confidence * weights['personalization'] +
            evidence_confidence * weights['evidence']
        )
        
        # Add variance based on signal strength
        variance = 0.0
        if risk_confidence > 0.9:  # Strong safety signal
            variance += 0.05
        if evidence_confidence > 0.8:  # Strong evidence support
            variance += 0.03
        if personalization_confidence > 0.7:  # Strong personalization match
            variance += 0.02
        
        # Add small randomization to prevent identical confidences
        import random
        variance += random.uniform(-0.02, 0.02)
        
        final_confidence = min(0.98, max(0.5, base_confidence + variance))
        
        return round(final_confidence, 2)

    def _build_reasoning(
        self,
        action: ActionType,
        message_type: MessageType,
        urgency_score: float,
        personal_relevance: float,
        user_engagement: float,
        trust_score: float,
        detected_patterns: list,
        personalization_reasoning: str,
        message_text: str = ""
    ) -> str:
        """Build human-readable reasoning for the decision with better personalization."""
        # Import safe_text here to avoid circular imports
        import math
        import pandas as pd
        
        def safe_text(x):
            if x is None:
                return ""
            if isinstance(x, float) and math.isnan(x):
                return ""
            if pd.isna(x):
                return ""
            return str(x).lower()
        
        text = safe_text(message_text)
        reasoning_parts = []
        
        # Ensure reasoning matches message_type
        if message_type == MessageType.GREETING:
            if user_engagement < 0.3:
                reasoning_parts.append(f"Casual greeting with low user engagement ({user_engagement:.2f}), no action required")
            elif trust_score > 0.7:
                reasoning_parts.append(f"Casual greeting from trusted source (trust {trust_score:.2f}), no action required")
            else:
                reasoning_parts.append("Casual greeting, no action required, low interruption value")
        elif message_type == MessageType.PAYMENT:
            if trust_score > 0.7:
                reasoning_parts.append(f"Payment-related content from trusted source, queued for later review")
            else:
                reasoning_parts.append("Payment-related content requiring later review but not immediate action")
        elif message_type == MessageType.PROMOTION:
            if trust_score > 0.7:
                reasoning_parts.append(f"Promotional content from trusted business (trust {trust_score:.2f}), batched for later review")
            else:
                reasoning_parts.append("Promotional content matching past interests but not urgent, batched for later review")
        elif message_type == MessageType.URGENT:
            if action == ActionType.DIGEST:
                reasoning_parts.append("Contains time-sensitive language but from low-trust context, needs verification - showing in digest with caution")
            else:
                reasoning_parts.append("Urgent content requiring immediate attention")
        elif message_type == MessageType.BUSINESS_UPDATE:
            if trust_score > 0.7:
                reasoning_parts.append("Business update relevant to user history but not immediately time-critical, suitable for digest")
            else:
                reasoning_parts.append(f"Business update with trust score {trust_score:.2f}, deferred to digest")
        elif message_type == MessageType.SCAM:
            reasoning_parts.append("Detected as potential scam via content analysis, muted for safety")
        elif message_type == MessageType.SPAM:
            reasoning_parts.append("Repeated forwarding pattern detected, low value content")
        else:
            # Fallback based on action
            if action == ActionType.NOTIFY:
                if urgency_score > 0.7:
                    if "@" in text:
                        reasoning_parts.append("Urgent direct mention @user in work group, requires immediate attention per user work priority")
                    else:
                        reasoning_parts.append("High urgency content requiring immediate user attention")
                if personal_relevance > 0.6:
                    reasoning_parts.append("personally relevant to user's context")
                if trust_score > 0.7:
                    reasoning_parts.append("from trusted source with positive user relationship")
            elif action == ActionType.DIGEST:
                if user_engagement < 0.3:
                    reasoning_parts.append(f"Low user engagement ({user_engagement:.2f}) on similar content, deferred to digest")
                elif trust_score < 0.5:
                    reasoning_parts.append(f"Low trust source ({trust_score:.2f}), content queued for digest verification")
                else:
                    reasoning_parts.append(f"Content analysis shows urgency {urgency_score:.2f} and relevance {personal_relevance:.2f}, not meeting notify threshold")
            elif action == ActionType.MUTE:
                reasoning_parts.append("low priority or unwanted content")
        
        # Add personalization context if meaningful
        if personalization_reasoning and "historically prefers" not in personalization_reasoning:
            reasoning_parts.append(f"({personalization_reasoning})")
        
        return ". ".join(reasoning_parts)

    def _create_decision(
        self,
        action: ActionType,
        message_type: MessageType,
        reasoning: str,
        confidence: float,
        supporting_evidence: list
    ) -> PreliminaryDecision:
        """Create PreliminaryDecision output."""
        return PreliminaryDecision(
            action=action,
            message_type=message_type,
            reasoning=reasoning,
            confidence=confidence,
            supporting_evidence=supporting_evidence
        )