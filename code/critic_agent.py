"""
Adversarial Critic Agent - Challenges and validates preliminary decisions.
Key component of winner strategy: ensures no single LLM makes final call.
Uses adversarial testing to catch potential errors before final output.
"""

from typing import Dict, Any, List
from base_agent import BaseAgent
from models import CriticReview, PreliminaryDecision, ActionType, MessageType
import google.genai as genai


class CriticAgent(BaseAgent):
    """
    Adversarial critic that challenges preliminary decisions.
    Looks for inconsistencies, safety issues, or weak reasoning.
    """

    def __init__(self, api_key: str = None):
        super().__init__("critic_agent", api_key)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set for critic agent")
        self.client = genai.Client(api_key=self.api_key)

    def process(self, input_data: Dict[str, Any]) -> CriticReview:
        """
        Critique the preliminary decision and suggest improvements.
        
        Args:
            input_data: Contains 'preliminary_decision' (PreliminaryDecision),
                       'message' (Message), 'risk_assessment' (RiskAssessment)
            
        Returns:
            CriticReview with approval status and criticisms
        """
        self._log_processing(input_data)
        
        preliminary_decision: PreliminaryDecision = input_data.get('preliminary_decision')
        message = input_data.get('message')
        risk_assessment = input_data.get('risk_assessment')
        
        if not preliminary_decision or not message:
            return self._create_review(
                approved=False,
                criticisms=["Invalid input data"],
                suggested_action=ActionType.DIGEST,
                confidence_in_critique=0.0
            )

        # Step 1: Deterministic safety checks
        safety_criticisms = self._safety_checks(preliminary_decision, risk_assessment)
        if safety_criticisms:
            return self._create_review(
                approved=False,
                criticisms=safety_criticisms,
                suggested_action=ActionType.MUTE,
                confidence_in_critique=0.95
            )

        # Step 2: Consistency checks
        consistency_criticisms = self._consistency_checks(preliminary_decision, message)
        
        # Step 3: LLM-based adversarial review
        llm_criticisms = self._llm_adversarial_review(preliminary_decision, message)
        
        # Combine all criticisms
        all_criticisms = safety_criticisms + consistency_criticisms + llm_criticisms
        
        # Determine if decision should be approved
        if len(all_criticisms) == 0:
            return self._create_review(
                approved=True,
                criticisms=[],
                suggested_action=None,
                confidence_in_critique=0.90
            )
        
        # If there are criticisms, determine if they're severe enough to block
        severe_criticisms = [c for c in all_criticisms if "severe" in c.lower() or "safety" in c.lower()]
        
        if severe_criticisms:
            # Suggest safer action
            suggested_action = self._suggest_safer_action(preliminary_decision.action)
            return self._create_review(
                approved=False,
                criticisms=all_criticisms,
                suggested_action=suggested_action,
                confidence_in_critique=0.85
            )
        else:
            # Minor criticisms, but still approve
            return self._create_review(
                approved=True,
                criticisms=all_criticisms,
                suggested_action=None,
                confidence_in_critique=0.75
            )

    def _safety_checks(
        self,
        decision: PreliminaryDecision,
        risk_assessment
    ) -> List[str]:
        """Deterministic safety checks."""
        criticisms = []
        
        # If risk assessment said to block, but decision is notify/digest
        if risk_assessment and risk_assessment.should_block:
            if decision.action in [ActionType.NOTIFY, ActionType.DIGEST]:
                criticisms.append("SEVERE: Risk assessment recommended blocking, but decision allows notification")
        
        # Scam messages should always be muted
        if decision.message_type == MessageType.SCAM and decision.action != ActionType.MUTE:
            criticisms.append("SEVERE: Scam message type should always be muted")
        
        # Spam messages should generally be muted
        if decision.message_type == MessageType.SPAM and decision.action == ActionType.NOTIFY:
            criticisms.append("SEVERE: Spam message type should not be notified")
        
        return criticisms

    def _consistency_checks(self, decision: PreliminaryDecision, message) -> List[str]:
        """Check for internal consistency in the decision."""
        criticisms = []
        
        # Low confidence with notify action is suspicious
        if decision.action == ActionType.NOTIFY and decision.confidence < 0.5:
            criticisms.append("Low confidence for notify action - consider digest instead")
        
        # High confidence without supporting evidence is suspicious
        if decision.confidence > 0.8 and not decision.supporting_evidence:
            criticisms.append("High confidence without supporting evidence - verify reasoning")
        
        # Urgent type with digest action needs strong reasoning
        if decision.message_type == MessageType.URGENT and decision.action == ActionType.DIGEST:
            if "urgent" not in decision.reasoning.lower():
                criticisms.append("Urgent message type with digest action - check if this is appropriate")
        
        return criticisms

    def _llm_adversarial_review(
        self,
        decision: PreliminaryDecision,
        message
    ) -> List[str]:
        """Use LLM to adversarially review the decision."""
        
        prompt = self._build_adversarial_prompt(decision, message)
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            
            critique_text = response.text
            
            if "APPROVED" in critique_text.upper():
                return []
            else:
                # Parse criticisms from response
                return self._parse_criticisms(critique_text)
                
        except Exception as e:
            print(f"Error in LLM adversarial review: {e}")
            return []  # Fail silently - don't block on LLM errors

    def _build_adversarial_prompt(self, decision: PreliminaryDecision, message) -> str:
        """Build prompt for adversarial review."""
        return f"""
Review this routing decision for potential flaws:

Message: "{message.message_text if hasattr(message, 'message_text') else 'N/A'}"
Proposed Action: {decision.action.value}
Message Type: {decision.message_type.value}
Reasoning: {decision.reasoning}
Confidence: {decision.confidence}

Consider:
- Is the action consistent with the message type?
- Is the reasoning sound and evidence-based?
- Are there safety concerns being overlooked?
- Is the confidence level appropriate?

If the decision is sound, respond with "APPROVED".
If there are issues, list them concisely (max 3 items).
"""

    def _parse_criticisms(self, critique_text: str) -> List[str]:
        """Parse criticisms from LLM response."""
        criticisms = []
        lines = critique_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('If') and not line.startswith('The'):
                # Remove bullet points and numbers
                cleaned = line.lstrip('-*123456789.')
                if cleaned:
                    criticisms.append(cleaned)
        
        return criticisms[:3]  # Limit to 3 criticisms

    def _suggest_safer_action(self, current_action: ActionType) -> ActionType:
        """Suggest a safer action based on current action."""
        if current_action == ActionType.NOTIFY:
            return ActionType.DIGEST
        elif current_action == ActionType.DIGEST:
            return ActionType.MUTE
        else:
            return ActionType.MUTE  # Already safest

    def _create_review(
        self,
        approved: bool,
        criticisms: List[str],
        suggested_action: ActionType,
        confidence_in_critique: float
    ) -> CriticReview:
        """Create CriticReview output."""
        return CriticReview(
            approved=approved,
            criticisms=criticisms,
            suggested_action=suggested_action,
            confidence_in_critique=confidence_in_critique
        )