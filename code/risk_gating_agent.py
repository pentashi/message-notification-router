"""
Risk Gating Agent - First line of defense for safety.
Uses deterministic rules to catch clear scams, spam, and high-risk patterns.
Zero hallucination - only uses provided data and rule-based patterns.
"""

import re
import math
import pandas as pd
from typing import Dict, Any
from base_agent import BaseAgent
from models import RiskAssessment, RiskLevel, Message, BusinessAccount


def safe_text(x):
    """Safe text conversion that handles NaN and None values."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if pd.isna(x):
        return ""
    return str(x).lower()


class RiskGatingAgent(BaseAgent):
    """
    Pre-retrieval risk gating agent.
    Uses deterministic rules to identify high-risk messages before LLM processing.
    """

    # High-risk scam patterns (deterministic, no hallucination)
    SCAM_PATTERNS = [
        r'otp|one time password|verification code',
        r'account.*block|profile.*block|temporary.*block',
        r'password.*confirm|confirm.*password',
        r'security alert|support alert',
        r'verify now|verify immediately|urgent verification',
        r'account.*expire|access.*expire',
        r'login\.in|account-login\.in',  # Suspicious domains
        r'payment.*details|bank.*details',
        r'click.*link|tap.*below.*link',
    ]

    # Spam patterns
    SPAM_PATTERNS = [
        r'forward.*blessing|share.*blessing',
        r'good morning.*stay positive.*share blessings',
        r'forward.*good luck|send.*ten people',
        r'fwd as received|forwarding because',
        r'drink warm water|avoid cold food',
    ]

    def __init__(self, api_key: str = None):
        super().__init__("risk_gating_agent", api_key)

    def process(self, input_data: Dict[str, Any]) -> RiskAssessment:
        """
        Analyze message for risk patterns.
        
        Args:
            input_data: Contains 'message' (Message object) and optionally 'business' (BusinessAccount)
            
        Returns:
            RiskAssessment with risk level and blocking decision
        """
        self._log_processing(input_data)
        
        message: Message = input_data.get('message')
        business: BusinessAccount = input_data.get('business')
        
        if not message:
            return self._create_risk_assessment(
                RiskLevel.CRITICAL,
                "Invalid message data",
                True,
                0.0
            )

        # Check for scam patterns
        scam_risk = self._check_scam_patterns(message)
        if scam_risk['is_scam']:
            return self._create_risk_assessment(
                RiskLevel.CRITICAL,
                scam_risk['reason'],
                True,
                0.95
            )

        # Check for spam patterns
        spam_risk = self._check_spam_patterns(message)
        if spam_risk['is_spam']:
            return self._create_risk_assessment(
                RiskLevel.HIGH_RISK,
                spam_risk['reason'],
                True,
                0.85
            )

        # Check business verification
        if business and not business.verified:
            return self._create_risk_assessment(
                RiskLevel.MEDIUM_RISK,
                "Unverified business sender",
                False,  # Don't block, but flag as medium risk
                0.70
            )

        # Check forwarded count (high forwarding = potential spam)
        if message.forwarded_count > 5:
            return self._create_risk_assessment(
                RiskLevel.MEDIUM_RISK,
                f"High forward count ({message.forwarded_count})",
                False,
                0.65
            )

        # Check suspicious domain for business
        if business:
            domain_risk = self._check_domain_risk(business)
            if domain_risk['is_suspicious']:
                return self._create_risk_assessment(
                    RiskLevel.HIGH_RISK,
                    domain_risk['reason'],
                    True,
                    0.80
                )

        # Message appears safe
        return self._create_risk_assessment(
            RiskLevel.SAFE,
            "No high-risk patterns detected",
            False,
            0.90
        )

    def _check_scam_patterns(self, message: Message) -> Dict[str, Any]:
        """Check for scam patterns using deterministic rules."""
        text_lower = safe_text(message.message_text)
        
        for pattern in self.SCAM_PATTERNS:
            if re.search(pattern, text_lower):
                # Additional context checks
                if self._is_urgency_with_verification_request(text_lower):
                    return {
                        'is_scam': True,
                        'reason': f"Scam pattern detected: '{pattern}' with urgency and verification request"
                    }
        
        return {'is_scam': False, 'reason': ''}

    def _check_spam_patterns(self, message: Message) -> Dict[str, Any]:
        """Check for spam patterns using deterministic rules."""
        text_lower = safe_text(message.message_text)
        
        for pattern in self.SPAM_PATTERNS:
            if re.search(pattern, text_lower):
                return {
                    'is_spam': True,
                    'reason': f"Spam pattern detected: '{pattern}'"
                }
        
        return {'is_spam': False, 'reason': ''}

    def _check_domain_risk(self, business: BusinessAccount) -> Dict[str, Any]:
        """Check if business domain is suspicious."""
        # Domain mismatch
        if business.official_domain != business.domain_used_by_sender:
            # Check if sender domain looks suspicious
            if 'rewards' in business.domain_used_by_sender.lower():
                return {
                    'is_suspicious': True,
                    'reason': f"Domain mismatch: official={business.official_domain}, sender={business.domain_used_by_sender}"
                }
        
        # New domain (low age)
        if business.domain_used_by_sender_age_days < 30:
            return {
                'is_suspicious': True,
                'reason': f"Suspicious: sender domain age only {business.domain_used_by_sender_age_days} days"
            }
        
        return {'is_suspicious': False, 'reason': ''}

    def _is_urgency_with_verification_request(self, text: str) -> bool:
        """Check if message combines urgency with verification requests."""
        urgency_indicators = ['immediately', 'now', 'urgent', 'hurry', 'expire soon', '2 hours']
        verification_indicators = ['verify', 'confirm', 'otp', 'password', 'login']
        
        has_urgency = any(indicator in text for indicator in urgency_indicators)
        has_verification = any(indicator in text for indicator in verification_indicators)
        
        return has_urgency and has_verification

    def _create_risk_assessment(
        self,
        risk_level: RiskLevel,
        reason: str,
        should_block: bool,
        confidence: float
    ) -> RiskAssessment:
        """Create RiskAssessment output."""
        return RiskAssessment(
            risk_level=risk_level,
            reason=reason,
            should_block=should_block,
            confidence=confidence
        )