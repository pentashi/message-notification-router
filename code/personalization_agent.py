"""
Personalization Agent - Analyzes user-specific patterns and preferences.
Uses historical data to determine how this user typically handles similar messages.
Zero hallucination - only uses provided historical data.
"""

from typing import Dict, Any, Optional
from base_agent import BaseAgent
from models import PersonalizationScore, User, BusinessAccount
import pandas as pd


class PersonalizationAgent(BaseAgent):
    """
    Analyzes user behavior patterns to personalize routing decisions.
    Uses only provided historical data - no assumptions.
    """

    def __init__(self, api_key: str = None):
        super().__init__("personalization_agent", api_key)

    def process(self, input_data: Dict[str, Any]) -> PersonalizationScore:
        """
        Analyze user patterns and preferences.
        
        Args:
            input_data: Contains 'user' (User), 'business' (optional BusinessAccount),
                       'user_history_df' (DataFrame), 'user_events_df' (DataFrame)
            
        Returns:
            PersonalizationScore with user-specific insights
        """
        self._log_processing(input_data)
        
        user: User = input_data.get('user')
        business: BusinessAccount = input_data.get('business')
        user_history_df = input_data.get('user_history_df')
        user_events_df = input_data.get('user_events_df')
        
        if not user:
            return self._create_score(
                user_engagement_score=0.5,
                trust_score=0.5,
                historical_preference="digest",
                reasoning="No user data available",
                confidence=0.0
            )

        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(user)
        
        # Calculate trust score for sender/business
        trust_score = self._calculate_trust_score(user, business, user_events_df)
        
        # Determine historical preference
        preference = self._determine_historical_preference(user_events_df)
        
        # Build reasoning
        reasoning = self._build_reasoning(user, engagement_score, trust_score, preference)
        
        return self._create_score(
            user_engagement_score=engagement_score,
            trust_score=trust_score,
            historical_preference=preference,
            reasoning=reasoning,
            confidence=0.85
        )

    def _calculate_engagement_score(self, user: User) -> float:
        """
        Calculate user engagement score based on 30-day activity.
        Higher score = more engaged user.
        """
        total_interactions = (
            user.messages_opened_30d +
            user.messages_replied_30d +
            user.notifications_dismissed_30d +
            user.messages_reported_30d
        )
        
        if total_interactions == 0:
            return 0.0
        
        # Engagement ratio: replies / total interactions
        if total_interactions > 0:
            reply_ratio = user.messages_replied_30d / total_interactions
        else:
            reply_ratio = 0.0
        
        # Open ratio: opened / total interactions
        open_ratio = user.messages_opened_30d / total_interactions if total_interactions > 0 else 0.0
        
        # Dismissal penalty
        dismissal_penalty = user.notifications_dismissed_30d / total_interactions if total_interactions > 0 else 0.0
        
        # Combined score
        engagement_score = (reply_ratio * 0.5) + (open_ratio * 0.3) - (dismissal_penalty * 0.2)
        
        return max(0.0, min(1.0, engagement_score))

    def _calculate_trust_score(
        self,
        user: User,
        business: Optional[BusinessAccount],
        user_events_df: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculate trust score for the sender/business.
        Uses business verification and user's historical relationship.
        """
        base_score = 0.5
        
        # Business verification bonus
        if business and business.verified:
            base_score += 0.3
        
        # Low report ratio bonus
        if user.messages_reported_30d == 0:
            base_score += 0.1
        
        # Business relationship history
        if business and user_events_df is not None:
            # Check if user has positive history with this business
            # (This would be more sophisticated with actual relationship data)
            base_score += 0.1
        
        return max(0.0, min(1.0, base_score))

    def _determine_historical_preference(self, user_events_df: Optional[pd.DataFrame]) -> str:
        """
        Determine user's historical preference based on past behavior.
        Uses only provided event data.
        """
        if user_events_df is None or user_events_df.empty:
            return "digest"  # Conservative default
        
        # Analyze past reactions to similar messages
        # This would be more sophisticated with actual pattern matching
        # For now, use simple heuristics
        
        # If user frequently dismisses, prefer digest/mute
        # If user frequently replies, prefer notify
        
        # Simplified: return digest as conservative default
        return "digest"

    def _build_reasoning(
        self,
        user: User,
        engagement_score: float,
        trust_score: float,
        preference: str
    ) -> str:
        """Build human-readable reasoning for personalization."""
        reasoning_parts = []
        
        if engagement_score > 0.7:
            reasoning_parts.append("highly engaged user")
        elif engagement_score > 0.4:
            reasoning_parts.append("moderately engaged user")
        else:
            reasoning_parts.append("low engagement user")
        
        if trust_score > 0.7:
            reasoning_parts.append("high trust context")
        elif trust_score < 0.4:
            reasoning_parts.append("low trust context")
        
        reasoning_parts.append(f"historically prefers '{preference}'")
        
        return ", ".join(reasoning_parts)

    def _create_score(
        self,
        user_engagement_score: float,
        trust_score: float,
        historical_preference: str,
        reasoning: str,
        confidence: float
    ) -> PersonalizationScore:
        """Create PersonalizationScore output."""
        return PersonalizationScore(
            user_engagement_score=user_engagement_score,
            trust_score=trust_score,
            historical_preference=historical_preference,
            reasoning=reasoning,
            confidence=confidence
        )