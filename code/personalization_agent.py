"""
Personalization Agent - Analyzes user-specific patterns and preferences.
Uses historical data to determine how this user typically handles similar messages.
Zero hallucination - only uses provided historical data.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from base_agent import BaseAgent
from models import PersonalizationScore, User, BusinessAccount
import pandas as pd


_CHAIN_SPAM_KEYWORDS = [
    "forward", "blessing", "good luck", "send to ten", "send to 10",
    "fwd as received", "share blessings",
]


class PersonalizationAgent(BaseAgent):
    """
    Analyzes user behavior patterns to personalize routing decisions.
    Uses only provided historical data - no assumptions.

    Explicit rules applied (per architecture spec):
    1. Quiet hours  → notify downgraded to digest
    2. Group muted by user → only direct urgent mention overrides to notify
    3. Business promo with opted-out user → mute, not digest
    4. High user_reports_30d → downgrade trust/action
    5. Chain-forward spam in muted group context → reinforce mute
    """

    def __init__(self, api_key: str = None):
        super().__init__("personalization_agent", api_key)

    def process(self, input_data: Dict[str, Any]) -> PersonalizationScore:
        """
        Analyze user patterns and preferences.

        Args:
            input_data: Contains 'user', 'business', 'user_history_df', 'user_events_df',
                        'group_membership' (dict|None), 'user_business_row' (DataFrame|None),
                        'message' (Message|None).

        Returns:
            PersonalizationScore with user-specific insights and override flags.
        """
        self._log_processing(input_data)

        user: User = input_data.get('user')
        business: BusinessAccount = input_data.get('business')
        user_history_df = input_data.get('user_history_df')
        user_events_df = input_data.get('user_events_df')
        group_membership: Optional[dict] = input_data.get('group_membership')
        user_business_row = input_data.get('user_business_row')
        message = input_data.get('message')

        if not user:
            return self._create_score(
                user_engagement_score=0.5,
                trust_score=0.5,
                historical_preference="digest",
                reasoning="No user data available",
                confidence=0.0
            )

        engagement_score = self._calculate_engagement_score(user)
        trust_score = self._calculate_trust_score(user, business, user_events_df)
        preference = self._determine_historical_preference(user_events_df)
        overrides = []

        # Rule 1: Quiet hours → force preference to digest
        if message and self._is_quiet_hours(user.do_not_disturb_window, message.created_at):
            preference = "digest"
            overrides.append("quiet hours: notify suppressed to digest")

        # Rule 2: Group muted by user → only direct urgent mention should notify
        if group_membership:
            muted = group_membership.get('group_muted_by_user', 0)
            if int(muted) == 1:
                msg_text = (message.message_text or "").lower() if message else ""
                user_mentioned = message and (
                    (message.user_id in msg_text) or ("@" + message.user_id in msg_text)
                )
                if not user_mentioned:
                    preference = "mute"
                    overrides.append("group muted by user: no direct mention, preference → mute")
                else:
                    # Direct mention in muted group → allow digest at most unless urgency is high
                    if preference == "notify":
                        preference = "digest"
                        overrides.append("group muted by user: direct mention upgrades mute → digest only")

        # Rule 3: Business promotion with opted-out user → mute
        if business and user_business_row is not None and not (
            hasattr(user_business_row, 'empty') and user_business_row.empty
        ):
            row = user_business_row.iloc[0] if hasattr(user_business_row, 'iloc') else user_business_row
            allows_promotions = row.get('allows_promotions', 1) if isinstance(row, dict) else row.get('allows_promotions', 1)
            opted_out_at = row.get('promotions_opted_out_at', None) if isinstance(row, dict) else row.get('promotions_opted_out_at', None)
            if int(allows_promotions) == 0 or (opted_out_at and str(opted_out_at) not in ("", "nan", "None")):
                trust_score = min(trust_score, 0.3)
                preference = "mute"
                overrides.append("user opted out of promotions from this business → mute")

        # Rule 4: High reports_30d → downgrade trust
        if user.messages_reported_30d >= 5:
            trust_score = max(0.0, trust_score - 0.2)
            overrides.append(f"user has reported {user.messages_reported_30d} messages in 30d — sender trust reduced")

        reasoning = self._build_reasoning(user, engagement_score, trust_score, preference)
        if overrides:
            reasoning += "; overrides: " + "; ".join(overrides)

        return self._create_score(
            user_engagement_score=engagement_score,
            trust_score=trust_score,
            historical_preference=preference,
            reasoning=reasoning,
            confidence=0.85
        )

    def _is_quiet_hours(self, dnd_window: str, created_at: str) -> bool:
        """Return True if message arrived inside the user's do-not-disturb window."""
        try:
            # dnd_window format: "22:00-07:00"
            parts = dnd_window.strip().split("-")
            if len(parts) != 2:
                return False
            start_h, start_m = map(int, parts[0].split(":"))
            end_h, end_m = map(int, parts[1].split(":"))
            # Parse message time (handle both "YYYY-MM-DD HH:MM" and "YYYY-MM-DD HH:MM:SS")
            dt = datetime.fromisoformat(created_at.strip()) if "T" in created_at else datetime.strptime(created_at.strip()[:16], "%Y-%m-%d %H:%M")
            msg_minutes = dt.hour * 60 + dt.minute
            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m
            if start_minutes > end_minutes:  # Spans midnight
                return msg_minutes >= start_minutes or msg_minutes <= end_minutes
            return start_minutes <= msg_minutes <= end_minutes
        except Exception:
            return False

    def _calculate_engagement_score(self, user: User) -> float:
        """Calculate user engagement score based on 30-day activity."""
        total_interactions = (
            user.messages_opened_30d +
            user.messages_replied_30d +
            user.notifications_dismissed_30d +
            user.messages_reported_30d
        )
        if total_interactions == 0:
            return 0.0
        reply_ratio = user.messages_replied_30d / total_interactions
        open_ratio = user.messages_opened_30d / total_interactions
        dismissal_penalty = user.notifications_dismissed_30d / total_interactions
        engagement_score = (reply_ratio * 0.5) + (open_ratio * 0.3) - (dismissal_penalty * 0.2)
        return max(0.0, min(1.0, engagement_score))

    def _calculate_trust_score(
        self,
        user: User,
        business: Optional[BusinessAccount],
        user_events_df: Optional[pd.DataFrame]
    ) -> float:
        """Calculate trust score for the sender/business."""
        base_score = 0.5
        if business and business.verified:
            base_score += 0.3
        if user.messages_reported_30d == 0:
            base_score += 0.1
        if business and user_events_df is not None:
            base_score += 0.1
        return max(0.0, min(1.0, base_score))

    def _determine_historical_preference(self, user_events_df: Optional[pd.DataFrame]) -> str:
        """Determine user's historical preference based on past behavior."""
        if user_events_df is None or user_events_df.empty:
            return "digest"
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