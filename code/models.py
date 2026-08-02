"""
Pydantic models for data validation and structured outputs.
Ensures zero hallucination by enforcing strict schema validation.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Literal
from enum import Enum


class ActionType(str, Enum):
    """Allowed routing actions."""
    NOTIFY = "notify"
    DIGEST = "digest"
    MUTE = "mute"


class MessageType(str, Enum):
    """Allowed message type classifications — exact 11-type set from problem_statement.md."""
    PERSONAL = "personal"
    URGENT = "urgent"
    EVENT = "event"
    PAYMENT = "payment"
    BUSINESS_UPDATE = "business_update"
    PROMOTION = "promotion"
    GREETING = "greeting"
    FORWARD = "forward"
    SPAM = "spam"
    SCAM = "scam"
    UNKNOWN = "unknown"


# Type mapping for normalization — maps legacy/invented names to official set
TYPE_MAP = {
    "operational": "urgent",
    "informational": "business_update",
    "business": "business_update",
    "question": "unknown",
    "reminder": "unknown",
}


class ConversationType(str, Enum):
    """Allowed conversation types."""
    PERSONAL = "personal"
    GROUP = "group"
    BUSINESS = "business"


class MediaType(str, Enum):
    """Allowed media types."""
    TEXT = ""
    IMAGE = "image"
    VOICE = "voice"


class RiskLevel(str, Enum):
    """Risk classification for safety gating."""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


class Message(BaseModel):
    """Incoming message schema from messages.csv."""
    message_id: str
    user_id: str
    conversation_type: ConversationType
    group_id: Optional[str] = None
    business_id: Optional[str] = None
    sender_user_id: Optional[str] = None
    created_at: str
    message_text: str
    media_type: MediaType = MediaType.TEXT
    media_id: Optional[str] = None
    forwarded_count: int = 0


class User(BaseModel):
    """User behavior schema from users.csv."""
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int


class Group(BaseModel):
    """Group metadata schema from groups.csv."""
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: str
    messages_30d: int


class BusinessAccount(BaseModel):
    """Business account schema from business_accounts.csv."""
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int


class RoutingDecision(BaseModel):
    """Final routing decision output."""
    message_id: str
    action: ActionType
    message_type: MessageType
    reason: str = Field(..., description="Short human-readable explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    evidence_message_ids: str = Field(..., description="Semicolon-separated historical message IDs or 'none'")

    @validator('message_type', pre=True)
    def normalize_type(cls, v):
        """Normalize message types to allowed enum values."""
        if isinstance(v, str):
            return TYPE_MAP.get(v.lower(), v.lower())
        return v

    @validator('confidence')
    def confidence_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

    @validator('evidence_message_ids')
    def evidence_must_be_valid(cls, v):
        if v.lower() != 'none':
            # Validate semicolon-separated format
            for msg_id in v.split(';'):
                if not msg_id.strip():
                    raise ValueError('Evidence message IDs must be non-empty or "none"')
        return v


class RiskAssessment(BaseModel):
    """Risk gating agent output."""
    risk_level: RiskLevel
    reason: str
    should_block: bool
    confidence: float


class ContentAnalysis(BaseModel):
    """Content analysis agent output."""
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    personal_relevance: float = Field(..., ge=0.0, le=1.0)
    action_required: bool
    topic_keywords: List[str]
    detected_patterns: List[str]
    confidence: float


class PersonalizationScore(BaseModel):
    """Personalization agent output."""
    user_engagement_score: float = Field(..., ge=0.0, le=1.0)
    trust_score: float = Field(..., ge=0.0, le=1.0)
    historical_preference: Literal["notify", "digest", "mute"]
    reasoning: str
    confidence: float


class Evidence(BaseModel):
    """Evidence retrieval agent output."""
    relevant_message_ids: List[str]
    similarity_scores: List[float]
    retrieval_reason: str
    confidence: float


class PreliminaryDecision(BaseModel):
    """Decision agent output before critic review."""
    action: ActionType
    message_type: MessageType
    reasoning: str
    confidence: float
    supporting_evidence: List[str]


class CriticReview(BaseModel):
    """Adversarial critic agent output."""
    approved: bool
    criticisms: List[str]
    suggested_action: Optional[ActionType] = None
    confidence_in_critique: float


class AgentOutput(BaseModel):
    """Standardized output for all agents."""
    agent_name: str
    success: bool
    data: dict
    confidence: float
    error_message: Optional[str] = None