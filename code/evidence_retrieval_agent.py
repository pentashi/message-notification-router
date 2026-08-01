"""
Evidence Retrieval Agent - Finds relevant historical messages for transparency.
Uses only provided historical data - no external knowledge.
Ensures decisions are grounded in actual user history.
"""

from typing import Dict, Any, List
from base_agent import BaseAgent
from models import Evidence, Message
import pandas as pd
from difflib import SequenceMatcher


class EvidenceRetrievalAgent(BaseAgent):
    """
    Retrieves relevant historical messages to support routing decisions.
    Uses similarity matching and user relationship data.
    """

    def __init__(self, api_key: str = None):
        super().__init__("evidence_retrieval_agent", api_key)

    def process(self, input_data: Dict[str, Any]) -> Evidence:
        """
        Retrieve relevant historical messages.
        
        Args:
            input_data: Contains 'message' (current Message), 'user_history_df' (DataFrame),
                       'sender_user_id' (optional), 'business_id' (optional)
            
        Returns:
            Evidence with relevant message IDs and similarity scores
        """
        self._log_processing(input_data)
        
        message: Message = input_data.get('message')
        user_history_df = input_data.get('user_history_df')
        sender_user_id = input_data.get('sender_user_id')
        business_id = input_data.get('business_id')
        
        if not message or user_history_df is None or user_history_df.empty:
            return self._create_evidence(
                relevant_message_ids=[],
                similarity_scores=[],
                retrieval_reason="No historical data available",
                confidence=0.0
            )

        # Strategy 1: Same sender
        same_sender_evidence = self._find_same_sender_messages(
            message, user_history_df, sender_user_id
        )
        
        # Strategy 2: Same business
        same_business_evidence = self._find_same_business_messages(
            message, user_history_df, business_id
        )
        
        # Strategy 3: Similar content
        similar_content_evidence = self._find_similar_content_messages(
            message, user_history_df
        )
        
        # Combine and rank evidence
        all_evidence = []
        all_evidence.extend(same_sender_evidence)
        all_evidence.extend(same_business_evidence)
        all_evidence.extend(similar_content_evidence)
        
        # Remove duplicates and keep top 3
        unique_evidence = self._deduplicate_and_rank(all_evidence)
        top_evidence = unique_evidence[:3]
        
        if not top_evidence:
            return self._create_evidence(
                relevant_message_ids=[],
                similarity_scores=[],
                retrieval_reason="No relevant historical messages found",
                confidence=0.0
            )
        
        message_ids = [e['message_id'] for e in top_evidence]
        similarity_scores = [e['similarity'] for e in top_evidence]
        
        reason = self._build_retrieval_reason(top_evidence, sender_user_id, business_id)
        
        return self._create_evidence(
            relevant_message_ids=message_ids,
            similarity_scores=similarity_scores,
            retrieval_reason=reason,
            confidence=0.80
        )

    def _find_same_sender_messages(
        self,
        message: Message,
        user_history_df: pd.DataFrame,
        sender_user_id: str
    ) -> List[Dict[str, Any]]:
        """Find historical messages from the same sender."""
        if not sender_user_id:
            return []
        
        same_sender = user_history_df[
            user_history_df['sender_user_id'] == sender_user_id
        ]
        
        if same_sender.empty:
            return []
        
        # Get recent messages from same sender
        recent_messages = same_sender.tail(5)  # Last 5 messages
        
        evidence = []
        for _, row in recent_messages.iterrows():
            evidence.append({
                'message_id': row['message_id'],
                'similarity': 0.9,  # High similarity for same sender
                'reason': 'same_sender'
            })
        
        return evidence

    def _find_same_business_messages(
        self,
        message: Message,
        user_history_df: pd.DataFrame,
        business_id: str
    ) -> List[Dict[str, Any]]:
        """Find historical messages from the same business."""
        if not business_id:
            return []
        
        same_business = user_history_df[
            user_history_df['business_id'] == business_id
        ]
        
        if same_business.empty:
            return []
        
        # Get recent messages from same business
        recent_messages = same_business.tail(3)  # Last 3 messages
        
        evidence = []
        for _, row in recent_messages.iterrows():
            evidence.append({
                'message_id': row['message_id'],
                'similarity': 0.85,  # High similarity for same business
                'reason': 'same_business'
            })
        
        return evidence

    def _find_similar_content_messages(
        self,
        message: Message,
        user_history_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Find historical messages with similar content."""
        if not message.message_text:
            return []
        
        similar_messages = []
        
        # Compare with recent historical messages
        recent_history = user_history_df.tail(20)  # Last 20 messages
        
        for _, row in recent_history.iterrows():
            historical_text = row.get('message_text', '')
            if not historical_text:
                continue
            
            similarity = self._calculate_text_similarity(
                message.message_text,
                historical_text
            )
            
            if similarity > 0.5:  # Threshold for similarity
                similar_messages.append({
                    'message_id': row['message_id'],
                    'similarity': similarity,
                    'reason': 'similar_content'
                })
        
        # Sort by similarity and keep top matches
        similar_messages.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_messages[:3]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _deduplicate_and_rank(self, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate message IDs and rank by similarity."""
        seen_ids = set()
        unique_evidence = []
        
        for e in evidence:
            message_id = e['message_id']
            if message_id not in seen_ids:
                seen_ids.add(message_id)
                unique_evidence.append(e)
        
        # Sort by similarity (highest first)
        unique_evidence.sort(key=lambda x: x['similarity'], reverse=True)
        return unique_evidence

    def _build_retrieval_reason(
        self,
        evidence: List[Dict[str, Any]],
        sender_user_id: str,
        business_id: str
    ) -> str:
        """Build human-readable reason for evidence retrieval."""
        if not evidence:
            return "No relevant historical messages found"
        
        reasons = []
        for e in evidence:
            if e['reason'] == 'same_sender':
                reasons.append("same sender")
            elif e['reason'] == 'same_business':
                reasons.append("same business")
            elif e['reason'] == 'similar_content':
                reasons.append("similar content")
        
        # Count unique reasons
        unique_reasons = list(set(reasons))
        
        if len(unique_reasons) == 1:
            return f"Found {len(evidence)} relevant message(s) based on {unique_reasons[0]}"
        else:
            return f"Found {len(evidence)} relevant message(s) based on {', '.join(unique_reasons)}"

    def _create_evidence(
        self,
        relevant_message_ids: List[str],
        similarity_scores: List[float],
        retrieval_reason: str,
        confidence: float
    ) -> Evidence:
        """Create Evidence output."""
        return Evidence(
            relevant_message_ids=relevant_message_ids,
            similarity_scores=similarity_scores,
            retrieval_reason=retrieval_reason,
            confidence=confidence
        )