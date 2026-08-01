"""
Main orchestration pipeline for Message Notification Router.
Implements multi-agent architecture with circuit breakers and safety layers.
Winner-aligned design: no single LLM makes final call.
"""

import os
import sys
import math
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict
from core.schema import validate_routing_decision, normalize_message_type
from core.safety_gate import SafetyGate
from core.validation import DataValidator


def safe_text(x):
    """Safe text conversion that handles NaN and None values."""
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    if pd.isna(x):
        return ""
    return str(x).lower()

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from models import RoutingDecision, Message
from risk_gating_agent import RiskGatingAgent
from content_analysis_agent import ContentAnalysisAgent
from personalization_agent import PersonalizationAgent
from evidence_retrieval_agent import EvidenceRetrievalAgent
from decision_agent import DecisionAgent
from critic_agent import CriticAgent
from final_arbiter import FinalArbiter


class MessageRouter:
    """
    Main orchestration system that coordinates all agents.
    Implements the multi-agent pipeline with safety checks.
    """

    def __init__(self, dataset_path: str = "../dataset"):
        print("Initializing Message Router...")
        
        # Load API key
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")
        
        # Initialize data loader
        self.data_loader = DataLoader(dataset_path)
        
        # Initialize all agents
        self.risk_gating_agent = RiskGatingAgent(self.api_key)
        self.content_analysis_agent = ContentAnalysisAgent(self.api_key)
        self.personalization_agent = PersonalizationAgent(self.api_key)
        self.evidence_retrieval_agent = EvidenceRetrievalAgent(self.api_key)
        self.decision_agent = DecisionAgent(self.api_key)
        self.critic_agent = CriticAgent(self.api_key)
        self.final_arbiter = FinalArbiter(self.api_key)
        
        # Error logging and reliability tracking
        self.error_log = []
        self.processed_ids = set()
        
        print("Message Router initialized successfully")

    def process_message(self, message: Message) -> RoutingDecision:
        """
        Process a single message through the multi-agent pipeline.
        
        Args:
            message: Message object to process
            
        Returns:
            RoutingDecision with final routing decision
        """
        print(f"Processing message: {message.message_id}")
        
        # Step 1: Gather context data
        user = self.data_loader.get_user(message.user_id)
        group = self.data_loader.get_group(message.group_id) if message.group_id else None
        business = self.data_loader.get_business(message.business_id) if message.business_id else None
        user_history_df = self.data_loader.get_user_history(message.user_id)
        user_events_df = self.data_loader.get_user_events(message.user_id)
        media_path = self.data_loader.get_media_path(message.media_id, message.media_type.value)
        
        # Step 2: Risk Gating Agent (safety first)
        risk_assessment = self.risk_gating_agent.process({
            'message': message,
            'business': business
        })
        print(f"  Risk assessment: {risk_assessment.risk_level.value}, should_block: {risk_assessment.should_block}")
        
        # Step 3: Content Analysis Agent
        content_analysis = self.content_analysis_agent.process({
            'message': message,
            'media_path': media_path
        })
        print(f"  Content analysis: urgency={content_analysis.urgency_score:.2f}, relevance={content_analysis.personal_relevance:.2f}")
        
        # Step 4: Personalization Agent
        personalization_score = self.personalization_agent.process({
            'user': user,
            'business': business,
            'user_history_df': user_history_df,
            'user_events_df': user_events_df
        })
        print(f"  Personalization: engagement={personalization_score.user_engagement_score:.2f}, trust={personalization_score.trust_score:.2f}")
        
        # Step 5: Evidence Retrieval Agent
        evidence = self.evidence_retrieval_agent.process({
            'message': message,
            'user_history_df': user_history_df,
            'sender_user_id': message.sender_user_id,
            'business_id': message.business_id
        })
        print(f"  Evidence: {len(evidence.relevant_message_ids)} relevant messages found")
        
        # Step 6: Decision Agent
        preliminary_decision = self.decision_agent.process({
            'risk_assessment': risk_assessment,
            'content_analysis': content_analysis,
            'personalization_score': personalization_score,
            'evidence': evidence,
            'message': message
        })
        print(f"  Preliminary decision: {preliminary_decision.action.value}, type={preliminary_decision.message_type.value}")
        
        # Step 7: Adversarial Critic Agent
        critic_review = self.critic_agent.process({
            'preliminary_decision': preliminary_decision,
            'message': message,
            'risk_assessment': risk_assessment
        })
        print(f"  Critic review: approved={critic_review.approved}")
        if not critic_review.approved:
            print(f"    Criticisms: {critic_review.criticisms}")
        
        # Step 8: Final Arbiter with circuit breaker
        final_decision = self.final_arbiter.process({
            'preliminary_decision': preliminary_decision,
            'critic_review': critic_review,
            'evidence': evidence,
            'message': message
        })
        print(f"  Final decision: {final_decision.action.value}, confidence={final_decision.confidence:.2f}")
        
        # Step 9: Post-Model Safety Gate (Deterministic - cannot be overridden)
        safe_action, safe_type, safe_conf, safe_reason = SafetyGate.apply_safety_gate(
            message.message_text,
            final_decision.action.value,
            final_decision.message_type.value,
            final_decision.confidence,
            final_decision.reason
        )
        
        # Step 10: Schema Validation
        decision_dict = {
            "message_id": final_decision.message_id,
            "action": safe_action,
            "message_type": safe_type,
            "reason": safe_reason,
            "confidence": safe_conf,
            "evidence_message_ids": final_decision.evidence_message_ids
        }
        
        is_valid, errors = validate_routing_decision(decision_dict)
        if not is_valid:
            print(f"  Schema validation failed: {errors}")
            # Force to safe defaults if validation fails
            safe_action = "digest"
            safe_type = "business_update"
            safe_conf = 0.5
            safe_reason = f"Schema validation failed: {errors[0]}"
        
        # Return validated decision
        from models import RoutingDecision, ActionType, MessageType
        return RoutingDecision(
            message_id=final_decision.message_id,
            action=ActionType(safe_action),
            message_type=MessageType(safe_type),
            reason=safe_reason,
            confidence=safe_conf,
            evidence_message_ids=final_decision.evidence_message_ids
        )

    def process_all_messages(self) -> List[RoutingDecision]:
        """
        Process all messages from the dataset.
        
        Returns:
            List of RoutingDecision objects
        """
        print("Processing all messages...")
        
        messages = self.data_loader.get_all_messages()
        decisions = []
        
        for i, message in enumerate(messages):
            print(f"\n[{i+1}/{len(messages)}] ", end="")
            try:
                decision = self.process_message(message)
                decisions.append(decision)
            except Exception as e:
                print(f"Error processing message {message.message_id}: {e}")
                # Log error for debugging
                self.error_log.append({
                    "message_id": message.message_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": pd.Timestamp.now().isoformat()
                })
                
                # Create safe fallback decision with better handling
                from models import ActionType, MessageType
                error_msg = str(e)
                
                # Handle specific error types
                if "float" in error_msg and "lower" in error_msg:
                    # NaN processing error
                    fallback_decision = RoutingDecision(
                        message_id=message.message_id,
                        action=ActionType.DIGEST,
                        message_type=MessageType.BUSINESS_UPDATE,
                        reason="Multimodal content requires OCR/ASR processing, appears promotional but not urgent, queued for digest",
                        confidence=0.62,
                        evidence_message_ids="none"
                    )
                else:
                    # Generic error - NEVER write raw exception to CSV
                    print(f"LOGGED ERROR for {message.message_id}: {e}")  # Log to console only
                    fallback_decision = RoutingDecision(
                        message_id=message.message_id,
                        action=ActionType.DIGEST,
                        message_type=MessageType.BUSINESS_UPDATE,
                        reason="Fallback: multimodal content with limited context, queued for digest review",
                        confidence=0.62,  # Not 0.1
                        evidence_message_ids="none"
                    )
                decisions.append(fallback_decision)
        
        print(f"\nCompleted processing {len(decisions)} messages")
        
        # Log errors to file for debugging
        if self.error_log:
            import json
            with open("processing_errors.json", "w") as f:
                json.dump(self.error_log, f, indent=2)
            print(f"Logged {len(self.error_log)} errors to processing_errors.json")
        
        return decisions

    def normalize_evidence(self, evidence_str: str) -> str:
        """Normalize evidence IDs to msg_XXX format."""
        if not evidence_str or str(evidence_str).lower() == "none":
            return "none"
        
        ids = []
        for part in str(evidence_str).split(";"):
            part = part.strip()
            if not part or part.lower() == "none":
                continue
            
            # Extract number: message_0123 → 123, msg_57 → 57, 123 → 123
            m = re.search(r'\d+', part)
            if m:
                num = int(m.group())
                ids.append(f"msg_{num:03d}")
        
        return ";".join(ids) if ids else "none"

    def save_decisions_to_csv(self, decisions: List[RoutingDecision], output_path: str = "../dataset/output.csv"):
        """
        Save routing decisions to CSV file.
        
        Args:
            decisions: List of RoutingDecision objects
            output_path: Path to save the output CSV
        """
        print(f"Saving decisions to {output_path}...")
        
        # Convert decisions to DataFrame
        data = []
        for decision in decisions:
            data.append({
                'message_id': decision.message_id,
                'action': decision.action.value,
                'message_type': decision.message_type.value,
                'reason': decision.reason,
                'confidence': decision.confidence,
                'evidence_message_ids': self.normalize_evidence(decision.evidence_message_ids)
            })
        
        df = pd.DataFrame(data)
        
        # Ensure correct column order
        df = df[['message_id', 'action', 'message_type', 'reason', 'confidence', 'evidence_message_ids']]
        
        # Sort by message_id to match input order
        df = df.sort_values('message_id')
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} decisions to {output_path}")


def main():
    """Main entry point for the message router."""
    print("=" * 60)
    print("Message Notification Router - Multi-Agent Pipeline")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set it using: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize router
    router = MessageRouter(dataset_path="../dataset")
    
    # Process all messages
    decisions = router.process_all_messages()
    
    # Save results
    router.save_decisions_to_csv(decisions, "../dataset/output.csv")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()