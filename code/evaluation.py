"""
Evaluation pipeline for testing the message router on sample_messages.csv.
Compares predictions against known labels to validate system performance.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List
import os

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from models import Message, ActionType, MessageType
from main import MessageRouter


class Evaluator:
    """Evaluates routing decisions against known labels."""

    def __init__(self, dataset_path: str = "../dataset"):
        self.dataset_path = dataset_path
        self.data_loader = DataLoader(dataset_path)
        
    def load_sample_messages(self) -> pd.DataFrame:
        """Load sample_messages.csv with known labels."""
        sample_path = Path(self.dataset_path) / "sample_messages.csv"
        return pd.read_csv(sample_path)

    def convert_sample_to_message(self, row: pd.Series) -> Message:
        """Convert a sample message row to Message object."""
        from models import ConversationType, MediaType
        import pandas as pd
        
        # Handle NaN values for media_type
        media_type_value = row.get('media_type', '')
        if pd.isna(media_type_value):
            media_type_value = ''
        
        return Message(
            message_id=row['message_id'],
            user_id=row['user_id'],
            conversation_type=ConversationType(row['conversation_type']),
            group_id=row.get('group_id', None) if pd.notna(row.get('group_id', None)) else None,
            business_id=row.get('business_id', None) if pd.notna(row.get('business_id', None)) else None,
            sender_user_id=row.get('sender_user_id', None) if pd.notna(row.get('sender_user_id', None)) else None,
            created_at=row['created_at'],
            message_text=row.get('message_text', '') if pd.notna(row.get('message_text', '')) else '',
            media_type=MediaType(str(media_type_value)),
            media_id=row.get('media_id', None) if pd.notna(row.get('media_id', None)) else None,
            forwarded_count=int(row.get('forwarded_count', 0)) if pd.notna(row.get('forwarded_count', 0)) else 0
        )

    def evaluate_sample_messages(self, router: MessageRouter, limit: int = None) -> Dict:
        """
        Evaluate router on sample messages with known labels.
        
        Args:
            router: MessageRouter instance
            limit: Optional limit on number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("Loading sample messages with known labels...")
        sample_df = self.load_sample_messages()
        
        if limit:
            sample_df = sample_df.head(limit)
        
        print(f"Evaluating {len(sample_df)} sample messages...")
        
        results = {
            'total': len(sample_df),
            'action_correct': 0,
            'message_type_correct': 0,
            'confidence_avg': 0.0,
            'errors': 0,
            'detailed_results': []
        }
        
        for idx, row in sample_df.iterrows():
            print(f"\n[{idx+1}/{len(sample_df)}] Evaluating {row['message_id']}")
            
            try:
                # Convert sample to message format
                message = self.convert_sample_to_message(row)
                
                # Get prediction from router
                prediction = router.process_message(message)
                
                # Compare with known labels
                expected_action = row['action']
                expected_type = row['message_type']
                
                action_match = prediction.action.value == expected_action
                type_match = prediction.message_type.value == expected_type
                
                if action_match:
                    results['action_correct'] += 1
                if type_match:
                    results['message_type_correct'] += 1
                
                results['confidence_avg'] += prediction.confidence
                
                # Store detailed result
                results['detailed_results'].append({
                    'message_id': row['message_id'],
                    'expected_action': expected_action,
                    'predicted_action': prediction.action.value,
                    'action_match': action_match,
                    'expected_type': expected_type,
                    'predicted_type': prediction.message_type.value,
                    'type_match': type_match,
                    'confidence': prediction.confidence,
                    'expected_reason': row['reason'],
                    'predicted_reason': prediction.reason
                })
                
                print(f"  Action: {prediction.action.value} (expected: {expected_action}) - {'[OK]' if action_match else '[MISMATCH]'}")
                print(f"  Type: {prediction.message_type.value} (expected: {expected_type}) - {'[OK]' if type_match else '[MISMATCH]'}")
                print(f"  Confidence: {prediction.confidence:.2f}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results['errors'] += 1
        
        # Calculate final metrics
        if results['total'] > 0:
            results['action_accuracy'] = results['action_correct'] / results['total']
            results['type_accuracy'] = results['message_type_correct'] / results['total']
            results['confidence_avg'] = results['confidence_avg'] / results['total']
        
        return results

    def print_evaluation_report(self, results: Dict):
        """Print detailed evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"Total samples evaluated: {results['total']}")
        print(f"Errors encountered: {results['errors']}")
        print(f"\nAction accuracy: {results['action_accuracy']:.2%} ({results['action_correct']}/{results['total']})")
        print(f"Type accuracy: {results['type_accuracy']:.2%} ({results['message_type_correct']}/{results['total']})")
        print(f"Average confidence: {results['confidence_avg']:.2f}")
        
        # Show mismatched examples
        mismatches = [r for r in results['detailed_results'] if not r['action_match']]
        if mismatches:
            print(f"\nAction mismatches ({len(mismatches)}):")
            for mismatch in mismatches[:5]:  # Show first 5
                print(f"  {mismatch['message_id']}: expected {mismatch['expected_action']}, got {mismatch['predicted_action']}")
                print(f"    Expected: {mismatch['expected_reason']}")
                print(f"    Predicted: {mismatch['predicted_reason']}")
        
        type_mismatches = [r for r in results['detailed_results'] if not r['type_match']]
        if type_mismatches:
            print(f"\nType mismatches ({len(type_mismatches)}):")
            for mismatch in type_mismatches[:5]:  # Show first 5
                print(f"  {mismatch['message_id']}: expected {mismatch['expected_type']}, got {mismatch['predicted_type']}")


def main():
    """Main evaluation entry point."""
    print("=" * 60)
    print("Message Router Evaluation Pipeline")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set it using: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = Evaluator(dataset_path="../dataset")
    
    # Initialize router
    router = MessageRouter(dataset_path="../dataset")
    
    # Evaluate on sample messages (limit to first 10 for quick testing)
    print("\nRunning evaluation on sample messages...")
    results = evaluator.evaluate_sample_messages(router, limit=10)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()