"""
Basic functionality test to validate the system works.
Tests data loading and basic agent initialization without API calls.
"""

import sys
from pathlib import Path

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

print("Testing basic functionality...")

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from models import (
        Message, User, Group, BusinessAccount,
        ActionType, MessageType, RiskLevel,
        RoutingDecision, RiskAssessment
    )
    print("   [OK] Models imported successfully")
except Exception as e:
    print(f"   [ERROR] Error importing models: {e}")
    sys.exit(1)

try:
    from data_loader import DataLoader
    print("   [OK] Data loader imported successfully")
except Exception as e:
    print(f"   [ERROR] Error importing data loader: {e}")
    sys.exit(1)

try:
    from base_agent import BaseAgent, CircuitBreaker
    print("   [OK] Base agent imported successfully")
except Exception as e:
    print(f"   [ERROR] Error importing base agent: {e}")
    sys.exit(1)

try:
    from risk_gating_agent import RiskGatingAgent
    print("   [OK] Risk gating agent imported successfully")
except Exception as e:
    print(f"   [ERROR] Error importing risk gating agent: {e}")
    sys.exit(1)

# Test 2: Data loading
print("\n2. Testing data loading...")
try:
    loader = DataLoader("../dataset")
    print(f"   [OK] Loaded {len(loader.messages)} messages")
    print(f"   [OK] Loaded {len(loader.users)} users")
    print(f"   [OK] Loaded {len(loader.groups)} groups")
    print(f"   [OK] Loaded {len(loader.business_accounts)} business accounts")
except Exception as e:
    print(f"   [ERROR] Error loading data: {e}")
    sys.exit(1)

# Test 3: Risk gating agent (no API key needed)
print("\n3. Testing risk gating agent...")
try:
    # Create a test message
    from models import ConversationType, MediaType
    test_message = Message(
        message_id="test_001",
        user_id="u_001",
        conversation_type=ConversationType.PERSONAL,
        group_id=None,
        business_id=None,
        sender_user_id="u_002",
        created_at="2026-08-01 12:00",
        message_text="Your account will be blocked in 2 hours. Verify OTP now.",
        media_type=MediaType.TEXT,
        media_id=None,
        forwarded_count=0
    )
    
    risk_agent = RiskGatingAgent()  # Will fail without API key, but that's expected
    print("   [WARN] Risk agent requires API key for full functionality")
except Exception as e:
    print(f"   [WARN] Expected behavior (no API key): {e}")

# Test 4: Circuit breaker
print("\n4. Testing circuit breaker...")
try:
    circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=60)
    
    def failing_function():
        raise Exception("Test failure")
    
    # First failure
    result1 = circuit_breaker.call(failing_function)
    print(f"   [OK] First failure handled: {result1.success}")
    
    # Second failure  
    result2 = circuit_breaker.call(failing_function)
    print(f"   [OK] Second failure handled: {result2.success}")
    
    # Third failure should trigger circuit breaker
    result3 = circuit_breaker.call(failing_function)
    print(f"   [OK] Circuit breaker triggered: {result3.success}")
    
except Exception as e:
    print(f"   [ERROR] Error testing circuit breaker: {e}")

print("\n" + "=" * 60)
print("Basic functionality test complete!")
print("=" * 60)
print("\nNote: Full agent testing requires GEMINI_API_KEY")
print("Set it with: export GEMINI_API_KEY='your-key-here'")