"""
Base agent class for all specialized agents.
Provides common functionality and enforces agent contract.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from models import AgentOutput
import os


class BaseAgent(ABC):
    """Abstract base class for all agents in the pipeline."""

    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        # Don't raise error here - let individual agents handle missing API key
        # Some agents (like risk gating) don't need API keys

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentOutput:
        """
        Process input data and return structured output.
        
        Args:
            input_data: Dictionary containing relevant data for this agent
            
        Returns:
            AgentOutput with success status, data, confidence, and optional error
        """
        pass

    def _create_output(
        self,
        success: bool,
        data: Dict[str, Any],
        confidence: float,
        error_message: Optional[str] = None
    ) -> AgentOutput:
        """Create standardized AgentOutput."""
        return AgentOutput(
            agent_name=self.name,
            success=success,
            data=data,
            confidence=confidence,
            error_message=error_message
        )

    def _validate_confidence(self, confidence: float) -> bool:
        """Validate confidence score is between 0 and 1."""
        return 0.0 <= confidence <= 1.0

    def _log_processing(self, input_data: Dict[str, Any]):
        """Log processing details for transparency."""
        print(f"[{self.name}] Processing input with keys: {list(input_data.keys())}")


class CircuitBreaker:
    """
    Circuit breaker pattern for agent failures.
    Prevents cascading failures and provides fallback behavior.
    """

    def __init__(self, failure_threshold: int = 3, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def call(self, agent_func, *args, **kwargs):
        """
        Execute agent function with circuit breaker protection.
        
        Args:
            agent_func: The agent function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            AgentOutput or fallback response
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                return self._fallback_response()

        try:
            result = agent_func(*args, **kwargs)
            if result.success:
                self._on_success()
                return result
            else:
                self._on_failure()
                return self._fallback_response()
        except Exception as e:
            self._on_failure()
            return self._fallback_response(str(e))

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        import time
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) > self.timeout

    def _on_success(self):
        """Handle successful agent call."""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed agent call."""
        import time
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def _fallback_response(self, error: Optional[str] = None) -> AgentOutput:
        """Return safe fallback response when circuit is open."""
        return AgentOutput(
            agent_name="circuit_breaker",
            success=False,
            data={
                "action": "digest",  # Safe default
                "message_type": "unknown", 
                "reasoning": "Circuit breaker triggered - using safe default",
                "confidence": 0.1
            },
            confidence=0.1,
            error_message=error or "Circuit breaker is open"
        )