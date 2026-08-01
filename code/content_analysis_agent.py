"""
Content Analysis Agent - Analyzes message content without making decisions.
Extracts features and patterns for downstream agents to use.
Uses LLM but with strict constraints to prevent hallucination.
"""

from typing import Dict, Any, List
from base_agent import BaseAgent
from models import ContentAnalysis, Message
import google.genai as genai
import os


class ContentAnalysisAgent(BaseAgent):
    """
    Analyzes message content to extract features.
    Does NOT make routing decisions - only provides analysis.
    """

    def __init__(self, api_key: str = None):
        super().__init__("content_analysis_agent", api_key)
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set for content analysis")
        self.client = genai.Client(api_key=self.api_key)

    def process(self, input_data: Dict[str, Any]) -> ContentAnalysis:
        """
        Analyze message content and extract features.
        
        Args:
            input_data: Contains 'message' (Message object) and optionally 'media_path' for multimodal
            
        Returns:
            ContentAnalysis with extracted features
        """
        self._log_processing(input_data)
        
        message: Message = input_data.get('message')
        media_path: str = input_data.get('media_path')
        
        if not message:
            return self._create_analysis(
                urgency_score=0.0,
                personal_relevance=0.0,
                action_required=False,
                topic_keywords=[],
                detected_patterns=[],
                confidence=0.0
            )

        # Analyze based on media type
        if message.media_type.value == "image":
            return self._analyze_image_message(message, media_path)
        elif message.media_type.value == "voice":
            return self._analyze_voice_message(message, media_path)
        else:
            return self._analyze_text_message(message)

    def _analyze_text_message(self, message: Message) -> ContentAnalysis:
        """Analyze text message using LLM with strict constraints."""
        
        prompt = self._build_text_analysis_prompt(message.message_text)
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt
            )
            
            analysis_text = response.text
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            print(f"Error in text analysis: {e}")
            return self._create_fallback_analysis()

    def _analyze_image_message(self, message: Message, media_path: str) -> ContentAnalysis:
        """Analyze image message with OCR + VLM."""
        # For now, analyze the text description if available
        # In production, would use OCR + vision model
        
        if message.message_text:
            return self._analyze_text_message(message)
        else:
            return self._create_fallback_analysis()

    def _analyze_voice_message(self, message: Message, media_path: str) -> ContentAnalysis:
        """Analyze voice message with ASR + LLM."""
        # For now, analyze any accompanying text
        # In production, would use speech-to-text
        
        if message.message_text:
            return self._analyze_text_message(message)
        else:
            return self._create_fallback_analysis()

    def _build_text_analysis_prompt(self, text: str) -> str:
        """Build prompt for text analysis with strict constraints."""
        return f"""
Analyze this message text and extract the following features:

Message: "{text}"

Provide analysis in this exact format:
Urgency: [0.0-1.0]
Personal Relevance: [0.0-1.0] 
Action Required: [yes/no]
Topic Keywords: [keyword1, keyword2, ...]
Detected Patterns: [pattern1, pattern2, ...]

Rules:
- Urgency: 1.0 if immediate action needed, 0.0 if no time pressure
- Personal Relevance: 1.0 if directly addresses recipient, 0.0 if generic
- Action Required: yes if asks for response/action, no if informational
- Topic Keywords: Extract 2-3 main topics (e.g., "delivery", "meeting", "payment")
- Detected Patterns: Note patterns like "deadline", "mention", "promotion", "greeting"

Be factual and conservative. If uncertain, use lower scores.
"""

    def _parse_analysis_response(self, response_text: str) -> ContentAnalysis:
        """Parse LLM response into structured format."""
        try:
            # Parse the structured response
            urgency = self._extract_score(response_text, "Urgency")
            personal_relevance = self._extract_score(response_text, "Personal Relevance")
            action_required = "yes" in response_text.lower().split("action required")[1].split("\n")[0] if "action required" in response_text.lower() else False
            keywords = self._extract_list(response_text, "Topic Keywords")
            patterns = self._extract_list(response_text, "Detected Patterns")
            
            return self._create_analysis(
                urgency_score=urgency,
                personal_relevance=personal_relevance,
                action_required=action_required,
                topic_keywords=keywords,
                detected_patterns=patterns,
                confidence=0.75  # Moderate confidence for LLM analysis
            )
        except Exception as e:
            print(f"Error parsing analysis response: {e}")
            return self._create_fallback_analysis()

    def _extract_score(self, text: str, label: str) -> float:
        """Extract numerical score from response."""
        try:
            line = [l for l in text.split('\n') if label in l][0]
            score = float(line.split(':')[1].strip())
            return max(0.0, min(1.0, score))  # Clamp to [0,1]
        except:
            return 0.5  # Conservative default

    def _extract_list(self, text: str, label: str) -> List[str]:
        """Extract list from response."""
        try:
            line = [l for l in text.split('\n') if label in l][0]
            content = line.split(':')[1].strip()
            # Remove brackets and split
            content = content.replace('[', '').replace(']', '')
            items = [item.strip() for item in content.split(',')]
            return [item for item in items if item]  # Filter empty
        except:
            return []

    def _create_analysis(
        self,
        urgency_score: float,
        personal_relevance: float,
        action_required: bool,
        topic_keywords: List[str],
        detected_patterns: List[str],
        confidence: float
    ) -> ContentAnalysis:
        """Create ContentAnalysis output."""
        return ContentAnalysis(
            urgency_score=urgency_score,
            personal_relevance=personal_relevance,
            action_required=action_required,
            topic_keywords=topic_keywords,
            detected_patterns=detected_patterns,
            confidence=confidence
        )

    def _create_fallback_analysis(self) -> ContentAnalysis:
        """Create conservative fallback analysis."""
        return ContentAnalysis(
            urgency_score=0.3,
            personal_relevance=0.3,
            action_required=False,
            topic_keywords=[],
            detected_patterns=[],
            confidence=0.2
        )