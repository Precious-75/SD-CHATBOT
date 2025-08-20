# sentiment_analyzer.py
# PURPOSE: Provides the SentimentAnalyzer class for rule-based sentiment, urgency,
# confusion, and frustration detection used by the CSV-based chatbot.
from __future__ import annotations
import re
from typing import Dict, Any


class SentimentAnalyzer:
    def __init__(self):
        # Simple rule-based sentiment analysis
        self.positive_words = {
            'good', 'great', 'excellent', 'awesome', 'amazing', 'fantastic', 'wonderful',
            'perfect', 'love', 'like', 'happy', 'satisfied', 'pleased', 'thank', 'thanks',
            'helpful', 'useful', 'appreciate', 'brilliant', 'outstanding', 'superb'
        }

        # Single-word negatives
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'angry', 'frustrated',
            'annoyed', 'upset', 'disappointed', 'useless', 'stupid', 'dumb', 'broken',
            'problem', 'issue', 'error', 'fail', 'failure', 'crash', 'stuck', 'urgent',
            'emergency', 'critical', 'serious', 'wrong', 'offline'
        }

        # Multi-word negative phrases (counted separately)
        self.negative_phrases = {
            'not working', 'not connecting', "doesn't work", 'does not work',
            "can't connect", 'cannot connect', "won't connect", 'failed to',
            'no sound', 'no internet'
        }

        self.urgency_words = {
            'urgent', 'emergency', 'asap', 'immediately', 'critical', 'serious',
            'important', 'deadline', 'quick', 'fast', 'hurry', 'rush'
        }

        self.confusion_phrases = {
            'confused', 'lost', 'stuck', 'dont understand', "don't understand",
            'how do i', 'what is', 'explain', 'help me understand', 'unclear'
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotional state of text."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        positive_count = sum(1 for w in words if w in self.positive_words)
        negative_word_count = sum(1 for w in words if w in self.negative_words)
        negative_phrase_count = sum(1 for p in self.negative_phrases if p in text_lower)
        negative_count = negative_word_count + negative_phrase_count

        urgency_count = sum(1 for w in words if w in self.urgency_words)
        confusion_count = sum(1 for p in self.confusion_phrases if p in text_lower)

        # Determine overall sentiment
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Determine emotional states
        is_urgent = urgency_count > 0
        is_confused = confusion_count > 0
        is_frustrated = (
            negative_count >= 2
            or any(tok in text_lower for tok in ['frustrated', 'angry', 'annoyed'])
        )

        # Intensity (0.0 to 1.0)
        total_emotional_words = positive_count + negative_count + urgency_count
        intensity = min(total_emotional_words / max(len(words), 1), 1.0)

        return {
            'sentiment': sentiment,
            'is_urgent': is_urgent,
            'is_confused': is_confused,
            'is_frustrated': is_frustrated,
            'intensity': intensity,
            'positive_score': positive_count,
            'negative_score': negative_count
        }
