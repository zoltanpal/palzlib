import threading
from typing import List

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sentiment_analyzer.models.sentiments import Sentiments


class EnglishSentimentAnalyzer:
    """
    A class for analyzing English text sentiment using VADER
    (Valence Aware Dictionary and sEntiment Reasoner).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        self.sid = SentimentIntensityAnalyzer()

    def _map_sentiment_result(self, results: dict) -> Sentiments:
        # Determine compound label based on compound score thresholds
        compound_score = results.get("compound", 0.0)
        if compound_score >= 0.05:
            compound_label = "positive"
        elif compound_score <= -0.05:
            compound_label = "negative"
        else:            
            compound_label = "neutral"

        return Sentiments(
            negative=results.get("neg", 0.0),
            neutral=results.get("neu", 0.0),
            positive=results.get("pos", 0.0),
            compound=compound_score,
            compound_label=compound_label,
        )

    def analyze_text(self, text: str) -> Sentiments:
        if not text:
            raise ValueError("Missing text to analyze")

        results = self.sid.polarity_scores(text)
        return self._map_sentiment_result(results)

    def analyze_batch(self, texts: List[str]) -> List[Sentiments]:
        return [
            self._map_sentiment_result(self.sid.polarity_scores(text))
            for text in texts
            if text
        ]