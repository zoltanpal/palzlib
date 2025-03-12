import nltk
import threading
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from palzlib.sentiment_analyzers.models.sentiments import Sentiments


class EnglishSentimentAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Check if the VADER lexicon is downloaded, if not, download it
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon")

        # Initialize SentimentIntensityAnalyzer
        self.sid = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str):
        if not text:
            raise ValueError("Missing text to analyze")

        results = self.sid.polarity_scores(text)

        return Sentiments(
            negative=results.get("neg", 0.0),
            neutral=results.get("neu", 0.0),
            positive=results.get("pos", 0.0),
            compound=results.get("compound", 0.0),
        )
