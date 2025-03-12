from palzlib.sentiment_analyzers.analyzers.base_analyzer import SentimentAnalyzerSingleton
from palzlib.sentiment_analyzers.models.sentiments import Sentiments, LABEL_MAPPING_ROBERTA


class HungarianSentimentAnalyzer(SentimentAnalyzerSingleton):
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls, "NYTK/sentiment-hts5-xlm-roberta-hungarian")
        return cls._instance

    def analyze_text(self, text: str):
        sentiment_prediction = self.analyze(text)
        sentiment_results = {
            LABEL_MAPPING_ROBERTA[item["label"]]: round(item["score"], 4)
            for item in sentiment_prediction[0]
        }
        return Sentiments(**sentiment_results)