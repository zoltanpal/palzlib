from palzlib.sentiment_analyzers.analyzers.base_analyzer import SentimentAnalyzerSingleton
from palzlib.sentiment_analyzers.models.sentiments import Sentiments


class DanishSentimentAnalyzer(SentimentAnalyzerSingleton):

    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls, "larskjeldgaard/senda")
        return cls._instance

    def analyze_text(self, text: str):
        sentiment_prediction = self.analyze(text)
        sentiment_results = {
            item["label"]: round(item["score"], 4) for item in sentiment_prediction[0]
        }
        # sentiment_results["negative"] = sentiment_results.pop("negativ", 0.0)
        # sentiment_results["positive"] = sentiment_results.pop("positiv", 0.0)
        # #
        sentiment_results["negative"] = sentiment_results["negativ"]
        sentiment_results["positive"] = sentiment_results["positiv"]

        del sentiment_results["negativ"]
        del sentiment_results["positiv"]

        return Sentiments(**sentiment_results)