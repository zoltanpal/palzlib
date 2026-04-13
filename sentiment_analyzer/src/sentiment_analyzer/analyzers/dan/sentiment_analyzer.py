from typing import List

from sentiment_analyzer.analyzers.base_analyzer import SentimentAnalyzerSingleton
from sentiment_analyzer.models.sentiments import Sentiments


class DanishSentimentAnalyzer(SentimentAnalyzerSingleton):
    """
    A singleton class for analyzing Danish text sentiment using a pre-trained LLM.

    This class inherits from `SentimentAnalyzerSingleton` to ensure only one instance of the model
    is used for Danish sentiment analysis.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls, "larskjeldgaard/senda")
            # cls._instance = super().__new__(cls, "NbAiLab/nb-bert-base-sentiment")
        return cls._instance

    def _map_sentiment_result(self, prediction) -> Sentiments:
        sentiment_results = {
            item["label"]: round(item["score"], 4)
            for item in prediction
        }

        sentiment_results["negative"] = sentiment_results.pop("negativ")
        sentiment_results["positive"] = sentiment_results.pop("positiv")

        return Sentiments(**sentiment_results)

    def analyze_text(self, text: str) -> Sentiments:
        """
        Analyzes the sentiment of a given Danish text.
        """
        sentiment_prediction = self.analyze(text)
        return self._map_sentiment_result(sentiment_prediction[0])

    def analyze_batch(self, texts: List[str]) -> List[Sentiments]:
        """
        Analyzes a batch of Danish texts.
        """
        predictions = self.pipeline(texts)
        return [self._map_sentiment_result(prediction) for prediction in predictions]