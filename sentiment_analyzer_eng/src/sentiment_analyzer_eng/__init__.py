__all__ = [
    "SentimentAnalyzer",
    "SentimentAnalyzerFactory",
    "Sentiments",
]


def __getattr__(name: str):
    if name == "SentimentAnalyzer":
        from sentiment_analyzer_eng.analyzers.sentiment_analyzer import SentimentAnalyzer

        return SentimentAnalyzer
    if name == "SentimentAnalyzerFactory":
        from sentiment_analyzer_eng.factory.sentiment_factory import (
            SentimentAnalyzerFactory,
        )

        return SentimentAnalyzerFactory
    if name == "Sentiments":
        from sentiment_analyzer_eng.models.sentiments import Sentiments

        return Sentiments
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
