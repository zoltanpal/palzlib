__all__ = [
    "SentimentAnalyzer",
    "Sentiments",
]


def __getattr__(name: str):
    if name == "SentimentAnalyzer":
        from sentiment_analyzer_finbert.analyzers.sentiment_analyzer import (
            SentimentAnalyzer,
        )

        return SentimentAnalyzer
    if name == "Sentiments":
        from sentiment_analyzer_finbert.models.sentiments import Sentiments

        return Sentiments
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
