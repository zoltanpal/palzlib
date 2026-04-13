from sentiment_analyzer_eng.analyzers.sentiment_analyzer import SentimentAnalyzer


class SentimentAnalyzerFactory:
    """
    Compatibility factory for retrieving an English sentiment analyzer.
    """

    @staticmethod
    def get_analyzer(
        language: str = "eng", model: str = "vader"
    ) -> SentimentAnalyzer:
        if language != "eng":
            raise ValueError(f"Unsupported language: {language}")
        return SentimentAnalyzer(model=model)
