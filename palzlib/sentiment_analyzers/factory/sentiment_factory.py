from palzlib.sentiment_analyzers.analyzers.dan.sentiment_analyzer import DanishSentimentAnalyzer
from palzlib.sentiment_analyzers.analyzers.eng.sentiment_analyzer import EnglishSentimentAnalyzer
from palzlib.sentiment_analyzers.analyzers.hun.sentiment_analyzer import HungarianSentimentAnalyzer
#from palzlib.sentiment_analyzers.analyzers.eng.sentiment_analyzer import EnglishSentimentAnalyzer

class SentimentAnalyzerFactory:
    _analyzers = {
        "hun": HungarianSentimentAnalyzer(),
        "dan": DanishSentimentAnalyzer(),
        "eng": EnglishSentimentAnalyzer(),
    }

    @staticmethod
    def get_analyzer(language: str):
        if language not in SentimentAnalyzerFactory._analyzers:
            raise ValueError(f"Unsupported language: {language}")
        return SentimentAnalyzerFactory._analyzers[language]