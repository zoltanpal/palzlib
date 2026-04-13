from ai_assistant.factory.sentiment_factory import SentimentAnalyzerFactory
from ai_assistant.models.sentiments import Sentiments


def test_sentiments_from_vader_positive():
    result = Sentiments.from_vader(
        {"neg": 0.1, "neu": 0.2, "pos": 0.7, "compound": 0.8}
    )

    assert result.negative == 0.1
    assert result.neutral == 0.2
    assert result.positive == 0.7
    assert result.compound == 0.8
    assert result.compound_label == "positive"
    assert result.sentiment_label == "positive"
    assert result.sentiment_value == 0.8


def test_factory_returns_english_analyzer():
    analyzer = SentimentAnalyzerFactory.get_analyzer()
    assert analyzer.__class__.__name__ == "SentimentAnalyzer"
    assert analyzer.model == "vader"


def test_factory_rejects_non_english_language():
    try:
        SentimentAnalyzerFactory.get_analyzer("hun")
    except ValueError as exc:
        assert str(exc) == "Unsupported language: hun"
    else:
        raise AssertionError("Expected ValueError for unsupported language")


def test_factory_rejects_unknown_model():
    try:
        SentimentAnalyzerFactory.get_analyzer(model="custom")
    except ValueError as exc:
        assert str(exc) == "Unsupported model: custom. Supported models: vader"
    else:
        raise AssertionError("Expected ValueError for unsupported model")
