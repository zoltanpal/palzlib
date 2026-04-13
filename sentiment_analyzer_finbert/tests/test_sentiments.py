import pytest

from sentiment_analyzer_finbert.models.sentiments import Sentiments


def test_sentiments_from_finbert_positive():
    result = Sentiments.from_finbert(
        [
            {"label": "positive", "score": 0.92},
            {"label": "neutral", "score": 0.06},
            {"label": "negative", "score": 0.02},
        ]
    )

    assert result.negative == 0.02
    assert result.neutral == 0.06
    assert result.positive == 0.92
    assert result.compound == 0.9
    assert result.compound_label == "positive"
    assert result.sentiment_label == "positive"
    assert result.sentiment_value == 0.92


def test_sentiments_from_finbert_nested_pipeline_result():
    result = Sentiments.from_finbert(
        [
            [
                {"label": "negative", "score": 0.7},
                {"label": "neutral", "score": 0.2},
                {"label": "positive", "score": 0.1},
            ]
        ]
    )

    assert result.sentiment_label == "negative"
    assert result.sentiment_value == 0.7
    assert result.compound == -0.6
    assert result.compound_label == "negative"


def test_sentiments_rejects_invalid_finbert_result():
    with pytest.raises(TypeError, match="FinBERT result must be a dictionary or list"):
        Sentiments.from_finbert("positive")
