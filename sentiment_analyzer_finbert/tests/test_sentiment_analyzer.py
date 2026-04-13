import pytest

from sentiment_analyzer_finbert import SentimentAnalyzer


def test_analyzer_rejects_unknown_model_before_loading_backend():
    with pytest.raises(
        ValueError, match="Unsupported model: custom. Supported models: finbert"
    ):
        SentimentAnalyzer(model="custom")
