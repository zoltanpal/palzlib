import threading
from typing import Iterable

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from sentiment_analyzer_eng.models.sentiments import Sentiments


class _VaderBackend:
    name = "vader"

    def __init__(self) -> None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        self._analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Sentiments:
        return Sentiments.from_vader(self._analyzer.polarity_scores(text))


class SentimentAnalyzer:
    """
    Thread-safe English sentiment analyzer.

    The public API is model-oriented so additional English backends can be added
    later without changing how callers construct or use the analyzer.
    """

    _instances: dict[str, "SentimentAnalyzer"] = {}
    _lock = threading.Lock()
    _supported_models = {
        "vader": _VaderBackend,
    }

    def __new__(cls, model: str = "vader"):
        with cls._lock:
            if model not in cls._instances:
                instance = super().__new__(cls)
                instance._initialize(model)
                cls._instances[model] = instance
        return cls._instances[model]

    def _initialize(self, model: str) -> None:
        backend_cls = self._supported_models.get(model)
        if backend_cls is None:
            supported = ", ".join(sorted(self._supported_models))
            raise ValueError(
                f"Unsupported model: {model}. Supported models: {supported}"
            )
        self.model = model
        self._backend = backend_cls()

    def _score_text(self, text: str) -> Sentiments:
        if not isinstance(text, str):
            raise TypeError("Text to analyze must be a string")

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Missing text to analyze")

        return self._backend.analyze_text(normalized_text)

    def analyze_text(self, text: str) -> Sentiments:
        return self._score_text(text)

    def analyze_batch(self, texts: Iterable[str], *, skip_invalid: bool = True) -> list[Sentiments]:
        results: list[Sentiments] = []

        for text in texts:
            try:
                results.append(self._score_text(text))
            except (TypeError, ValueError):
                if not skip_invalid:
                    raise

        return results
