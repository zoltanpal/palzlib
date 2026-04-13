import threading
from collections.abc import Iterable
from dataclasses import dataclass

from sentiment_analyzer_finbert.models.sentiments import Sentiments


@dataclass(frozen=True, slots=True)
class _ModelConfig:
    model_id: str


class _FinbertBackend:
    name = "finbert"

    def __init__(self, model_id: str) -> None:
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT analysis requires the 'transformers' and 'torch' packages"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self._classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
        )

    def analyze_text(self, text: str) -> Sentiments:
        return Sentiments.from_finbert(self._classifier(text))

    def analyze_batch(self, texts: list[str]) -> list[Sentiments]:
        raw_results = self._classifier(texts)
        return [Sentiments.from_finbert(result) for result in raw_results]


class SentimentAnalyzer:
    """
    Thread-safe financial sentiment analyzer.

    The public API is model-oriented so additional finance sentiment backends can
    be added later without changing how callers construct or use the analyzer.
    """

    _instances: dict[tuple[str, str | None], "SentimentAnalyzer"] = {}
    _lock = threading.Lock()
    _supported_models = {
        "finbert": _ModelConfig(model_id="ProsusAI/finbert"),
    }

    def __new__(cls, model: str = "finbert", model_id: str | None = None):
        cache_key = (model, model_id)
        with cls._lock:
            if cache_key not in cls._instances:
                instance = super().__new__(cls)
                instance._initialize(model, model_id)
                cls._instances[cache_key] = instance
        return cls._instances[cache_key]

    def _initialize(self, model: str, model_id: str | None) -> None:
        config = self._supported_models.get(model)
        if config is None:
            supported = ", ".join(sorted(self._supported_models))
            raise ValueError(
                f"Unsupported model: {model}. Supported models: {supported}"
            )

        self.model = model
        self.model_id = model_id or config.model_id
        self._backend = _FinbertBackend(self.model_id)

    def _normalize_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Text to analyze must be a string")

        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Missing text to analyze")

        return normalized_text

    def analyze_text(self, text: str) -> Sentiments:
        return self._backend.analyze_text(self._normalize_text(text))

    def analyze_batch(
        self, texts: Iterable[str], *, skip_invalid: bool = True
    ) -> list[Sentiments]:
        normalized_texts: list[str] = []

        for text in texts:
            try:
                normalized_texts.append(self._normalize_text(text))
            except (TypeError, ValueError):
                if not skip_invalid:
                    raise

        if not normalized_texts:
            return []

        return self._backend.analyze_batch(normalized_texts)
