import math
from dataclasses import asdict, dataclass, field
from typing import Optional


LABEL_MAPPING_ROBERTA = {
    "LABEL_0": "very_negative",
    "LABEL_1": "negative",
    "LABEL_2": "neutral",
    "LABEL_3": "positive",
    "LABEL_4": "very_positive",
}


@dataclass
class Sentiments:
    negative: float = field(default=0.0)
    very_negative: float = field(default=0.0)
    neutral: float = field(default=0.0)
    positive: float = field(default=0.0)
    very_positive: float = field(default=0.0)

    # None means not provided by the model
    compound: Optional[float] = field(default=None)
    compound_label: str = field(default="")

    sentiment_label: str = field(default="")
    sentiment_value: float = field(default=0.0)

    def asdict(self) -> dict:
        return asdict(self)

    def calculate_compound(self) -> float:
        score = self.positive - self.negative
        return round(math.tanh(score), 4)

    def normalize_label(self, label: str) -> str:
        if label == "very_negative":
            return "negative"
        if label == "very_positive":
            return "positive"
        return label

    def get_max_sentiment(self) -> tuple[str, float]:
        candidates = {
            "very_negative": self.very_negative,
            "negative": self.negative,
            "neutral": self.neutral,
            "positive": self.positive,
            "very_positive": self.very_positive,
        }

        label, value = max(candidates.items(), key=lambda x: x[1])
        return self.normalize_label(label), round(value, 4)

    def get_vader_label(self) -> str:
        if self.compound is None:
            return ""

        if self.compound >= 0.05:
            return "positive"
        elif self.compound <= -0.05:
            return "negative"
        return "neutral"

    def __post_init__(self):
        self.negative = round(self.negative, 4)
        self.very_negative = round(self.very_negative, 4)
        self.neutral = round(self.neutral, 4)
        self.positive = round(self.positive, 4)
        self.very_positive = round(self.very_positive, 4)

        # VADER-style result: compound explicitly provided
        if self.compound is not None:
            self.compound = round(self.compound, 4)
            self.compound_label = self.get_vader_label()
            self.sentiment_label = self.compound_label
            self.sentiment_value = self.compound
        else:
            self.compound = self.calculate_compound()
            self.compound_label = ""
            self.sentiment_label, self.sentiment_value = self.get_max_sentiment()