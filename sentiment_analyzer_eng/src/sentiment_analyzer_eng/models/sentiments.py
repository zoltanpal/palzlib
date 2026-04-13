from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class Sentiments:
    negative: float = field(default=0.0)
    neutral: float = field(default=0.0)
    positive: float = field(default=0.0)
    compound: float = field(default=0.0)
    compound_label: str = field(default="")
    sentiment_label: str = field(default="")
    sentiment_value: float = field(default=0.0)

    def __post_init__(self) -> None:
        self.negative = round(self.negative, 4)
        self.neutral = round(self.neutral, 4)
        self.positive = round(self.positive, 4)
        self.compound = round(self.compound, 4)

        derived_label = self._label_from_compound(self.compound)
        if not self.compound_label:
            self.compound_label = derived_label
        if not self.sentiment_label:
            self.sentiment_label = self.compound_label
        if self.sentiment_value == 0.0:
            self.sentiment_value = self.compound

    @staticmethod
    def _label_from_compound(compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        if compound <= -0.05:
            return "negative"
        return "neutral"

    @classmethod
    def from_vader(cls, scores: dict) -> "Sentiments":
        return cls(
            negative=scores.get("neg", 0.0),
            neutral=scores.get("neu", 0.0),
            positive=scores.get("pos", 0.0),
            compound=scores.get("compound", 0.0),
        )

    def asdict(self) -> dict:
        return asdict(self)
