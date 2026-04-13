from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Sentiments:
    negative: float = field(default=0.0)
    neutral: float = field(default=0.0)
    positive: float = field(default=0.0)
    compound: float = field(default=0.0)
    compound_label: str = field(default="")

    def __post_init__(self) -> None:

        # if not self.compound_label:
        #     self.compound_label = self._label_from_compound(self.compound)
        # if not self.sentiment_label:
        #     self.sentiment_label, self.sentiment_value = self._max_sentiment()
        pass

    @classmethod
    def from_finbert(cls, result: Any) -> "Sentiments":
        scores = cls._scores_by_label(result)
        positive = scores.get("positive", 0.0)
        negative = scores.get("negative", 0.0)

        return cls(
            negative=negative,
            neutral=scores.get("neutral", 0.0),
            positive=positive,
            compound=positive - negative,
        )

    @classmethod
    def _scores_by_label(cls, result: Any) -> dict[str, float]:
        rows = cls._flatten_pipeline_result(result)
        scores: dict[str, float] = {}


        print(scores)

        for row in rows:
            label = str(row.get("label", "")).lower()
            if label in {"negative", "neutral", "positive"}:
                scores[label] = float(row.get("score", 0.0))

        return scores

    @classmethod
    def _flatten_pipeline_result(cls, result: Any) -> list[dict[str, Any]]:
        if isinstance(result, dict):
            return [result]

        if not isinstance(result, list):
            raise TypeError("FinBERT result must be a dictionary or list")

        if not result:
            return []

        if all(isinstance(item, dict) for item in result):
            return result

        if len(result) == 1:
            return cls._flatten_pipeline_result(result[0])

        flattened: list[dict[str, Any]] = []
        for item in result:
            flattened.extend(cls._flatten_pipeline_result(item))
        return flattened


    '''
    @staticmethod
    def _label_from_compound(compound: float) -> str:
        if compound > 0:
            return "positive"
        if compound < 0:
            return "negative"
        return "neutral"

    def _max_sentiment(self) -> tuple[str, float]:
        label, value = max(
            {
                "negative": self.negative,
                "neutral": self.neutral,
                "positive": self.positive,
            }.items(),
            key=lambda item: item[1],
        )
        return label, round(value, 4)






    '''

    def asdict(self) -> dict:
        return asdict(self)
