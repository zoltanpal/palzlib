import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import spacy
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

INTENT_SENTIMENT_SUMMARY = "sentiment_summary"
INTENT_TREND = "trend"
INTENT_COMPARISON = "comparison"
INTENT_TOPICS = "top_topics"
INTENT_UNKNOWN = "unknown"

RELEVANT_ENTITY_LABELS = {
    "ORG",
    "PERSON",
    "GPE",
    "PRODUCT",
    "MONEY",
    "NORP",
}

_F = re.IGNORECASE

@dataclass
class ParsedPrompt:
    intent: str
    entities: List[str]
    time_range: Optional[str]
    is_comparison: bool
    raw_query: str

_TIME_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bday before yesterday\b", _F), "2_days_ago"),
    (re.compile(r"\byesterday\b", _F), "yesterday"),
    (re.compile(r"\btoday\b|\bright now\b|\bat the moment\b|\bcurrently\b|\bnow\b", _F), "today"),
]

_COMPARISON_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bcompare[sd]?\b", _F),
    re.compile(r"\bvs\.?\b|\bversus\b", _F),
    re.compile(r"\bbetween\b.+\band\b", _F),
]

_TOPIC_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(?:main|top|key|biggest|major|most\s+discussed)\s+topics?\b", _F),
]

_TREND_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bhow\s+has\b", _F),
    re.compile(r"\bchanged?\b|\bchanging\b", _F),
    re.compile(r"\btrend(?:ing|ed|s)?\b", _F),
]

_SUMMARY_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bwhat(?:'s|\s+is|\s+are)\b", _F),
    re.compile(r"\bsentiment\b|\bopinion[s]?\b|\bperception\b", _F),
    re.compile(r"\bnews\s+(?:about|on|for|around)\b", _F),
]

_WHITESPACE = re.compile(r"\s+")
_TICKER_PREFIX = re.compile(r"\$(?=[A-Z])", _F)
_HASHTAG_PREFIX = re.compile(r"#(?=\w)")
_SLASH_WITH = re.compile(r"\bw/")
_AMPERSAND = re.compile(r"\s*&\s*")


class PromptParser:
    def __init__(
        self,
        model: str = "en_core_web_lg",
        relevant_labels: Optional[Set[str]] = None,
    ) -> None:
        self.relevant_labels = relevant_labels or RELEVANT_ENTITY_LABELS
        self._nlp = self._load_model(model)

    @staticmethod
    def _load_model(model: str):
        nlp = spacy.load(model, disable=["tagger", "parser", "lemmatizer"])
        logger.info("Loaded spaCy model: %s", model)
        return nlp

    def parse(self, query: str) -> ParsedPrompt:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query must be a non-empty string, got: {query!r}")

        normalized = self._normalize(query)
        doc = self._nlp(normalized)

        entities = self._extract_entities(doc)
        time_range = self._extract_time_range(normalized)
        is_comparison = self._is_comparison(normalized, entities)
        intent = self._extract_intent(normalized, is_comparison)

        return ParsedPrompt(
            intent=intent,
            entities=entities,
            time_range=time_range,
            is_comparison=is_comparison,
            raw_query=query.strip(),
        )

    @staticmethod
    def _normalize(query: str) -> str:
        query = _TICKER_PREFIX.sub("", query)
        query = _HASHTAG_PREFIX.sub("", query)
        query = _SLASH_WITH.sub("with ", query)
        query = _AMPERSAND.sub(" and ", query)
        return _WHITESPACE.sub(" ", query.strip())

    @staticmethod
    def _extract_time_range(query: str) -> Optional[str]:
        for pattern, value in _TIME_PATTERNS:
            if pattern.search(query):
                return value
        return None

    def _extract_entities(self, doc: Doc) -> List[str]:
        seen: Set[str] = set()
        entities: List[str] = []

        for ent in doc.ents:
            if ent.label_ not in self.relevant_labels:
                continue
            key = ent.text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                entities.append(ent.text.strip())

        return entities

    @staticmethod
    def _is_comparison(query: str, entities: List[str]) -> bool:
        if any(pattern.search(query) for pattern in _COMPARISON_PATTERNS):
            return True
        return len(entities) >= 2

    @staticmethod
    def _extract_intent(query: str, is_comparison: bool) -> str:
        if is_comparison:
            return INTENT_COMPARISON
        if any(pattern.search(query) for pattern in _TOPIC_PATTERNS):
            return INTENT_TOPICS
        if any(pattern.search(query) for pattern in _TREND_PATTERNS):
            return INTENT_TREND
        if any(pattern.search(query) for pattern in _SUMMARY_PATTERNS):
            return INTENT_SENTIMENT_SUMMARY
        return INTENT_UNKNOWN


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = PromptParser(model="en_core_web_lg")
    result = parser.parse("What is the market saying about Tesla today?")
    print(result)