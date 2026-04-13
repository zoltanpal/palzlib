from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple, Set

import spacy
from spacy.language import Language
from spacy.tokens import Doc

from .models import ParsedPrompt

logger = logging.getLogger(__name__)

INTENT_SENTIMENT_SUMMARY = "sentiment_summary"
INTENT_TREND = "trend"
INTENT_COMPARISON = "comparison"
INTENT_TOPICS = "top_topics"
INTENT_UNKNOWN = "unknown"

# spaCy entity labels we care about in a financial news context.
RELEVANT_ENTITY_LABELS = {
    "ORG",
    "PERSON",
    "GPE",
    "PRODUCT",
    "MONEY",
    "NORP",
}

_F = re.IGNORECASE

# ---------------------------------------------------------------------------
# TIME RANGE PATTERNS
# ---------------------------------------------------------------------------
_TIME_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bday before yesterday\b", _F), "2_days_ago"),
    (re.compile(r"\byesterday\b", _F), "yesterday"),
    (re.compile(r"\btoday\b|\bright now\b|\bat the moment\b|\bcurrently\b|\bnow\b", _F), "today"),

    (re.compile(r"\bpast\s+(?:7|seven)\s+days?\b", _F), "last_week"),
    (re.compile(r"\blast\s+(?:7|seven)\s+days?\b", _F), "last_week"),
    (re.compile(r"\bthis\s+week\b", _F), "this_week"),
    (re.compile(r"\blast\s+week\b", _F), "last_week"),
    (re.compile(r"\bpast\s+week\b", _F), "last_week"),

    (re.compile(r"\bpast\s+(?:30|thirty)\s+days?\b", _F), "last_month"),
    (re.compile(r"\blast\s+(?:30|thirty)\s+days?\b", _F), "last_month"),
    (re.compile(r"\bthis\s+month\b", _F), "this_month"),
    (re.compile(r"\blast\s+month\b", _F), "last_month"),
    (re.compile(r"\bpast\s+month\b", _F), "last_month"),

    (re.compile(r"\bthis\s+year\b|\bso\s+far\s+this\s+year\b|\bYTD\b", _F), "this_year"),
    (re.compile(r"\blast\s+year\b|\bpast\s+year\b|\bpast\s+12\s+months?\b", _F), "last_year"),

    (re.compile(r"\bthis\s+quarter\b", _F), "this_quarter"),
    (re.compile(r"\blast\s+quarter\b", _F), "last_quarter"),
    (re.compile(r"\bQ[1-4]\b", _F), "this_quarter"),

    (re.compile(r"\bearnings\s+(?:season|call|report|release)\b", _F), "this_quarter"),

    (re.compile(r"\brecently\b|\blately\b|\bthese\s+days\b|\bof\s+late\b", _F), "recent"),
    (re.compile(r"\bever\b|\ball\s+time\b|\bhistorically\b|\bover\s+the\s+years\b", _F), "all_time"),
]


# ---------------------------------------------------------------------------
# COMPARISON PATTERNS
# ---------------------------------------------------------------------------
_COMPARISON_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bcompare[sd]?\b", _F),
    re.compile(r"\bvs\.?\b|\bversus\b", _F),
    re.compile(r"\bbetween\b.+\band\b", _F),
    re.compile(r"\bdifference\s+between\b", _F),
    re.compile(r"\bhow\s+does\b.+\bcompare\b", _F),
    re.compile(r"\bwhich\s+(?:is|performs?|does)\s+better\b", _F),
    re.compile(r"\b(?:better|worse)\s+than\b", _F),
    re.compile(r"\brank(?:ing|ed)?\b", _F),
    re.compile(r"\bside\s+by\s+side\b", _F),
    re.compile(r"\b\w+\s+(?:&|and)\s+\w+\s+(?:sentiment|comparison|vs|versus)\b", _F),
]


# ---------------------------------------------------------------------------
# TOPIC PATTERNS
# ---------------------------------------------------------------------------
_TOPIC_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\b(?:main|top|key|biggest|major|most\s+discussed)\s+topics?\b", _F),
    re.compile(r"\bthemes?\b", _F),
    re.compile(r"\bwhat(?:'s|\s+is|\s+are)\s+(?:people\s+)?talking\s+about\b", _F),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:buzz|narrative|chatter|discussion)\b", _F),
    re.compile(r"\btrending\s+(?:topics?|stories|news)\b", _F),
    re.compile(r"\bmost\s+mentioned\b", _F),
    re.compile(r"\bkey\s+(?:issues?|concerns?|narratives?)\b", _F),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:the\s+)?(?:story|angle|narrative)\b", _F),
    re.compile(r"\bwhat(?:'s|\s+is)\s+driving\b", _F),
    re.compile(r"\bwhy\s+is\b.+\b(?:up|down|falling|rising|dropping|surging)\b", _F),
    re.compile(r"\bwhat\s+(?:caused|triggered|sparked)\b", _F),
]


# ---------------------------------------------------------------------------
# TREND PATTERNS
# ---------------------------------------------------------------------------
_TREND_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bhow\s+has\b", _F),
    re.compile(r"\bhas\s+(?:it|this|the\s+\w+)\s+(?:changed|improved|worsened|shifted)\b", _F),
    re.compile(r"\bchanged?\b|\bchanging\b", _F),
    re.compile(r"\btrend(?:ing|ed|s)?\b", _F),
    re.compile(r"\bevol(?:ving|ved|ution)\b", _F),
    re.compile(r"\bimproving\b|\bimproved\b|\bimprovement\b", _F),
    re.compile(r"\bworsening\b|\bworsened\b", _F),
    re.compile(r"\bmoving\b|\bmovement\b|\bmomentum\b", _F),
    re.compile(r"\bshift(?:ing|ed|s)?\b", _F),
    re.compile(r"\bgrown?\b|\bgrowing\b|\bgrowth\b", _F),
    re.compile(r"\bdeclin(?:ing|ed|e)\b|\bfalling\b|\bfell\b|\bdrop(?:ping|ped)?\b", _F),
    re.compile(r"\brising\b|\brose\b|\brise\b|\bsurging\b|\bsurge[sd]?\b", _F),
    re.compile(r"\bover\s+time\b|\bover\s+the\s+(?:past|last)\b", _F),
    re.compile(r"\b(?:week|month|year)\s+over\s+(?:week|month|year)\b", _F),
    re.compile(r"\bhistory\b|\bhistorical(?:ly)?\b", _F),
    re.compile(r"\bbullish\b|\bbearish\b", _F),
    re.compile(r"\bturning\s+(?:bullish|bearish|positive|negative)\b", _F),
    re.compile(r"\bsentiment\s+(?:shift|swing|turn|change|reversal)\b", _F),
    re.compile(r"\brecovering\b|\brecovery\b|\bbouncing\s+back\b", _F),
    re.compile(r"\bpeak(?:ing|ed)?\b|\bbottom(?:ing|ed)?\b", _F),
]


# ---------------------------------------------------------------------------
# SENTIMENT SUMMARY PATTERNS
# ---------------------------------------------------------------------------
_SUMMARY_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bwhat(?:'s|\s+is|\s+are)\b", _F),
    re.compile(r"\bhow\s+(?:is|are|do|does)\b", _F),
    re.compile(r"\bsaying\s+about\b|\bthink(?:ing)?\s+about\b|\bfeel(?:ing)?\s+about\b", _F),
    re.compile(r"\bmarket\s+(?:saying|feeling|sentiment|view|opinion|reaction)\b", _F),
    re.compile(r"\bsentiment\b|\bopinion[s]?\b|\bperception\b|\battitude\b", _F),
    re.compile(r"\bpublic\s+(?:view|reaction|response|opinion)\b", _F),
    re.compile(r"\bpeople\s+(?:say|think|feel|react)\b", _F),
    re.compile(r"\boverall\s+(?:view|feeling|mood|tone|outlook)\b", _F),
    re.compile(r"\bsummary\b|\bsummariz(?:e|ing)\b|\boverview\b|\bbrief(?:ing)?\b", _F),
    re.compile(r"\banalysis\b|\banalyz(?:e|ing)\b|\banalys(?:e|ing)\b|\bassessment\b", _F),
    re.compile(r"\bnews\s+(?:about|on|for|around)\b", _F),
    re.compile(r"\bwhat(?:'s|\s+is)\s+(?:going\s+on|happening)\b", _F),
    re.compile(r"\btell\s+me\s+about\b|\bgive\s+me\s+(?:a\s+)?(?:summary|overview|update)\b", _F),
    re.compile(r"\boutlook\b|\bforecast\b|\bprospects?\b", _F),
    re.compile(r"\bwhere\s+(?:is|are|does|do)\b.+\b(?:stand|sit|head)\b", _F),
    re.compile(r"\bany\s+(?:news|updates?|developments?)\b", _F),
    re.compile(r"\bcatch\s+me\s+up\b|\bfill\s+me\s+in\b", _F),
    re.compile(r"\bw\/\b|\bwith\s+regards?\s+to\b|\bregarding\b", _F),
    re.compile(r"\bin\s+trouble\b|\bat\s+risk\b|\bunder\s+pressure\b", _F),
    re.compile(r"\bwhat\s+went\s+wrong\b|\bwhat\s+happened\s+(?:to|with)\b", _F),
    re.compile(r"\bwhy\s+(?:is|are|did|was|were)\b", _F),
]

_WHITESPACE = re.compile(r"\s+")
_TICKER_PREFIX = re.compile(r"\$(?=[A-Z])", _F)
_HASHTAG_PREFIX = re.compile(r"#(?=\w)")
_SLASH_WITH = re.compile(r"\bw/")
_AMPERSAND = re.compile(r"\s*&\s*")


class PromptParser:
    """
    Rule-based NLP parser for sentiment/news dashboard queries.
    """

    def __init__(
        self,
        model: str = "en_core_web_lg",
        relevant_labels: Optional[Set[str]] = None,
    ) -> None:
        self.relevant_labels = relevant_labels or RELEVANT_ENTITY_LABELS
        self._nlp: Language = self._load_model(model)

    @staticmethod
    def _load_model(model: str) -> Language:
        try:
            nlp = spacy.load(model, disable=["tagger", "parser", "lemmatizer"])
            logger.info("Loaded spaCy model: %s", model)
            return nlp
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' is not installed. "
                f"Run: python -m spacy download {model}"
            ) from exc

    def parse(self, query: str) -> ParsedPrompt:
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Query must be a non-empty string, got: {query!r}")

        normalized = self._normalize(query)
        doc = self._nlp(normalized)

        entities = self._extract_entities(doc)
        time_range = self._extract_time_range(normalized)
        is_comparison = self._is_comparison(normalized, entities)
        intent = self._extract_intent(normalized, is_comparison)

        if intent == INTENT_UNKNOWN:
            logger.debug("Intent undetermined for query: %r", query)

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
            if not key:
                continue

            if key not in seen:
                seen.add(key)
                entities.append(ent.text.strip())
                logger.debug("Entity found: %r [%s]", ent.text, ent.label_)

        return entities

    @staticmethod
    def _is_comparison(query: str, entities: List[str]) -> bool:
        if any(pattern.search(query) for pattern in _COMPARISON_PATTERNS):
            return True

        if len(entities) >= 2:
            logger.debug("Implicit comparison inferred from entities: %s", entities)
            return True

        return False

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

    parser = PromptParser()

    examples = [
        "What is the market saying about Tesla today?",
        "how is the market with Nvidia stock today?",
        "Compare Tesla and BYD today",
        "What are the main topics about Apple?",
        "What is happening with Microsoft recently?",
        "$TSLA sentiment this week",
        "#Apple news today",
        "NVDA w/ earnings this quarter",
        "Tesla & BYD last month",
        "Is Tesla turning bullish this week?",
        "Why is Nvidia dropping today?",
        "What went wrong with BYD last week?",
        "Elon Musk and Jensen Huang in the news this week",
        "US vs China EV sentiment this month",
        "Catch me up on Amazon this week",
        "is crypto sentiment improving today?",
    ]

    for example in examples:
        result = parser.parse(example)
        print(f"Q : {example}")
        print(result)
        print("-" * 80)        