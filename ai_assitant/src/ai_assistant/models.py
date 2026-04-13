from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedPrompt:
    intent: str
    entities: List[str]
    time_range: Optional[str]
    is_comparison: bool
    raw_query: str
