from palzlib.sentiment_analyzers.langs.dan.sentiment_analyzer import (
    get_sentiments as get_sentiments_dan,
)
from palzlib.sentiment_analyzers.langs.eng.sentiment_analyzer import (
    get_sentiments as get_sentiments_eng,
)
from palzlib.sentiment_analyzers.langs.hun.sentiment_analyzer import (
    get_sentiments as get_sentiments_hun,
)

__all__ = ["get_sentiments_eng", "get_sentiments_hun", "get_sentiments_dan"]
