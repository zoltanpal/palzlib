from nltk.sentiment.vader import SentimentIntensityAnalyzer as VaderSentimentAnalyzer

from palzlib.sentiment_analyzers_old.models import Sentiments

sid = VaderSentimentAnalyzer()


def get_sentiments(text: str) -> Sentiments:
    """
    Analyze the sentiment of a given text using VADER.

    Args:
        text (str): The input text to analyze.
    Returns:
        Sentiments: A Sentiments object containing the calculated
                    negative, neutral, positive, and compound scores.
    Raises:
        ValueError: If the input `text` is empty or None.
    """
    if not text:
        raise ValueError("Missing text to analyze")

    # Perform sentiment analysis
    results = sid.polarity_scores(text)

    # Map results to a Sentiments object
    return Sentiments(
        negative=results.get("neg", 0.0),
        neutral=results.get("neu", 0.0),
        positive=results.get("pos", 0.0),
        compound=results.get("compound", 0.0),
    )
