# Sentiment Analyzer FinBERT

`sentiment_analyzer_finbert` is a small Python package for financial sentiment analysis.

The package is intentionally narrow:
- English financial text
- FinBERT as the built-in backend
- Structured return values
- A stable API that can support additional finance sentiment models later

## Requirements

- Python 3.13+
- `torch`
- `transformers`

## Installation

From source:

```bash
pip install -e .
```

The analyzer downloads the default FinBERT model from Hugging Face on first use:

```text
ProsusAI/finbert
```

If your environment has no network access, download or cache the model before using the analyzer.

## Usage

The main API is `SentimentAnalyzer`:

```python
from sentiment_analyzer_finbert import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_text("Earnings beat expectations and guidance improved.")

print(result.sentiment_label)
print(result.sentiment_value)
print(result.asdict())
```

Batch analysis:

```python
from sentiment_analyzer_finbert import SentimentAnalyzer

analyzer = SentimentAnalyzer()
results = analyzer.analyze_batch(
    [
        "Margins expanded after the restructuring.",
        "The company lowered full-year revenue guidance.",
        "Management reiterated its prior outlook.",
    ]
)
```

If you want to be explicit about the backend:

```python
from sentiment_analyzer_finbert import SentimentAnalyzer

analyzer = SentimentAnalyzer(model="finbert")
```

## Result Object

Each analysis returns a `Sentiments` object with these fields:
- `negative`
- `neutral`
- `positive`
- `compound`
- `compound_label`
- `sentiment_label`
- `sentiment_value`

Example:

```python
{
    "negative": 0.01,
    "neutral": 0.04,
    "positive": 0.95,
    "compound": 0.94,
    "compound_label": "positive",
    "sentiment_label": "positive",
    "sentiment_value": 0.95,
}
```

For FinBERT, `sentiment_label` is the highest-confidence label. `compound` is calculated as `positive - negative`, which gives a compact directional finance sentiment score while keeping the original class probabilities available.
