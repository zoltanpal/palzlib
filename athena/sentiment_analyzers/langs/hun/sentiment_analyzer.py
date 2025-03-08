from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from athena.sentiment_analyzers.models import LABEL_MAPPING_ROBERTA, Sentiments

# Initialize the tokenizer and model for Hungarian sentiment analysis
model_name = "NYTK/sentiment-hts5-xlm-roberta-hungarian"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create a sentiment-analysis pipeline
sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    top_k=None,  # Ensures all sentiment labels are returned
)


def get_sentiments(text: str) -> Sentiments:
    """
    Analyze the sentiment of a given text using a fine-tuned RoBERTa model.

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

    # Perform sentiment prediction using the classifier
    sentiment_prediction = sentiment_classifier(text)

    # Map the model's predictions to the Sentiments object fields
    sentiment_results = {
        LABEL_MAPPING_ROBERTA[item["label"]]: round(item["score"], 4)
        for item in sentiment_prediction[0]
    }

    # Return the Sentiments object with the mapped results
    return Sentiments(**sentiment_results)
