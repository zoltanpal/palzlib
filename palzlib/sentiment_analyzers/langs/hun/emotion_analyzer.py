from transformers import pipeline

# Initialize model for emotion classification
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    top_k=None,
)


def get_emotion_prediction(text: str) -> dict:
    """
    Predicts emotion scores for a given text.

    Args:
        text (str): The input text for emotion classification.

    Returns:
        dict: A dictionary containing the emotion scores for 'anger',
            'fear', 'joy', 'sadness', 'love', and 'surprise'.
    """

    if not text:
        return {}
    emotion_prediction = emotion_classifier(text)

    anger, fear, joy, sadness, love, surprise = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for prediction in emotion_prediction[0]:
        match prediction["label"]:
            case "anger":
                anger = prediction["score"]
            case "fear":
                fear = prediction["score"]
            case "joy":
                joy = prediction["score"]
            case "sadness":
                sadness = prediction["score"]
            case "love":
                love = prediction["score"]
            case "surprise":
                surprise = prediction["score"]

    return {
        "anger": anger,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "love": love,
        "surprise": surprise,
    }
