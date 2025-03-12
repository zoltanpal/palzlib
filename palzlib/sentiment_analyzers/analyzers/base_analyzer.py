from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import threading

class SentimentAnalyzerSingleton:
    _instances = {}
    _lock = threading.Lock()  # Ensures thread safety

    def __new__(cls, model_name):
        with cls._lock:
            if model_name not in cls._instances:
                instance = super().__new__(cls)
                instance._init_model(model_name)
                cls._instances[model_name] = instance
            return cls._instances[model_name]

    def _init_model(self, model_name):
        """
        Initializes the model, tokenizer, and pipeline.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,  # Ensures all sentiment labels are returned
        )

    def analyze(self, text: str):
        """
        Analyzes the sentiment of a given text.
        """
        if not text:
            raise ValueError("Missing text to analyze")

        sentiment_prediction = self.pipeline(text)
        return sentiment_prediction