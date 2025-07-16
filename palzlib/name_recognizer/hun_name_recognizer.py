from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


class HungarianNameRecognizer:
    def __init__(
        self, model_name: str = "NYTK/named-entity-recognition-nerkor-hubert-hungarian"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0,
        )

    def extract_names(self, text: str):
        # Run NER, then filter for person entities
        entities = self.pipeline(text)
        return [ent["word"] for ent in entities if ent["entity_group"] == "PER"]


if __name__ == "__main__":
    recognizer = HungarianNameRecognizer()
    sample = "Orbán Viktor új gazdasági csomagot jelentett be Zalaegerszegen"
    names = recognizer.extract_names(sample)
    print({"entities": names})
