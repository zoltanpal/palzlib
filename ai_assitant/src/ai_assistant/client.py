from openai import OpenAI

class OpenAIAssistant:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required to initialize OpenAIAssistant")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4.1-mini"

    def send_message(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt
        )
        return response.output_text
    

assistant = OpenAIAssistant(api_key="")

examples = [
    "Sugar Prices Fall as Crude Oil Slumps",
    "U.S. Postal Service seeks 8% fuel surcharge for package deliveries as Iran war raises oil prices",
    "Gold Price Rebounds Toward $4,550 As Oil Slide Revives Safe-Haven Bid",
    "Crude Oil Weakness Pressures Sugar Prices",
    "Crude Oil Prices Fall on the Outlook for a US-Iran Truce",
    "Goldman Sachs: US stocks rise as oil prices retreat from recent highs",
    "The Strait of Hormuz Blockade Is Affecting More Than Just Oil Prices. Here Are 4 Stocks That Could Get Hit in 2026.",
    "Mozambique Dollar Bond Selloff Extends as Oil Price Shock Deepens Crisis",
]

query = "oil prices"
window_hours = 6
prompt = f"""
    These headlines are about {query} topic.
    The headlines are the last {window_hours} hours.
    Please provide a concise summary of the following headlines in 2-3 sentences.
"""
   
response = assistant.send_message(f"{prompt}\n:{examples}")
print(response)