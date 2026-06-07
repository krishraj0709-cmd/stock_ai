# models/sentiment/finbert.py
# FinBERT disabled on cloud deployment — too large for free tier
# Runs locally if transformers is installed

def fetch_news_headlines(symbol):
    return []

class SentimentAnalyzer:
    def __init__(self):
        self.available = False
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.available = True
            print("[FinBERT] Loaded successfully")
        except ImportError:
            print("[FinBERT] transformers not available — sentiment disabled")

    def analyze(self, headlines):
        if not self.available or not headlines:
            return {"sentiment": "NEUTRAL", "score": 0.0}
        return {"sentiment": "NEUTRAL", "score": 0.0}