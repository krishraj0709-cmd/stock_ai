# models/sentiment/finbert.py

import torch
import requests
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class SentimentAnalyzer:
    """
    Uses ProsusAI/finbert — a BERT model fine-tuned on
    financial text. Returns positive / negative / neutral scores.
    First run will download ~400 MB model weights.
    """

    def __init__(self):
        model_name = "ProsusAI/finbert"
        print("Loading FinBERT model (first run downloads ~400MB)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.eval()
        self.labels = ["positive", "negative", "neutral"]
        print("FinBERT loaded.")

    def score(self, texts: list) -> list:
        """Returns list of dicts with positive/negative/neutral probabilities."""
        results = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()
            results.append(dict(zip(self.labels, probs)))
        return results

    def aggregate_score(self, texts: list) -> float:
        """
        Returns a single composite score:
         +1.0 = very bullish
          0.0 = neutral
         -1.0 = very bearish
        """
        if not texts:
            return 0.0
        scores = self.score(texts)
        total  = sum(s["positive"] - s["negative"] for s in scores)
        return round(total / len(scores), 4)


def fetch_news_headlines(query: str) -> list:
    """
    Fetch recent news headlines for a stock symbol.
    Set NEWS_API_KEY in your .env file (free at newsapi.org).
    Falls back to empty list if key not set.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        print("  NEWS_API_KEY not set — skipping news sentiment")
        return []

    try:
        url  = "https://newsapi.org/v2/everything"
        resp = requests.get(url, params={
            "q":        query,
            "language": "en",
            "sortBy":   "publishedAt",
            "pageSize": 20,
            "apiKey":   api_key,
        }, timeout=10)
        articles = resp.json().get("articles", [])
        return [
            a["title"] + ". " + (a.get("description") or "")
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        print(f"  News fetch failed: {e}")
        return []


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    headlines = [
        "Reliance Industries posts record quarterly profit",
        "TCS misses earnings estimates amid global slowdown",
        "HDFC Bank announces dividend, stock stays flat",
    ]
    print(analyzer.score(headlines))
    print("Composite:", analyzer.aggregate_score(headlines))