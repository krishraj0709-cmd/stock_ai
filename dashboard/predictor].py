# engine/predictor.py

import numpy as np
import pandas as pd
import torch
import os

from data.ingestion.yfinance_connector import fetch_historical, fetch_fundamentals
from data.feature_store.technical import add_technical_indicators, FEATURE_COLS
from models.sentiment.finbert import SentimentAnalyzer, fetch_news_headlines
from models.intraday.lstm_model import load_lstm
from models.longterm.xgb_model import load_xgboost

# For intraday: SELL means short sell (allowed same day)
# For longterm: SELL means EXIT / AVOID (no shorting overnight)
SIGNAL_MAP_INTRADAY = {0: "SELL",  1: "HOLD", 2: "BUY"}
SIGNAL_MAP_LONGTERM = {0: "EXIT",  1: "HOLD", 2: "BUY"}


class PredictionEngine:

    def __init__(self):
        self.sentiment = SentimentAnalyzer()

        # Load LSTM if trained
        lstm_path = "models/intraday/best_lstm.pt"
        if os.path.exists(lstm_path):
            self.lstm, self.lstm_scaler, self.seq_len = load_lstm(lstm_path)
            print("  LSTM loaded.")
        else:
            self.lstm = None
            print("  LSTM not found — train first with train_all.py")

        # Load XGBoost if trained
        xgb_path = "models/longterm/xgb_model.json"
        if os.path.exists(xgb_path):
            self.xgb, self.xgb_features = load_xgboost(xgb_path)
            print("  XGBoost loaded.")
        else:
            self.xgb = None
            print("  XGBoost not found — train first with train_all.py")

    def predict(
        self,
        symbol: str,
        mode: str = "longterm"    # "intraday" or "longterm"
    ) -> dict:
        nse_symbol = symbol if symbol.endswith(".NS") else symbol + ".NS"

        # Fetch data
        period = "3mo" if mode == "intraday" else "2y"
        df     = fetch_historical(nse_symbol, period=period)
        df     = add_technical_indicators(df)

        entry_price = float(df["close"].iloc[-1])
        atr         = float(df["atr"].iloc[-1])

        # Get signal
        if mode == "intraday" and self.lstm is not None:
            signal, confidence, probs = self._intraday_signal(df)
        elif self.xgb is not None:
            signal, confidence, probs = self._longterm_signal(df, symbol)
        else:
            return {"error": "No trained model found. Run train_all.py first."}

        stop_loss, target = self._risk_levels(entry_price, atr, signal)

        return {
            "symbol":       symbol.upper(),
            "mode":         mode,
            "signal":       signal,
            "confidence":   round(confidence * 100, 1),
            "entry_price":  round(entry_price, 2),
            "target":       round(target, 2),
            "stop_loss":    round(stop_loss, 2),
            "expected_pct": round((target / entry_price - 1) * 100, 2),
            "risk_level":   self._risk_level(atr, entry_price),
            "probabilities": {
                "BUY":  round(probs[2] * 100, 1),
                "HOLD": round(probs[1] * 100, 1),
                "SELL": round(probs[0] * 100, 1),
            }
        }

    # ── Intraday (LSTM) ───────────────────────────────────────
    def _intraday_signal(self, df):
        feat = df[FEATURE_COLS].values[-self.seq_len:]
        if len(feat) < self.seq_len:
            return "HOLD", 0.5, [0.25, 0.5, 0.25]
        n, f = feat.shape
        scaled = self.lstm_scaler.transform(feat)
        X = torch.tensor(scaled[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(self.lstm(X), dim=1).numpy()[0]
        idx = int(np.argmax(probs))
        return SIGNAL_MAP_INTRADAY[idx], float(probs[idx]), probs.tolist()

    def _longterm_signal(self, df, symbol):
        headlines = fetch_news_headlines(symbol)
        row = df[self.xgb_features].iloc[-1:].copy()
        probs = self.xgb.predict_proba(row)[0]
        idx = int(np.argmax(probs))
        return SIGNAL_MAP_LONGTERM[idx], float(probs[idx]), probs.tolist()
   
    # ── Risk helpers ──────────────────────────────────────────

    def _risk_levels(self, price, atr, signal):
        mult = 2.0
        if signal == "BUY":
            return round(price - mult * atr, 2), round(price + 3 * atr, 2)
        elif signal == "SELL":
            return round(price + mult * atr, 2), round(price - 3 * atr, 2)
        return round(price * 0.97, 2), round(price * 1.03, 2)

    def _risk_level(self, atr, price):
        ratio = atr / price if price > 0 else 0
        if ratio > 0.025:
            return "HIGH"
        if ratio > 0.012:
            return "MEDIUM"
        return "LOW"


if __name__ == "__main__":
    engine = PredictionEngine()
    result = engine.predict("RELIANCE", mode="longterm")
    for k, v in result.items():
        print(f"  {k:15s}: {v}")