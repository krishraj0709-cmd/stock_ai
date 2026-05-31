# data/feature_store/technical.py

import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds RSI, MACD, Bollinger Bands, VWAP, ATR, EMAs, volume features.
    Input df must have columns: open, high, low, close, volume
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── RSI (14) ──────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands (20, 2σ) ──────────────────────────────
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid
    df["bb_pct"]   = (c - df["bb_lower"]) / (
        (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    )

    # ── VWAP ──────────────────────────────────────────────────
    typical_price = (h + l + c) / 3
    df["vwap"] = (typical_price * v).cumsum() / v.cumsum()

    # ── ATR (14) ──────────────────────────────────────────────
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # ── EMAs ──────────────────────────────────────────────────
    for span in [9, 20, 50, 200]:
        df[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()

    # ── Volume features ───────────────────────────────────────
    df["vol_sma20"]    = v.rolling(20).mean()
    df["vol_ratio"]    = v / df["vol_sma20"].replace(0, np.nan)
    df["price_change"] = c.pct_change()
    df["log_return"]   = np.log(c / c.shift(1))

    # ── Candlestick features ──────────────────────────────────
    df["candle_body"]  = (c - df["open"]).abs()
    df["candle_range"] = h - l
    df["upper_shadow"] = h - pd.concat([c, df["open"]], axis=1).max(axis=1)
    df["lower_shadow"] = pd.concat([c, df["open"]], axis=1).min(axis=1) - l

    # ── Momentum ──────────────────────────────────────────────
    df["roc_5"]  = c.pct_change(5)
    df["roc_10"] = c.pct_change(10)
    df["roc_20"] = c.pct_change(20)

    return df.dropna()


FEATURE_COLS = [
    "rsi", "macd", "macd_hist", "bb_pct", "bb_width",
    "atr", "vol_ratio", "ema_9", "ema_20", "ema_50",
    "price_change", "log_return", "candle_body",
    "candle_range", "upper_shadow", "lower_shadow",
    "roc_5", "roc_10", "roc_20",
]


if __name__ == "__main__":
    from data.ingestion.yfinance_connector import fetch_historical
    df = fetch_historical("TCS.NS", period="6mo")
    df = add_technical_indicators(df)
    print(df[FEATURE_COLS].tail())
    print(f"\nFeatures: {len(FEATURE_COLS)}, Rows: {len(df)}")