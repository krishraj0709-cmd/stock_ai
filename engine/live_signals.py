# engine/live_signals.py
import time
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, time as dtime
from data.feature_store.technical import add_technical_indicators

# NSE market hours
MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def is_market_open() -> bool:
    now = datetime.now().time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def fetch_live_ohlcv(ticker: str, interval: str = "5m") -> pd.DataFrame:
    """
    Fetch live intraday data.
    interval options: 1m, 2m, 5m, 15m, 30m
    Yahoo Finance allows up to 7 days of minute data.
    """
    df = yf.download(
        ticker, period="2d",
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.dropna(inplace=True)
    return df


def get_intraday_signal(ticker: str) -> dict:
    """
    Real-time intraday signal using 5-minute candles.
    Returns entry/exit recommendation with reason.
    """
    try:
        df = fetch_live_ohlcv(ticker, interval="5m")
        if len(df) < 20:
            return {"error": "Not enough intraday data yet"}

        df = add_technical_indicators(df)
        latest = df.iloc[-1]
        prev   = df.iloc[-2]

        price   = float(latest["close"])
        rsi     = float(latest["rsi"])
        macd    = float(latest["macd"])
        macd_sig= float(latest["macd_signal"])
        macd_h  = float(latest["macd_hist"])
        bb_pct  = float(latest["bb_pct"])
        vol_r   = float(latest["vol_ratio"])
        ema9    = float(latest["ema_9"])
        ema20   = float(latest["ema_20"])
        atr     = float(latest["atr"])

        # ── Entry signals ─────────────────────────────────────
        buy_signals  = []
        sell_signals = []

        # RSI
        if rsi < 35:
            buy_signals.append("RSI oversold (" + str(round(rsi,1)) + ")")
        elif rsi > 65:
            sell_signals.append("RSI overbought (" + str(round(rsi,1)) + ")")

        # MACD crossover
        if macd > macd_sig and float(prev["macd"]) < float(prev["macd_signal"]):
            buy_signals.append("MACD bullish crossover")
        elif macd < macd_sig and float(prev["macd"]) > float(prev["macd_signal"]):
            sell_signals.append("MACD bearish crossover")

        # Bollinger Band bounce
        if bb_pct < 0.1:
            buy_signals.append("Price near lower Bollinger Band")
        elif bb_pct > 0.9:
            sell_signals.append("Price near upper Bollinger Band")

        # EMA trend
        if ema9 > ema20:
            buy_signals.append("EMA9 above EMA20 (uptrend)")
        else:
            sell_signals.append("EMA9 below EMA20 (downtrend)")

        # Volume confirmation
        vol_confirmed = vol_r > 1.3

        # ── Decision ─────────────────────────────────────────
        buy_score  = len(buy_signals)
        sell_score = len(sell_signals)

        if buy_score >= 3 and vol_confirmed:
            action     = "ENTER LONG"
            emoji      = "🟢"
            confidence = min(95, 50 + buy_score * 12)
        elif buy_score >= 2:
            action     = "WEAK BUY"
            emoji      = "🟡"
            confidence = 40 + buy_score * 8
        elif sell_score >= 3 and vol_confirmed:
            action     = "EXIT / SHORT"
            emoji      = "🔴"
            confidence = min(95, 50 + sell_score * 12)
        elif sell_score >= 2:
            action     = "WEAK SELL"
            emoji      = "🟡"
            confidence = 40 + sell_score * 8
        else:
            action     = "WAIT"
            emoji      = "⚪"
            confidence = 30

        # ── Risk levels ───────────────────────────────────────
        stop_loss = round(price - 1.5 * atr, 2)
        target1   = round(price + 2.0 * atr, 2)
        target2   = round(price + 3.5 * atr, 2)

        # ── Exit conditions ───────────────────────────────────
        exit_conditions = []
        if rsi > 70:
            exit_conditions.append("RSI above 70 — take profit")
        if bb_pct > 0.95:
            exit_conditions.append("Price at upper BB — consider exit")
        if macd_h < 0 and float(prev["macd_hist"]) > 0:
            exit_conditions.append("MACD histogram turned negative — exit long")
        if not exit_conditions:
            exit_conditions.append("Hold until target or stop-loss hit")

        return {
            "ticker":          ticker,
            "timestamp":       datetime.now().strftime("%H:%M:%S"),
            "price":           round(price, 2),
            "action":          action,
            "emoji":           emoji,
            "confidence":      confidence,
            "stop_loss":       stop_loss,
            "target_1":        target1,
            "target_2":        target2,
            "atr":             round(atr, 2),
            "rsi":             round(rsi, 1),
            "macd_hist":       round(macd_h, 4),
            "volume_spike":    vol_confirmed,
            "buy_signals":     buy_signals,
            "sell_signals":    sell_signals,
            "exit_conditions": exit_conditions,
            "market_open":     is_market_open(),
        }

    except Exception as e:
        return {"error": str(e)}