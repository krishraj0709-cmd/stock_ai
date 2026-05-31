# training/write_dashboard.py
# Run this once: python training/write_dashboard.py

code = """\
# dashboard/app.py
import os
import sys
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingestion.yfinance_connector import fetch_historical
from data.feature_store.technical import add_technical_indicators

API_URL = "http://localhost:8000"

st.set_page_config(page_title="India Stock AI", page_icon="📈", layout="wide")
st.title("📈 India Stock Market AI")
st.caption("NSE and BSE — Buy / Sell / Hold signals — Intraday and Long-term")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("Settings")

symbol = st.sidebar.text_input(
    "Stock symbol", "RELIANCE",
    help="e.g. RELIANCE, TCS, INFY, HDFCBANK"
).upper().strip()

exchange = st.sidebar.radio(
    "Exchange",
    ["NSE", "BSE"],
    help="NSE suffix = .NS   BSE suffix = .BO"
)

mode = st.sidebar.selectbox("Prediction mode", ["longterm", "intraday"])
period = st.sidebar.selectbox("Chart period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

suffix = ".NS" if exchange == "NSE" else ".BO"
ticker_symbol = symbol + suffix

POPULAR = ["RELIANCE", "TCS", "INFY", "HDFCBANK",
           "ICICIBANK", "SBIN", "BAJFINANCE", "WIPRO"]
st.sidebar.markdown("**Quick pick:**")
cols = st.sidebar.columns(2)
for i, s in enumerate(POPULAR):
    if cols[i % 2].button(s, key=s):
        symbol = s
        ticker_symbol = symbol + suffix

# ── Predict button ────────────────────────────────────────────
predict_btn = st.button("Get Prediction", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Analysing " + symbol + " on " + exchange + "..."):
        try:
            resp = requests.post(
                API_URL + "/predict",
                json={"symbol": symbol, "mode": mode},
                timeout=60
            )
            data = resp.json()
        except Exception as e:
            st.error("API error: " + str(e) + ". Make sure uvicorn is running.")
            st.stop()

    if "error" in data:
        st.error(data["error"])
    else:
        color_map = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
        signal = data["signal"]
        st.markdown("### " + color_map[signal] + " " + signal + " — " + symbol + " (" + exchange + ")")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Entry price",  "Rs " + str(data["entry_price"]))
        c2.metric("Target",       "Rs " + str(data["target"]),
                  str(data["expected_pct"]) + "%")
        c3.metric("Stop-loss",    "Rs " + str(data["stop_loss"]))
        c4.metric("Confidence",   str(data["confidence"]) + "%")
        c5.metric("Risk level",   data["risk_level"])

        st.markdown("**Signal probabilities**")
        probs = data.get("probabilities", {})
        pc1, pc2, pc3 = st.columns(3)
        buy_val  = int(probs.get("BUY",  0))
        hold_val = int(probs.get("HOLD", 0))
        sell_val = int(probs.get("SELL", 0))
        pc1.progress(buy_val,  text="BUY  "  + str(buy_val)  + "%")
        pc2.progress(hold_val, text="HOLD " + str(hold_val) + "%")
        pc3.progress(sell_val, text="SELL "  + str(sell_val)  + "%")

        st.info(
            "Expected move: " + str(data["expected_pct"]) + "% | "
            "Risk: " + data["risk_level"] + " | "
            "Mode: " + mode
        )
        st.divider()

# ── Price chart ───────────────────────────────────────────────
st.subheader(symbol + " (" + exchange + ") — Price chart")
try:
    df = fetch_historical(ticker_symbol, period=period)
    df = add_technical_indicators(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"],
        low=df["low"],   close=df["close"],
        name="Price",
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626"
    ))
    if "ema_20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_20"],
            name="EMA 20",
            line=dict(color="#3b82f6", width=1), opacity=0.8
        ))
    if "ema_50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["ema_50"],
            name="EMA 50",
            line=dict(color="#f59e0b", width=1), opacity=0.8
        ))
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"],
            name="BB Upper",
            line=dict(color="gray", width=1, dash="dot"), opacity=0.5
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"],
            name="BB Lower",
            line=dict(color="gray", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.05)",
            opacity=0.5
        ))
    fig.update_layout(
        height=500,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    if "rsi" in df.columns:
        st.markdown("**RSI — Relative Strength Index**")
        st.caption("Above 70 = Overbought (price may fall)  |  Below 30 = Oversold (price may rise)  |  50 = Neutral")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"],
            name="RSI",
            line=dict(color="#8b5cf6", width=1.5)
        ))
        rsi_fig.add_hline(
            y=70, line_dash="dot", line_color="red",
            opacity=0.6, annotation_text="Overbought 70"
        )
        rsi_fig.add_hline(
            y=30, line_dash="dot", line_color="green",
            opacity=0.6, annotation_text="Oversold 30"
        )
        rsi_fig.update_layout(
            height=220,
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(rsi_fig, use_container_width=True)

except Exception as e:
    st.warning(
        "Could not load chart for " + ticker_symbol + ": " + str(e) +
        ". If using BSE, try switching to NSE for this stock."
    )

# ── Backtesting ───────────────────────────────────────────────
st.divider()
st.subheader("Backtesting")
st.caption(
    "Walk-forward backtest shows how the model would have performed "
    "on historical data. Higher Sharpe and CAGR is better. "
    "Lower drawdown is better."
)

bt_period = st.selectbox("Backtest period", ["1y", "2y", "3y", "5y"], index=1)
if st.button("Run Backtest"):
    with st.spinner("Running walk-forward backtest — this takes 1-2 minutes..."):
        try:
            resp = requests.post(
                API_URL + "/backtest",
                json={"symbol": symbol, "period": bt_period},
                timeout=180
            )
            stats = resp.json()
        except Exception as e:
            st.error(str(e))
            st.stop()

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Sharpe ratio",  str(stats.get("sharpe_ratio")),
              help="Above 1.0 good, above 2.0 excellent")
    b2.metric("Max drawdown",  str(stats.get("max_drawdown_pct")) + "%",
              help="Worst loss from peak. Under -20% is acceptable")
    b3.metric("CAGR",          str(stats.get("cagr_pct")) + "%",
              help="Yearly return. Above 15% beats most mutual funds")
    b4.metric("Win rate",      str(stats.get("win_rate_pct")) + "%",
              help="Trades that were profitable. Above 55% is good")

    total  = stats.get("total_trades", 0)
    pnl    = stats.get("net_pnl", 0)
    final  = stats.get("final_capital", 0)
    st.caption(
        "Total trades: " + str(total) +
        "  |  Net PnL: Rs " + str(pnl) +
        "  |  Final capital: Rs " + str(final)
    )
"""

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(code)

print("dashboard/app.py written successfully!")
print("Now run: streamlit run dashboard/app.py")