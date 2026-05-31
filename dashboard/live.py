# dashboard/live.py
# Run with: streamlit run dashboard/live.py
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from engine.live_signals import get_intraday_signal, fetch_live_ohlcv, is_market_open
from data.feature_store.technical import add_technical_indicators

st.set_page_config(
    page_title="Live Intraday AI",
    page_icon="⚡",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("⚡ Live Intraday Signal Engine")
st.caption("Auto-refreshes every 60 seconds during market hours (9:15 AM – 3:30 PM IST)")

# ── Market status ─────────────────────────────────────────────
now = datetime.now()
if is_market_open():
    st.success("🟢 Market is OPEN — " + now.strftime("%H:%M:%S IST"))
else:
    st.warning("🔴 Market is CLOSED — Showing last available data")

# ── Sidebar controls ──────────────────────────────────────────
st.sidebar.header("Live Settings")

symbols_input = st.sidebar.text_area(
    "Watchlist (one per line)",
    "RELIANCE\nTCS\nINFY\nHDFCBANK\nICICIBANK",
    height=150
)
symbols = [s.strip().upper() for s in symbols_input.strip().split("\n") if s.strip()]

interval = st.sidebar.selectbox(
    "Candle interval",
    ["1m", "2m", "5m", "15m", "30m"],
    index=2,
    help="5m recommended for intraday signals"
)

auto_refresh = st.sidebar.toggle("Auto-refresh (60s)", value=True)
refresh_secs = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)

st.sidebar.divider()
st.sidebar.markdown("**Signal guide**")
st.sidebar.markdown("🟢 ENTER LONG — Strong buy")
st.sidebar.markdown("🔴 EXIT / SHORT — Strong sell")
st.sidebar.markdown("🟡 WEAK signal — Wait for confirmation")
st.sidebar.markdown("⚪ WAIT — No clear signal")

# ── Signal cards ──────────────────────────────────────────────
st.subheader("Live Signals — " + now.strftime("%d %b %Y %H:%M:%S"))

if st.button("🔄 Refresh Now", type="primary"):
    st.rerun()

cols = st.columns(min(len(symbols), 3))

all_signals = {}
for i, sym in enumerate(symbols):
    ticker = sym + ".NS"
    with st.spinner("Fetching " + sym + "..."):
        signal = get_intraday_signal(ticker)
    all_signals[sym] = signal

    col = cols[i % 3]
    with col:
        if "error" in signal:
            st.error(sym + ": " + signal["error"])
            continue

        action = signal["action"]
        emoji  = signal["emoji"]
        conf   = signal["confidence"]

        # Card border color
        if "ENTER" in action:
            border = "#16a34a"
        elif "EXIT" in action or "SHORT" in action:
            border = "#dc2626"
        else:
            border = "#6b7280"

        st.markdown(
            f"""
            <div style='border:2px solid {border};border-radius:10px;
                        padding:14px;margin-bottom:10px;'>
            <h3 style='margin:0;color:{border}'>{emoji} {sym}</h3>
            <h4 style='margin:4px 0;color:white'>{action}</h4>
            <p style='margin:2px 0;font-size:13px;color:#9ca3af'>
                Price: ₹{signal["price"]} &nbsp;|&nbsp;
                RSI: {signal["rsi"]} &nbsp;|&nbsp;
                Conf: {conf}%
            </p>
            <p style='margin:2px 0;font-size:12px;color:#4ade80'>
                Target 1: ₹{signal["target_1"]} &nbsp;
                Target 2: ₹{signal["target_2"]}
            </p>
            <p style='margin:2px 0;font-size:12px;color:#f87171'>
                Stop-loss: ₹{signal["stop_loss"]}
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ── Detailed view for selected stock ─────────────────────────
st.divider()
st.subheader("Detailed Analysis")
selected = st.selectbox("Select stock for detail", symbols)

if selected and selected in all_signals:
    sig = all_signals[selected]
    if "error" not in sig:
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Price",      "₹" + str(sig["price"]))
        d2.metric("RSI",        str(sig["rsi"]),
                  "Oversold" if sig["rsi"] < 30 else ("Overbought" if sig["rsi"] > 70 else "Normal"))
        d3.metric("Confidence", str(sig["confidence"]) + "%")
        d4.metric("Stop-loss",  "₹" + str(sig["stop_loss"]))
        d5.metric("ATR",        "₹" + str(sig["atr"]))

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**✅ Buy signals detected**")
            if sig["buy_signals"]:
                for s in sig["buy_signals"]:
                    st.markdown("• " + s)
            else:
                st.markdown("• None")

        with col_b:
            st.markdown("**❌ Sell signals detected**")
            if sig["sell_signals"]:
                for s in sig["sell_signals"]:
                    st.markdown("• " + s)
            else:
                st.markdown("• None")

        with col_c:
            st.markdown("**🚪 Exit conditions**")
            for s in sig["exit_conditions"]:
                st.markdown("• " + s)

        st.markdown(
            "**Volume spike:** " +
            ("✅ Yes — signal confirmed by volume" if sig["volume_spike"]
             else "❌ No — treat signal with caution")
        )

        # Live chart
        st.subheader("Live 5-min chart — " + selected)
        try:
            df = fetch_live_ohlcv(selected + ".NS", interval=interval)
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
            if "ema_9" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["ema_9"],
                    name="EMA 9", line=dict(color="#3b82f6", width=1)
                ))
            if "ema_20" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["ema_20"],
                    name="EMA 20", line=dict(color="#f59e0b", width=1)
                ))
            if "vwap" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["vwap"],
                    name="VWAP", line=dict(color="#a855f7", width=1.5, dash="dot")
                ))

            # Mark entry price
            fig.add_hline(
                y=sig["price"],
                line_dash="solid", line_color="white",
                opacity=0.4, annotation_text="Current"
            )
            fig.add_hline(
                y=sig["stop_loss"],
                line_dash="dot", line_color="red",
                opacity=0.7, annotation_text="Stop-loss"
            )
            fig.add_hline(
                y=sig["target_1"],
                line_dash="dot", line_color="green",
                opacity=0.7, annotation_text="Target 1"
            )
            fig.add_hline(
                y=sig["target_2"],
                line_dash="dot", line_color="#4ade80",
                opacity=0.5, annotation_text="Target 2"
            )

            fig.update_layout(
                height=450,
                xaxis_rangeslider_visible=False,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", y=1.05)
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning("Chart error: " + str(e))

# ── Signal history log ────────────────────────────────────────
st.divider()
st.subheader("Signal log — today")

if "signal_log" not in st.session_state:
    st.session_state.signal_log = []

for sym, sig in all_signals.items():
    if "error" not in sig and sig["action"] not in ["WAIT"]:
        st.session_state.signal_log.append({
            "Time":       sig["timestamp"],
            "Stock":      sym,
            "Signal":     sig["action"],
            "Price":      sig["price"],
            "Target 1":   sig["target_1"],
            "Stop-loss":  sig["stop_loss"],
            "Confidence": str(sig["confidence"]) + "%",
        })

if st.session_state.signal_log:
    log_df = pd.DataFrame(st.session_state.signal_log).drop_duplicates()
    st.dataframe(log_df.tail(20), use_container_width=True)
else:
    st.info("No actionable signals yet. Signals appear here when ENTER or EXIT conditions are met.")

# ── Auto refresh ──────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()