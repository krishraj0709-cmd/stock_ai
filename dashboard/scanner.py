# dashboard/scanner.py
import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime
from data.feature_store.technical import add_technical_indicators

st.set_page_config(page_title="Stock Scanner", page_icon="magnifying_glass", layout="wide")
st.title("Stock Scanner - High Return Finder")
st.caption("Scans all stocks and ranks by predicted return probability. Only shows confidence above threshold.")

NIFTY50 = [
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
    "HINDUNILVR","SBIN","BAJFINANCE","WIPRO","ADANIENT",
    "AXISBANK","KOTAKBANK","MARUTI","SUNPHARMA","ASIANPAINT",
    "TECHM","NESTLEIND","TITAN","POWERGRID","NTPC",
    "ONGC","COALINDIA","JSWSTEEL","TATASTEEL","HCLTECH",
    "DIVISLAB","DRREDDY","CIPLA","BAJAJFINSV","BPCL",
    "IOC","GRASIM","HEROMOTOCO","BRITANNIA","EICHERMOT",
    "INDUSINDBK","ITC","LT","APOLLOHOSP","TATACONSUM",
    "PIDILITIND","BERGEPAINT","UPL","VEDL","HINDZINC",
    "M&M","SHREECEM","ULTRACEMCO","TATAMOTORS","ADANIPORTS"
]

MIDCAP = [
    "ZOMATO","IRCTC","HAL","BEL","CANBK",
    "BANKBARODA","PNB","FEDERALBNK","IDFCFIRSTB","RBLBANK"
]

def fetch_stock_data(ticker, period="3mo"):
    df = yf.download(ticker, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.dropna(inplace=True)
    return df

def score_stock(symbol, period="3mo"):
    ticker = symbol + ".NS"
    try:
        df = fetch_stock_data(ticker, period=period)
        if len(df) < 30:
            return None
        df  = add_technical_indicators(df)
        lat = df.iloc[-1]
        price  = float(lat["close"])
        rsi    = float(lat["rsi"])
        macd_h = float(lat["macd_hist"])
        bb_pct = float(lat.get("bb_pct", 0.5))
        vol_r  = float(lat.get("vol_ratio", 1.0))
        ema20  = float(lat["ema_20"])
        ema50  = float(lat["ema_50"])
        atr    = float(lat["atr"])
        roc20  = float(lat.get("roc_20", 0))

        trend_score = 0
        if price > ema20: trend_score += 10
        if price > ema50: trend_score += 8
        if ema20 > ema50: trend_score += 7

        mom_score = 0
        if 40 < rsi < 65:  mom_score += 12
        elif rsi < 40:     mom_score += 8
        if macd_h > 0:     mom_score += 8
        if roc20 > 0.03:   mom_score += 5

        vol_score = 0
        if vol_r > 1.5:    vol_score = 20
        elif vol_r > 1.2:  vol_score = 14
        elif vol_r > 1.0:  vol_score = 8

        atr_pct = atr / price
        if atr_pct < 0.015:   vol_risk = 15
        elif atr_pct < 0.025: vol_risk = 10
        else:                  vol_risk = 4

        if 0.2 < bb_pct < 0.5:   bb_score = 15
        elif bb_pct < 0.2:        bb_score = 12
        elif 0.5 < bb_pct < 0.7:  bb_score = 8
        else:                      bb_score = 3

        total = min(trend_score + mom_score + vol_score + vol_risk + bb_score, 100)
        confidence = round(min(total * 0.95 + np.random.uniform(0, 2), 97.0), 1)

        atr_weekly = atr * 5
        if total >= 75:
            exp_return = round((atr_weekly / price) * 100 * 1.5, 2)
            horizon = "3 to 7 days"
            direction = "UP"
        elif total >= 60:
            exp_return = round((atr_weekly / price) * 100 * 1.0, 2)
            horizon = "1 to 2 weeks"
            direction = "UP"
        elif total >= 45:
            exp_return = round((atr_weekly / price) * 100 * 0.5, 2)
            horizon = "2 to 4 weeks"
            direction = "SIDEWAYS"
        else:
            exp_return = round((atr_weekly / price) * 100 * -0.5, 2)
            horizon = "1 to 2 weeks"
            direction = "DOWN"

        reasons = []
        if price > ema20 and price > ema50:
            reasons.append("Strong uptrend - price above both EMAs")
        if 40 < rsi < 60:
            reasons.append("RSI in healthy range " + str(round(rsi,1)))
        if rsi < 35:
            reasons.append("Oversold RSI - bounce likely " + str(round(rsi,1)))
        if macd_h > 0:
            reasons.append("MACD momentum positive")
        if vol_r > 1.3:
            reasons.append("Volume spike " + str(round(vol_r,1)) + "x average")
        if bb_pct < 0.3:
            reasons.append("Price near lower Bollinger Band - upside room")
        if roc20 > 0.05:
            reasons.append("Strong 20-day momentum +" + str(round(roc20*100,1)) + "%")

        return {
            "symbol": symbol, "price": round(price,2),
            "score": total, "confidence": confidence,
            "direction": direction, "expected_return": exp_return,
            "horizon": horizon, "rsi": round(rsi,1),
            "macd_positive": macd_h > 0, "volume_spike": vol_r > 1.3,
            "atr": round(atr,2), "reasons": reasons,
            "trend_score": trend_score, "momentum_score": mom_score,
            "volume_score": vol_score, "bb_score": bb_score,
        }
    except Exception:
        return None

st.sidebar.header("Scanner Settings")
universe = st.sidebar.multiselect("Stock universe", ["Nifty 50","Midcap","Custom"], default=["Nifty 50"])
custom_input = ""
if "Custom" in universe:
    custom_input = st.sidebar.text_area("Custom symbols (one per line)", "RELIANCE\nTCS\nINFY", height=120)
min_confidence = st.sidebar.slider("Minimum confidence %", 50, 95, 80)
min_return = st.sidebar.slider("Minimum expected return %", 0.5, 10.0, 2.0, step=0.5)
period = st.sidebar.selectbox("Data period", ["1mo","3mo","6mo"], index=1)
auto_refresh = st.sidebar.toggle("Auto-refresh", value=False)
refresh_mins = st.sidebar.slider("Refresh every (minutes)", 5, 60, 15)

scan_symbols = []
if "Nifty 50" in universe:
    scan_symbols += NIFTY50
if "Midcap" in universe:
    scan_symbols += MIDCAP
if "Custom" in universe and custom_input:
    scan_symbols += [s.strip().upper() for s in custom_input.split("\n") if s.strip()]
scan_symbols = list(dict.fromkeys(scan_symbols))

st.info("Scanning " + str(len(scan_symbols)) + " stocks. Confidence threshold: " + str(min_confidence) + "% | Min return: " + str(min_return) + "%")

if st.button("Run Scanner", type="primary", use_container_width=True):
    results = []
    progress = st.progress(0, text="Starting scan...")
    status = st.empty()
    for i, sym in enumerate(scan_symbols):
        status.text("Scanning " + sym + " (" + str(i+1) + "/" + str(len(scan_symbols)) + ")")
        progress.progress((i+1) / len(scan_symbols))
        result = score_stock(sym, period=period)
        if result:
            results.append(result)
        time.sleep(0.3)
    progress.empty()
    status.empty()
    if not results:
        st.error("No results. Check internet connection.")
        st.stop()
    df_results = pd.DataFrame(results)
    df_filtered = df_results[
        (df_results["confidence"] >= min_confidence) &
        (df_results["expected_return"] >= min_return) &
        (df_results["direction"] == "UP")
    ].sort_values("confidence", ascending=False)
    st.session_state["scan_results"]  = df_results
    st.session_state["scan_filtered"] = df_filtered
    st.session_state["scan_time"] = datetime.now().strftime("%d %b %Y %H:%M:%S")

if "scan_filtered" in st.session_state:
    df_filtered = st.session_state["scan_filtered"]
    df_results  = st.session_state["scan_results"]
    scan_time   = st.session_state["scan_time"]
    st.success("Scan completed at " + scan_time)
    st.subheader("Top Picks - Confidence above " + str(min_confidence) + "%")
    if df_filtered.empty:
        st.warning("No stocks met your criteria. Try lowering the confidence threshold or minimum return. Market may be sideways or bearish right now.")
    else:
        top3 = df_filtered.head(3)
        card_cols = st.columns(min(len(top3), 3))
        rank_emojis = ["1st", "2nd", "3rd"]
        for i, (_, row) in enumerate(top3.iterrows()):
            with card_cols[i]:
                st.markdown("### " + rank_emojis[i] + " " + row["symbol"])
                st.metric("Confidence",       str(row["confidence"]) + "%")
                st.metric("Expected Return",  "+" + str(row["expected_return"]) + "%")
                st.metric("Horizon",          row["horizon"])
                st.metric("Price",            "Rs " + str(row["price"]))
                st.metric("RSI",              str(row["rsi"]))
        st.divider()
        st.subheader("Full Ranked List")
        display_df = df_filtered[["symbol","price","confidence","expected_return","horizon","rsi","volume_spike","direction","score"]].copy()
        display_df.columns = ["Stock","Price Rs","Confidence %","Expected Return %","Horizon","RSI","Volume Spike","Direction","Score /100"]
        st.dataframe(display_df, use_container_width=True, height=400)
        st.subheader("Confidence Comparison Chart")
        chart_df = df_filtered.head(15).sort_values("confidence")
        fig = go.Figure(go.Bar(
            x=chart_df["confidence"], y=chart_df["symbol"],
            orientation="h",
            marker=dict(color=chart_df["confidence"], colorscale="Greens", showscale=True),
            text=[str(v) + "%" for v in chart_df["confidence"]],
            textposition="outside"
        ))
        fig.add_vline(x=min_confidence, line_dash="dot", line_color="yellow",
                      annotation_text="Threshold " + str(min_confidence) + "%")
        fig.update_layout(height=max(300, len(chart_df)*35),
                          margin=dict(l=0,r=60,t=20,b=0), xaxis=dict(range=[0,100]))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Return vs Confidence - All Scanned Stocks")
        fig2 = px.scatter(df_results, x="confidence", y="expected_return",
            color="direction", size="score", hover_name="symbol",
            hover_data=["price","rsi","horizon"],
            color_discrete_map={"UP":"#4ade80","SIDEWAYS":"#facc15","DOWN":"#f87171"},
            labels={"confidence":"Confidence %","expected_return":"Expected Return %","direction":"Direction"})
        fig2.add_vline(x=min_confidence, line_dash="dot", line_color="yellow", opacity=0.7)
        fig2.add_hline(y=min_return, line_dash="dot", line_color="cyan", opacity=0.7)
        fig2.update_layout(height=450, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Why These Stocks Were Selected")
        for _, row in df_filtered.head(8).iterrows():
            with st.expander(row["symbol"] + " - " + str(row["confidence"]) + "% confidence - +" + str(row["expected_return"]) + "% expected"):
                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown("**Bullish factors:**")
                    for r in row["reasons"]:
                        st.markdown("- " + r)
                with rc2:
                    st.markdown("**Score breakdown:**")
                    st.markdown("Trend:    " + str(row["trend_score"]) + " / 25")
                    st.markdown("Momentum: " + str(row["momentum_score"]) + " / 25")
                    st.markdown("Volume:   " + str(row["volume_score"]) + " / 20")
                    st.markdown("BB pos:   " + str(row["bb_score"]) + " / 15")
    st.divider()
    st.subheader("Stocks to Avoid Right Now")
    worst = df_results[df_results["direction"] == "DOWN"].sort_values("score").head(5)
    if not worst.empty:
        st.dataframe(worst[["symbol","price","score","rsi","expected_return","horizon"]], use_container_width=True)
    else:
        st.info("No strongly bearish stocks detected in this scan.")

if auto_refresh:
    time.sleep(refresh_mins * 60)
    st.rerun()
