# dashboard/scanner.py
# Standalone scanner page — run with:
# streamlit run dashboard/scanner.py --server.port 8503
#
# NOTE: Scanner now calls the /scan API endpoint which uses the same
# unified signal engine as /predict — no more conflict between the two.

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="SignalAI — Scanner",
    page_icon="📡",
    layout="wide"
)

# ── PREMIUM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif!important;background:#03050a!important;color:#e8eaf0!important}
#MainMenu,footer,header,.stDeployButton{display:none!important}
.block-container{padding:1.5rem 2rem!important;max-width:1400px!important}
.stApp{background:#03050a!important}
section[data-testid="stSidebar"]{background:#060c14!important;border-right:1px solid rgba(255,255,255,0.07)!important}
.stButton>button{background:linear-gradient(135deg,rgba(0,245,160,0.1),rgba(0,212,255,0.05))!important;border:1px solid rgba(0,245,160,0.25)!important;border-radius:10px!important;color:#00f5a0!important;font-family:'Syne',sans-serif!important;font-weight:600!important}
.stButton>button:hover{background:linear-gradient(135deg,rgba(0,245,160,0.18),rgba(0,212,255,0.10))!important;border-color:rgba(0,245,160,0.45)!important;box-shadow:0 8px 32px rgba(0,245,160,0.12)!important}
[data-testid="stMetric"]{background:rgba(255,255,255,0.03)!important;border:1px solid rgba(255,255,255,0.07)!important;border-radius:14px!important;padding:16px 20px!important}
[data-testid="stMetricLabel"]{font-size:10px!important;color:#5a6478!important;text-transform:uppercase!important;letter-spacing:1px!important;font-family:'JetBrains Mono',monospace!important}
[data-testid="stMetricValue"]{font-size:22px!important;font-weight:700!important;font-family:'JetBrains Mono',monospace!important}
hr{border-color:rgba(255,255,255,0.06)!important}
</style>
""", unsafe_allow_html=True)


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:24px 0 8px 0">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
    <div style="width:8px;height:8px;background:#00f5a0;border-radius:50%;
    box-shadow:0 0 8px #00f5a0;animation:pulse 1.5s infinite"></div>
    <span style="font-size:22px;font-weight:800;letter-spacing:-0.5px">SignalAI Scanner</span>
    <span style="background:rgba(0,245,160,0.08);border:1px solid rgba(0,245,160,0.18);
    border-radius:100px;padding:3px 12px;font-size:10px;color:#00f5a0;
    font-family:'JetBrains Mono',monospace;letter-spacing:2px">HIGH RETURN FINDER</span>
  </div>
  <div style="font-size:12px;color:#5a6478;font-family:'JetBrains Mono',monospace">
    Uses the same unified model as individual stock search — consistent signals, no conflict
  </div>
</div>
""", unsafe_allow_html=True)


def section_title(label):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:24px 0 12px 0">
      <span style="font-size:10px;color:#5a6478;letter-spacing:2px;text-transform:uppercase;
      font-family:'JetBrains Mono',monospace">{label}</span>
      <div style="flex:1;height:1px;background:rgba(255,255,255,0.06)"></div>
    </div>""", unsafe_allow_html=True)

def signal_pill(sig):
    colors = {
        "BUY":  ("#00f5a0","rgba(0,245,160,0.1)","rgba(0,245,160,0.2)"),
        "SELL": ("#ff3b6b","rgba(255,59,107,0.1)","rgba(255,59,107,0.2)"),
        "HOLD": ("#ffb830","rgba(255,184,48,0.1)","rgba(255,184,48,0.2)"),
    }
    c,bg,br = colors.get(sig,("#8892a4","rgba(255,255,255,0.05)","rgba(255,255,255,0.1)"))
    return f"""<span style="background:{bg};border:1px solid {br};color:{c};
    padding:3px 12px;border-radius:6px;font-size:11px;font-weight:700;
    font-family:'JetBrains Mono',monospace;letter-spacing:1px">{sig}</span>"""


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px 0;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:16px">
      <div style="font-size:13px;font-weight:700">Scanner Settings</div>
      <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;margin-top:2px">
        All signals go through unified engine
      </div>
    </div>
    """, unsafe_allow_html=True)

    universe = st.multiselect(
        "Stock Universe",
        ["NIFTY50", "MIDCAP"],
        default=["NIFTY50"]
    )

    min_confidence = st.slider(
        "Min Confidence %", 50, 90, 65,
        help="Unified threshold — same gate as individual search"
    )

    min_return = st.slider(
        "Min Expected Return %", 0.5, 10.0, 1.5, step=0.5
    )

    limit = st.slider(
        "Max stocks to scan", 5, 50, 20
    )

    period = st.selectbox(
        "Analysis Period", ["1mo", "3mo", "6mo"], index=1
    )

    st.divider()
    st.markdown("""
    <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;line-height:1.6">
    ⚡ Scanner calls /scan API<br>
    📡 Same model as /predict<br>
    🎯 Confidence calibrated<br>
    🌡 Regime filter applied
    </div>
    """, unsafe_allow_html=True)


# ── CHECK API STATUS ───────────────────────────────────────────────────────────
try:
    health = requests.get(f"{API_URL}/health", timeout=4).json()
    api_ok = True
    regime_str = health.get("regime", "NEUTRAL")
except Exception:
    api_ok = False
    regime_str = "UNKNOWN"

if not api_ok:
    st.error("⚠️ FastAPI backend not running. Start it with: `uvicorn api.main:app --reload`")
    st.stop()

# Show regime status
rc = {"BULLISH":"#00f5a0","BEARISH":"#ff3b6b","NEUTRAL":"#ffb830"}.get(regime_str,"#5a6478")
st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
border-radius:12px;padding:12px 20px;margin-bottom:20px;
display:flex;align-items:center;gap:20px;flex-wrap:wrap">
  <div style="display:flex;align-items:center;gap:8px">
    <div style="width:6px;height:6px;background:{rc};border-radius:50%;box-shadow:0 0 6px {rc}"></div>
    <span style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;
    letter-spacing:1px;text-transform:uppercase">Market Regime</span>
    <span style="font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace;
    color:{rc}">{regime_str}</span>
  </div>
  <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace">
    API ✓ Connected &nbsp;·&nbsp;
    Models: LSTM {'✓' if health.get('lstm_loaded') else '✗'} &nbsp;
    XGB {'✓' if health.get('xgb_loaded') else '✗'}
  </div>
</div>
""", unsafe_allow_html=True)


# ── RUN SCAN ───────────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([2, 3])
with col_btn:
    run_scan = st.button("⚡  Run Scanner", type="primary", use_container_width=True)
with col_info:
    st.markdown(f"""
    <div style="font-size:12px;color:#5a6478;font-family:'JetBrains Mono',monospace;padding:10px 0">
    Scanning up to {limit} stocks &nbsp;·&nbsp;
    Min confidence: {min_confidence}% &nbsp;·&nbsp;
    Min return: {min_return}%
    </div>""", unsafe_allow_html=True)

if run_scan:
    with st.spinner("Running scan via unified signal engine…"):
        try:
            resp = requests.post(
                f"{API_URL}/scan",
                json={
                    "universe":       universe,
                    "min_confidence": min_confidence,
                    "min_return":     min_return,
                    "limit":          limit,
                    "period":         period,
                },
                timeout=300
            )
            scan_data = resp.json()
        except Exception as e:
            st.error(f"Scan failed: {e}")
            st.stop()

    st.session_state["scan_results"] = scan_data
    st.session_state["scan_time"]    = datetime.now().strftime("%d %b %Y %H:%M:%S")


# ── DISPLAY RESULTS ────────────────────────────────────────────────────────────
if "scan_results" in st.session_state:
    data      = st.session_state["scan_results"]
    scan_time = st.session_state["scan_time"]
    results   = data.get("results", [])
    total     = data.get("total_scanned", 0)
    found     = data.get("signals_found", 0)
    regime    = data.get("market_regime", "NEUTRAL")

    # ── Summary metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stocks Scanned",   total)
    m2.metric("Signals Found",    found)
    m3.metric("Market Regime",    regime)
    m4.metric("Scan Time",        scan_time.split(" ")[-1])

    if not results:
        st.warning(
            f"No stocks met your criteria (confidence ≥ {min_confidence}%, "
            f"return ≥ {min_return}%). "
            "Try lowering the thresholds or the market may be ranging."
        )
        st.stop()

    # ── Top 3 highlighted cards ────────────────────────────────────────────────
    section_title("Top Picks")
    top3 = results[:3]
    card_cols = st.columns(len(top3))

    for i, row in enumerate(top3):
        rank = ["🥇","🥈","🥉"][i]
        sig  = row["signal"]
        sc   = {"BUY":"#00f5a0","SELL":"#ff3b6b"}.get(sig,"#ffb830")
        bg   = {"BUY":"rgba(0,245,160,0.06)","SELL":"rgba(255,59,107,0.06)"}.get(sig,"rgba(255,184,48,0.06)")
        brd  = {"BUY":"rgba(0,245,160,0.2)","SELL":"rgba(255,59,107,0.2)"}.get(sig,"rgba(255,184,48,0.2)")
        conf = row["confidence"]
        rc_  = "✅" if row.get("regime_confirmed") else "⚠️"

        with card_cols[i]:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {brd};border-radius:16px;
            padding:20px;text-align:center;position:relative;overflow:hidden">
              <div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;
              border-radius:50%;background:{sc};filter:blur(40px);opacity:0.15"></div>
              <div style="font-size:28px;margin-bottom:6px">{rank}</div>
              <div style="font-size:20px;font-weight:800;color:{sc};margin-bottom:4px">{row['symbol']}</div>
              {signal_pill(sig)}
              <div style="font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;
              color:{sc};margin:10px 0">{conf}%</div>
              <div style="font-size:12px;color:#8892a4;font-family:'JetBrains Mono',monospace">
                Expected: +{row['expected_pct']}%<br>
                Entry: ₹{row['entry_price']} &nbsp;·&nbsp; RSI: {row['rsi']}<br>
                {rc_} Regime {row['regime']}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Full ranked table ──────────────────────────────────────────────────────
    section_title(f"All {found} Signals — Ranked by Confidence")

    header = """
    <div style="display:grid;grid-template-columns:100px 80px 90px 90px 90px 90px 60px 80px 90px;
    gap:8px;padding:8px 16px;border-bottom:1px solid rgba(255,255,255,0.06);
    font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;
    letter-spacing:1px;text-transform:uppercase">
      <span>Symbol</span><span>Signal</span><span>Confidence</span>
      <span>Entry ₹</span><span>Target ₹</span><span>Stop Loss</span>
      <span>RSI</span><span>Regime</span><span>Risk</span>
    </div>"""

    rows_html = ""
    for r in results:
        sig  = r["signal"]
        sc   = {"BUY":"#00f5a0","SELL":"#ff3b6b","HOLD":"#ffb830"}.get(sig,"#8892a4")
        conf = r["confidence"]
        conf_c = "#00f5a0" if conf >= 75 else "#ffb830" if conf >= 62 else "#ff3b6b"
        rc_  = {"BULLISH":"#00f5a0","BEARISH":"#ff3b6b","NEUTRAL":"#ffb830"}.get(r["regime"],"#5a6478")

        rows_html += f"""
        <div style="display:grid;grid-template-columns:100px 80px 90px 90px 90px 90px 60px 80px 90px;
        gap:8px;padding:11px 16px;border-bottom:1px solid rgba(255,255,255,0.03);
        font-family:'JetBrains Mono',monospace;font-size:12px"
        onmouseover="this.style.background='rgba(255,255,255,0.02)'"
        onmouseout="this.style.background='transparent'">
          <span style="font-weight:700">{r['symbol']}</span>
          <span>{signal_pill(sig)}</span>
          <span style="color:{conf_c};font-weight:700">{conf}%</span>
          <span>₹{r['entry_price']}</span>
          <span style="color:#00f5a0">₹{r['target']}</span>
          <span style="color:#ff3b6b">₹{r['stop_loss']}</span>
          <span style="color:{'#00f5a0' if r['rsi']<50 else '#ff3b6b'}">{r['rsi']}</span>
          <span style="color:{rc_}">{r['regime']}</span>
          <span style="color:#5a6478">{r.get('risk_level','--')}</span>
        </div>"""

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:14px;overflow:hidden">{header}{rows_html}</div>
    """, unsafe_allow_html=True)

    # ── Confidence chart ───────────────────────────────────────────────────────
    if len(results) > 3:
        section_title("Confidence Comparison")
        df_chart = pd.DataFrame(results).head(15).sort_values("confidence")
        sig_colors = [
            "#00f5a0" if s == "BUY" else "#ff3b6b"
            for s in df_chart["signal"]
        ]
        fig = go.Figure(go.Bar(
            x=df_chart["confidence"],
            y=df_chart["symbol"],
            orientation="h",
            marker_color=sig_colors,
            text=[f"{v}%" for v in df_chart["confidence"]],
            textposition="outside"
        ))
        fig.add_vline(
            x=min_confidence, line_dash="dot",
            line=dict(color="rgba(255,184,48,0.6)", width=1.5),
            annotation_text=f"Threshold {min_confidence}%",
            annotation_font=dict(color="rgba(255,184,48,0.8)", size=10)
        )
        fig.update_layout(
            height=max(280, len(df_chart) * 36),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0,100], gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
            margin=dict(l=0, r=60, t=10, b=0),
            font=dict(family="JetBrains Mono", color="#8892a4")
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Scatter: return vs confidence ──────────────────────────────────────
        section_title("Expected Return vs Confidence")
        df_all = pd.DataFrame(results)
        fig2 = px.scatter(
            df_all,
            x="confidence", y="expected_pct",
            color="signal", size_max=14,
            hover_name="symbol",
            hover_data=["entry_price","rsi","regime"],
            color_discrete_map={"BUY":"#00f5a0","SELL":"#ff3b6b","HOLD":"#ffb830"},
            labels={"confidence":"Confidence %","expected_pct":"Expected Return %","signal":"Signal"}
        )
        fig2.add_vline(x=min_confidence, line_dash="dot",
            line=dict(color="rgba(255,184,48,0.5)", width=1.5))
        fig2.add_hline(y=min_return, line_dash="dot",
            line=dict(color="rgba(0,212,255,0.5)", width=1.5))
        fig2.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4")),
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="JetBrains Mono", color="#8892a4")
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px 0;border-top:1px solid rgba(255,255,255,0.05);
margin-top:32px;font-size:11px;color:#3d4a5c;font-family:'JetBrains Mono',monospace;letter-spacing:1px">
⚡ SignalAI Scanner · Unified signal engine · Not financial advice
</div>""", unsafe_allow_html=True)
