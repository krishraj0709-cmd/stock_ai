# dashboard/app.py  ── SignalAI Premium Dashboard v2
import os, sys, requests, time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components
from datetime import datetime
load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingestion.yfinance_connector import fetch_historical
from data.feature_store.technical import add_technical_indicators

API_URL = "http://localhost:8000"

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignalAI — Trading Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── PREMIUM CSS INJECTION ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── ROOT ── */
:root {
  --void: #03050a;
  --deep: #070d18;
  --glass: rgba(255,255,255,0.03);
  --glass2: rgba(255,255,255,0.06);
  --border: rgba(255,255,255,0.07);
  --border2: rgba(255,255,255,0.13);
  --green: #00f5a0;
  --green2: #00c97a;
  --red: #ff3b6b;
  --amber: #ffb830;
  --blue: #4f8cff;
  --cyan: #00d4ff;
  --muted: #5a6478;
  --dim: #8892a4;
}

/* ── GLOBAL ── */
html, body, [class*="css"] {
  font-family: 'Syne', sans-serif !important;
  background-color: #03050a !important;
  color: #e8eaf0 !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }
.stApp { background: #03050a !important; }
section[data-testid="stSidebar"] {
  background: #060c14 !important;
  border-right: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── SIDEBAR WIDGETS ── */
section[data-testid="stSidebar"] * { color: #8892a4 !important; }
section[data-testid="stSidebar"] label { color: #5a6478 !important; font-size: 11px !important; letter-spacing: 1px !important; text-transform: uppercase !important; font-family: 'JetBrains Mono', monospace !important; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] .stSelectbox div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 10px !important;
  color: #e8eaf0 !important;
}

/* ── BUTTONS ── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,245,160,0.1), rgba(0,212,255,0.05)) !important;
  border: 1px solid rgba(0,245,160,0.25) !important;
  border-radius: 10px !important;
  color: #00f5a0 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.5px !important;
  transition: all 0.3s !important;
  padding: 8px 20px !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,245,160,0.18), rgba(0,212,255,0.10)) !important;
  border-color: rgba(0,245,160,0.45) !important;
  box-shadow: 0 8px 32px rgba(0,245,160,0.12) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, rgba(0,245,160,0.15), rgba(0,212,255,0.08)) !important;
  border-color: rgba(0,245,160,0.35) !important;
  font-size: 14px !important;
  padding: 10px 24px !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: 14px !important;
  padding: 16px 20px !important;
  backdrop-filter: blur(20px) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 10px !important;
  color: #5a6478 !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
  font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricValue"] {
  font-size: 22px !important;
  font-weight: 700 !important;
  font-family: 'JetBrains Mono', monospace !important;
  color: #e8eaf0 !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── CHARTS ── */
[data-testid="stPlotlyChart"] {
  border-radius: 14px !important;
  overflow: hidden !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  overflow: hidden !important;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: rgba(255,255,255,0.03) !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 9px !important;
  color: #5a6478 !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12px !important;
  letter-spacing: 0.5px !important;
  padding: 6px 16px !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(255,255,255,0.07) !important;
  color: #e8eaf0 !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #00f5a0 !important; }

/* ── SELECT / INPUT ── */
.stTextInput input, .stSelectbox select {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.09) !important;
  border-radius: 10px !important;
  color: #e8eaf0 !important;
  font-family: 'Syne', sans-serif !important;
}
.stTextInput input:focus {
  border-color: rgba(0,245,160,0.3) !important;
  box-shadow: 0 0 0 3px rgba(0,245,160,0.06) !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── PROGRESS BAR ── */
.stProgress > div > div {
  background: linear-gradient(90deg, #00c97a, #00f5a0) !important;
  border-radius: 4px !important;
}
.stProgress > div {
  background: rgba(255,255,255,0.05) !important;
  border-radius: 4px !important;
}

/* ── ALERTS ── */
.stAlert {
  border-radius: 12px !important;
  border: 1px solid !important;
  backdrop-filter: blur(20px) !important;
}
.stAlert[data-baseweb="notification"] { background: rgba(79,140,255,0.06) !important; border-color: rgba(79,140,255,0.2) !important; }
.element-container .stSuccess { background: rgba(0,245,160,0.05) !important; border-color: rgba(0,245,160,0.2) !important; }
.element-container .stError { background: rgba(255,59,107,0.06) !important; border-color: rgba(255,59,107,0.2) !important; }
.element-container .stWarning { background: rgba(255,184,48,0.06) !important; border-color: rgba(255,184,48,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ── ANTIGRAVITY HERO ───────────────────────────────────────────────────────────
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@800&family=JetBrains+Mono:wght@400;500&display=swap');
*{margin:0;padding:0;box-sizing:border-box}
body{background:transparent;overflow:hidden}
.hero{
  width:100%;height:260px;position:relative;
  display:flex;flex-direction:column;justify-content:center;align-items:center;
  background:radial-gradient(ellipse 80% 100% at 50% 0%, rgba(79,140,255,0.07) 0%, transparent 70%);
}
canvas{position:absolute;inset:0;width:100%;height:100%}
.hero-content{position:relative;z-index:2;text-align:center}
.badge{
  display:inline-flex;align-items:center;gap:7px;
  background:rgba(0,245,160,0.07);border:1px solid rgba(0,245,160,0.18);
  border-radius:100px;padding:5px 14px;
  font-family:'JetBrains Mono',monospace;font-size:10px;
  color:#00f5a0;letter-spacing:2px;text-transform:uppercase;margin-bottom:18px;
}
.pulse{width:5px;height:5px;background:#00f5a0;border-radius:50%;animation:p 1.5s infinite}
@keyframes p{0%,100%{box-shadow:0 0 0 0 rgba(0,245,160,0.6)}50%{box-shadow:0 0 0 5px rgba(0,245,160,0)}}
.title{
  font-family:'Syne',sans-serif;font-size:64px;font-weight:800;
  line-height:1;letter-spacing:-3px;margin-bottom:10px;
  background:linear-gradient(135deg,#fff 0%,rgba(255,255,255,0.65) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.title span{
  background:linear-gradient(135deg,#00f5a0,#00d4ff);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.sub{
  font-family:'JetBrains Mono',monospace;font-size:12px;
  color:#5a6478;letter-spacing:1px;
}
</style>
<div class="hero">
  <canvas id="c"></canvas>
  <div class="hero-content">
    <div class="badge"><span class="pulse"></span>Live Market Intelligence · NSE · BSE</div>
    <div class="title">Signal<span>AI</span></div>
    <div class="sub">XGBoost + LSTM Ensemble &nbsp;·&nbsp; Real-time Signals &nbsp;·&nbsp; India Markets</div>
  </div>
</div>
<script>
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
canvas.width=canvas.parentElement.offsetWidth||900;
canvas.height=260;
let mx=canvas.width/2,my=130;
document.querySelector('.hero').addEventListener('mousemove',e=>{
  const r=canvas.getBoundingClientRect();
  mx=e.clientX-r.left; my=e.clientY-r.top;
});
class B{
  constructor(){this.reset(true)}
  reset(init=false){
    this.x=Math.random()*canvas.width;
    this.y=init?Math.random()*canvas.height:canvas.height+10;
    this.r=1.5+Math.random()*4;
    this.vx=(Math.random()-.5)*.4;
    this.vy=-(0.3+Math.random()*.6);
    this.a=0.08+Math.random()*.2;
    this.h=140+Math.random()*80;
  }
  tick(){
    const dx=this.x-mx,dy=this.y-my;
    const d=Math.sqrt(dx*dx+dy*dy);
    if(d<160&&d>0){const s=(160-d)/160;this.vx+=dx/d*s*.7;this.vy+=dy/d*s*.7;}
    this.vx*=.97;this.vy*=.97;this.vy-=.007;
    this.x+=this.vx;this.y+=this.vy;
    const g=ctx.createRadialGradient(this.x-.3*this.r,this.y-.3*this.r,0,this.x,this.y,this.r);
    g.addColorStop(0,`hsla(${this.h},100%,78%,${this.a*1.6})`);
    g.addColorStop(.5,`hsla(${this.h},90%,55%,${this.a})`);
    g.addColorStop(1,`hsla(${this.h},80%,35%,0)`);
    ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);
    ctx.fillStyle=g;ctx.fill();
    ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);
    ctx.strokeStyle=`hsla(${this.h},100%,70%,${this.a*.35})`;
    ctx.lineWidth=.4;ctx.stroke();
    if(this.y<-15||this.x<-40||this.x>canvas.width+40)this.reset();
  }
}
const bubbles=Array.from({length:70},()=>new B());
function anim(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  bubbles.forEach(b=>b.tick());
  requestAnimationFrame(anim);
}
anim();
</script>
""", height=270)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def card(html_content):
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:16px;padding:20px 24px;backdrop-filter:blur(20px);margin-bottom:16px;
    position:relative;overflow:hidden;">
    <div style="position:absolute;inset:0;background:radial-gradient(ellipse 80% 50% at 50% 0%,
    rgba(255,255,255,0.012),transparent);pointer-events:none;"></div>
    <div style="position:relative;z-index:1">{html_content}</div></div>
    """, unsafe_allow_html=True)

def section_title(label):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:28px 0 14px 0;">
      <span style="font-size:10px;color:#5a6478;letter-spacing:2px;text-transform:uppercase;
      font-family:'JetBrains Mono',monospace;">{label}</span>
      <div style="flex:1;height:1px;background:rgba(255,255,255,0.06)"></div>
    </div>""", unsafe_allow_html=True)

def signal_pill(signal):
    colors = {
        "BUY":  ("#00f5a0", "rgba(0,245,160,0.1)",  "rgba(0,245,160,0.18)"),
        "SELL": ("#ff3b6b", "rgba(255,59,107,0.1)",  "rgba(255,59,107,0.18)"),
        "HOLD": ("#ffb830", "rgba(255,184,48,0.1)",  "rgba(255,184,48,0.18)"),
        "EXIT": ("#ff7a30", "rgba(255,122,48,0.1)",  "rgba(255,122,48,0.18)"),
    }
    c, bg, br = colors.get(signal, ("#8892a4","rgba(255,255,255,0.05)","rgba(255,255,255,0.1)"))
    return f"""<span style="background:{bg};border:1px solid {br};color:{c};
    padding:3px 12px;border-radius:6px;font-size:11px;font-weight:700;
    font-family:'JetBrains Mono',monospace;letter-spacing:1px;">{signal}</span>"""

def regime_color(r):
    return {"BULLISH":"#00f5a0","BEARISH":"#ff3b6b","NEUTRAL":"#ffb830"}.get(r,"#5a6478")

def fetch_regime():
    try:
        r = requests.get(f"{API_URL}/market/regime", timeout=8)
        return r.json()
    except Exception:
        return {"regime":"NEUTRAL","vix":15.0,"nifty_trend":"NEUTRAL","bnf_trend":"NEUTRAL"}

def fetch_analytics():
    try:
        r = requests.get(f"{API_URL}/analytics", timeout=8)
        return r.json().get("analytics", [])
    except Exception:
        return []

def fetch_recent():
    try:
        r = requests.get(f"{API_URL}/predictions/recent", timeout=8)
        return r.json().get("predictions", [])
    except Exception:
        return []


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 24px 0;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:20px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <div style="width:7px;height:7px;background:#00f5a0;border-radius:50%;
        box-shadow:0 0 8px #00f5a0;animation:p 1.5s infinite"></div>
        <span style="font-size:15px;font-weight:800;letter-spacing:-0.5px">SignalAI</span>
      </div>
      <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;
      letter-spacing:1px;padding-left:15px">TRADING INTELLIGENCE v2</div>
    </div>
    """, unsafe_allow_html=True)

    symbol = st.text_input("Stock Symbol", "RELIANCE",
        help="e.g. RELIANCE, TCS, INFY, HDFCBANK").upper().strip()

    exchange = st.radio("Exchange", ["NSE", "BSE"], horizontal=True)
    mode = st.selectbox("Prediction Mode", ["intraday", "longterm"])
    period = st.selectbox("Chart Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

    suffix = ".NS" if exchange == "NSE" else ".BO"
    ticker_symbol = symbol + suffix

    st.markdown("<div style='margin:16px 0 8px 0;font-size:10px;color:#5a6478;letter-spacing:1px;text-transform:uppercase;font-family:JetBrains Mono,monospace'>Quick Pick</div>", unsafe_allow_html=True)
    POPULAR = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","WIPRO"]
    cols = st.columns(2)
    for i, s in enumerate(POPULAR):
        if cols[i%2].button(s, key=f"qp_{s}", use_container_width=True):
            symbol = s
            ticker_symbol = symbol + suffix

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↻  Refresh Market Regime", use_container_width=True):
        st.cache_data.clear()


# ── MARKET REGIME BAR ─────────────────────────────────────────────────────────
regime = fetch_regime()
nifty_t = regime.get("nifty_trend", "NEUTRAL")
bnf_t   = regime.get("bnf_trend",   "NEUTRAL")
vix_v   = regime.get("vix", 15.0)
overall = regime.get("regime", "NEUTRAL")

vix_color = "#00f5a0" if vix_v < 15 else "#ffb830" if vix_v < 20 else "#ff3b6b"

st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
border-radius:14px;padding:14px 22px;margin-bottom:24px;
display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;
position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:1px;
background:linear-gradient(90deg,transparent,rgba(0,245,160,0.25),transparent)"></div>
<div style="display:flex;align-items:center;gap:28px;flex-wrap:wrap">
  <div style="display:flex;align-items:center;gap:8px">
    <div style="width:7px;height:7px;border-radius:50%;background:{regime_color(nifty_t)};box-shadow:0 0 7px {regime_color(nifty_t)}"></div>
    <span style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase">NIFTY</span>
    <span style="font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{regime_color(nifty_t)}">{nifty_t}</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <div style="width:7px;height:7px;border-radius:50%;background:{regime_color(bnf_t)};box-shadow:0 0 7px {regime_color(bnf_t)}"></div>
    <span style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase">BANKNIFTY</span>
    <span style="font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{regime_color(bnf_t)}">{bnf_t}</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <span style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase">INDIA VIX</span>
    <span style="background:rgba(255,184,48,0.08);border:1px solid rgba(255,184,48,0.18);
    border-radius:6px;padding:3px 10px;font-size:12px;font-family:'JetBrains Mono',monospace;color:{vix_color}">{vix_v}</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <span style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase">OVERALL</span>
    <span style="font-size:13px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{regime_color(overall)}">{overall}</span>
  </div>
</div>
<div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#5a6478;font-family:'JetBrains Mono',monospace">
  <div style="width:5px;height:5px;background:#00f5a0;border-radius:50%;animation:pulse 1.5s infinite"></div>
  LIVE
</div>
</div>
""", unsafe_allow_html=True)


# ── MAIN TABS ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["⚡  Signal", "📡  Scanner", "📋  History", "📊  Analytics"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
# ── SEARCH BAR ────────────────────────────────────────────
    section_title("Search & Predict")

    # symbol state
    if "active_symbol" not in st.session_state:
        st.session_state.active_symbol = "RELIANCE"

    search_col, btn_col = st.columns([4, 1])
    with search_col:
        typed = st.text_input(
            label="Stock Symbol",
            value=st.session_state.active_symbol,
            placeholder="Type symbol — RELIANCE, TCS, ZOMATO, INFY...",
            key="signal_search_input"
        )
        if typed.strip():
            st.session_state.active_symbol = typed.upper().strip()
    with btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(
            f"⚡ Analyse",
            type="primary",
            use_container_width=True,
            key="predict_main_btn"
        )

    # Quick pick chips
    st.markdown("""
    <div style="font-size:10px;color:#5a6478;letter-spacing:1px;
    font-family:'JetBrains Mono',monospace;margin:10px 0 6px 0">
    Quick Pick:</div>""", unsafe_allow_html=True)

    CHIPS = [
        "RELIANCE","TCS","INFY","HDFCBANK",
        "ZOMATO","BAJFINANCE","WIPRO","SBIN",
        "TATAMOTORS","ICICIBANK","ADANIENT","SUNPHARMA"
    ]
    chip_rows = [CHIPS[:6], CHIPS[6:]]
    for row in chip_rows:
        cols = st.columns(len(row))
        for i, s in enumerate(row):
            if cols[i].button(s, key=f"chip2_{s}", use_container_width=True):
                st.session_state.active_symbol = s
                st.rerun()

    # Use active symbol
    symbol        = st.session_state.active_symbol
    ticker_symbol = symbol + suffix

    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:10px;padding:10px 16px;margin:10px 0;
    font-family:'JetBrains Mono',monospace;font-size:12px;color:#8892a4;
    display:flex;align-items:center;gap:8px">
      <div style="width:6px;height:6px;background:#00f5a0;border-radius:50%"></div>
      Analysing: <span style="color:#00f5a0;font-weight:700">{symbol}</span>
      &nbsp;·&nbsp; {exchange} &nbsp;·&nbsp; {mode.upper()}
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        with st.spinner(f"Analysing {symbol} on {exchange}…"):
            try:
                resp = requests.post(f"{API_URL}/predict",
                    json={"symbol": symbol, "mode": mode}, timeout=60)
                data = resp.json()
            except Exception as e:
                st.error(f"API error: {e}. Make sure uvicorn is running.")
                st.stop()

        if "error" in data:
            st.error(data["error"])
        else:
            signal = data["signal"]
            sig_colors = {"BUY":"rgba(0,245,160","SELL":"rgba(255,59,107","HOLD":"rgba(255,184,48"}
            sig_c = sig_colors.get(signal, "rgba(88,144,255")
            sig_text = {"BUY":"#00f5a0","SELL":"#ff3b6b","HOLD":"#ffb830"}.get(signal,"#8892a4")

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{sig_c},0.07),{sig_c},0.03));
            border:1px solid {sig_c},0.22);border-radius:18px;padding:26px 28px;
            margin-bottom:20px;position:relative;overflow:hidden;">
              <div style="position:absolute;top:-50px;right:-50px;width:160px;height:160px;
              border-radius:50%;background:{sig_text};filter:blur(50px);opacity:0.15"></div>
              <div style="position:relative;z-index:1">
                <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:20px">
                  <div>
                    {signal_pill(signal)}
                    <div style="font-size:26px;font-weight:800;letter-spacing:-1px;margin-top:10px">{symbol}</div>
                    <div style="font-size:11px;color:#5a6478;font-family:'JetBrains Mono',monospace">{exchange} · {mode.upper()} · {regime.get('regime','NEUTRAL')} market</div>
                  </div>
                  <div style="text-align:right">
                    <div style="font-size:10px;color:#5a6478;text-transform:uppercase;letter-spacing:1px;font-family:'JetBrains Mono',monospace">Entry Price</div>
                    <div style="font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace">₹{data['entry_price']}</div>
                  </div>
                </div>
                <div style="margin-bottom:18px">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span style="font-size:10px;color:#5a6478;text-transform:uppercase;letter-spacing:1px;font-family:'JetBrains Mono',monospace">Signal Confidence</span>
                    <span style="font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{sig_text}">{data['confidence']}%</span>
                  </div>
                  <div style="height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden">
                    <div style="width:{data['confidence']}%;height:100%;background:linear-gradient(90deg,{sig_text}aa,{sig_text});border-radius:3px;transition:width 1s ease"></div>
                  </div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Entry",    f"₹{data['entry_price']}")
            col2.metric("Target",   f"₹{data['target']}",     f"+{data['expected_pct']}%")
            col3.metric("Stop Loss",f"₹{data['stop_loss']}")
            col4.metric("Confidence",f"{data['confidence']}%")
            col5.metric("Risk",     data['risk_level'])

            section_title("Signal Probabilities")
            probs = data.get("probabilities", {})
            b_val = int(probs.get("BUY", 0))
            h_val = int(probs.get("HOLD", 0))
            s_val = int(probs.get("SELL", 0))

            st.markdown(f"""
            <div style="display:flex;flex-direction:column;gap:10px;margin-bottom:20px">
              <div style="display:flex;align-items:center;gap:12px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#00f5a0;width:36px">BUY</span>
                <div style="flex:1;height:7px;background:rgba(255,255,255,0.05);border-radius:4px">
                  <div style="width:{b_val}%;height:100%;background:linear-gradient(90deg,#00c97a,#00f5a0);border-radius:4px"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00f5a0;width:35px">{b_val}%</span>
              </div>
              <div style="display:flex;align-items:center;gap:12px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#ffb830;width:36px">HOLD</span>
                <div style="flex:1;height:7px;background:rgba(255,255,255,0.05);border-radius:4px">
                  <div style="width:{h_val}%;height:100%;background:linear-gradient(90deg,#cc8800,#ffb830);border-radius:4px"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#ffb830;width:35px">{h_val}%</span>
              </div>
              <div style="display:flex;align-items:center;gap:12px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#ff3b6b;width:36px">SELL</span>
                <div style="flex:1;height:7px;background:rgba(255,255,255,0.05);border-radius:4px">
                  <div style="width:{s_val}%;height:100%;background:linear-gradient(90deg,#cc1f4a,#ff3b6b);border-radius:4px"></div>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#ff3b6b;width:35px">{s_val}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.info(f"Expected move: **{data['expected_pct']}%**  ·  Risk: **{data['risk_level']}**  ·  Mode: **{mode}**  ·  Regime: **{regime.get('regime')}**")

    st.divider()

    # ── PRICE CHART ────────────────────────────────────────────────────────────
    section_title(f"{symbol} Price Chart")
    try:
        df = fetch_historical(ticker_symbol, period=period)
        df = add_technical_indicators(df)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="Price",
            increasing_line_color="#00f5a0", decreasing_line_color="#ff3b6b",
            increasing_fillcolor="rgba(0,245,160,0.15)",
            decreasing_fillcolor="rgba(255,59,107,0.15)",
        ))
        if "ema_20" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ema_20"], name="EMA 20",
                line=dict(color="#4f8cff", width=1.2), opacity=0.8))
        if "ema_50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["ema_50"], name="EMA 50",
                line=dict(color="#ffb830", width=1.2), opacity=0.8))
        if "bb_upper" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], name="BB Upper",
                line=dict(color="rgba(148,163,184,0.5)", width=1, dash="dot")))
            fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], name="BB Lower",
                line=dict(color="rgba(148,163,184,0.5)", width=1, dash="dot"),
                fill="tonexty", fillcolor="rgba(148,163,184,0.03)"))
        fig.update_layout(
            height=480, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4", size=11)),
            margin=dict(l=0, r=0, t=10, b=0),
            font=dict(family="JetBrains Mono", color="#8892a4")
        )
        st.plotly_chart(fig, use_container_width=True)

        if "rsi" in df.columns:
            section_title("RSI — Relative Strength Index")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI",
                line=dict(color="#9b6dff", width=1.5),
                fill="tozeroy", fillcolor="rgba(155,109,255,0.04)"))
            rsi_fig.add_hline(y=70, line_dash="dot",
                line=dict(color="rgba(255,59,107,0.5)", width=1),
                annotation_text="Overbought 70",
                annotation_font=dict(color="rgba(255,59,107,0.7)", size=10))
            rsi_fig.add_hline(y=30, line_dash="dot",
                line=dict(color="rgba(0,245,160,0.5)", width=1),
                annotation_text="Oversold 30",
                annotation_font=dict(color="rgba(0,245,160,0.7)", size=10))
            rsi_fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,59,107,0.03)", line_width=0)
            rsi_fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,245,160,0.03)", line_width=0)
            rsi_fig.update_layout(
                height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478", range=[0, 100]),
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
                font=dict(family="JetBrains Mono", color="#8892a4")
            )
            st.plotly_chart(rsi_fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Chart unavailable for {ticker_symbol}: {e}")

    st.divider()

    # ── BACKTEST ───────────────────────────────────────────────────────────────
    section_title("Backtesting")
    bt_period = st.selectbox("Backtest Period", ["1y", "2y", "3y", "5y"], index=1)
    if st.button("▶  Run Walk-Forward Backtest", use_container_width=True):
        with st.spinner("Running walk-forward backtest — 1–2 minutes…"):
            try:
                resp = requests.post(f"{API_URL}/backtest",
                    json={"symbol": symbol, "period": bt_period}, timeout=180)
                stats = resp.json()
            except Exception as e:
                st.error(str(e)); st.stop()

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Sharpe Ratio",  stats.get("sharpe_ratio"), help=">1.0 good, >2.0 excellent")
        b2.metric("Max Drawdown",  f"{stats.get('max_drawdown_pct')}%", help="Under -20% acceptable")
        b3.metric("CAGR",          f"{stats.get('cagr_pct')}%", help=">15% beats most mutual funds")
        b4.metric("Win Rate",      f"{stats.get('win_rate_pct')}%", help=">55% is good")
        st.caption(f"Trades: {stats.get('total_trades')}  ·  Net PnL: ₹{stats.get('net_pnl')}  ·  Final capital: ₹{stats.get('final_capital')}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCANNER
# ════════════════════════════════════════════════════════════════════════════════
with tab2:

    STOCK_UNIVERSE = {
        "Nifty 50":   ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","HINDUNILVR",
                       "SBIN","BAJFINANCE","WIPRO","ADANIENT","AXISBANK","KOTAKBANK",
                       "MARUTI","SUNPHARMA","TATAMOTORS","ASIANPAINT","TECHM",
                       "NESTLEIND","ULTRACEMCO","TITAN","POWERGRID","NTPC","ONGC",
                       "COALINDIA","JSWSTEEL","TATASTEEL","HCLTECH","DIVISLAB",
                       "DRREDDY","CIPLA","BAJAJFINSV","BPCL","IOC","GRASIM",
                       "HEROMOTOCO","BRITANNIA","EICHERMOT","INDUSINDBK","M&M",
                       "ITC","LT","APOLLOHOSP","TATACONSUM","PIDILITIND","UPL"],
        "Midcap":     ["ZOMATO","IRCTC","HAL","BEL","CANBK","BANKBARODA","PNB",
                       "FEDERALBNK","IDFCFIRSTB","RBLBANK","PAYTM","NYKAA",
                       "DELHIVERY","POLICYBZR","MUTHOOTFIN"],
        "Smallcap":   ["HAPPSTMNDS","ROUTE","INTELLECT","TANLA","MASTEK",
                       "LATENTVIEW","KPITTECH","TATAELXSI","CYIENT","MPHASIS",
                       "PERSISTENT","COFORGE","SONACOMS","KAYNES","DOMS"],
        "Banknifty":  ["HDFCBANK","ICICIBANK","KOTAKBANK","SBIN","AXISBANK",
                       "INDUSINDBK","BANDHANBNK","FEDERALBNK","IDFCFIRSTB",
                       "RBLBANK","PNB","BANKBARODA","CANBK","AUBANK"],
        "IT":         ["TCS","INFY","WIPRO","HCLTECH","TECHM","MPHASIS","LTTS",
                       "COFORGE","PERSISTENT","KPITTECH","TATAELXSI","CYIENT"],
        "Pharma":     ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","BIOCON","LUPIN",
                       "AUROPHARMA","TORNTPHARM","ALKEM","ZYDUSLIFE"],
        "FMCG":       ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO",
                       "COLPAL","GODREJCP","EMAMILTD","TATACONSUM","VARUNBEV"],
        "Auto":       ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO",
                       "EICHERMOT","TVSMOTORS","ASHOKLEY","BOSCHLTD","MRF"],
    }

    # ── FILTER PANEL ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
    border-radius:16px;padding:22px 24px;margin-bottom:20px;position:relative;overflow:hidden">
    <div style="position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(79,140,255,0.3),transparent)"></div>
    <div style="font-size:11px;color:#4f8cff;font-family:'JetBrains Mono',monospace;
    letter-spacing:2px;text-transform:uppercase;margin-bottom:16px">
    ⚙ Scanner Filters</div>
    """, unsafe_allow_html=True)

    # Row 1 — Category + Signal filter
    r1c1, r1c2 = st.columns([3, 1])
    with r1c1:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">Stock Categories</div>""", unsafe_allow_html=True)
        selected_categories = st.multiselect(
            label="", label_visibility="collapsed",
            options=list(STOCK_UNIVERSE.keys()),
            default=["Nifty 50"],
            key="scan_categories"
        )
    with r1c2:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">Signal Filter</div>""", unsafe_allow_html=True)
        signal_filter = st.selectbox(
            label="", label_visibility="collapsed",
            options=["BUY + SELL", "BUY only", "SELL only"],
            key="scan_signal_filter"
        )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Row 2 — Confidence + Min Return + Num stocks
    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">Min Confidence %</div>""", unsafe_allow_html=True)
        min_conf = st.slider("", 50, 90, 65,
            key="scan_conf", label_visibility="collapsed",
            help="Only show signals above this confidence threshold")
    with r2c2:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">Min Expected Return %</div>""", unsafe_allow_html=True)
        min_ret = st.slider("", 0.5, 10.0, 1.5, step=0.5,
            key="scan_ret", label_visibility="collapsed")
    with r2c3:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">Stocks to Scan</div>""", unsafe_allow_html=True)
        scan_limit = st.slider("", 5, 60, 20,
            key="scan_limit", label_visibility="collapsed")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Row 3 — RSI range + Risk level
    r3c1, r3c2 = st.columns([2, 1])
    with r3c1:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">RSI Range</div>""", unsafe_allow_html=True)
        rsi_range = st.slider("", 0, 100, (30, 70),
            key="scan_rsi", label_visibility="collapsed",
            help="Only include stocks with RSI in this range")
    with r3c2:
        st.markdown("""<div style="font-size:10px;color:#5a6478;letter-spacing:1px;
        text-transform:uppercase;font-family:'JetBrains Mono',monospace;
        margin-bottom:6px">Risk Level</div>""", unsafe_allow_html=True)
        risk_filter = st.multiselect(
            label="", label_visibility="collapsed",
            options=["LOW", "MEDIUM", "HIGH"],
            default=["LOW", "MEDIUM"],
            key="scan_risk"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── BUILD SYMBOL LIST FROM SELECTED CATEGORIES ────────────────────────────
    scan_symbols = []
    for cat in (selected_categories or ["Nifty 50"]):
        scan_symbols += STOCK_UNIVERSE.get(cat, [])
    scan_symbols = list(dict.fromkeys(scan_symbols))  # deduplicate

    # ── SCAN SUMMARY LINE ─────────────────────────────────────────────────────
    sig_label = signal_filter.replace(" only","").replace(" + "," & ")
    cat_str   = ", ".join(selected_categories) if selected_categories else "None"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;
    flex-wrap:wrap;gap:10px;margin-bottom:16px">
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <span style="background:rgba(79,140,255,0.08);border:1px solid rgba(79,140,255,0.18);
        border-radius:8px;padding:4px 12px;font-size:11px;color:#4f8cff;
        font-family:'JetBrains Mono',monospace">📦 {len(scan_symbols)} stocks</span>
        <span style="background:rgba(0,245,160,0.08);border:1px solid rgba(0,245,160,0.15);
        border-radius:8px;padding:4px 12px;font-size:11px;color:#00f5a0;
        font-family:'JetBrains Mono',monospace">🎯 Conf ≥ {min_conf}%</span>
        <span style="background:rgba(255,184,48,0.08);border:1px solid rgba(255,184,48,0.15);
        border-radius:8px;padding:4px 12px;font-size:11px;color:#ffb830;
        font-family:'JetBrains Mono',monospace">📈 Return ≥ {min_ret}%</span>
        <span style="background:rgba(155,109,255,0.08);border:1px solid rgba(155,109,255,0.18);
        border-radius:8px;padding:4px 12px;font-size:11px;color:#9b6dff;
        font-family:'JetBrains Mono',monospace">📊 RSI {rsi_range[0]}–{rsi_range[1]}</span>
        <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
        border-radius:8px;padding:4px 12px;font-size:11px;color:#8892a4;
        font-family:'JetBrains Mono',monospace">⚡ {sig_label}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SCAN BUTTON ───────────────────────────────────────────────────────────
    if not selected_categories:
        st.warning("Select at least one stock category to scan.")
    else:
        run_scan = st.button(
            f"⚡  Scan {min(len(scan_symbols), scan_limit)} Stocks",
            type="primary", use_container_width=True, key="run_scan_btn"
        )

        if run_scan:
            with st.spinner(f"Scanning {min(len(scan_symbols), scan_limit)} stocks…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/scan",
                        json={
                            "universe":       selected_categories,
                            "min_confidence": float(min_conf),
                            "min_return":     float(min_ret),
                            "limit":          scan_limit,
                        },
                        timeout=300
                    )
                    raw = resp.json()
                    all_results = raw.get("results", [])
                except Exception as e:
                    st.error(f"Scan error: {e}. Make sure FastAPI is running.")
                    all_results = []

            # ── CLIENT-SIDE FILTERS ───────────────────────────────────────────
            filtered = []
            for r in all_results:
                # Signal filter
                if signal_filter == "BUY only"  and r["signal"] != "BUY":  continue
                if signal_filter == "SELL only" and r["signal"] != "SELL": continue
                # RSI range filter
                rsi_val = r.get("rsi", 50)
                if not (rsi_range[0] <= rsi_val <= rsi_range[1]):          continue
                # Risk level filter
                if risk_filter and r.get("risk_level","MEDIUM") not in risk_filter: continue
                filtered.append(r)

            st.session_state["scan_results_v2"] = filtered
            st.session_state["scan_meta"] = {
                "total":   raw.get("total_scanned", 0),
                "found":   len(filtered),
                "regime":  raw.get("market_regime","NEUTRAL"),
                "vix":     raw.get("vix", 15.0),
                "time":    datetime.now().strftime("%H:%M:%S"),
                "cats":    cat_str,
            }

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if "scan_results_v2" in st.session_state:
        results = st.session_state["scan_results_v2"]
        meta    = st.session_state["scan_meta"]

        # Summary metrics
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Scanned",  meta["total"])
        mc2.metric("Signals",  meta["found"])
        mc3.metric("Regime",   meta["regime"])
        mc4.metric("VIX",      meta["vix"])
        mc5.metric("Time",     meta["time"])

        if not results:
            st.warning(
                f"No stocks matched all your filters. Try: "
                f"lowering confidence below {min_conf}%, "
                f"widening RSI range, or adding more categories."
            )
        else:
            # ── Top picks ─────────────────────────────────────────────────────
            section_title(f"Top {min(3, len(results))} Picks")
            top3      = results[:3]
            card_cols = st.columns(len(top3))

            for i, row in enumerate(top3):
                sig  = row["signal"]
                sc   = {"BUY":"#00f5a0","SELL":"#ff3b6b"}.get(sig,"#ffb830")
                bg   = {"BUY":"rgba(0,245,160,0.06)","SELL":"rgba(255,59,107,0.06)"}.get(sig,"rgba(255,184,48,0.06)")
                brd  = {"BUY":"rgba(0,245,160,0.2)","SELL":"rgba(255,59,107,0.2)"}.get(sig,"rgba(255,184,48,0.2)")
                rank = ["🥇","🥈","🥉"][i]
                rc_  = "✅" if row.get("regime_confirmed") else "⚠️"

                with card_cols[i]:
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {brd};
                    border-radius:16px;padding:20px;text-align:center;
                    position:relative;overflow:hidden">
                      <div style="position:absolute;top:-30px;right:-30px;
                      width:100px;height:100px;border-radius:50%;
                      background:{sc};filter:blur(40px);opacity:0.15"></div>
                      <div style="font-size:26px;margin-bottom:4px">{rank}</div>
                      <div style="font-size:18px;font-weight:800;color:{sc}">{row['symbol']}</div>
                      <div style="margin:6px 0">{signal_pill(sig)}</div>
                      <div style="font-size:20px;font-weight:700;
                      font-family:'JetBrains Mono',monospace;color:{sc}">{row['confidence']}%</div>
                      <div style="font-size:11px;color:#8892a4;
                      font-family:'JetBrains Mono',monospace;margin-top:8px;line-height:1.8">
                        Return: +{row['expected_pct']}%<br>
                        Entry: ₹{row['entry_price']}<br>
                        Target: ₹{row['target']} &nbsp;·&nbsp; SL: ₹{row['stop_loss']}<br>
                        RSI: {row['rsi']} &nbsp;·&nbsp; Risk: {row.get('risk_level','--')}<br>
                        {rc_} {row['regime']}
                      </div>
                    </div>""", unsafe_allow_html=True)

            # ── Full table ────────────────────────────────────────────────────
            section_title(f"All {len(results)} Results")

            hdr = """
            <div style="display:grid;
            grid-template-columns:110px 75px 95px 90px 90px 90px 55px 85px 80px;
            gap:6px;padding:8px 16px;border-bottom:1px solid rgba(255,255,255,0.06);
            font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;
            letter-spacing:1px;text-transform:uppercase">
              <span>Symbol</span><span>Signal</span><span>Confidence</span>
              <span>Entry ₹</span><span>Target ₹</span><span>Stop Loss</span>
              <span>RSI</span><span>Risk</span><span>Regime</span>
            </div>"""

            rows_html = ""
            for r in results:
                sig  = r["signal"]
                sc   = {"BUY":"#00f5a0","SELL":"#ff3b6b"}.get(sig,"#ffb830")
                cc   = "#00f5a0" if r["confidence"]>=75 else "#ffb830" if r["confidence"]>=62 else "#ff3b6b"
                rsk  = r.get("risk_level","--")
                rsk_c= {"LOW":"#00f5a0","MEDIUM":"#ffb830","HIGH":"#ff3b6b"}.get(rsk,"#5a6478")
                rgm_c= {"BULLISH":"#00f5a0","BEARISH":"#ff3b6b","NEUTRAL":"#ffb830"}.get(r["regime"],"#5a6478")
                rsi_c= "#00f5a0" if r["rsi"]<50 else "#ff3b6b"

                rows_html += f"""
                <div style="display:grid;
                grid-template-columns:110px 75px 95px 90px 90px 90px 55px 85px 80px;
                gap:6px;padding:11px 16px;border-bottom:1px solid rgba(255,255,255,0.03);
                font-family:'JetBrains Mono',monospace;font-size:12px;cursor:default"
                onmouseover="this.style.background='rgba(255,255,255,0.02)'"
                onmouseout="this.style.background='transparent'">
                  <span style="font-weight:700">{r['symbol']}</span>
                  <span>{signal_pill(sig)}</span>
                  <span style="color:{cc};font-weight:700">{r['confidence']}%</span>
                  <span>₹{r['entry_price']}</span>
                  <span style="color:#00f5a0">₹{r['target']}</span>
                  <span style="color:#ff3b6b">₹{r['stop_loss']}</span>
                  <span style="color:{rsi_c}">{r['rsi']}</span>
                  <span style="color:{rsk_c}">{rsk}</span>
                  <span style="color:{rgm_c}">{r['regime']}</span>
                </div>"""

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.07);
            border-radius:14px;overflow:hidden">{hdr}{rows_html}</div>
            """, unsafe_allow_html=True)

            # ── Confidence bar chart ──────────────────────────────────────────
            if len(results) > 3:
                section_title("Confidence Chart")
                df_c = pd.DataFrame(results).head(15).sort_values("confidence")
                fig  = go.Figure(go.Bar(
                    x=df_c["confidence"], y=df_c["symbol"],
                    orientation="h",
                    marker_color=[
                        "#00f5a0" if s=="BUY" else "#ff3b6b"
                        for s in df_c["signal"]
                    ],
                    text=[f"{v}%" for v in df_c["confidence"]],
                    textposition="outside"
                ))
                fig.add_vline(x=min_conf, line_dash="dot",
                    line=dict(color="rgba(255,184,48,0.6)", width=1.5),
                    annotation_text=f"Min {min_conf}%",
                    annotation_font=dict(color="rgba(255,184,48,0.8)", size=10))
                fig.update_layout(
                    height=max(250, len(df_c)*34),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(range=[0,100],
                        gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
                    margin=dict(l=0, r=60, t=10, b=0),
                    font=dict(family="JetBrains Mono", color="#8892a4")
                )
                st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    section_title("Prediction Memory")
    preds = fetch_recent()

    if preds:
        outcome_color = {
            "TARGET_HIT": "#00f5a0",
            "SL_HIT":     "#ff3b6b",
            "PARTIAL":    "#ffb830",
            "NEUTRAL":    "#5a6478",
            "PENDING":    "#8892a4",
        }
        outcome_icon = {
            "TARGET_HIT": "✓", "SL_HIT": "✗",
            "PARTIAL": "~", "NEUTRAL": "–", "PENDING": "…",
        }

        header_html = """
        <div style="display:grid;grid-template-columns:100px 100px 80px 80px 90px 90px 110px 80px;
        gap:8px;padding:8px 16px;border-bottom:1px solid rgba(255,255,255,0.06);
        font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;
        letter-spacing:1px;text-transform:uppercase">
          <span>Time</span><span>Stock</span><span>Signal</span>
          <span>Conf</span><span>Entry</span><span>Target</span>
          <span>Outcome</span><span>P&L</span>
        </div>"""

        rows_html = ""
        for p in preds:
            sig = p.get("signal","HOLD")
            out = p.get("outcome","PENDING")
            pnl = p.get("pnl_pct")
            oc  = outcome_color.get(out,"#5a6478")
            oi  = outcome_icon.get(out,"")
            sc  = {"BUY":"#00f5a0","SELL":"#ff3b6b","HOLD":"#ffb830"}.get(sig,"#8892a4")
            pnl_str = f"{'+'if pnl and pnl>0 else ''}{pnl:.2f}%" if pnl else "--"
            pnl_c   = "#00f5a0" if pnl and pnl>0 else "#ff3b6b" if pnl and pnl<0 else "#5a6478"
            conf    = p.get("calibrated_confidence",0)
            conf_pct= int(conf*100) if isinstance(conf,float) else (conf if conf else 0)
            time_str = str(p.get("predicted_at",""))[:16].replace("T"," ")

            rows_html += f"""
            <div style="display:grid;grid-template-columns:100px 100px 80px 80px 90px 90px 110px 80px;
            gap:8px;padding:11px 16px;border-bottom:1px solid rgba(255,255,255,0.03);
            font-family:'JetBrains Mono',monospace;font-size:12px;
            transition:background 0.15s" onmouseover="this.style.background='rgba(255,255,255,0.02)'"
            onmouseout="this.style.background='transparent'">
              <span style="color:#5a6478">{time_str}</span>
              <span style="font-weight:600">{p.get('stock_symbol','')}</span>
              <span style="color:{sc};font-weight:600">{sig}</span>
              <span style="color:{sc}">{conf_pct}%</span>
              <span>₹{p.get('entry_price','--')}</span>
              <span style="color:#00f5a0">₹{p.get('target_price','--')}</span>
              <span style="color:{oc}">{out.replace('_',' ')} {oi}</span>
              <span style="color:{pnl_c}">{pnl_str}</span>
            </div>"""

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
        border-radius:14px;overflow:hidden">
          {header_html}{rows_html}
        </div>""", unsafe_allow_html=True)
    else:
        card("""<div style='text-align:center;padding:32px 0;color:#5a6478;
        font-family:JetBrains Mono,monospace;font-size:13px'>
        No predictions recorded yet — run a signal to start logging</div>""")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    section_title("Performance Analytics")
    analytics = fetch_analytics()

    if analytics:
        total_trades = sum(a.get("total",0) for a in analytics)
        buy_data  = next((a for a in analytics if a["signal"]=="BUY"),  {})
        sell_data = next((a for a in analytics if a["signal"]=="SELL"), {})
        hold_data = next((a for a in analytics if a["signal"]=="HOLD"), {})

        overall_wr = buy_data.get("win_rate_pct", 0)
        avg_pnl    = buy_data.get("avg_pnl", 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate",     f"{overall_wr}%",
            delta="Good" if overall_wr >= 55 else "Needs improvement")
        col2.metric("Total Trades", total_trades)
        col3.metric("Avg PnL",      f"{avg_pnl}%")
        col4.metric("Model Health",
            "🟢 Strong" if overall_wr >= 60 else "🟡 OK" if overall_wr >= 45 else "🔴 Weak")

        section_title("Signal Breakdown")
        for row in analytics:
            sig = row.get("signal","")
            sig_c = {"BUY":"#00f5a0","SELL":"#ff3b6b","HOLD":"#ffb830"}.get(sig,"#8892a4")
            wr    = row.get("win_rate_pct", 0)
            apnl  = row.get("avg_pnl", 0)
            ttl   = row.get("total", 0)
            dacc  = row.get("directional_accuracy", 0)

            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
            border-radius:12px;padding:16px 20px;margin-bottom:10px;
            display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
              <div style="display:flex;align-items:center;gap:12px">
                {signal_pill(sig)}
                <span style="font-size:11px;color:#5a6478;font-family:'JetBrains Mono',monospace">{ttl} trades</span>
              </div>
              <div style="display:flex;gap:28px;flex-wrap:wrap">
                <div style="text-align:center">
                  <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px">Win Rate</div>
                  <div style="font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{sig_c}">{wr}%</div>
                </div>
                <div style="text-align:center">
                  <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px">Avg PnL</div>
                  <div style="font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:{'#00f5a0' if apnl and apnl>0 else '#ff3b6b'}">{'+'if apnl and apnl>0 else ''}{apnl}%</div>
                </div>
                <div style="text-align:center">
                  <div style="font-size:10px;color:#5a6478;font-family:'JetBrains Mono',monospace;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px">Dir. Acc</div>
                  <div style="font-size:18px;font-weight:700;font-family:'JetBrains Mono',monospace;color:#8892a4">{dacc}%</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        section_title("Confidence Reliability")
        st.caption("When AI says X% confidence, does it actually win X% of the time?")
        try:
            r = requests.get(f"{API_URL}/analytics/confidence-reliability", timeout=8)
            buckets = r.json().get("buckets", [])
            if buckets:
                df_b = pd.DataFrame(buckets)
                fig_rel = go.Figure()
                fig_rel.add_trace(go.Bar(x=df_b["conf_bucket"]*100, y=df_b["actual_accuracy"]*100,
                    name="Actual Win Rate", marker_color="#00f5a0",
                    marker_line_color="rgba(0,245,160,0.3)", marker_line_width=1))
                fig_rel.add_trace(go.Scatter(x=df_b["conf_bucket"]*100, y=df_b["conf_bucket"]*100,
                    name="Ideal (Calibrated)", line=dict(color="#4f8cff", dash="dot", width=1.5)))
                fig_rel.update_layout(
                    height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Confidence (%)", gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
                    yaxis=dict(title="Actual Win %", gridcolor="rgba(255,255,255,0.04)", color="#5a6478"),
                    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4")),
                    margin=dict(l=0, r=0, t=10, b=0),
                    font=dict(family="JetBrains Mono", color="#8892a4"),
                    bargap=0.3
                )
                st.plotly_chart(fig_rel, use_container_width=True)
        except Exception:
            st.info("Confidence reliability chart available after 20+ resolved predictions.")
    else:
        card("""<div style='text-align:center;padding:40px 0;color:#5a6478;
        font-family:JetBrains Mono,monospace;font-size:13px'>
        Analytics available after 10+ resolved predictions<br>
        <span style='font-size:11px;color:#3d4a5c'>Run predictions, mark outcomes in History tab</span>
        </div>""")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:32px 0 16px 0;
border-top:1px solid rgba(255,255,255,0.05);margin-top:32px;
font-size:11px;color:#3d4a5c;font-family:'JetBrains Mono',monospace;letter-spacing:1px">
⚡ SignalAI · For educational purposes only · Not financial advice · Trade responsibly
</div>""", unsafe_allow_html=True)
