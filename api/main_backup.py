# api/main.py
import os, sys
sys.path.insert(0, r'C:\Users\krish\stock_ai')

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List

from engine.predictor import PredictionEngine
from engine.backtester import WalkForwardBacktester
from data.ingestion.yfinance_connector import fetch_historical
from data.feature_store.technical import add_technical_indicators

# ── NEW IMPORTS ────────────────────────────────────────────────────────────────
from apscheduler.schedulers.background import BackgroundScheduler
from db.prediction_store import (
    init_db, log_prediction, get_analytics,
    get_recent_predictions, get_db
)
from db.outcome_checker import check_pending_outcomes
from db.market_regime import get_market_regime
from db.confidence import calibrate_confidence, get_signal_decision, build_context

# ── INIT ───────────────────────────────────────────────────────────────────────
init_db()

app    = FastAPI(title="SignalAI — India Stock AI", version="2.0.0")
engine = PredictionEngine()

# ── BACKGROUND SCHEDULER ──────────────────────────────────────────────────────
_scheduler = BackgroundScheduler()
_scheduler.add_job(check_pending_outcomes, 'interval', minutes=30, id='outcome_checker')
_scheduler.add_job(
    lambda: get_market_regime(force_refresh=True),
    'interval', hours=1, id='regime_refresh'
)
_scheduler.start()


# ── REQUEST MODELS ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    symbol: str
    mode:   str = "longterm"

class BacktestRequest(BaseModel):
    symbol: str
    period: str = "3y"

class ScanRequest(BaseModel):
    universe:       Optional[List[str]] = ["NIFTY50"]
    min_confidence: Optional[float]     = 62.0   # unified threshold (not 80%)
    min_return:     Optional[float]     = 1.5
    period:         Optional[str]       = "3mo"
    limit:          Optional[int]       = 20


# ── NIFTY50 UNIVERSE ──────────────────────────────────────────────────────────
NIFTY50 = [
    "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
    "HINDUNILVR","SBIN","BAJFINANCE","WIPRO","ADANIENT",
    "AXISBANK","KOTAKBANK","MARUTI","SUNPHARMA","TATAMOTORS",
    "ASIANPAINT","TECHM","NESTLEIND","ULTRACEMCO","TITAN",
    "POWERGRID","NTPC","ONGC","COALINDIA","JSWSTEEL",
    "TATASTEEL","HCLTECH","DIVISLAB","DRREDDY","CIPLA",
    "BAJAJFINSV","BPCL","IOC","GRASIM","HEROMOTOCO",
    "BRITANNIA","EICHERMOT","SHREECEM","INDUSINDBK","M&M",
    "ITC","LT","HINDZINC","VEDL","APOLLOHOSP",
    "TATACONSUM","PIDILITIND","BERGEPAINT","MCDOWELL-N","UPL"
]

MIDCAP = [
    "ZOMATO","IRCTC","HAL","BEL","CANBK",
    "BANKBARODA","PNB","FEDERALBNK","IDFCFIRSTB","RBLBANK"
]

UNIVERSE_MAP = {
    "NIFTY50": NIFTY50,
    "MIDCAP":  MIDCAP,
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE UNIFIED SIGNAL FUNCTION
# Both /predict and /scan use this — single source of truth
# ══════════════════════════════════════════════════════════════════════════════
def _generate_signal(symbol: str, mode: str = "longterm") -> dict:
    """
    Unified signal generator used by BOTH individual predict and scanner.
    Applies: ML model → confidence calibration → regime filter → final signal
    """
    # 1. Get market regime (cached, refreshed hourly)
    regime = get_market_regime()

    # 2. Run ML model
    result = engine.predict(symbol, mode)
    if "error" in result:
        return result

    # 3. Get technical context for calibration
    try:
        nse_symbol = symbol if symbol.endswith(".NS") else symbol + ".NS"
        period     = "3mo" if mode == "intraday" else "2y"
        df         = fetch_historical(nse_symbol, period=period)
        df         = add_technical_indicators(df)
        lat        = df.iloc[-1]

        rsi          = float(lat.get("rsi",          50))
        vol_ratio    = float(lat.get("vol_ratio",    1.0))
        bb_pct       = float(lat.get("bb_pct",       0.5))
        macd_hist    = float(lat.get("macd_hist",    0.0))
    except Exception:
        rsi, vol_ratio, bb_pct, macd_hist = 50, 1.0, 0.5, 0.0

    raw_conf = result["confidence"] / 100.0   # convert % → 0-1

    # 4. Calibrate confidence with regime + technical context
    from datetime import datetime
    now = datetime.now()
    context = build_context(
        rsi=rsi,
        vix=regime.get("vix", 15),
        market_regime=regime.get("regime", "NEUTRAL"),
        volume_ratio=vol_ratio,
        hour=now.hour,
        minute=now.minute
    )
    calibrated_conf = calibrate_confidence(raw_conf, result["signal"], context)

    # 5. Apply signal gate — below threshold → HOLD
    final_signal = get_signal_decision(calibrated_conf, result["signal"])

    # 6. Override if signal contradicts regime strongly
    regime_str = regime.get("regime", "NEUTRAL")
    if final_signal == "BUY" and regime_str == "BEARISH":
        # Don't block entirely, but reduce confidence more
        calibrated_conf = round(calibrated_conf * 0.80, 4)
        if calibrated_conf < 0.62:
            final_signal = "HOLD"
    elif final_signal == "SELL" and regime_str == "BULLISH":
        calibrated_conf = round(calibrated_conf * 0.80, 4)
        if calibrated_conf < 0.62:
            final_signal = "HOLD"

    # 7. Build final response
    result.update({
        "signal":              final_signal,
        "raw_confidence":      result["confidence"],
        "confidence":          round(calibrated_conf * 100, 1),
        "market_regime":       regime_str,
        "nifty_trend":         regime.get("nifty_trend", "NEUTRAL"),
        "vix":                 regime.get("vix", 15.0),
        "rsi":                 round(rsi, 1),
        "volume_ratio":        round(vol_ratio, 2),
        "regime_confirmed":    (
            (final_signal == "BUY"  and regime_str in ("BULLISH","NEUTRAL")) or
            (final_signal == "SELL" and regime_str in ("BEARISH","NEUTRAL")) or
            final_signal == "HOLD"
        )
    })

    # 8. Auto-log to prediction DB (only non-HOLD signals)
    if final_signal != "HOLD":
        try:
            log_prediction({
                "stock_symbol":          symbol,
                "timeframe":             "15m" if mode == "intraday" else "1d",
                "signal":                final_signal,
                "raw_confidence":        raw_conf,
                "calibrated_confidence": calibrated_conf,
                "entry_price":           result["entry_price"],
                "target_price":          result["target"],
                "stop_loss":             result["stop_loss"],
                "rsi":                   rsi,
                "macd_hist":             macd_hist,
                "bb_position":           bb_pct,
                "volume_ratio":          vol_ratio,
                "market_regime":         regime_str,
                "vix_at_signal":         regime.get("vix", 15.0),
                "nifty_trend":           regime.get("nifty_trend","NEUTRAL"),
                "xgb_proba":             raw_conf,
                "lstm_proba":            raw_conf,
                "voter_count":           3 if calibrated_conf > 0.75 else 2,
            })
        except Exception as db_err:
            print(f"[DB] Log failed for {symbol}: {db_err}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "message": "SignalAI is running",
        "version": "2.0.0",
        "docs":    "/docs"
    }


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "lstm_loaded": engine.lstm is not None,
        "xgb_loaded":  engine.xgb  is not None,
        "regime":      get_market_regime().get("regime", "UNKNOWN")
    }


# ── INDIVIDUAL STOCK PREDICT ──────────────────────────────────────────────────
@app.post("/predict")
def predict(req: PredictRequest):
    """
    Individual stock signal.
    Uses unified _generate_signal() — same logic as scanner.
    """
    try:
        return _generate_signal(req.symbol, req.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── SCANNER ───────────────────────────────────────────────────────────────────
@app.post("/scan")
@app.get("/scan")
def scan_stocks(req: ScanRequest = None):
    """
    Scans a universe of stocks using the SAME unified signal engine as /predict.
    No more conflict — both use identical logic.
    """
    if req is None:
        req = ScanRequest()

    # Build symbol list from requested universes
    symbols = []
    for u in req.universe:
        symbols += UNIVERSE_MAP.get(u.upper(), [])
    symbols = list(dict.fromkeys(symbols))  # deduplicate

    if not symbols:
        symbols = NIFTY50[:20]  # fallback

    results   = []
    regime    = get_market_regime()

    for sym in symbols[:req.limit]:
        try:
            result = _generate_signal(sym, mode="longterm")

            if "error" in result:
                continue

            # Filter: only show actionable signals above threshold
            if (result["signal"] in ("BUY", "SELL") and
                result["confidence"] >= req.min_confidence and
                abs(result.get("expected_pct", 0)) >= req.min_return):

                results.append({
                    "symbol":          sym,
                    "signal":          result["signal"],
                    "confidence":      result["confidence"],
                    "entry_price":     result["entry_price"],
                    "target":          result["target"],
                    "stop_loss":       result["stop_loss"],
                    "expected_pct":    result.get("expected_pct", 0),
                    "rsi":             result.get("rsi", 0),
                    "regime":          result.get("market_regime", "NEUTRAL"),
                    "regime_confirmed":result.get("regime_confirmed", False),
                    "risk_level":      result.get("risk_level", "MEDIUM"),
                })

        except Exception as e:
            print(f"[SCAN] Error for {sym}: {e}")
            continue

    # Sort by confidence descending
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "results":       results,
        "total_scanned": len(symbols),
        "signals_found": len(results),
        "market_regime": regime.get("regime", "NEUTRAL"),
        "vix":           regime.get("vix", 15.0),
    }


# ── BACKTEST ──────────────────────────────────────────────────────────────────
@app.post("/backtest")
def backtest(req: BacktestRequest):
    try:
        nse_symbol = req.symbol if req.symbol.endswith(".NS") else req.symbol + ".NS"
        df         = fetch_historical(nse_symbol, period=req.period)
        df         = add_technical_indicators(df)
        bt         = WalkForwardBacktester()
        stats      = bt.run(df, engine.xgb)
        stats.pop("equity_curve", None)
        return {"symbol": req.symbol, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── ANALYTICS ENDPOINTS ───────────────────────────────────────────────────────
@app.get("/analytics")
def analytics():
    return {"analytics": get_analytics()}


@app.get("/predictions/recent")
def recent_predictions():
    return {"predictions": get_recent_predictions(20)}


@app.get("/market/regime")
def market_regime_endpoint():
    return get_market_regime()


@app.get("/analytics/confidence-reliability")
def confidence_reliability():
    """
    Shows: when model says X% confidence, what % actually win?
    Use this to spot calibration issues.
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT
                ROUND(calibrated_confidence * 10) / 10.0 as conf_bucket,
                COUNT(*) as count,
                ROUND(AVG(directional_correct), 3) as actual_accuracy
            FROM predictions
            WHERE outcome != 'PENDING'
              AND signal != 'HOLD'
              AND calibrated_confidence IS NOT NULL
            GROUP BY conf_bucket
            ORDER BY conf_bucket
        """).fetchall()
    return {"buckets": [dict(r) for r in rows]}


# ── SYMBOLS ───────────────────────────────────────────────────────────────────
@app.get("/symbols")
def popular_symbols():
    return {
        "nifty50":    NIFTY50,
        "midcap":     MIDCAP,
        "format_note":"Pass symbol without .NS suffix"
    }


# ── RETRAIN ───────────────────────────────────────────────────────────────────
@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_job)
    return {"message": "Retraining scheduled in background"}

async def _retrain_job():
    from training.train_all import run_training
    run_training(symbols=["RELIANCE.NS", "TCS.NS", "INFY.NS"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
