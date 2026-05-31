# api/main.py

import os
from dotenv import load_dotenv
load_dotenv()

from engine.evaluate import evaluate_model
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from engine.predictor import PredictionEngine
from engine.backtester import WalkForwardBacktester
from data.ingestion.yfinance_connector import fetch_historical
from data.feature_store.technical import add_technical_indicators

app    = FastAPI(title="India Stock AI", version="1.0.0")
engine = PredictionEngine()


# ── Request / Response schemas ────────────────────────────────

class PredictRequest(BaseModel):
    symbol: str
    mode:   str = "longterm"   # "intraday" or "longterm"

class BacktestRequest(BaseModel):
    symbol: str
    period: str = "3y"


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "India Stock AI is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "lstm_loaded": engine.lstm is not None,
        "xgb_loaded":  engine.xgb  is not None,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = engine.predict(req.symbol, req.mode)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
def backtest(req: BacktestRequest):
    try:
        nse_symbol = req.symbol if req.symbol.endswith(".NS") \
                     else req.symbol + ".NS"
        df  = fetch_historical(nse_symbol, period=req.period)
        df  = add_technical_indicators(df)

        bt    = WalkForwardBacktester()
        stats = bt.run(df, engine.xgb)
        stats.pop("equity_curve", None)   # keep response small
        return {"symbol": req.symbol, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols")
def popular_symbols():
    return {
        "nifty50_samples": [
            "RELIANCE", "TCS", "INFY", "HDFCBANK",
            "ICICIBANK", "HINDUNILVR", "SBIN", "BAJFINANCE",
            "WIPRO", "ADANIENT"
        ],
        "format_note": "Pass symbol without .NS — the API adds it automatically"
    }


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks):
    background_tasks.add_task(_retrain_job)
    return {"message": "Retraining scheduled in background"}


async def _retrain_job():
    """Nightly continuous learning job."""
    from training.train_all import run_training
    run_training(symbols=["RELIANCE.NS", "TCS.NS", "INFY.NS"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)