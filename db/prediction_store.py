import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

def init_db():
    with get_db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '15m',
            predicted_at TEXT DEFAULT (datetime('now','localtime')),
            signal TEXT NOT NULL,
            raw_confidence REAL,
            calibrated_confidence REAL,
            entry_price REAL,
            target_price REAL,
            stop_loss REAL,
            rsi REAL,
            macd_hist REAL,
            ema_diff REAL,
            bb_position REAL,
            volume_ratio REAL,
            pattern_name TEXT DEFAULT 'NONE',
            market_regime TEXT DEFAULT 'NEUTRAL',
            vix_at_signal REAL,
            nifty_trend TEXT DEFAULT 'NEUTRAL',
            xgb_proba REAL,
            lstm_proba REAL,
            voter_count INTEGER DEFAULT 0,
            outcome TEXT DEFAULT 'PENDING',
            actual_price_at_check REAL,
            outcome_checked_at TEXT,
            pnl_pct REAL,
            directional_correct INTEGER
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON predictions(stock_symbol, predicted_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON predictions(outcome)")
        print("[DB] Database initialized successfully.")

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def log_prediction(data: dict) -> int:
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions
            (stock_symbol, timeframe, signal, raw_confidence, calibrated_confidence,
             entry_price, target_price, stop_loss, rsi, macd_hist, ema_diff,
             bb_position, volume_ratio, pattern_name, market_regime,
             vix_at_signal, nifty_trend, xgb_proba, lstm_proba, voter_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get('stock_symbol',''), data.get('timeframe','15m'),
            data.get('signal','HOLD'), data.get('raw_confidence', 0.5),
            data.get('calibrated_confidence', 0.5),
            data.get('entry_price', 0), data.get('target_price', 0),
            data.get('stop_loss', 0), data.get('rsi', 50),
            data.get('macd_hist', 0), data.get('ema_diff', 0),
            data.get('bb_position', 0.5), data.get('volume_ratio', 1.0),
            data.get('pattern_name', 'NONE'), data.get('market_regime', 'NEUTRAL'),
            data.get('vix_at_signal', 15.0), data.get('nifty_trend', 'NEUTRAL'),
            data.get('xgb_proba', 0.5), data.get('lstm_proba', 0.5),
            data.get('voter_count', 0)
        ))
        return cursor.lastrowid

def get_pending_predictions(older_than_minutes=30):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            WHERE outcome = 'PENDING'
            AND predicted_at < datetime('now','localtime',? || ' minutes')
            LIMIT 50
        """, (f'-{older_than_minutes}',)).fetchall()
        return [dict(r) for r in rows]

def update_outcome(pred_id: int, outcome: str, actual_price: float, pnl: float, dir_correct: int):
    with get_db() as conn:
        conn.execute("""
            UPDATE predictions SET
                outcome=?, actual_price_at_check=?, pnl_pct=?,
                directional_correct=?, outcome_checked_at=datetime('now','localtime')
            WHERE id=?
        """, (outcome, actual_price, round(pnl,3), dir_correct, pred_id))

def get_analytics():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT signal,
                COUNT(*) as total,
                ROUND(AVG(CASE WHEN outcome='TARGET_HIT' THEN 1.0 ELSE 0.0 END)*100,1) as win_rate_pct,
                ROUND(AVG(pnl_pct),2) as avg_pnl,
                ROUND(AVG(calibrated_confidence)*100,1) as avg_confidence,
                ROUND(AVG(directional_correct)*100,1) as directional_accuracy
            FROM predictions
            WHERE outcome != 'PENDING'
            GROUP BY signal
        """).fetchall()
        return [dict(r) for r in rows]

def get_recent_predictions(limit=20):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, stock_symbol, signal, calibrated_confidence,
                   entry_price, target_price, stop_loss, outcome, pnl_pct, predicted_at
            FROM predictions
            ORDER BY predicted_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

if __name__ == "__main__":
    init_db()
