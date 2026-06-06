# db/prediction_store.py — PostgreSQL version (Supabase)
import os
from contextlib import contextmanager
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")

@contextmanager
def get_db():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_db():
    print("[DB] Connected to Supabase PostgreSQL")

def log_prediction(data: dict) -> int:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions
            (stock_symbol, timeframe, signal, raw_confidence,
             calibrated_confidence, entry_price, target_price, stop_loss,
             rsi, macd_hist, bb_position, volume_ratio, pattern_name,
             market_regime, vix_at_signal, nifty_trend,
             xgb_proba, lstm_proba, voter_count)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            data.get('stock_symbol',''),
            data.get('timeframe','15m'),
            data.get('signal','HOLD'),
            data.get('raw_confidence', 0.5),
            data.get('calibrated_confidence', 0.5),
            data.get('entry_price', 0),
            data.get('target_price', 0),
            data.get('stop_loss', 0),
            data.get('rsi', 50),
            data.get('macd_hist', 0),
            data.get('bb_position', 0.5),
            data.get('volume_ratio', 1.0),
            data.get('pattern_name','NONE'),
            data.get('market_regime','NEUTRAL'),
            data.get('vix_at_signal', 15.0),
            data.get('nifty_trend','NEUTRAL'),
            data.get('xgb_proba', 0.5),
            data.get('lstm_proba', 0.5),
            data.get('voter_count', 0)
        ))
        return cur.fetchone()['id']

def get_pending_predictions(older_than_minutes=30):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM predictions
            WHERE outcome = 'PENDING'
            AND predicted_at < NOW() - INTERVAL '%s minutes'
            LIMIT 50
        """, (older_than_minutes,))
        return cur.fetchall()

def update_outcome(pred_id, outcome, actual_price, pnl, dir_correct):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE predictions SET
                outcome=%s, actual_price_at_check=%s,
                pnl_pct=%s, directional_correct=%s,
                outcome_checked_at=NOW()
            WHERE id=%s
        """, (outcome, actual_price, round(pnl,3), dir_correct, pred_id))

def get_analytics():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT signal,
                COUNT(*) as total,
                ROUND(AVG(CASE WHEN outcome='TARGET_HIT'
                    THEN 1.0 ELSE 0.0 END)*100, 1) as win_rate_pct,
                ROUND(AVG(pnl_pct)::numeric, 2) as avg_pnl,
                ROUND(AVG(calibrated_confidence)*100, 1) as avg_confidence,
                ROUND(AVG(directional_correct)*100, 1) as directional_accuracy
            FROM predictions
            WHERE outcome != 'PENDING'
            GROUP BY signal
        """)
        return cur.fetchall()

def get_recent_predictions(limit=20):
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, stock_symbol, signal, calibrated_confidence,
                   entry_price, target_price, stop_loss,
                   outcome, pnl_pct, predicted_at
            FROM predictions
            ORDER BY predicted_at DESC LIMIT %s
        """, (limit,))
        return cur.fetchall()

def get_db_for_query():
    return get_db()