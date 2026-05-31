"""
Runs as background job via APScheduler.
Checks pending predictions and marks WIN/LOSS.
"""
import yfinance as yf
from db.prediction_store import get_pending_predictions, update_outcome
from datetime import datetime

def check_pending_outcomes():
    print(f"[CHECKER] Running outcome check at {datetime.now().strftime('%H:%M:%S')}")
    pending = get_pending_predictions(older_than_minutes=30)

    if not pending:
        print("[CHECKER] No pending predictions to check.")
        return

    for pred in pending:
        try:
            sym = pred['stock_symbol']
            # Try NSE suffix first, then plain
            for suffix in ['.NS', '']:
                try:
                    ticker = yf.Ticker(sym + suffix)
                    hist = ticker.history(period='1d', interval='5m')
                    if not hist.empty:
                        break
                except Exception:
                    continue

            if hist.empty:
                continue

            current_price = float(hist['Close'].iloc[-1])
            entry = pred['entry_price'] or current_price
            target = pred['target_price'] or 0
            sl = pred['stop_loss'] or 0

            if not target or not sl:
                continue

            high_since = float(hist['High'].max())
            low_since = float(hist['Low'].min())

            if pred['signal'] == 'BUY':
                target_hit = high_since >= target
                sl_hit = low_since <= sl
                dir_correct = int(current_price > entry)
                pnl = (current_price - entry) / entry * 100
            elif pred['signal'] == 'SELL':
                target_hit = low_since <= target
                sl_hit = high_since >= sl
                dir_correct = int(current_price < entry)
                pnl = (entry - current_price) / entry * 100
            else:
                continue

            if target_hit and not sl_hit:
                outcome = 'TARGET_HIT'
            elif sl_hit and not target_hit:
                outcome = 'SL_HIT'
            elif target_hit and sl_hit:
                outcome = 'PARTIAL'
            else:
                outcome = 'NEUTRAL'

            update_outcome(pred['id'], outcome, current_price, pnl, dir_correct)
            print(f"[CHECKER] {sym} | {pred['signal']} | {outcome} | PnL: {pnl:.2f}%")

        except Exception as e:
            print(f"[CHECKER] Error for {pred.get('stock_symbol','?')}: {e}")

if __name__ == "__main__":
    check_pending_outcomes()
