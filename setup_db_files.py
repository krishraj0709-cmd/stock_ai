import os

BASE = r'C:\Users\krish\stock_ai\db'
os.makedirs(BASE, exist_ok=True)

# ── __init__.py ──────────────────────────────────────────
open(os.path.join(BASE, '__init__.py'), 'w').close()

# ── market_regime.py ─────────────────────────────────────
with open(os.path.join(BASE, 'market_regime.py'), 'w') as f:
    f.write("""import yfinance as yf
import json, os
from datetime import datetime

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regime_cache.json')

def get_market_regime(force_refresh=False):
    if not force_refresh and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                cache = json.load(f)
            age = (datetime.now() - datetime.fromisoformat(cache['cached_at'])).seconds / 60
            if age < 60:
                return cache
        except Exception:
            pass

    result = {
        'regime': 'NEUTRAL',
        'vix': 15.0,
        'nifty_trend': 'NEUTRAL',
        'bnf_trend': 'NEUTRAL',
        'cached_at': datetime.now().isoformat()
    }

    try:
        nifty = yf.download('^NSEI', period='5d', interval='1d', progress=False)
        if not nifty.empty:
            price = float(nifty['Close'].iloc[-1])
            ema20 = float(nifty['Close'].ewm(span=20).mean().iloc[-1])
            if price > ema20 * 1.002:
                result['nifty_trend'] = 'BULLISH'
            elif price < ema20 * 0.998:
                result['nifty_trend'] = 'BEARISH'
            result['regime'] = result['nifty_trend']
    except Exception as e:
        print(f'NIFTY error: {e}')

    try:
        vix = yf.download('^INDIAVIX', period='2d', interval='1d', progress=False)
        if not vix.empty:
            result['vix'] = round(float(vix['Close'].iloc[-1]), 2)
    except Exception as e:
        print(f'VIX error: {e}')

    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(result, f)
    except Exception:
        pass

    return result
""")

# ── confidence.py ─────────────────────────────────────────
with open(os.path.join(BASE, 'confidence.py'), 'w') as f:
    f.write("""def calibrate_confidence(raw_proba, signal, context):
    conf = raw_proba
    rsi = context.get('rsi', 50)
    vix = context.get('vix', 15)
    regime = context.get('market_regime', 'NEUTRAL')
    volume_ratio = context.get('volume_ratio', 1.0)
    hour = context.get('hour', 12)

    if vix > 25:
        conf *= 0.65
    elif vix > 20:
        conf *= 0.80

    if signal == 'BUY' and regime == 'BEARISH':
        conf *= 0.75
    elif signal == 'SELL' and regime == 'BULLISH':
        conf *= 0.75

    if signal == 'BUY' and rsi > 70:
        conf *= 0.70
    elif signal == 'SELL' and rsi < 30:
        conf *= 0.70

    if volume_ratio < 0.8:
        conf *= 0.85

    return round(min(conf, 0.97), 4)


def get_signal_decision(calibrated_conf, signal, min_threshold=0.62):
    if calibrated_conf < min_threshold:
        return 'HOLD'
    return signal


def build_context(rsi, vix, market_regime, volume_ratio=1.0, hour=12, minute=0):
    return {
        'rsi': rsi,
        'vix': vix,
        'market_regime': market_regime,
        'volume_ratio': volume_ratio,
        'hour': hour,
        'minute': minute
    }
""")

# ── outcome_checker.py ────────────────────────────────────
with open(os.path.join(BASE, 'outcome_checker.py'), 'w') as f:
    f.write("""import yfinance as yf
from db.prediction_store import get_pending_predictions, update_outcome
from datetime import datetime

def check_pending_outcomes():
    print(f'[CHECKER] Running at {datetime.now().strftime(\"%H:%M:%S\")}')
    pending = get_pending_predictions(older_than_minutes=30)

    if not pending:
        print('[CHECKER] No pending predictions.')
        return

    for pred in pending:
        try:
            sym = pred['stock_symbol']
            hist = None
            for suffix in ['.NS', '']:
                try:
                    ticker = yf.Ticker(sym + suffix)
                    hist = ticker.history(period='1d', interval='5m')
                    if not hist.empty:
                        break
                except Exception:
                    continue

            if hist is None or hist.empty:
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
            print(f'[CHECKER] {sym} | {pred[\"signal\"]} | {outcome} | PnL: {pnl:.2f}%')

        except Exception as e:
            print(f'[CHECKER] Error for {pred.get(\"stock_symbol\",\"?\")}: {e}')
""")

print("=" * 50)
print("ALL FILES CREATED SUCCESSFULLY:")
for f in os.listdir(BASE):
    print(f"  {BASE}\\{f}")
print("=" * 50)