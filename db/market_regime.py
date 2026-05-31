import yfinance as yf
import json, os
from datetime import datetime

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'regime_cache.json')

def _get_close(df):
    """Handles both old and new yfinance MultiIndex DataFrames."""
    col = df['Close']
    if hasattr(col, 'columns'):   # new yfinance returns DataFrame
        col = col.squeeze()       # flatten to Series
    return col

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
            close = _get_close(nifty)
            price = float(close.iloc[-1])
            ema20 = float(close.ewm(span=20).mean().iloc[-1])
            if price > ema20 * 1.002:
                result['nifty_trend'] = 'BULLISH'
            elif price < ema20 * 0.998:
                result['nifty_trend'] = 'BEARISH'
            result['regime'] = result['nifty_trend']
            print(f"[REGIME] NIFTY: {price:.0f} | EMA20: {ema20:.0f} | Trend: {result['nifty_trend']}")
    except Exception as e:
        print(f'[REGIME] NIFTY error: {e}')

    try:
        vix = yf.download('^INDIAVIX', period='2d', interval='1d', progress=False)
        if not vix.empty:
            result['vix'] = round(float(_get_close(vix).iloc[-1]), 2)
            print(f"[REGIME] VIX: {result['vix']}")
    except Exception as e:
        print(f'[REGIME] VIX error: {e}')

    try:
        bnf = yf.download('^NSEBANK', period='5d', interval='1d', progress=False)
        if not bnf.empty:
            close = _get_close(bnf)
            bnf_price = float(close.iloc[-1])
            bnf_ema = float(close.ewm(span=20).mean().iloc[-1])
            if bnf_price > bnf_ema * 1.002:
                result['bnf_trend'] = 'BULLISH'
            elif bnf_price < bnf_ema * 0.998:
                result['bnf_trend'] = 'BEARISH'
            print(f"[REGIME] BNF: {bnf_price:.0f} | Trend: {result['bnf_trend']}")
    except Exception as e:
        print(f'[REGIME] BNF error: {e}')

    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(result, f)
    except Exception:
        pass

    return result

if __name__ == "__main__":
    print(get_market_regime(force_refresh=True))