"""
Confidence calibration + signal gating.
Drop-in replacement for raw predict_proba confidence.
"""

def calibrate_confidence(raw_proba: float, signal: str, context: dict) -> float:
    """
    raw_proba  : float 0-1 from model.predict_proba
    signal     : 'BUY', 'SELL', or 'HOLD'
    context    : dict with keys: rsi, vix, market_regime, volume_ratio, hour
    Returns calibrated confidence float 0-1
    """
    conf = raw_proba

    rsi          = context.get('rsi', 50)
    vix          = context.get('vix', 15)
    regime       = context.get('market_regime', 'NEUTRAL')
    volume_ratio = context.get('volume_ratio', 1.0)
    hour         = context.get('hour', 12)

    # --- Penalty 1: VIX too high = uncertain market ---
    if vix > 25:
        conf *= 0.65
    elif vix > 20:
        conf *= 0.80

    # --- Penalty 2: Signal against market regime ---
    if signal == 'BUY' and regime == 'BEARISH':
        conf *= 0.75
    elif signal == 'SELL' and regime == 'BULLISH':
        conf *= 0.75

    # --- Penalty 3: RSI overbought/oversold gate ---
    if signal == 'BUY' and rsi > 70:
        conf *= 0.70   # chasing overbought
    elif signal == 'SELL' and rsi < 30:
        conf *= 0.70   # chasing oversold

    # --- Penalty 4: Low volume = less conviction ---
    if volume_ratio < 0.8:
        conf *= 0.85

    # --- Penalty 5: Session edge (first 15min / last 30min) ---
    if hour == 9 or (hour == 15 and context.get('minute', 0) >= 0):
        conf *= 0.70

    return round(min(conf, 0.97), 4)


def get_signal_decision(calibrated_conf: float, signal: str,
                        min_threshold: float = 0.62) -> str:
    """
    Applies minimum threshold gate.
    Returns final signal: BUY / SELL / HOLD
    """
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
