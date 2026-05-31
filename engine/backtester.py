# engine/backtester.py

import numpy as np
import pandas as pd
from data.ingestion.yfinance_connector import fetch_historical
from data.feature_store.technical import add_technical_indicators, FEATURE_COLS


class WalkForwardBacktester:
    """
    Walk-forward backtester.
    Trains on a rolling window, tests on the next period,
    then advances forward — no look-ahead bias.
    """

    def __init__(
        self,
        train_window: int = 252,   # ~1 trading year
        test_window:  int = 63,    # ~1 quarter
        initial_capital: float = 100_000.0,
        position_pct: float = 0.10  # 10% of capital per trade
    ):
        self.train_window    = train_window
        self.test_window     = test_window
        self.initial_capital = initial_capital
        self.position_pct    = position_pct

    def run(self, df: pd.DataFrame, model) -> dict:
        equity = [self.initial_capital]
        trades = []
        n      = len(df)

        for start in range(self.train_window, n - self.test_window, self.test_window):
            test_df = df.iloc[start: start + self.test_window]

            for i in range(len(test_df) - 1):
                hist   = df.iloc[:start + i + 1]
                feat   = hist[FEATURE_COLS].iloc[-1:].values
                feat   = np.nan_to_num(feat)

                try:
                    probs  = model.predict_proba(feat)[0]
                    action = int(np.argmax(probs))
                    conf   = float(probs[action])
                except Exception:
                    action, conf = 1, 0.5

                price_today     = float(test_df["close"].iloc[i])
                price_tomorrow  = float(test_df["close"].iloc[i + 1])
                ret             = (price_tomorrow - price_today) / price_today

                capital = equity[-1]
                if action == 2:          # BUY
                    pnl = capital * self.position_pct * ret
                elif action == 0:        # SELL (short proxy)
                    pnl = capital * self.position_pct * (-ret)
                else:
                    pnl = 0.0

                equity.append(capital + pnl)
                trades.append({
                    "date":       test_df.index[i],
                    "action":     ["SELL", "HOLD", "BUY"][action],
                    "confidence": round(conf * 100, 1),
                    "price":      round(price_today, 2),
                    "pnl":        round(pnl, 2),
                })

        return self._metrics(equity, trades)

    def _metrics(self, equity: list, trades: list) -> dict:
        eq   = pd.Series(equity)
        rets = eq.pct_change().dropna()

        wins     = [t for t in trades if t["pnl"] > 0]
        losses   = [t for t in trades if t["pnl"] < 0]
        dd       = float((eq / eq.cummax() - 1).min())
        n_years  = max(len(equity) / 252, 0.01)
        cagr     = (eq.iloc[-1] / eq.iloc[0]) ** (1 / n_years) - 1

        sharpe = (
            rets.mean() / rets.std() * np.sqrt(252)
            if rets.std() > 0 else 0.0
        )

        return {
            "sharpe_ratio":     round(float(sharpe), 2),
            "max_drawdown_pct": round(dd * 100, 2),
            "cagr_pct":         round(cagr * 100, 2),
            "win_rate_pct":     round(len(wins) / max(len(trades), 1) * 100, 1),
            "total_trades":     len(trades),
            "winning_trades":   len(wins),
            "losing_trades":    len(losses),
            "initial_capital":  round(equity[0], 0),
            "final_capital":    round(float(eq.iloc[-1]), 0),
            "net_pnl":          round(float(eq.iloc[-1]) - equity[0], 0),
            "equity_curve":     [round(e, 0) for e in equity],
        }


if __name__ == "__main__":
    from models.longterm.xgb_model import load_xgboost
    df    = fetch_historical("TCS.NS", period="3y")
    df    = add_technical_indicators(df)
    model, _ = load_xgboost()
    bt    = WalkForwardBacktester()
    stats = bt.run(df, model)
    for k, v in stats.items():
        if k != "equity_curve":
            print(f"  {k:22s}: {v}")