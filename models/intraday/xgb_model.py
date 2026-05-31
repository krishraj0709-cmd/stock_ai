# models/longterm/xgb_model.py

import numpy as np
import pandas as pd
import xgboost as xgb
import json
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

FEATURE_COLS = [
    "rsi", "macd", "macd_hist", "bb_pct", "bb_width",
    "atr", "vol_ratio", "ema_9", "ema_20", "ema_50",
    "price_change", "log_return", "candle_body",
    "candle_range", "upper_shadow", "lower_shadow",
    "roc_5", "roc_10", "roc_20",
]


def create_labels(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Label based on 20-day forward return:
      0 = SELL  (< -5%)
      1 = HOLD  (-5% to +5%)
      2 = BUY   (> +5%)
    """
    fwd = df["close"].pct_change(horizon).shift(-horizon)
    df  = df.copy()
    df["label"] = pd.cut(
        fwd,
        bins=[-np.inf, -0.05, 0.05, np.inf],
        labels=[0, 1, 2]
    ).astype("Int64")
    return df.dropna(subset=["label"])


def train_xgboost(
    df: pd.DataFrame,
    horizon: int = 20,
    save_path: str = "models/longterm/xgb_model.json"
) -> xgb.XGBClassifier:

    df = create_labels(df, horizon=horizon)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].values
    y = df["label"].astype(int).values

    params = dict(
        n_estimators       = 500,
        max_depth          = 6,
        learning_rate      = 0.05,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        min_child_weight   = 5,
        gamma              = 0.1,
        reg_alpha          = 0.1,
        reg_lambda         = 1.0,
        objective          = "multi:softprob",
        num_class          = 3,
        eval_metric        = "mlogloss",
        early_stopping_rounds = 30,
        tree_method        = "hist",
        random_state       = 42,
        verbosity          = 0,
    )

    tscv        = TimeSeriesSplit(n_splits=5)
    oof_preds   = np.zeros((len(X), 3))
    final_model = None

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        oof_preds[val_idx] = model.predict_proba(X_val)
        final_model = model
        print(f"  Fold {fold}/5 done")

    print("\nOut-of-fold classification report:")
    print(classification_report(
        y, oof_preds.argmax(axis=1),
        target_names=["SELL", "HOLD", "BUY"]
    ))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_model.save_model(save_path)
    print(f"  Model saved to {save_path}")

    meta = {"feature_cols": available, "horizon": horizon}
    with open(save_path.replace(".json", "_meta.json"), "w") as f:
        json.dump(meta, f)

    return final_model


def load_xgboost(path: str = "models/longterm/xgb_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(path)
    meta_path = path.replace(".json", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta["feature_cols"]


if __name__ == "__main__":
    print("XGBoost model definitions loaded OK.")