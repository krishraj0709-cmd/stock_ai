# training/train_lstm_only.py
# Dedicated LSTM training script
# Run with: python training/train_lstm_only.py
# Takes 1-2 hours on CPU. Leave it running overnight.

import sys, os, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from data.feature_store.technical import add_technical_indicators, FEATURE_COLS
from models.intraday.lstm_model import prepare_sequences, train_lstm

SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "HINDUNILVR.NS", "SUNPHARMA.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "ASIANPAINT.NS", "ITC.NS", "LT.NS"
]

def fetch(ticker):
    df = yf.download(ticker, period="5y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.dropna(inplace=True)
    return df

print("=" * 55)
print("  LSTM TRAINING — India Stock AI")
print("  This will take 1-2 hours on CPU")
print("  DO NOT close this window")
print("=" * 55)

# Step 1: Download and build sequences
print("\nStep 1: Downloading data and building sequences...")
all_X, all_y = [], []

for sym in SYMBOLS:
    try:
        print(f"  Fetching {sym}...", end=" ")
        df = fetch(sym)
        df = add_technical_indicators(df)

        avail = [c for c in FEATURE_COLS if c in df.columns]
        if len(avail) < 10:
            print("skipped — too few features")
            continue

        X, y = prepare_sequences(df, avail, seq_len=60, horizon=1)
        if len(X) < 50:
            print("skipped — too few sequences")
            continue

        all_X.append(X)
        all_y.append(y)
        print(f"OK — {len(X)} sequences, {len(avail)} features")

    except Exception as e:
        print(f"FAILED — {e}")

if not all_X:
    print("\nERROR: No sequences built. Check internet and data.")
    sys.exit(1)

# Step 2: Combine all stocks
X_all = np.concatenate(all_X, axis=0)
y_all = np.concatenate(all_y, axis=0)

# Shuffle
idx   = np.random.permutation(len(X_all))
X_all = X_all[idx]
y_all = y_all[idx]

counts = np.bincount(y_all)
print(f"\nTotal sequences : {len(X_all)}")
print(f"Input shape     : {X_all.shape}")
print(f"Class balance   : SELL={counts[0]}  HOLD={counts[1]}  BUY={counts[2]}")

# Step 3: Train
print("\nStep 2: Training LSTM model...")
print("  Each epoch will print its validation loss.")
print("  Training stops early if no improvement for 10 epochs.\n")

os.makedirs("models/intraday", exist_ok=True)

model, scaler = train_lstm(
    X_all, y_all,
    epochs    = 80,
    batch     = 64,
    lr        = 1e-3,
    patience  = 12,
    save_path = "models/intraday/best_lstm.pt"
)

# Step 4: Verify
if os.path.exists("models/intraday/best_lstm.pt"):
    size = os.path.getsize("models/intraday/best_lstm.pt") / 1024 / 1024
    print(f"\nLSTM saved successfully!")
    print(f"File: models/intraday/best_lstm.pt")
    print(f"Size: {size:.1f} MB")
else:
    print("\nERROR: File not saved. Check errors above.")
    sys.exit(1)

# Step 5: Quick test
print("\nStep 3: Quick inference test...")
try:
    from models.intraday.lstm_model import load_lstm
    model, scaler, seq_len = load_lstm("models/intraday/best_lstm.pt")
    test_input = torch.randn(1, seq_len, X_all.shape[2])
    with torch.no_grad():
        out = torch.softmax(model(test_input), dim=1)
    probs = out.numpy()[0]
    labels = ["SELL", "HOLD", "BUY"]
    print(f"  Test prediction: {labels[probs.argmax()]} "
          f"(SELL={probs[0]:.2f} HOLD={probs[1]:.2f} BUY={probs[2]:.2f})")
    print("  Inference test PASSED")
except Exception as e:
    print(f"  Inference test failed: {e}")

print("\n" + "=" * 55)
print("  LSTM TRAINING COMPLETE")
print("=" * 55)
print("""
Next steps:
  1. Restart uvicorn:
     uvicorn api.main:app --reload --port 8000

  2. You will now see:
     LSTM loaded.    OK
     XGBoost loaded. OK

  3. Intraday mode will now use the real LSTM model.
""")