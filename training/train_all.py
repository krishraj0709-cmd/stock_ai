import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBRegressor

import torch
import torch.nn as nn


# =========================================
# NSE STOCK LIST
# =========================================
STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "WIPRO.NS",
    "ADANIENT.NS",
    "AXISBANK.NS",
    "KOTAKBANK.NS",
    "MARUTI.NS",
    "TATAMOTORS.NS",
    "SUNPHARMA.NS"
]


# =========================================
# FEATURE ENGINEERING
# =========================================
def add_features(df):
    df = df.copy()

    # Flatten columns if yfinance returns MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Return"] = df["Close"].pct_change()

    df["MA_10"] = df["Close"].rolling(10).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / loss

    df["RSI"] = 100 - (100 / (1 + rs))

    df["Volatility"] = df["Return"].rolling(10).std()

    df = df.dropna()

    return df


# =========================================
# XGBOOST MODEL
# =========================================
def train_xgb(df, symbol):

    features = [
        "MA_10",
        "MA_20",
        "MA_50",
        "RSI",
        "Volatility"
    ]

    X = df[features]
    y = df["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)

    os.makedirs("models/longterm", exist_ok=True)

    model_path = f"models/longterm/{symbol.replace('.', '_')}_xgb.pkl"

    joblib.dump(model, model_path)

    print(f"✅ XGB trained for {symbol} | MSE: {mse:.2f}")

    return mse


# =========================================
# LSTM
# =========================================
class LSTMModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=50):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        out = self.fc(out[:, -1, :])

        return out


def create_sequences(data, seq_length=20):

    X = []
    y = []

    for i in range(len(data) - seq_length):

        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(X), np.array(y)


def train_lstm(df, symbol, epochs=5):

    data = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data)

    X, y = create_sequences(data_scaled)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = LSTMModel()

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01
    )

    for epoch in range(epochs):

        output = model(X)

        loss = criterion(output, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        print(
            f"{symbol} | Epoch {epoch+1}/{epochs} | Loss: {loss.item():.6f}"
        )

    os.makedirs("models/intraday", exist_ok=True)

    model_path = f"models/intraday/{symbol.replace('.', '_')}_lstm.pt"

    torch.save(model.state_dict(), model_path)

    return loss.item()


# =========================================
# MAIN TRAINING LOOP
# =========================================
def main():

    print("\n🚀 INDIA STOCK AI MASTER TRAINING\n")

    results = {}

    success = 0

    for symbol in STOCKS:

        print(f"\n📊 Downloading {symbol}...")

        try:

            df = yf.download(
                symbol,
                period="5y",
                interval="1d",
                auto_adjust=True,
                progress=False
            )

            if df.empty:

                print(f"❌ No data for {symbol}")

                continue

            # Fix column structure
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = add_features(df)

            if len(df) < 100:

                print(f"❌ Not enough data for {symbol}")

                continue

            # Train XGB
            xgb_mse = train_xgb(df, symbol)

            # Train LSTM
            lstm_loss = train_lstm(df, symbol)

            results[symbol] = {
                "xgb_mse": float(xgb_mse),
                "lstm_loss": float(lstm_loss)
            }

            success += 1

        except Exception as e:

            print(f"❌ Error with {symbol}: {e}")

    os.makedirs("models", exist_ok=True)

    with open("models/training_results.json", "w") as f:

        json.dump(results, f, indent=4)

    print("\n===================================")
    print(f"✅ TRAINING COMPLETE")
    print(f"✅ Successfully trained: {success}/{len(STOCKS)}")
    print("===================================")


if __name__ == "__main__":
    main()