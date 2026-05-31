# models/intraday/lstm_model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── Model definition ──────────────────────────────────────────

class LSTMPredictor(nn.Module):
    def __init__(
        self,
        input_size: int = 19,
        hidden: int = 128,
        layers: int = 3,
        dropout: float = 0.2,
        output_size: int = 3        # sell=0  hold=1  buy=2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden,
            num_layers=layers,
            dropout=dropout,
            batch_first=True
        )
        self.attention = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True, dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]      # last timestep only
        return self.fc(out)


# ── Data preparation ──────────────────────────────────────────

def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int = 60,
    horizon: int = 1
):
    """
    Build sliding-window sequences for LSTM training.
    Labels:
      0 = SELL  (forward return < -0.5%)
      1 = HOLD  (forward return between -0.5% and +0.5%)
      2 = BUY   (forward return > +0.5%)
    """
    feat    = df[feature_cols].values
    returns = df["close"].pct_change(horizon).shift(-horizon)
    labels  = pd.cut(
        returns,
        bins=[-np.inf, -0.005, 0.005, np.inf],
        labels=[0, 1, 2]
    ).astype("Int64")

    X, y = [], []
    for i in range(seq_len, len(feat) - horizon):
        if pd.isna(labels.iloc[i]):
            continue
        X.append(feat[i - seq_len:i])
        y.append(int(labels.iloc[i]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ── Training ──────────────────────────────────────────────────

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch: int = 64,
    lr: float = 1e-3,
    patience: int = 10,
    save_path: str = "models/intraday/best_lstm.pt"
):
    n, s, f = X.shape

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, f)).reshape(n, s, f)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=batch, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch
    )

    model   = LSTMPredictor(input_size=f)
    optim   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val, no_imp = np.inf, 0

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_dl:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
        sched.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_loss += loss_fn(model(xb), yb).item()

        print(f"  Epoch {ep:3d}/{epochs} | "
              f"train {train_loss/len(train_dl):.4f} | "
              f"val {val_loss/len(val_dl):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "scaler":      scaler,
                "input_size":  f,
                "seq_len":     s,
            }, save_path)
            no_imp = 0
        else:
            no_imp += 1

        if no_imp >= patience:
            print(f"  Early stopping at epoch {ep}")
            break

    print(f"  Best val loss: {best_val:.4f} — saved to {save_path}")
    return model, scaler


# ── Inference ─────────────────────────────────────────────────

def load_lstm(path: str = "models/intraday/best_lstm.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    model = LSTMPredictor(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["scaler"], checkpoint["seq_len"]


if __name__ == "__main__":
    print("LSTM model definitions loaded OK.")