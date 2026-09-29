import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


class LSTMBaseline(nn.Module):
    """Two-layer LSTM autoencoder. Train on normal data only;
    high reconstruction error at inference means anomaly."""

    def __init__(self, n_features, hidden=64, layers=2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.linear = nn.Linear(hidden, n_features)

    def forward(self, x):                                   # (B, T, F)
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder(x)
        dec_in = h_n[-1].unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, H)
        dec_out, _ = self.decoder(dec_in)
        return self.linear(dec_out)                          # (B, T, F)

    def fit(self, data, val=None, epochs=50, batch_size=32, lr=1e-3, verbose=True):
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            total = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = loss_fn(self(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)
            train_loss = total / len(data)

            msg = f"epoch {epoch + 1:3d}  train {train_loss:.6f}"
            if val is not None:
                self.eval()
                with torch.no_grad():
                    msg += f"  val {loss_fn(self(val), val).item():.6f}"
            if verbose:
                print(msg)

    @torch.no_grad()
    def score(self, data):
        """Per-window reconstruction error. Returns (n_windows,)."""
        self.eval()
        return ((self(data) - data) ** 2).mean(dim=(1, 2))

def load_grid_data(csv, window=24, train_frac=0.7):
    df = pd.read_csv(csv).drop(columns=["timestep"])
    df = df[df.columns[df.std() > 1e-9]]  # safety net
    arr = df.to_numpy(dtype=np.float32)

    split = int(len(arr) * train_frac)
    mu = arr[:split].mean(axis=0)
    sigma = arr[:split].std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    scaled = (arr - mu) / sigma

    def windows(a):
        return torch.from_numpy(
            np.stack([a[i:i + window] for i in range(len(a) - window + 1)]))

    return windows(scaled[:split]), windows(scaled[split:]), (mu, sigma)

if __name__ == "__main__":
    train_windows, val_windows, scaler = load_grid_data("grid_dataset.csv")

    model = LSTMBaseline(n_features=train_windows.shape[2])
    model.fit(train_windows)

    errors = model.score(val_windows)
    threshold = errors.mean() + 3 * errors.std()
    print(f"threshold: {threshold:.4f}  (val error mean {errors.mean():.4f})")

    torch.save({"state_dict": model.state_dict(),
                "threshold": threshold,
                "mu": scaler[0], "sigma": scaler[1]}, "baseline.pt")