import torch
import torch.nn as nn
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

    def fit(self, data, epochs=50, batch_size=32, lr=1e-3, verbose=True):
        """data: FloatTensor (n_windows, seq_len, n_features), normal operation."""
        loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()
        for epoch in range(epochs):
            total = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = loss_fn(self(batch), batch)
                loss.backward()
                opt.step()
                total += loss.item() * batch.size(0)
            if verbose:
                print(f"epoch {epoch + 1:3d}  loss {total / len(data):.6f}")

    @torch.no_grad()
    def score(self, data):
        """Per-window reconstruction error. Returns (n_windows,)."""
        self.eval()
        return ((self(data) - data) ** 2).mean(dim=(1, 2))