try:
    from .Training import ReconstructionTraining
except ImportError:
    from Training import ReconstructionTraining
import torch
import torch.nn as nn


class LSTMBaseline(ReconstructionTraining, nn.Module):
    """Two-layer LSTM autoencoder over flattened bus features (no graph).
    Train on normal data only; high reconstruction error means anomaly."""

    def __init__(self, node_count, n_features, hidden=64, layers=2):
        super().__init__()
        d = node_count * n_features
        self.encoder = nn.LSTM(d, hidden, layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.linear = nn.Linear(hidden, d)

        self.register_buffer("threshold", torch.tensor(float("inf")))

    def forward(self, x, edge_index=None):
        # edge_index is accepted for interface parity and ignored.
        B, T, N, F = x.shape
        _, (h_n, _) = self.encoder(x.reshape(B, T, N * F))
        dec_in = h_n[-1].unsqueeze(1).repeat(1, T, 1)  # (B, T, H)
        dec_out, _ = self.decoder(dec_in)
        return self.linear(dec_out).view(B, T, N, F), h_n[-1]


if __name__ == "__main__":
    B, T, N, F = 2, 5, 6, 3
    ei = torch.arange(N).repeat(2, 1)

    model = LSTMBaseline(N, F, hidden=16)
    x = torch.randn(B, T, N, F)
    recon, z = model(x, ei)

    assert recon.shape == x.shape
    assert z.shape == (B, 16)

    nn.functional.mse_loss(recon, x).backward()
    assert all(p.grad is not None for p in model.parameters())

    w, b = model.score(x, ei)
    assert w.shape == (B,)
    assert b.shape == (B, N)

    thr = model.fit_threshold(x, ei)
    flags, scores = model.detect(x, ei)
    print(thr, flags)
    assert flags.shape == (B,)

    print("ok")
