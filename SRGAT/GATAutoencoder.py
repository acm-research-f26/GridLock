from GATv2Layer import GATv2Layer
import torch
import torch.nn as nn
import torch.nn.functional as f

def batch_edge_index(edge_index, num_graphs, num_nodes):
    offsets = (torch.arange(num_graphs, device=edge_index.device) * num_nodes).view(-1, 1, 1)
    shifted = edge_index.unsqueeze(0) + offsets

    return shifted.permute(1, 0, 2).reshape(2, -1)

class GATAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=16, heads=4 ,latent=32, dropout=0.0):
        super().__init__()
        self.gat1 = GATv2Layer(n_features, hidden, heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Layer(heads * hidden, hidden, heads, concat=False, dropout=dropout)
        self.enc_lstm = nn.LSTM(hidden, latent, batch_first=True)
        self.dec_lstm = nn.LSTM(latent, latent, batch_first=True)
        self.out = nn.Linear(latent, n_features)

        self.register_buffer("threshold", torch.tensor(float("inf")))

    def encode(self, x, edge_index):
        B, T, N, F = x.shape

        ei = batch_edge_index(edge_index, B * T, N)
        h = x.reshape(B * T * N, F)
        h = f.elu(self.gat1(h, ei))
        h = f.elu(self.gat2(h, ei))

        h = h.view(B, T, N, -1).permute(0, 2, 1, 3)
        h = h.reshape(B * N, T, -1)

        _, (h_n, _) = self.enc_lstm(h)
        return h_n[-1].view(B, N, -1)

    def decode(self, z, T):
        B, N, L = z.shape
        z = z.reshape(B * N, 1, L).expand(-1, T, -1)
        out, _ = self.dec_lstm(z)
        out = self.out(out)

        return out.view(B, N, T, -1).permute(0, 2, 1, 3)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.decode(z, x.size(1)), z

    def score(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            recon, _ = self(x, edge_index)

        err = (recon - x) ** 2
        bus_score = err.mean(dim=(1, 3))
        window_score = bus_score.mean(1)

        self.train()

        return window_score, bus_score

    def fit_threshold(self, x_val, edge_index, q=0.99):
        window_score, _ = self.score(x_val, edge_index)
        threshold = torch.quantile(window_score, q)
        self.threshold = threshold

        return threshold

    def detect(self, x, edge_index):
        window_score, _ = self.score(x, edge_index)
        flags = window_score > self.threshold

        return flags, window_score


if __name__ == "__main__":
    B, T, N, F = 2, 5, 6, 3
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    ei = torch.cat([ei, torch.arange(N).repeat(2, 1)], dim=1)

    model = GATAutoencoder(F)
    x = torch.randn(B, T, N, F)
    recon, z = model(x, ei)

    assert recon.shape == x.shape
    assert z.shape == (B, N, 32)

    f.mse_loss(recon, x).backward()
    assert all(p.grad is not None for p in model.parameters())

    w, b = model.score(x, ei)
    print(w.shape, b.shape)
    assert w.shape == (B,)
    assert b.shape == (B, N)

    thr = model.fit_threshold(x, ei)
    flags, scores = model.detect(x, ei)
    print(thr, flags)
    assert flags.shape == (B,)

    model.eval()
    r_batch, z_batch = model(x, ei)
    r_one, z_one = model(x[:1], ei)
    assert torch.allclose(z_batch[0], z_one[0], atol=1e-5)
    assert torch.allclose(r_batch[0], r_one[0], atol=1e-5)

    print("ok")