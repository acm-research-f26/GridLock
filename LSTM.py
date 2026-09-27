import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, layers, batch_first=True)
        self.linear = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, (h_n, _) = self.encoder(x)
