import torch
import torch.nn as nn
import torch.nn.functional as f


class GATv2Layer(nn.Module):
    def __init__(self, in_features, out_features, heads=4,
                 concat=True, negative_slope=0.2, dropout=0.0):
        super().__init__()

        self.lin_src = nn.Linear(in_features, heads*out_features, bias=False)
        self.lin_tgt = nn.Linear(in_features, heads*out_features, bias=False)

        self.att = nn.Parameter(torch.zeros(1, heads, out_features))
        nn.init.xavier_uniform_(self.att)

        out_dim = heads * out_features if concat else out_features
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.out_features = out_features
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)
        self.concat = concat
        self.heads = heads

    def forward(self, x, edge_index):
        src = self.lin_src(x).view(-1, self.heads, self.out_features)
        tgt = self.lin_tgt(x).view(-1, self.heads, self.out_features)

        src_i, tgt_i = edge_index[0], edge_index[1]
        x_src = src[src_i]
        x_tgt = tgt[tgt_i]

        relu = f.leaky_relu(x_src + x_tgt, self.negative_slope)
        scores = (self.att * relu).sum(dim=-1)


