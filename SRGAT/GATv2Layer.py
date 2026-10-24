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
        idx = tgt_i.unsqueeze(-1).expand_as(scores)

        neg_inf = float("-inf")
        group_max = scores.new_full((x.size(0), self.heads), neg_inf)
        group_max = group_max.scatter_reduce(0, idx, scores, reduce="amax", include_self=False)

        scores = scores - group_max[tgt_i]
        exp = scores.exp()

        den = scores.new_zeros(x.size(0), self.heads).scatter_add(0, idx, exp)
        alpha = exp / (den[tgt_i] + 1e-16)

        alpha = self.dropout(alpha)
        messages = alpha.unsqueeze(-1) * x_src

        m_idx = tgt_i.view(-1, 1, 1).expand_as(messages)

        out = messages.new_zeros(x.size(0), self.heads, self.out_features).scatter_add(0, m_idx, messages)
        if self.concat:
            out = out.reshape(out.size(0), self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        return out + self.bias

if __name__ == "__main__":
    torch.manual_seed(0)
    layer = GATv2Layer(in_features=4, out_features=16, heads=4)
    x = torch.randn(14, 4)
    edges = torch.randint(0, 14, (2, 40))
    loops = torch.arange(14).repeat(2, 1)
    edge_index = torch.cat([edges, loops], dim=1)

    out = layer(x, edge_index)
    print(out.shape)
    assert out.shape == (14, 64)
    assert not torch.isnan(out).any()

    out.sum().backward()
    assert layer.att.grad is not None
    assert layer.lin_src.weight.grad is not None
    assert layer.lin_tgt.weight.grad is not None

    layer2 = GATv2Layer(4, 16, heads=4, concat=False)
    out2 = layer2(x, edge_index)
    print(out2.shape)
    assert out2.shape == (14, 16)

    print("all checks passed")

