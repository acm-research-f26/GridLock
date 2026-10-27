# GATv2Layer.py
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def validate_edge_index(
    edge_index: Tensor,
    num_nodes: int,
    device: torch.device,
) -> None:
    """Validate a COO graph: edge_index[0] -> edge_index[1]."""
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges].")

    if edge_index.dtype != torch.long:
        raise TypeError("edge_index must have dtype torch.long.")

    if edge_index.device != device:
        raise ValueError("edge_index and node features must be on the same device.")

    if edge_index.numel():
        if edge_index.min().item() < 0:
            raise ValueError("edge_index contains negative node indices.")
        if edge_index.max().item() >= num_nodes:
            raise ValueError("edge_index contains an out-of-range node index.")


def with_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Replace existing self-loops with exactly one self-loop per node."""
    non_loop_edges = edge_index[:, edge_index[0] != edge_index[1]]
    nodes = torch.arange(num_nodes, device=edge_index.device)
    loops = torch.stack((nodes, nodes))
    return torch.cat((non_loop_edges, loops), dim=1)


class GATv2Layer(nn.Module):
    """
    Sparse multi-head GATv2 using only PyTorch.

    Inputs:
        x: [num_nodes, in_features]
        edge_index: [2, num_edges], with source -> target edges

    Output:
        [num_nodes, heads * out_features] if concat=True
        [num_nodes, out_features] otherwise

    Non-self duplicate edges are retained and count as separate edges.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        *,
        add_self_loops: bool = True,
        residual: bool = False,
        bias: bool = True,
        validate_inputs: bool = True,
    ):
        super().__init__()

        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("heads", heads),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not math.isfinite(negative_slope) or negative_slope < 0:
            raise ValueError("negative_slope must be finite and nonnegative.")

        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.add_self_loops = add_self_loops
        self.validate_inputs = validate_inputs

        projected_dim = heads * out_features
        output_dim = projected_dim if concat else out_features

        self.lin_src = nn.Linear(in_features, projected_dim, bias=False)
        self.lin_tgt = nn.Linear(in_features, projected_dim, bias=False)

        self.att = nn.Parameter(torch.empty(1, heads, out_features))
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Linear(in_features, output_dim, bias=False)
            if residual else None
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_tgt.weight)
        nn.init.xavier_uniform_(self.att)

        if self.residual is not None:
            nn.init.xavier_uniform_(self.residual.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.validate_inputs:
            if x.ndim != 2 or x.size(1) != self.in_features:
                raise ValueError(
                    f"x must have shape [num_nodes, {self.in_features}]."
                )
            if not x.is_floating_point():
                raise TypeError("x must be floating point.")

            validate_edge_index(edge_index, x.size(0), x.device)

        num_nodes = x.size(0)

        if self.add_self_loops:
            edge_index = with_self_loops(edge_index, num_nodes)

        source, target = edge_index

        src = self.lin_src(x).reshape(
            num_nodes, self.heads, self.out_features
        )
        tgt = self.lin_tgt(x).reshape(
            num_nodes, self.heads, self.out_features
        )

        # Preserve float64; promote low-precision attention math to float32.
        work_dtype = (
            torch.float32
            if src.dtype in (torch.float16, torch.bfloat16)
            else src.dtype
        )

        src_edges = src[source].to(work_dtype)
        tgt_edges = tgt[target].to(work_dtype)

        attention_features = F.leaky_relu(
            src_edges + tgt_edges,
            negative_slope=self.negative_slope,
        )
        logits = (
            attention_features * self.att.to(work_dtype)
        ).sum(dim=-1)

        index = target[:, None].expand(-1, self.heads)

        # Destination-wise stable softmax for each attention head.
        maxima = logits.new_full(
            (num_nodes, self.heads), -torch.inf
        )
        maxima.scatter_reduce_(
            0,
            index,
            logits.detach(),
            reduce="amax",
            include_self=True,
        )

        weights = (logits - maxima[target]).exp()
        denominator = logits.new_zeros((num_nodes, self.heads))
        denominator.scatter_add_(0, index, weights)

        # Every represented target has at least one exponent equal to 1.
        alpha = weights / denominator[target].clamp_min(
            torch.finfo(work_dtype).tiny
        )
        alpha = self.dropout(alpha)

        messages = alpha.unsqueeze(-1) * src_edges

        out = messages.new_zeros(
            (num_nodes, self.heads, self.out_features)
        )
        out.index_add_(0, target, messages)

        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_features)
        else:
            out = out.mean(dim=1)

        out = out.to(src.dtype)

        if self.residual is not None:
            out = out + self.residual(x)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"heads={self.heads}, concat={self.concat}, "
            f"add_self_loops={self.add_self_loops}"
        )