#GAT Autoencoder
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from GATv2Layer import (
    GATv2Layer,
    validate_edge_index,
    with_self_loops,
)


def batch_edge_index(
    edge_index: Tensor,
    num_graphs: int,
    num_nodes: int,
) -> Tensor:
    """Create a disjoint union of equally sized copies of one graph."""
    if num_graphs <= 0 or num_nodes <= 0:
        raise ValueError("num_graphs and num_nodes must be positive.")

    offsets = (
        torch.arange(num_graphs, device=edge_index.device)
        .reshape(num_graphs, 1, 1)
        * num_nodes
    )
    shifted = edge_index.unsqueeze(0) + offsets
    return shifted.permute(1, 0, 2).reshape(2, -1)


@contextmanager
def evaluating(module: nn.Module):
    """Temporarily evaluate, restoring even mixed submodule modes."""
    previous_modes = [(child, child.training) for child in module.modules()]
    try:
        module.eval()
        yield
    finally:
        for child, was_training in previous_modes:
            child.training = was_training


class GATAutoencoder(nn.Module):
    """
    Spatial GATv2 encoder followed by a per-node temporal LSTM autoencoder.

    Input:
        x: [batch, time, nodes, features]
        edge_index: [2, edges], shared by all windows and time steps

    Returns:
        reconstruction: [batch, time, nodes, features]
        latent: [batch, nodes, latent_dim]
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 16,
        heads: int = 4,
        latent: int = 32,
        dropout: float = 0.0,
        *,
        feature_dropout: float = 0.0,
        residual: bool = False,
        layer_norm: bool = False,
        add_self_loops: bool = True,
        validate_inputs: bool = True,
    ):
        super().__init__()

        for name, value in (
            ("n_features", n_features),
            ("hidden", hidden),
            ("heads", heads),
            ("latent", latent),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if not 0.0 <= feature_dropout < 1.0:
            raise ValueError("feature_dropout must be in [0, 1).")

        self.n_features = n_features
        self.latent = latent
        self.add_self_loops = add_self_loops
        self.validate_inputs = validate_inputs

        # Validate and add loops once, before batching, rather than in each GAT.
        gat_options = dict(
            heads=heads,
            dropout=dropout,
            residual=residual,
            add_self_loops=False,
            validate_inputs=False,
        )

        self.gat1 = GATv2Layer(
            n_features, hidden, concat=True, **gat_options
        )
        self.gat2 = GATv2Layer(
            heads * hidden, hidden, concat=False, **gat_options
        )

        self.norm1 = (
            nn.LayerNorm(heads * hidden) if layer_norm else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(hidden) if layer_norm else nn.Identity()
        self.feature_dropout = nn.Dropout(feature_dropout)

        self.enc_lstm = nn.LSTM(hidden, latent, batch_first=True)
        self.dec_lstm = nn.LSTM(latent, latent, batch_first=True)
        self.out = nn.Linear(latent, n_features)

        # NaN explicitly represents an uncalibrated detector.
        self.register_buffer("threshold", torch.tensor(float("nan")))

    def _validate_x(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, time, nodes, features].")
        if any(size == 0 for size in x.shape):
            raise ValueError("All dimensions of x must be nonzero.")
        if x.size(-1) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.size(-1)}."
            )
        if not x.is_floating_point():
            raise TypeError("x must be floating point.")
        if x.device != self.out.weight.device:
            raise ValueError("x and model parameters must be on the same device.")

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        if self.validate_inputs:
            self._validate_x(x)
            validate_edge_index(edge_index, x.size(2), x.device)

        batch, time, nodes, features = x.shape

        if self.add_self_loops:
            edge_index = with_self_loops(edge_index, nodes)

        batched_edges = batch_edge_index(edge_index, batch * time, nodes)

        h = x.reshape(batch * time * nodes, features)
        h = self.feature_dropout(F.elu(self.norm1(self.gat1(h, batched_edges))))
        h = self.feature_dropout(F.elu(self.norm2(self.gat2(h, batched_edges))))

        # Independent temporal sequences, one per node in each batch item.
        h = (
            h.reshape(batch, time, nodes, -1)
            .permute(0, 2, 1, 3)
            .reshape(batch * nodes, time, -1)
        )

        _, (hidden, _) = self.enc_lstm(h)
        return hidden[-1].reshape(batch, nodes, self.latent)

    def decode(self, z: Tensor, T: int) -> Tensor:
        if z.ndim != 3 or z.size(-1) != self.latent:
            raise ValueError(
                f"z must have shape [batch, nodes, {self.latent}]."
            )
        if not isinstance(T, int) or isinstance(T, bool) or T <= 0:
            raise ValueError("T must be a positive integer.")

        batch, nodes, latent = z.shape

        decoder_input = (
            z.reshape(batch * nodes, 1, latent)
            .expand(-1, T, -1)
            .contiguous()
        )

        decoded, _ = self.dec_lstm(decoder_input)
        reconstruction = self.out(decoded)

        return (
            reconstruction.reshape(batch, nodes, T, self.n_features)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        z = self.encode(x, edge_index)
        return self.decode(z, x.size(1)), z

    @torch.no_grad()
    def score(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        """Return window MSE [batch] and per-node MSE [batch, nodes]."""
        with evaluating(self):
            reconstruction, _ = self(x, edge_index)

            work_dtype = (
                torch.float32
                if x.dtype in (torch.float16, torch.bfloat16)
                else x.dtype
            )
            error = (
                reconstruction.to(work_dtype) - x.to(work_dtype)
            ).square()

            bus_score = error.mean(dim=(1, 3))
            window_score = bus_score.mean(dim=1)

        return window_score, bus_score

    @torch.no_grad()
    def fit_threshold(
        self,
        x_val: Tensor,
        edge_index: Tensor,
        q: float = 0.99,
    ) -> Tensor:
        """Calibrate using reconstruction scores from normal validation data."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must be strictly between 0 and 1.")

        window_score, _ = self.score(x_val, edge_index)

        if not torch.isfinite(window_score).all().item():
            raise ValueError("Validation scores contain NaN or infinity.")

        # quantile requires float32/float64 input.
        quantile_scores = (
            window_score
            if window_score.dtype == torch.float64
            else window_score.float()
        )
        threshold = torch.quantile(quantile_scores, q)

        # Preserve the registered buffer rather than replacing its identity.
        self.threshold.copy_(threshold.to(self.threshold))

        if not torch.isfinite(self.threshold).item():
            raise ValueError("Threshold is not finite in the buffer's dtype.")

        return self.threshold.detach().clone()

    @torch.no_grad()
    def detect(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if not torch.isfinite(self.threshold).item():
            raise RuntimeError(
                "Threshold is not calibrated. Call fit_threshold() first."
            )

        window_score, _ = self.score(x, edge_index)

        if not torch.isfinite(window_score).all().item():
            raise ValueError("Detection scores contain NaN or infinity.")

        flags = window_score > self.threshold
        return flags, window_score