#Smoke Test for GATv2Layer and GATAutoencoder

import torch
import torch.nn.functional as F

from GATv2Layer import GATv2Layer
from GATAutoencoder import GATAutoencoder


def smoke_test():
    torch.manual_seed(0)

    B, T, N, D = 3, 5, 6, 3
    x = torch.randn(B, T, N, D)

    # Directed edges. Self-loops are added automatically.
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 0]],
        dtype=torch.long,
    )

    # Standalone GAT checks.
    for concat in (True, False):
        layer = GATv2Layer(D, 8, heads=4, concat=concat)
        result = layer(x[0, 0], edge_index)
        expected_width = 32 if concat else 8
        assert result.shape == (N, expected_width)
        assert torch.isfinite(result).all()

        result.square().mean().backward()
        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    # Empty graph is valid; automatic loops retain each node.
    empty_edges = torch.empty((2, 0), dtype=torch.long)
    assert layer(x[0, 0], empty_edges).shape == (N, 8)

    model = GATAutoencoder(
        D,
        hidden=16,
        heads=4,
        latent=32,
        dropout=0.1,
        residual=True,
        layer_norm=True,
    )

    reconstruction, latent = model(x, edge_index)
    assert reconstruction.shape == x.shape
    assert latent.shape == (B, N, 32)

    F.mse_loss(reconstruction, x).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name

    # Scoring must not unexpectedly change the model's mode.
    for training in (True, False):
        model.train(training)
        window_scores, bus_scores = model.score(x, edge_index)
        assert model.training == training
        assert window_scores.shape == (B,)
        assert bus_scores.shape == (B, N)
        assert not window_scores.requires_grad

    # Detecting before calibration should fail explicitly.
    try:
        model.detect(x, edge_index)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Uncalibrated detection should fail.")

    threshold = model.fit_threshold(x, edge_index)
    flags, scores = model.detect(x, edge_index)
    assert threshold.ndim == 0
    assert flags.shape == (B,)
    assert flags.dtype == torch.bool
    assert scores.shape == (B,)

    # Disjoint graph batching must not mix batch items.
    model.eval()
    with torch.no_grad():
        r_batch, z_batch = model(x, edge_index)
        r_one, z_one = model(x[:1], edge_index)

    torch.testing.assert_close(z_batch[:1], z_one, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(r_batch[:1], r_one, atol=1e-5, rtol=1e-4)

    print("All smoke tests passed.")


if __name__ == "__main__":
    smoke_test()