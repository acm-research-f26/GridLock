"""Train and compare GAT or flat LSTM autoencoders on identical IEEE39 node windows."""
import argparse
import json
import numpy as np
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / "logs"

from GridData import ML_OUTPUT
from SRGAT.Data import load_graph_data
from SRGAT.GATAutoencoder import GATAutoencoder
from SRGAT.LSTMAutoencoder import LSTMBaseline


def train_gat(
    csv=ML_OUTPUT,
    window=24,
    epochs=30,
    batch_size=32,
    lr=1e-3,
    hidden=16,
    heads=4,
    latent=32,
    dropout=0.0,
    quantile=0.99,
    save_path=LOG_DIR / "srgat_ieee39.pt",
    device=None,
    seed=0,
    threads=2,
    model_type="gat",
    lstm_hidden=64,
    lstm_layers=2,
):
    if model_type not in {"gat", "lstm"}:
        raise ValueError("model_type must be gat or lstm")
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading IEEE 39-bus dataset from: {csv}")
    splits, edges, end_timesteps, metadata = load_graph_data(csv, window=window)

    train_x = splits["train"]
    val_x = splits["validation"]
    test_x = splits["test"]

    print(f"Windows loaded (window={window} steps / {window * 15 / 60:.1f}h):")
    print(f"  Train:      {tuple(train_x.shape)} (B, T, N, F)")
    print(f"  Validation: {tuple(val_x.shape)}")
    print(f"  Test:       {tuple(test_x.shape)}")
    print(f"  Graph:      {metadata['graph']['node_count']} nodes, {edges.shape[1]} directed edges (incl. self-loops)")
    print(f"Using device: {device}")

    if any(len(x) == 0 for x in splits.values()):
        raise ValueError("Each split needs at least one complete window; generate more samples")
    n_features = train_x.shape[-1]
    if model_type == "lstm":
        model = LSTMBaseline(metadata['graph']['node_count'], n_features,
                             hidden=lstm_hidden, layers=lstm_layers).to(device)
    else:
        model = GATAutoencoder(n_features=n_features, hidden=hidden, heads=heads,
                              latent=latent, dropout=dropout).to(device)

    print(f"\n--- Training {model_type.upper()} autoencoder ---")
    history = model.fit(
        train_data=train_x,
        edge_index=edges,
        val_data=val_x,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )

    print("\n--- Calibrating Anomaly Detection Threshold ---")
    threshold = model.fit_threshold(val_x, edges, q=quantile)
    print(f"Empirical threshold ({quantile * 100:.1f}th quantile on validation): {float(threshold):.6f}")

    print("\n--- Evaluating on Held-Out Test Windows ---")
    flags, test_scores = model.detect(test_x, edges)
    flagged_count = int(flags.sum().item())
    print(f"Test windows: {len(test_x)}")
    print(f"Flagged anomalies: {flagged_count} ({flagged_count / len(test_x) * 100:.2f}%)")
    print(f"Test score range: min {float(test_scores.min()):.6f}, max {float(test_scores.max()):.6f}, mean {float(test_scores.mean()):.6f}")

    if save_path:
        checkpoint = {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "threshold": float(threshold),
            "hyperparameters": {
                "n_features": n_features,
                "lstm_hidden": lstm_hidden,
                "lstm_layers": lstm_layers,
                "hidden": hidden,
                "heads": heads,
                "latent": latent,
                "dropout": dropout,
                "window": window,
                "node_count": metadata["graph"]["node_count"],
            },
            "history": history,
            "preprocessing": metadata,
            "edge_index": edges.cpu(),
            "seed": seed,
            "training_config": dict(epochs=epochs, batch_size=batch_size, lr=lr,
                                    quantile=quantile, threads=threads, device=device),
            "test_end_timesteps": end_timesteps["test"].tolist(),
        }
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        report = dict(model_type=model_type, parameter_count=sum(p.numel() for p in model.parameters()), seed=seed, device=device, epochs=epochs, threshold=float(threshold),
                      split_windows={name: len(value) for name, value in splits.items()},
                      test_flagged=flagged_count, test_windows=len(test_x),
                      normal_test_flag_rate=flagged_count / len(test_x), history=history,
                      score_min=float(test_scores.min()), score_max=float(test_scores.max()),
                      interpretation="Normal synthetic test data only; flag rate is not attack detection accuracy.")
        Path(str(save_path) + '.metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        np.savetxt(str(save_path) + '.scores.csv',
                   np.column_stack([end_timesteps['test'], test_scores.numpy(), flags.numpy()]),
                   delimiter=',', header='end_timestep,score,flag', comments='')
        print(f"\nModel checkpoint saved to: {save_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=ML_OUTPUT, help="ML or raw dataset path")
    parser.add_argument("--window", type=int, default=24, help="Window size in 15-minute timesteps")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=16, help="GAT hidden dimension per head")
    parser.add_argument("--heads", type=int, default=4, help="Number of GAT attention heads")
    parser.add_argument("--latent", type=int, default=32, help="LSTM latent representation size")
    parser.add_argument("--dropout", type=float, default=0.0, help="GAT dropout rate")
    parser.add_argument("--quantile", type=float, default=0.99, help="Threshold quantile on validation data")
    parser.add_argument("--save", default=None, help="Path to save trained weights")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--model", choices=['gat', 'lstm'], default='gat')
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    args = parser.parse_args()

    train_gat(
        csv=args.csv,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        heads=args.heads,
        latent=args.latent,
        dropout=args.dropout,
        quantile=args.quantile,
        save_path=args.save or LOG_DIR / ("srgat_ieee39.pt" if args.model == "gat" else "lstm_ieee39.pt"),
        model_type=args.model,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        device=args.device,
        seed=args.seed,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
