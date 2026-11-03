"""Train and compare SRGAT, NAGAT or flat LSTM autoencoders on identical IEEE39 node windows."""
import argparse
import json
import numpy as np
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / "logs"

from GridData import ML_OUTPUT, NODE_FEATURES
from SRGAT.Data import load_graph_data
from SRGAT.Evaluate import inject_attack, detection_metrics
from SRGAT.GATAutoencoder import GATAutoencoder
from SRGAT.LSTMAutoencoder import LSTMBaseline
from SRGAT.NAGATModel import NAGATAutoencoder

DEFAULT_SAVE = {"gat": "srgat_ieee39.pt", "nagat": "nagat_ieee39.pt", "lstm": "lstm_ieee39.pt"}


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
    attack_buses=3,
    attack_magnitude=1.0,
    attack_features=("p_injection_mw", "q_injection_mvar"),
):
    if model_type not in DEFAULT_SAVE:
        raise ValueError(f"model_type must be one of {', '.join(DEFAULT_SAVE)}")
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
    noise = metadata.get("measurement_noise") or {}
    overrides = ", ".join(f"{name} {rel:.1%}" for name, rel in (noise.get("feature_rel") or {}).items())
    print(f"  Noise:      {noise.get('rel', 0.0):.0%} on {noise.get('columns', 0)} metered P/Q columns "
          f"(same draws as PhysicsDetector) + {noise.get('feature_columns', 0)} other node-feature columns"
          + (f" ({overrides})" if overrides else ""))
    print(f"Using device: {device}")

    if any(len(x) == 0 for x in splits.values()):
        raise ValueError("Each split needs at least one complete window; generate more samples")
    n_features = train_x.shape[-1]
    if model_type == "lstm":
        model = LSTMBaseline(metadata['graph']['node_count'], n_features,
                             hidden=lstm_hidden, layers=lstm_layers).to(device)
    elif model_type == "nagat":
        model = NAGATAutoencoder(n_features=n_features, hidden=hidden, heads=heads,
                                 latent=latent, dropout=dropout).to(device)
    else:
        model = GATAutoencoder(n_features=n_features, hidden=hidden, heads=heads,
                              latent=latent, dropout=dropout).to(device)

    print(f"\n--- Training {model_type.upper()} autoencoder ---")
    start = time.perf_counter()
    history = model.fit(
        train_data=train_x,
        edge_index=edges,
        val_data=val_x,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        verbose=True,
    )
    train_seconds = time.perf_counter() - start
    print(f"Training time: {train_seconds:.1f}s")

    print("\n--- Calibrating Anomaly Detection Threshold ---")
    threshold = model.fit_threshold(val_x, edges, q=quantile)
    print(f"Empirical threshold ({quantile * 100:.1f}th quantile on validation): {float(threshold):.6f}")

    print("\n--- Evaluating on Held-Out Test Windows ---")
    flags, test_scores = model.detect(test_x, edges)
    flagged_count = int(flags.sum().item())
    print(f"Test windows: {len(test_x)}")
    print(f"Flagged anomalies: {flagged_count} ({flagged_count / len(test_x) * 100:.2f}%)")
    print(f"Test score range: min {float(test_scores.min()):.6f}, max {float(test_scores.max()):.6f}, mean {float(test_scores.mean()):.6f}")

    print("\n--- Synthetic FDI Detection (normal test vs attacked copy) ---")
    feature_idx = [NODE_FEATURES.index(name) for name in attack_features]
    attacked_x, _ = inject_attack(test_x, feature_idx, attack_buses, attack_magnitude, seed)
    attack_flags, attack_scores = model.detect(attacked_x, edges)
    labels = torch.cat([torch.zeros(len(test_x)), torch.ones(len(attacked_x))])
    detection = detection_metrics(labels, torch.cat([test_scores, attack_scores]),
                                  torch.cat([flags, attack_flags]))
    print(f"Attack: ramp to +/-{attack_magnitude} (normalized) on {', '.join(attack_features)} "
          f"at {attack_buses} random buses per window")
    print(f"AUC: {detection['auc']:.4f} | F1: {detection['f1']:.4f} "
          f"(precision {detection['precision']:.4f}, recall {detection['recall']:.4f})")

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
                      csv=str(Path(csv).resolve()), window=window, train_seconds=train_seconds,
                      measurement_noise=metadata.get("measurement_noise"),
                      best_val_loss=min(history["val_loss"]),
                      split_windows={name: len(value) for name, value in splits.items()},
                      test_flagged=flagged_count, test_windows=len(test_x),
                      normal_test_flag_rate=flagged_count / len(test_x), history=history,
                      score_min=float(test_scores.min()), score_max=float(test_scores.max()),
                      attack=dict(buses=attack_buses, magnitude=attack_magnitude,
                                  features=list(attack_features)),
                      detection=detection,
                      interpretation="Flag rate is on normal test data. AUC/F1 compare normal test windows "
                                     "with a synthetic ramp-injection copy, not a stealthy FDI attack.")
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
    parser.add_argument("--model", choices=list(DEFAULT_SAVE), default='gat')
    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--attack-buses", type=int, default=3, help="Buses attacked per window")
    parser.add_argument("--attack-magnitude", type=float, default=1.0, help="Peak injected offset (normalized units)")
    parser.add_argument("--attack-features", nargs="+", choices=NODE_FEATURES,
                        default=["p_injection_mw", "q_injection_mvar"], help="Node features the attack modifies")
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
        save_path=args.save or LOG_DIR / DEFAULT_SAVE[args.model],
        model_type=args.model,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        attack_buses=args.attack_buses,
        attack_magnitude=args.attack_magnitude,
        attack_features=args.attack_features,
        device=args.device,
        seed=args.seed,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
