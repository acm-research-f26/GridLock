"""Graph window data loader for IEEE 39-bus network."""
import sys
from pathlib import Path
import torch

# Ensure GridLockFinal root is in sys.path for GridData import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GridData import load_arrays, NODE_FEATURES, ML_OUTPUT


def load_graph_data(
    csv=ML_OUTPUT,
    window=24,
    train_frac=0.7,
    val_frac=0.15,
    add_self_loops=True,
):
    """Load train/validation/test graph windows [B, T, N, F] and edge_index for IEEE 39-bus."""
    arrays, end_timesteps, metadata = load_arrays(
        csv, window=window, graph=True, train_frac=train_frac, val_frac=val_frac
    )
    graph = metadata.get("graph")
    if not graph:
        raise ValueError("Missing graph metadata in dataset manifest or report")

    node_count = graph["node_count"]
    if graph.get("node_features") != list(NODE_FEATURES):
        raise ValueError("Graph feature order differs from the loader schema")
    edges = torch.tensor(graph["edge_index"], dtype=torch.long)

    if edges.ndim != 2 or edges.shape[0] != 2 or edges.numel() == 0:
        raise ValueError("Graph edges must have shape [2, E]")
    if edges.min() < 0 or edges.max() >= node_count:
        raise ValueError("Graph edges reference nonexistent buses")

    if add_self_loops:
        self_loops = torch.arange(node_count, dtype=torch.long).repeat(2, 1)
        edges = torch.cat([edges, self_loops], dim=1)

    tensors = {key: torch.from_numpy(value) for key, value in arrays.items()}
    return tensors, edges, end_timesteps, metadata


def load_edge_data(csv=ML_OUTPUT, window=24, train_frac=0.7,
                   val_frac=0.15, add_self_loops=True):
    """Uniform [B,T,E,3] edge windows in exactly the edge_index order.

    Slots: loading %, source-terminal MW, source-terminal Mvar. Reverse edges
    use the actual opposite terminal, preserving losses/charging. Self-loop
    attributes are neutral normalized zeros and physical_mask marks them False.
    The team's node-only GAT does not consume these optional attributes.
    """
    import numpy as np
    from GridData import prepare_rows, sequence_windows
    frame, values, metadata = prepare_rows(csv, train_frac, val_frac)
    graph = metadata['graph']
    columns = [b[key] for key in ['feature_columns', 'reverse_feature_columns']
               for b in graph['branches']]
    positions = [[metadata['feature_columns'].index(c) for c in cols] for cols in columns]
    attributes = values[:, np.asarray(positions)]
    edges = torch.tensor(graph['edge_index'], dtype=torch.long)
    mask = torch.ones(edges.shape[1], dtype=torch.bool)
    if add_self_loops:
        count = graph['node_count']
        attributes = np.concatenate([attributes, np.zeros((len(frame), count, 3))], axis=1)
        edges = torch.cat([edges, torch.arange(count).repeat(2, 1)], dim=1)
        mask = torch.cat([mask, torch.zeros(count, dtype=torch.bool)])
    arrays, ends = sequence_windows(attributes, frame.timestep.to_numpy(),
                                     frame['split'].to_numpy(), window)
    return {k: torch.from_numpy(v) for k, v in arrays.items()}, edges, mask, ends, metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=ML_OUTPUT)
    args = parser.parse_args()

    splits, edges, ends, meta = load_graph_data(args.csv)
    print("Graph splits loaded:")
    for k, v in splits.items():
        print(f"  {k}: shape {tuple(v.shape)}")
    print("Edge index shape:", tuple(edges.shape))
