"""Score raw IEEE39 measurements using a frozen checkpoint scaler and graph."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOG_DIR = ROOT / 'logs'

from GridData import _check_time, _true_flags, sequence_windows, node_columns, NORMALIZATION_VERSION
from SRGAT.GATAutoencoder import GATAutoencoder
from SRGAT.LSTMAutoencoder import LSTMBaseline


def predict(csv, checkpoint=LOG_DIR / 'srgat_ieee39.pt', output=LOG_DIR / 'ieee39_predictions.csv', batch_size=8):
    torch.set_num_threads(2)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    meta, hp = saved['preprocessing'], saved['hyperparameters']
    if meta.get('normalization_version') != NORMALIZATION_VERSION:
        raise ValueError('Checkpoint uses old normalization; retrain on regenerated data')
    path = Path(csv)
    if Path(str(path) + '.manifest.json').exists():
        raise ValueError('Prediction expects raw physical units, not normalized ML data')
    report = json.loads(Path(str(path) + '.report.json').read_text())
    if report['graph'] != meta['graph']:
        raise ValueError('Input graph differs from the training topology/feature order')
    frame = pd.read_csv(path)
    steps = _check_time(frame)
    fields = meta['graph']['node_features']
    names = node_columns(hp['node_count'], fields)
    positions = [meta['feature_columns'].index(name) for name in names]
    raw = frame[names].to_numpy(dtype=float)
    valid = np.isfinite(raw).all(axis=1)
    if 'converged' in frame:
        valid &= _true_flags(frame['converged'])
    # Retain operating-limit violations for scoring; never fit on prediction data.
    values = ((raw[valid] - np.asarray(meta['mean'])[positions]) /
              np.asarray(meta['scale'])[positions]).reshape(-1, hp['node_count'], hp['n_features'])
    windows, ends = sequence_windows(values, steps[valid], ['test'] * int(valid.sum()), hp['window'])
    if not len(windows['test']):
        raise ValueError('No complete consecutive windows available for prediction')
    if saved.get('model_type', 'gat') == 'lstm':
        model = LSTMBaseline(hp['node_count'], hp['n_features'], hp['lstm_hidden'], hp['lstm_layers'])
    else:
        model = GATAutoencoder(hp['n_features'], hp['hidden'], hp['heads'], hp['latent'], hp['dropout'])
    model.load_state_dict(saved['model_state_dict'])
    score, bus_score = model.score(torch.from_numpy(windows['test']), saved['edge_index'], batch_size)
    result = pd.DataFrame({'end_timestep': ends['test'], 'score': score.numpy(),
                           'flag': (score > model.threshold.cpu()).numpy()})
    result = pd.concat([result, pd.DataFrame(bus_score.numpy(),
                          columns=[f'Bus{i}_score' for i in range(1, hp['node_count'] + 1)])], axis=1)
    if Path(output).resolve() in {path.resolve(), Path(checkpoint).resolve()}:
        raise ValueError('Prediction output must not overwrite input data or checkpoint')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f'Saved {len(result)} window scores to {output}; excluded {int((~valid).sum())} invalid rows')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=str(ROOT / 'Dataset' / 'grid_dataset_ieee39_raw.csv'))
    parser.add_argument('--checkpoint', default=str(LOG_DIR / 'srgat_ieee39.pt'))
    parser.add_argument('--output', default=str(LOG_DIR / 'ieee39_predictions.csv'))
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    predict(args.csv, args.checkpoint, args.output, args.batch_size)
