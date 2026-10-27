"""Shared, training-only preprocessing for legacy, raw IEEE39 and prepared ML CSVs."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'Dataset'
RAW_OUTPUT = str(DATA_DIR / 'grid_dataset_ieee39_raw.csv')
ML_OUTPUT = str(DATA_DIR / 'grid_dataset_ieee39_ml.csv')
SPLITS = ('train', 'validation', 'test')
NODE_FEATURES = ('vm_pu', 'va_degree', 'p_injection_mw', 'q_injection_mvar', 'load_p_mw', 'load_q_mvar', 'incident_current_max_ka', 'allocated_loss_mw')


NORMALIZATION_VERSION = 'shared_bus_type_v2'


def node_columns(node_count, fields=NODE_FEATURES):
    """Bus-major ordering; demand uses the team's Load{n}_... CSV names."""
    return [f'Load{bus}_{field[5:]}' if field.startswith('load_') else f'Bus{bus}_{field}'
            for bus in range(1, node_count + 1) for field in fields]


def shared_scaler(training, features, graph):
    """Fit each node feature type over TRAIN time x buses; retain constants."""
    mean, std = training.mean(axis=0), training.std(axis=0)
    groups = {}
    if graph:
        for field in NODE_FEATURES:
            groups['node:' + field] = node_columns(graph['node_count'], (field,))
        if graph.get('branches') and graph.get('edge_features'):
            for i, field in enumerate(graph['edge_features']):
                groups['edge:' + field] = list(dict.fromkeys(
                    b[key][i] for b in graph['branches']
                    for key in ['feature_columns', 'reverse_feature_columns']))
    stats = {}
    for name, columns in groups.items():
        indices = [features.index(c) for c in columns]
        pooled = training[:, indices]
        mu, sigma = float(pooled.mean()), float(pooled.std())
        mean[indices], std[indices] = mu, sigma
        stats[name] = dict(mean=mu, scale=sigma if sigma > 1e-9 else 1.0,
                           columns=columns)
    return mean, np.where(std > 1e-9, std, 1.0), stats


def physical_columns(frame):
    """Legacy fallback: exclude time, status, labels and solver diagnostics explicitly."""
    pattern = (r'^(Bus\d+_(vm_pu|vm_kv|va_degree|p_injection_mw|q_injection_mvar|incident_current_max_ka|allocated_loss_mw)|'
               r'(line\d+-\d+|transform\d+-\d+|line\d+|trafo\d+)(_branch\d+)?_'
               r'(p_from_mw|q_from_mvar|p_to_mw|q_to_mvar|p_hv_mw|q_hv_mvar|p_lv_mw|q_lv_mvar|pl_mw|ql_mvar|loading_pct|loading_percent|i_from_ka|i_to_ka|i_hv_ka|i_lv_ka)|'
               r'(Load\d+|Gen\d+|Slack\d+|slack)_([pq]_(mw|mvar))|'
               r'total_(load|generation|shunt)_(mw|mvar)|(line|trafo|total)_loss_mw)$')
    return [col for col in frame if re.fullmatch(pattern, col)]


def _true_flags(series):
    values = series.astype(str).str.lower()
    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError(f'Invalid boolean status in {series.name}')
    return values.isin(['true', '1']).to_numpy()


def _check_time(frame):
    steps = pd.to_numeric(frame['timestep'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(steps).all() or not np.equal(steps, np.floor(steps)).all():
        raise ValueError('Timesteps must be finite integers')
    if len(steps) > 1 and not (np.diff(steps) > 0).all():
        raise ValueError('Timesteps must be unique and strictly chronological')
    if 'simulation_time_minutes' in frame:
        if not np.allclose(frame.simulation_time_minutes.to_numpy(), steps * 15):
            raise ValueError('Simulation time must match 15-minute timesteps')
    return steps.astype(np.int64)


def prepare_rows(csv, train_frac=0.7, val_frac=0.15):
    """Return normalized physical rows and frozen preprocessing metadata.

    Split ORIGINAL chronological row positions before filtering. Invalid rows are
    omitted without imputation; their timestep gaps survive into window construction.
    Operational-limit violations are excluded from the normal training baseline.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac or train_frac + val_frac >= 1:
        raise ValueError('Require positive train, validation and test fractions')
    path = Path(csv)
    frame = pd.read_csv(path)
    steps = _check_time(frame)
    manifest_path = Path(str(path) + '.manifest.json')
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if metadata.get('kind') != 'normalized_gridlock_dataset':
            raise ValueError('Unknown prepared dataset format')
        if not np.isclose(train_frac, metadata['train_frac']) or not np.isclose(val_frac, metadata['val_frac']):
            raise ValueError('Prepared dataset has frozen split fractions; regenerate it to change them')
        if metadata.get('normalization_version') != NORMALIZATION_VERSION:
            raise ValueError('Outdated normalization; regenerate ML data from raw measurements')
        features = metadata['feature_columns']
        values = frame[features].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError('Prepared dataset contains invalid physical measurements')
        if not frame['split'].isin(SPLITS).all():
            raise ValueError('Unknown split in prepared dataset')
        expected = np.where(steps < metadata['split_timestep_bounds'][0], 'train',
                            np.where(steps < metadata['split_timestep_bounds'][1], 'validation', 'test'))
        if not np.array_equal(frame['split'], expected):
            raise ValueError('Prepared split assignments disagree with frozen boundaries')
        return frame, values, metadata

    report_path = Path(str(path) + '.report.json')
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    graph = report.get('graph')
    if graph and (graph.get('schema_version') != 2 or graph.get('node_features') != list(NODE_FEATURES)):
        raise ValueError('Outdated raw graph schema; regenerate data with PPSim39.py')
    features = report.get('feature_columns', physical_columns(frame))
    if not features or len(features) != len(set(features)):
        raise ValueError('No physical features or duplicate feature names')
    values = frame[features].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    accepted = np.isfinite(values).all(axis=1)
    for key in ['converged', 'measurements_valid', 'within_modeled_limits']:
        if key in frame:
            accepted &= _true_flags(frame[key])
    train_end = int(len(frame) * train_frac)
    val_end = int(len(frame) * (train_frac + val_frac))
    if not 0 < train_end < val_end < len(frame):
        raise ValueError('Dataset is too short for three chronological splits')
    splits = np.where(np.arange(len(frame)) < train_end, 'train',
                      np.where(np.arange(len(frame)) < val_end, 'validation', 'test'))
    training = values[accepted & (splits == 'train')]
    if len(training) == 0:
        raise ValueError('No valid training rows; cannot fit normalization')
    active = training.std(axis=0) > 1e-9
    mean, scale, shared_stats = shared_scaler(training, features, report.get('graph'))
    if not active.any():
        raise ValueError('No varying physical training features')
    selected = frame.loc[accepted].copy()
    selected['simulation_time_minutes'] = steps[accepted] * 15
    selected['split'] = splits[accepted]
    normalized = (values[accepted] - mean) / scale
    if not np.isfinite(normalized).all():
        raise ValueError('Nonfinite normalized measurements')
    metadata = dict(kind='normalized_gridlock_dataset', source=str(path),
                    feature_columns=features, active_lstm_features=features,
                    normalization_version=NORMALIZATION_VERSION, shared_scalers=shared_stats,
                    mean=mean.tolist(), scale=scale.tolist(), active_mask=active.tolist(),
                    train_frac=train_frac, val_frac=val_frac,
                    split_timestep_bounds=[int(steps[train_end]), int(steps[val_end])],
                    original_rows=len(frame), accepted_rows=int(accepted.sum()),
                    excluded_rows=int((~accepted).sum()),
                    excluded_timesteps=steps[~accepted].tolist(),
                    split_row_counts={name: int((selected['split'] == name).sum()) for name in SPLITS},
                    graph=report.get('graph'),
                    operational_policy='Keep numerically valid states within modeled limits for normal-only training; no attack labels inferred.',
                    window_policy='No windows across split boundaries or missing/nonconsecutive timesteps; no interpolation.',
                    normalization='Node feature types pooled over all buses and valid TRAIN timesteps; edge feature types pooled over physical directed branches and TRAIN. Other columns scaled individually. Zero std uses scale 1; all columns retained.')
    return selected, normalized, metadata


def export_ml_dataset(raw_csv, output_csv=ML_OUTPUT, train_frac=0.7, val_frac=0.15):
    if Path(raw_csv).resolve() == Path(output_csv).resolve():
        raise ValueError('Raw and ML output paths must differ')
    frame, normalized, metadata = prepare_rows(raw_csv, train_frac, val_frac)
    columns = ['timestep', 'simulation_time_minutes', 'split']
    if 'within_modeled_limits' in frame:
        columns.append('within_modeled_limits')
    out = frame[columns].reset_index(drop=True)
    out = pd.concat([out, pd.DataFrame(normalized, columns=metadata['feature_columns'])], axis=1)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    Path(str(output_csv) + '.manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved {output_csv}: {out.shape}; excluded {metadata["excluded_rows"]} failed/invalid rows; '
          f'{len(metadata["active_lstm_features"])} LSTM features', flush=True)
    return out, metadata


def sequence_windows(values, timesteps, splits, window):
    """Return windows and original end timesteps without ever bridging a gap."""
    if not isinstance(window, int) or window < 1:
        raise ValueError('Window length must be a positive integer')
    result, ends = {}, {}
    for name in SPLITS:
        choose = np.asarray(splits) == name
        arr, steps = values[choose], np.asarray(timesteps)[choose]
        starts = [start for start in range(max(0, len(arr) - window + 1))
                  if np.all(np.diff(steps[start:start + window]) == 1)]
        result[name] = (np.stack([arr[start:start + window] for start in starts]).astype(np.float32)
                        if starts else np.empty((0, window, *values.shape[1:]), dtype=np.float32))
        ends[name] = np.asarray([steps[start + window - 1] for start in starts], dtype=np.int64)
    return result, ends


def load_arrays(csv, window=24, graph=False, train_frac=0.7, val_frac=0.15):
    frame, values, metadata = prepare_rows(csv, train_frac, val_frac)
    if graph:
        node_count = (metadata.get('graph') or {}).get('node_count')
        if not isinstance(node_count, int) or node_count < 1:
            raise ValueError('Graph report/manifest must specify a positive node count')
        names = node_columns(node_count)
        positions = [metadata['feature_columns'].index(name) for name in names]
        values = values[:, positions].reshape(-1, node_count, len(NODE_FEATURES))
    windows, ends = sequence_windows(values, frame.timestep.to_numpy(), frame['split'].to_numpy(), window)
    if not np.isfinite(values.astype(np.float32)).all():
        raise ValueError('Physical measurements overflow float32 model inputs')
    return windows, ends, metadata



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default=RAW_OUTPUT)
    parser.add_argument('--output', default=ML_OUTPUT)
    args = parser.parse_args()
    export_ml_dataset(args.raw, args.output)
