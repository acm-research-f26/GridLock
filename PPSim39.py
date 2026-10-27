"""Reproducible IEEE 39-bus AC snapshots. See README.md for assumptions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

DATA_DIR = Path(__file__).resolve().parent / "Dataset"
DEFAULT_OUTPUT = str(DATA_DIR / "grid_dataset_ieee39_raw.csv")
BALANCE_TOL_MW = 1e-5
LIMIT_TOL = 1e-5
SOLVER = dict(calculate_voltage_angles=True, init="flat", verbose=False,
              OPF_VIOLATION=1e-8, PDIPM_FEASTOL=1e-9, PDIPM_GRADTOL=1e-8,
              PDIPM_COMPTOL=1e-8, PDIPM_COSTTOL=1e-8)


def solve_dispatch(net):
    """AC optimal power flow with published generator/voltage/branch limits."""
    pp.runopp(net, **SOLVER)
    # Refine the optimized operating point with a tight AC power-flow solve.
    net.gen['p_mw'] = net.res_gen.p_mw.to_numpy()
    net.gen['vm_pu'] = net.res_bus.loc[net.gen.bus, 'vm_pu'].to_numpy()
    net.ext_grid['vm_pu'] = net.res_bus.loc[net.ext_grid.bus, 'vm_pu'].to_numpy()
    pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10,
             max_iteration=50, enforce_q_lims=True, init='results', numba=False)

# Engineering scenario envelope, not an empirical confidence interval.
DEMAND_BOUNDS = (0.75, 1.15)


def create_ieee39_network():
    """Retain published topology, ratings and limits; bound the initial voltage guess."""
    net = pn.case39()
    if len(net.bus) != 39 or len(net.line) + len(net.trafo) != 46:
        raise ValueError("Unexpected IEEE 39-bus topology")
    net.gen['vm_pu'] = np.minimum(net.gen.vm_pu.to_numpy(), net.bus.loc[net.gen.bus, 'max_vm_pu'].to_numpy())
    return net


def daily_profile(n_samples, steps_per_day=96):
    """Midnight trough, broad daytime peak and secondary evening component."""
    phase = np.arange(n_samples) * 2 * np.pi / steps_per_day
    return 0.95 - 0.12 * np.cos(phase) - 0.03 * np.cos(2 * phase - np.pi / 3)


def ar1_noise(n_samples, n_series, rho, sigma, rng):
    """Stationary, reproducible AR(1) demand fluctuations."""
    noise = np.empty((n_samples, n_series))
    noise[0] = rng.normal(0, sigma, n_series)
    scale = sigma * np.sqrt(1 - rho ** 2)
    for k in range(1, n_samples):
        noise[k] = rho * noise[k - 1] + scale * rng.standard_normal(n_series)
    return noise


def demand_multipliers(n_samples, n_loads, seed=0, rho=0.9, noise_std=0.02):
    rng = np.random.default_rng(seed)
    raw = daily_profile(n_samples)[:, None] * (
        1 + ar1_noise(n_samples, n_loads, rho, noise_std, rng))
    return np.clip(raw, *DEMAND_BOUNDS)


def _limit_count(values, lower, upper):
    values = np.asarray(values, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    return int(((np.isfinite(lower) & (values < lower - LIMIT_TOL)) |
                (np.isfinite(upper) & (values > upper + LIMIT_TOL))).sum())


def _generator_violations(net, element):
    table, result = net[element], net['res_' + element]
    return sum(_limit_count(result[quantity], table['min_' + quantity],
                            table['max_' + quantity])
               for quantity in ['p_mw', 'q_mvar'])


def measurement_name_map(net):
    """Map earlier index-based exports to original GridLock naming conventions."""
    names = {}
    bus_number = {idx: number for number, idx in enumerate(net.bus.index, 1)}
    for element, old_prefix, new_prefix, left, right, fields in [
        ('line', 'line', 'line', 'from_bus', 'to_bus',
         ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar', 'pl_mw', 'ql_mvar', 'loading_percent', 'i_from_ka', 'i_to_ka']),
        ('trafo', 'trafo', 'transform', 'hv_bus', 'lv_bus',
         ['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar', 'pl_mw', 'ql_mvar', 'loading_percent', 'i_hv_ka', 'i_lv_ka'])]:
        used = set()
        for idx, branch in net[element].iterrows():
            prefix = f'{new_prefix}{bus_number[branch[left]]}-{bus_number[branch[right]]}'
            if prefix in used:
                prefix += f'_branch{idx}'
            used.add(prefix)
            for field in fields:
                suffix = 'loading_pct' if field == 'loading_percent' else field
                names[f'{old_prefix}{idx}_{field}'] = f'{prefix}_{suffix}'
    for idx, generator in net.gen.iterrows():
        for field in ['p_mw', 'q_mvar']:
            names[f'Gen{idx}_{field}'] = f'Gen{bus_number[generator.bus]}_{field}'
    for idx in net.ext_grid.index:
        for field in ['p_mw', 'q_mvar']:
            names[f'Slack{idx}_{field}'] = f'slack_{field}' if len(net.ext_grid) == 1 else f'slack{idx}_{field}'
    return names


def extract_measurements(net):
    """Power injection is positive INTO the network (opposite res_bus sign)."""
    row = {}
    features = []

    def put(key, value):
        row[key] = float(value)
        features.append(key)

    loads = net.res_load[['p_mw', 'q_mvar']].groupby(net.load.bus).sum()
    loads = loads.reindex(net.bus.index, fill_value=0.0)
    for number, idx in enumerate(net.bus.index, 1):
        result = net.res_bus.loc[idx]
        for field in ['vm_pu', 'va_degree']:
            put(f'Bus{number}_{field}', result[field])
        put(f'Bus{number}_vm_kv', result.vm_pu * net.bus.at[idx, 'vn_kv'])
        put(f'Bus{number}_p_injection_mw', -result.p_mw)
        put(f'Bus{number}_q_injection_mvar', -result.q_mvar)
        put(f'Load{number}_p_mw', loads.at[idx, 'p_mw'])
        put(f'Load{number}_q_mvar', loads.at[idx, 'q_mvar'])
    for idx, result in net.res_line.iterrows():
        for field in ['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar',
                      'pl_mw', 'ql_mvar', 'loading_percent', 'i_from_ka', 'i_to_ka']:
            put(f'line{idx}_{field}', result[field])
    for idx, result in net.res_trafo.iterrows():
        for field in ['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar',
                      'pl_mw', 'ql_mvar', 'loading_percent', 'i_hv_ka', 'i_lv_ka']:
            put(f'trafo{idx}_{field}', result[field])
    for element, prefix in [('gen', 'Gen'), ('ext_grid', 'Slack')]:
        for idx, result in net['res_' + element].iterrows():
            put(f'{prefix}{idx}_p_mw', result.p_mw)
            put(f'{prefix}{idx}_q_mvar', result.q_mvar)
    for quantity, unit in [('p', 'mw'), ('q', 'mvar')]:
        field = f'{quantity}_{unit}'
        put(f'total_load_{unit}', net.res_load[field].sum())
        put(f'total_shunt_{unit}', net.res_shunt[field].sum())
        put(f'total_generation_{unit}', net.res_gen[field].sum() + net.res_ext_grid[field].sum())
    put('line_loss_mw', net.res_line.pl_mw.sum())
    put('trafo_loss_mw', net.res_trafo.pl_mw.sum())
    put('total_loss_mw', row['line_loss_mw'] + row['trafo_loss_mw'])
    row['power_balance_error_mw'] = (row['total_generation_mw'] - row['total_load_mw']
                                    - row['total_shunt_mw'] - row['total_loss_mw'])
    row['reactive_balance_error_mvar'] = (row['total_generation_mvar'] - row['total_load_mvar']
        - row['total_shunt_mvar'] - net.res_line.ql_mvar.sum() - net.res_trafo.ql_mvar.sum())
    current_errors = []
    incident_current = np.zeros(len(net.bus))
    incident_loss = np.zeros(len(net.bus))
    nodal_p = net.res_bus.p_mw.copy()
    nodal_q = net.res_bus.q_mvar.copy()
    for element, terminals in [('line', [('from', 'from_bus'), ('to', 'to_bus')]),
                               ('trafo', [('hv', 'hv_bus'), ('lv', 'lv_bus')])]:
        for idx, branch in net[element].iterrows():
            result = net['res_' + element].loc[idx]
            for side, bus_field in terminals:
                bus = int(branch[bus_field])
                p, q = result[f'p_{side}_mw'], result[f'q_{side}_mvar']
                kv = net.res_bus.at[bus, 'vm_pu'] * net.bus.at[bus, 'vn_kv']
                current_errors.append(abs(result[f'i_{side}_ka'] - np.hypot(p, q) / (np.sqrt(3) * kv)))
                incident_current[bus] = max(incident_current[bus], result[f'i_{side}_ka'])
                incident_loss[bus] += result.pl_mw / 2
                nodal_p.loc[bus] += p
                nodal_q.loc[bus] += q
    for bus in range(len(net.bus)):
        put(f'Bus{bus + 1}_incident_current_max_ka', incident_current[bus])
        put(f'Bus{bus + 1}_allocated_loss_mw', incident_loss[bus])
    row['max_current_identity_error_ka'] = float(max(current_errors))
    row['max_nodal_balance_error_mva'] = float(np.hypot(nodal_p, nodal_q).max())
    row['negative_active_loss_count'] = int((net.res_line.pl_mw < -BALANCE_TOL_MW).sum()
                                          + (net.res_trafo.pl_mw < -BALANCE_TOL_MW).sum())
    row['voltage_min_pu'] = float(net.res_bus.vm_pu.min())
    row['voltage_max_pu'] = float(net.res_bus.vm_pu.max())
    row['voltage_violation_count'] = _limit_count(net.res_bus.vm_pu,
                                                net.bus.min_vm_pu, net.bus.max_vm_pu)
    for element in ['line', 'trafo']:
        loading = net['res_' + element].loading_percent
        row[element + '_loading_max_percent'] = float(loading.max())
        row[element + '_thermal_violation_count'] = _limit_count(
            loading, np.full(len(loading), -np.inf), net[element].max_loading_percent)
    row['generator_limit_violation_count'] = _generator_violations(net, 'gen')
    row['slack_limit_violation_count'] = _generator_violations(net, 'ext_grid')
    row['negative_load_count'] = int((net.res_load.p_mw < -LIMIT_TOL).sum())
    row['missing_or_invalid_measurements'] = int((~np.isfinite(list(row.values()))).sum())
    row['measurements_valid'] = bool(
        row['missing_or_invalid_measurements'] == 0
        and abs(row['power_balance_error_mw']) < BALANCE_TOL_MW
        and abs(row['reactive_balance_error_mvar']) < BALANCE_TOL_MW
        and row['max_nodal_balance_error_mva'] < BALANCE_TOL_MW
        and row['max_current_identity_error_ka'] < 1e-8
        and row['negative_active_loss_count'] == 0
        and row['negative_load_count'] == 0)
    row['within_modeled_limits'] = bool(all(row[key] == 0 for key in [
        'voltage_violation_count', 'line_thermal_violation_count',
        'trafo_thermal_violation_count', 'generator_limit_violation_count',
        'slack_limit_violation_count']))
    names = measurement_name_map(net)
    return {names.get(key, key): value for key, value in row.items()}, [names.get(key, key) for key in features]


def solve_and_validate(net):
    """Compatibility helper: solve and check numerical validity, retaining violations."""
    solve_dispatch(net)
    if not net.converged:
        raise pp.LoadflowNotConverged('Solver did not converge')
    row, _ = extract_measurements(net)
    if not row['measurements_valid']:
        raise ValueError('Nonfinite measurements, negative load, or power imbalance')
    return row


def graph_metadata(net):
    mapping = {int(idx): node for node, idx in enumerate(net.bus.index)}
    branches = []
    names = measurement_name_map(net)
    for element, left, right in [('line', 'from_bus', 'to_bus'), ('trafo', 'hv_bus', 'lv_bus')]:
        for idx, branch in net[element].iterrows():
            branches.append(dict(element=element, index=int(idx),
                                 source=mapping[int(branch[left])],
                                 target=mapping[int(branch[right])],
                                 feature_columns=[names[f'{element}{idx}_{field}'] for field in
                                     (['loading_percent', 'p_from_mw', 'q_from_mvar'] if element == 'line'
                                      else ['loading_percent', 'p_hv_mw', 'q_hv_mvar'])],
                                 reverse_feature_columns=[names[f'{element}{idx}_{field}'] for field in
                                     (['loading_percent', 'p_to_mw', 'q_to_mvar'] if element == 'line'
                                      else ['loading_percent', 'p_lv_mw', 'q_lv_mvar'])]))
    edges = [[b['source'], b['target']] for b in branches]
    edges += [[v, u] for u, v in edges.copy()]
    from GridData import NODE_FEATURES
    return dict(schema_version=2, node_count=len(net.bus), bus_index_to_node=mapping, branches=branches,
                edge_index=np.asarray(edges).T.tolist(),
                node_features=list(NODE_FEATURES),
                edge_features=['loading_pct', 'p_source_mw', 'q_source_mvar'],
                load_bus_numbers=[mapping[int(bus)] + 1 for bus in sorted(net.load.bus.unique())])


def _json_safe(value):
    """Use JSON null for unavailable summary statistics (e.g. all states failed)."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def summarize(df, baseline, net, features, seed, rho, noise_std, multipliers):
    solved = df.loc[df.converged]
    accepted = df.loc[df.converged & df.measurements_valid]
    ranges = {c: {'min': float(accepted[c].min()), 'max': float(accepted[c].max())}
              for c in features if c.startswith(('Gen', 'slack_'))}
    summaries = {}
    for col in ['voltage_min_pu', 'voltage_max_pu', 'total_load_mw', 'total_load_mvar',
                'total_generation_mw', 'total_loss_mw', 'line_loading_max_percent',
                'trafo_loading_max_percent']:
        summaries[col] = {key: float(value) for key, value in
                          accepted[col].describe()[['min', 'max', 'mean', 'std']].items()}
    counts = {key: int(solved[key].sum()) for key in [
        'voltage_violation_count', 'line_thermal_violation_count',
        'trafo_thermal_violation_count', 'generator_limit_violation_count',
        'slack_limit_violation_count', 'negative_load_count']}
    return dict(case='pandapower.networks.case39', pandapower_version=pp.__version__,
                numpy_version=np.__version__, pandas_version=pd.__version__,
                buses=len(net.bus), lines=len(net.line), transformers=len(net.trafo), branches=46,
                requested_timesteps=len(df), converged_timesteps=int(df.converged.sum()),
                convergence_percent=float(df.converged.mean() * 100),
                accepted_timesteps=len(accepted),
                failed_timesteps=df.loc[~df.converged, ['timestep', 'failure_reason']].to_dict('records'),
                invalid_solved_timesteps=int((df.converged & ~df.measurements_valid).sum()),
                missing_or_invalid_measurements=int((~np.isfinite(df[features].to_numpy())).sum()),
                baseline=baseline, statistics=summaries, generator_output_ranges=ranges,
                violation_counts=counts,
                timesteps_with_voltage_violations=int((solved.voltage_violation_count > 0).sum()),
                timesteps_with_thermal_violations=int(((solved.line_thermal_violation_count +
                                                       solved.trafo_thermal_violation_count) > 0).sum()),
                max_abs_power_balance_residual_mw=float(accepted.power_balance_error_mw.abs().max()),
                max_abs_reactive_balance_residual_mvar=float(accepted.reactive_balance_error_mvar.abs().max()),
                max_nodal_balance_error_mva=float(accepted.max_nodal_balance_error_mva.max()),
                max_current_identity_error_ka=float(accepted.max_current_identity_error_ka.max()),
                within_limits_timesteps=int((df.measurements_valid & df.within_modeled_limits).sum()),
                seed=seed, interval_minutes=15, rho=rho, noise_std=noise_std,
                demand_bounds=list(DEMAND_BOUNDS), actual_multiplier_range=[float(multipliers.min()),
                                                                          float(multipliers.max())],
                dispatch='AC OPF at each timestep using published polynomial costs and generator, voltage, and branch constraints; generator setpoints optimized.',
                thermal_rating_warning='Published case39 ratings are benchmark constraints, not verified field equipment ratings. OPF uses apparent-power branch constraints; exported current-based loading is checked independently.',
                comparison='Published case39 topology and impedances retained; constrained AC dispatch. No independent solver or field-data calibration.',
                synthetic_variations='15-minute quasi-static demand snapshots: 0.95 - 0.12*cos(phase) - 0.03*cos(2*phase-pi/3), multiplied by (1+AR1 noise); P and Q share each load multiplier. Bounds 0.75..1.15 are a moderate engineering scenario envelope, not measured demand bounds. Generator P/Q/voltage optimized at each snapshot; taps and shunts unchanged. No ramp-rate or transient dynamics modeled.',
                solver=SOLVER, feature_columns=features, measurement_name_map=measurement_name_map(net), graph=graph_metadata(net))


def generate_dataset(n_samples=2000, steps_per_day=96, rho=0.9, noise_std=0.02,
                     seed=0, output_csv=DEFAULT_OUTPUT, drop_constant=False, progress=False):
    """Keep every requested timestep. Failed states have NaN measurements and a reason.

    Constant physical features are retained for a stable schema. ML adapters should
    remove constants using training data only; see the validation documentation.
    """
    if n_samples < 1 or steps_per_day != 96 or not 0 <= rho < 1 or noise_std < 0:
        raise ValueError('Require n_samples>=1, 96 steps/day, 0<=rho<1 and noise_std>=0')
    if drop_constant:
        raise ValueError('Stable physical schema required; drop constants in the ML adapter')
    net = create_ieee39_network()
    baseline = solve_and_validate(net)
    _, features = extract_measurements(net)
    # Keep published baseline loads for each timestep; no cumulative perturbation.
    base_gen_vm = net.gen.vm_pu.to_numpy().copy()
    base_p = net.load.p_mw.to_numpy().copy()
    base_q = net.load.q_mvar.to_numpy().copy()
    multipliers = demand_multipliers(n_samples, len(base_p), seed, rho, noise_std)
    rows = []
    for step, multiplier in enumerate(multipliers):
        net.gen['vm_pu'] = base_gen_vm
        net.load['p_mw'] = base_p * multiplier
        net.load['q_mvar'] = base_q * multiplier
        row = dict(timestep=step, simulation_time_minutes=step * 15,
                   converged=False, measurements_valid=False, failure_reason='')
        try:
            solve_dispatch(net)
            if not net.converged:
                raise pp.LoadflowNotConverged('Solver did not converge')
            measurements, _ = extract_measurements(net)
            row.update(measurements, converged=True)
            if not row['measurements_valid']:
                row['failure_reason'] = 'Solved state failed finite-value/load/power-balance validation'
        except (pp.LoadflowNotConverged, pp.OPFNotConverged) as exc:
            # Never export stale results from the preceding timestep.
            row.update({key: np.nan for key in baseline})
            row.update(converged=False, measurements_valid=False,
                       within_modeled_limits=False,
                       failure_reason=f'{type(exc).__name__}: {exc}; load={float(net.load.p_mw.sum()):.6f} MW; multiplier={multiplier.min():.6f}..{multiplier.max():.6f}')
        rows.append(row)
        if progress and (step + 1) % 100 == 0:
            print(f'Solved {step + 1}/{n_samples} requested snapshots', flush=True)
    df = pd.DataFrame(rows)
    report = _json_safe(summarize(df, baseline, net, features, seed, rho, noise_std, multipliers))
    df.attrs['validation_report'] = report
    if output_csv:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        Path(str(path) + '.report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        df[features].describe().to_csv(str(path) + '.statistics.csv')
        print(f'Saved {path}: {df.shape[0]} rows, {df.shape[1]} columns', flush=True)
    return df


def print_report(report):
    keys = ['buses', 'branches', 'requested_timesteps', 'converged_timesteps',
            'convergence_percent', 'accepted_timesteps', 'invalid_solved_timesteps',
            'missing_or_invalid_measurements', 'max_abs_power_balance_residual_mw',
            'violation_counts', 'statistics', 'generator_output_ranges', 'failed_timesteps',
            'thermal_rating_warning', 'comparison']
    print(json.dumps({key: report[key] for key in keys}, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--ml-output', default=str(DATA_DIR / 'grid_dataset_ieee39_ml.csv'))
    parser.add_argument('--raw-only', action='store_true')
    args = parser.parse_args()
    data = generate_dataset(n_samples=args.samples, seed=args.seed,
                            output_csv=args.output, progress=True)
    print_report(data.attrs['validation_report'])
    preview = ['timestep', 'simulation_time_minutes', 'Bus1_vm_pu', 'Bus14_vm_pu',
               'Bus39_vm_pu', 'total_load_mw', 'total_loss_mw', 'converged']
    print('\nFirst five rows (selected columns):\n' + data[preview].head().to_string(index=False))

    if not args.raw_only:
        from GridData import export_ml_dataset
        export_ml_dataset(args.output, args.ml_output)
