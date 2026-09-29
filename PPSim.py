import numpy as np
import pandas as pd
import pandapower as pp

def create_ieee14_network():
    """Builds and returns the base IEEE 14-bus network."""
    S_BASE = 100.0
    VN_HV, VN_LV = 110.0, 20.0

    net = pp.create_empty_network(name="IEEE 14-bus", sn_mva=S_BASE, f_hz=60.0)

    HV_LINES = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
    LV_LINES = [(6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14)]
    GENS_IDX = [(2, 40.0, 1.045, -40, 50), (3, 0.0, 1.010, 0, 40), (6, 0.0, 1.070, -6, 24), (8, 0.0, 1.090, -6, 24)]
    TR_LINES = [(4, 7), (4, 9), (5, 6)]
    LOAD_IDX = [
        (2, 21.7, 12.7), (3, 94.2, 19), (4, 47.8, -3.9), (5, 7.6, 1.6), (6, 11.2, 7.5),
        (9, 29.5, 16.6), (10, 9.0, 5.8), (11, 3.5, 1.8), (12, 6.1, 1.6), (13, 13.5, 5.8),
        (14, 14.9, 5.0),
    ]

    bus = {}
    for i in range(1, 15):
        vn = VN_HV if i <= 5 else VN_LV
        bus[i] = pp.create_bus(net, vn_kv=vn, name=f"Bus{i}")

    for j, pm, qm in LOAD_IDX:
        pp.create_load(net, bus=bus[j], p_mw=pm, q_mvar=qm, name=f"Load{j}")

    for d, b in HV_LINES:
        pp.create_line(net, bus[d], bus[b], 15, "243-AL1/39-ST1A 110.0", name=f"line{d}-{b}")
    for m, n in LV_LINES:
        pp.create_line(net, bus[m], bus[n], 15, "243-AL1/39-ST1A 20.0", name=f"line{m}-{n}")
    for q, p in TR_LINES:
        pp.create_transformer(net, bus[q], bus[p], "63 MVA 110/20 kV", name=f"transform{q}-{p}")

    pp.create_ext_grid(net, bus[1], vm_pu=1.06, name="Slack")
    pp.create_shunt(net, bus[9], q_mvar=-19.0)
    pp.create_shunt(net, bus[14], q_mvar=-8.0)
    pp.create_shunt(net, bus[13], q_mvar=-6.0)

    for b, p, v, q_min, q_max in GENS_IDX:
        pp.create_gen(net, bus[b], p, v, min_q_mvar=q_min, max_q_mvar=q_max, name=f"Gen{b}")

    net.trafo["shift_degree"] = 0
    return net


def daily_profile(n_samples, steps_per_day):
    """Normalised demand curve: overnight trough, morning rise, evening peak."""
    t = np.arange(n_samples) / steps_per_day * 2 * np.pi
    return (1.0
            + 0.18 * np.sin(t - np.pi / 2)      # main daily swing
            + 0.07 * np.sin(2 * t - np.pi / 3))  # secondary evening peak


def ar1_noise(n_samples, n_series, rho, sigma, rng):
    """Autocorrelated noise: each series wanders instead of jumping."""
    e = np.zeros((n_samples, n_series))
    scale = sigma * np.sqrt(1 - rho ** 2)
    for k in range(1, n_samples):
        e[k] = rho * e[k - 1] + scale * rng.standard_normal(n_series)
    return e


def generate_dataset(n_samples=2000, steps_per_day=96, rho=0.9, noise_std=0.04,
                     seed=0, output_csv="grid_dataset.csv", drop_constant=True):
    """
    Generates a time-series dataset of grid states under a realistic demand cycle.

    n_samples     : number of timesteps to simulate
    steps_per_day : timesteps per 24h cycle (96 = 15-minute resolution)
    rho           : AR(1) coefficient for the noise; higher = smoother
    noise_std     : standard deviation of per-load noise
    seed          : RNG seed, for reproducibility
    drop_constant : remove columns that never vary (they break normalisation)
    """
    rng = np.random.default_rng(seed)

    net = create_ieee14_network()
    base_p_mw = net.load["p_mw"].to_numpy().copy()
    base_q_mvar = net.load["q_mvar"].to_numpy().copy()
    n_loads = len(base_p_mw)

    profile = daily_profile(n_samples, steps_per_day)
    noise = ar1_noise(n_samples, n_loads, rho, noise_std, rng)

    records, skipped = [], 0
    for step in range(n_samples):
        multiplier = np.clip(profile[step] * (1.0 + noise[step]), 0.2, 2.0)
        net.load["p_mw"] = base_p_mw * multiplier
        net.load["q_mvar"] = base_q_mvar * multiplier

        try:
            pp.runpp(net, numba=True, enforce_q_lims=True)
        except Exception:
            skipped += 1
            continue

        row = {"timestep": step}

        for idx, res in net.res_bus.iterrows():
            name = net.bus.at[idx, "name"]
            row[f"{name}_vm_pu"] = res["vm_pu"]
            row[f"{name}_va_degree"] = res["va_degree"]

        for idx, res in net.res_line.iterrows():
            name = net.line.at[idx, "name"]
            row[f"{name}_loading_pct"] = res["loading_percent"]
            row[f"{name}_p_from_mw"] = res["p_from_mw"]

        for idx, res in net.res_trafo.iterrows():
            row[f"{net.trafo.at[idx, 'name']}_loading_pct"] = res["loading_percent"]

        for idx, res in net.res_gen.iterrows():
            name = net.gen.at[idx, "name"]
            row[f"{name}_p_mw"] = res["p_mw"]
            row[f"{name}_q_mvar"] = res["q_mvar"]

        row["total_load_mw"] = float(net.load["p_mw"].sum())
        row["slack_p_mw"] = net.res_ext_grid["p_mw"].iloc[0]
        row["slack_q_mvar"] = net.res_ext_grid["q_mvar"].iloc[0]
        row["total_loss_mw"] = net.res_line["pl_mw"].sum()

        records.append(row)

    df = pd.DataFrame(records)

    if skipped:
        print(f"WARNING: {skipped}/{n_samples} steps did not converge and were dropped. "
              f"Timesteps are no longer evenly spaced.")

    if drop_constant:
        feats = df.drop(columns=["timestep"])
        const = feats.columns[feats.std() < 1e-9].tolist()
        if const:
            df = df.drop(columns=const)
            print(f"Dropped {len(const)} constant columns: {', '.join(const)}")

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved '{output_csv}' ({len(df)} samples, {len(df.columns)} columns).")

    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=2000, seed=0, output_csv="grid_dataset.csv")

    feats = df.drop(columns=["timestep"])
    ac = feats.apply(lambda s: s.autocorr(1))
    print(f"\nlag-1 autocorrelation: median {ac.median():.3f}, min {ac.min():.3f}")
    print("total load range: %.1f - %.1f MW" % (df.total_load_mw.min(), df.total_load_mw.max()))