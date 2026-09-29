import numpy as np
import pandas as pd
import pandapower as pp


def create_ieee14_network():
    """Builds and returns the base IEEE 14-bus network."""
    # Base values (feel free to change).
    S_BASE = 100.0
    VN_HV, VN_LV = 110.0, 20.0

    net = pp.create_empty_network(name="IEEE 14-bus", sn_mva=S_BASE, f_hz=60.0)

    # These are the indexes that connect all loads, generators, high V, low V, and transformers together.
    HV_LINES = [(1, 2), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (4, 5)]
    LV_LINES = [(6, 11), (6, 12), (6, 13), (7, 8), (7, 9), (9, 10), (9, 14), (10, 11), (12, 13), (13, 14)]
    GENS_IDX = [(2, 40.0, 1.045, -40, 50), (3, 0.0, 1.010, 0, 40), (6, 0.0, 1.070, -6, 24), (8, 0.0, 1.090, -6, 24)]
    TR_LINES = [(4, 7), (4, 9), (5, 6)]
    LOAD_IDX = [
        (2, 21.7, 12.7), (3, 94.2, 19), (4, 47.8, -3.9), (5, 7.6, 1.6), (6, 11.2, 7.5),
        (9, 29.5, 16.6), (10, 9.0, 5.8), (11, 3.5, 1.8), (12, 6.1, 1.6), (13, 13.5, 5.8),
        (14, 14.9, 5.0)
    ]

    # These are the populated lists (for living life secure).
    bus = {}
    load = {}
    lv_lines = {}
    hv_lines = {}
    tr_lines = {}

    # Here, we create the buses, from bus 1-14, where 1-5 are HV
    for i in range(1, 15):
        vn = VN_HV if i <= 5 else VN_LV
        bus[i] = pp.create_bus(net, vn_kv=vn, name=f"Bus{i}")

    # Here, we create the loads, which exist at certain buses.
    for j, pm, qm in LOAD_IDX:
        load[j] = pp.create_load(net, bus=bus[j], p_mw=pm, q_mvar=qm, name=f"Load{j}")

    # Here, we create the lines. We start with the high voltage lines, low voltage lines, and then the transformers.
    for d, b in HV_LINES:
        hv_lines[(d, b)] = pp.create_line(net, bus[d], bus[b], 15, "243-AL1/39-ST1A 110.0", name=f"line{d}-{b}")
    for m, n in LV_LINES:
        lv_lines[(m, n)] = pp.create_line(net, bus[m], bus[n], 15, "243-AL1/39-ST1A 20.0", name=f"line{m}-{n}")
    for q, p in TR_LINES:
        tr_lines[(q, p)] = pp.create_transformer(net, bus[q], bus[p], "63 MVA 110/20 kV", name=f"transform{q}-{p}")

    # Since bus one has the slack, we create the ext_grid at bus 1, then the rest at their respective index.
    pp.create_ext_grid(net, bus[1], vm_pu=1.06, name="Slack")
    pp.create_shunt(net, bus[9], q_mvar=-19.0)
    pp.create_shunt(net, bus[14], q_mvar=-8.0)
    pp.create_shunt(net, bus[13], q_mvar=-6.0)

    for b, p, v, q_min, q_max in GENS_IDX:
        pp.create_gen(net, bus[b], p, v, min_q_mvar=q_min, max_q_mvar=q_max, name=f"Gen{b}")

    net.trafo["shift_degree"] = 0
    return net


def print_single_snapshot(net):
    """Computes and prints diagnostic tables for a single snapshot."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    pp.runpp(net, numba=False, enforce_q_lims=True)
    t = net.res_line.assign(name=net.line.name)[["name", "loading_percent"]]
    print(t[t.loading_percent > 80].sort_values("loading_percent", ascending=False).round(1).to_string(
        index=False) + '\n')

    b = net.res_bus.assign(name=net.bus.name)[["name", "vm_pu"]]
    print(b[(b.vm_pu < 0.95) | (b.vm_pu > 1.05)].sort_values("vm_pu").round(3).to_string(index=False) + '\n')

    g = net.res_gen.assign(name=net.gen.name)[["name", "p_mw", "q_mvar", "vm_pu"]]
    print(g.round(1).to_string(index=False) + '\n')

    print(net.res_trafo.assign(name=net.trafo.name)[["name", "loading_percent"]].round(1).to_string(index=False) + '\n')

    print(f"losses: {net.res_line.pl_mw.sum():.1f} MW")
    print(f"slack:  {net.res_ext_grid.p_mw.iloc[0]:.1f} MW, {net.res_ext_grid.q_mvar.iloc[0]:.1f} MVAr\n")


def generate_dataset(n_samples: int = 500, noise_std: float = 0.05,
                     output_csv: str = "grid_dataset.csv") -> pd.DataFrame:
    """
    Generates a time-series dataset of grid states across varying load conditions.

    Parameters:
        n_samples: Number of power flow states / timesteps to simulate.
        noise_std: Standard deviation of Gaussian load perturbations around base values.
        output_csv: Path to save the resulting CSV dataset (optional).

    Returns:
        pd.DataFrame containing feature columns for each solved power flow state.
    """
    net = create_ieee14_network()
    base_p_mw = net.load["p_mw"].copy()
    base_q_mvar = net.load["q_mvar"].copy()

    records = []
    for step in range(n_samples):
        # Perturb loads randomly around baseline
        multiplier = np.random.normal(1.0, noise_std, size=len(net.load))
        # Ensure loads remain positive
        multiplier = np.clip(multiplier, 0.2, 2.0)
        net.load["p_mw"] = base_p_mw * multiplier
        net.load["q_mvar"] = base_q_mvar * multiplier

        try:
            pp.runpp(net, numba=False, enforce_q_lims=True)

            row = {"timestep": step}

            # 1. Bus voltages (magnitude vm_pu and angle va_degree)
            for idx, res in net.res_bus.iterrows():
                bus_name = net.bus.at[idx, "name"]
                row[f"{bus_name}_vm_pu"] = res["vm_pu"]
                row[f"{bus_name}_va_degree"] = res["va_degree"]

            # 2. Line loadings and power flows
            for idx, res in net.res_line.iterrows():
                line_name = net.line.at[idx, "name"]
                row[f"{line_name}_loading_pct"] = res["loading_percent"]
                row[f"{line_name}_p_from_mw"] = res["p_from_mw"]

            # 3. Transformer loadings
            for idx, res in net.res_trafo.iterrows():
                trafo_name = net.trafo.at[idx, "name"]
                row[f"{trafo_name}_loading_pct"] = res["loading_percent"]

            # 4. Generator outputs
            for idx, res in net.res_gen.iterrows():
                gen_name = net.gen.at[idx, "name"]
                row[f"{gen_name}_p_mw"] = res["p_mw"]
                row[f"{gen_name}_q_mvar"] = res["q_mvar"]

            # 5. Slack bus & total transmission losses
            row["slack_p_mw"] = net.res_ext_grid["p_mw"].iloc[0]
            row["slack_q_mvar"] = net.res_ext_grid["q_mvar"].iloc[0]
            row["total_loss_mw"] = net.res_line["pl_mw"].sum()

            records.append(row)
        except Exception:
            # Skip iterations that do not converge under extreme parameters
            continue

    df = pd.DataFrame(records)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(
            f"Dataset successfully created and saved to '{output_csv}' ({len(df)} samples, {len(df.columns)} columns).")

    return df


if __name__ == "__main__":
    # 1. Optional diagnostic view of single baseline snapshot
    base_net = create_ieee14_network()
    print("Baseline snapshot diagnostics:")
    print_single_snapshot(base_net)

    # 2. Generate and export the dataset deliverable
    print("Generating simulation dataset...")
    dataset = generate_dataset(n_samples=500, noise_std=0.05, output_csv="grid_dataset.csv")
    print(f"\nPreview of generated dataset (first 5 rows):")
    print(dataset.head())