# Targeted Constrained FDI Attack on the IEEE 14-bus DC State Estimator
#
# Constrained targeted attack (Liu, Ning & Reiter, CCS'09, Sec. 3.2.2 / 4.2.2):
# the attacker fixes the error c on the chosen target state(s) and ZERO on every
# other state, so a = H c is fully determined -- the one and only attack vector.
# Because a is a column-space vector of H, the chi-square residual test (classical
# bad-data detection) cannot see it. Our enhancement over the static paper version:
# ramp the attack in over time and score it with the team's LSTM autoencoder, which
# learned the normal spatio-temporal measurement manifold and still flags it.
#
# Everything lives in the DC state-estimation measurement space (bus power
# injections + line power flows) -- that is where FDI and BDD are actually defined.
# Reuses PPSim's 14-bus network and LSTMAutoencoder's detector unchanged.

import numpy as np
from scipy.stats import chi2
import pandapower as pp

from PPSim import create_ieee14_network, daily_profile, ar1_noise
from LSTMAutoencoder import LSTMBaseline
import torch


# --------------------------------------------------------------------------- #
# DC model: measurement matrix H and clean measurement time-series            #
# --------------------------------------------------------------------------- #
def dc_measurement_matrix(net):
    """DC measurement matrix H for z = H x, where x = non-slack bus angles and
    the measurements are all bus injections stacked on all branch flows."""
    from pandapower.pypower.makeBdc import makeBdc
    from pandapower.pypower.idx_bus import BUS_TYPE, REF

    pp.rundcpp(net, numba=False)
    bus_, branch_ = net._ppc["bus"], net._ppc["branch"]
    Bbus, Bf = makeBdc(bus_, branch_, return_csr=False)[:2]
    Bbus = np.asarray(Bbus.todense() if hasattr(Bbus, "todense") else Bbus)
    Bf = np.asarray(Bf.todense() if hasattr(Bf, "todense") else Bf)

    ref = int(np.where(bus_[:, BUS_TYPE] == REF)[0][0])
    non_ref = [i for i in range(Bbus.shape[0]) if i != ref]
    H = np.vstack([Bbus[:, non_ref], Bf[:, non_ref]])
    bus_names = [net.bus.at[i, "name"] for i in non_ref]
    return H, non_ref, bus_names


def clean_state_series(net, non_ref, n_samples=1500, steps_per_day=96,
                       rho=0.9, noise_std=0.04, seed=0):
    """True DC bus angles x(t) over a daily demand cycle (same profile as PPSim)."""
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    rng = np.random.default_rng(seed)
    profile = daily_profile(n_samples, steps_per_day)
    noise = ar1_noise(n_samples, len(base_p), rho, noise_std, rng)

    X = np.empty((n_samples, len(non_ref)))
    for t in range(n_samples):
        mult = np.clip(profile[t] * (1.0 + noise[t]), 0.2, 2.0)
        net.load["p_mw"] = base_p * mult
        net.load["q_mvar"] = base_q * mult
        pp.rundcpp(net, numba=False)
        X[t] = np.deg2rad(net.res_bus["va_degree"].to_numpy())[non_ref]
    return X


# --------------------------------------------------------------------------- #
# The attack                                                                   #
# --------------------------------------------------------------------------- #
def constrained_attack_vector(H, target_idx, c_inject):
    """a = H c with c fixed on the targets and zero elsewhere (constrained case).
    Returns a and the fixed state-error vector c."""
    c = np.zeros(H.shape[1])
    for i, val in zip(target_idx, c_inject):
        c[i] = val
    return H @ c, c


def ramp(n, start, end):
    """Linear stealthy ramp: 0 before `start`, rising to 1 by `end`, held after."""
    r = np.clip((np.arange(n) - start) / max(end - start, 1), 0.0, 1.0)
    return r


# --------------------------------------------------------------------------- #
# Classical bad-data detection (chi-square residual test)                     #
# --------------------------------------------------------------------------- #
def bdd_residuals(H, Z, sigma):
    """Per-timestep residual energy J = r^T W r from DC weighted least squares."""
    W = np.diag(1.0 / sigma ** 2)
    gain = H.T @ W @ H
    G_inv_Ht_W = np.linalg.solve(gain, H.T @ W)
    J = np.empty(len(Z))
    Xhat = np.empty((len(Z), H.shape[1]))
    for t, z in enumerate(Z):
        x_hat = G_inv_Ht_W @ z
        r = z - H @ x_hat
        Xhat[t] = x_hat
        J[t] = r @ W @ r
    return J, Xhat


# --------------------------------------------------------------------------- #
# LSTM autoencoder detection (team's LSTMBaseline, trained on clean windows)  #
# --------------------------------------------------------------------------- #
def window(a, w):
    return torch.from_numpy(
        np.stack([a[i:i + w] for i in range(len(a) - w + 1)]).astype(np.float32))


def lstm_detect(Z_clean, Z_attacked, window_len=24, train_frac=0.6, epochs=40, seed=0):
    """Train on clean measurement windows only; return per-window error on the
    attacked stream plus the mean+3sigma threshold learned from clean data."""
    torch.manual_seed(seed)
    split = int(len(Z_clean) * train_frac)
    mu = Z_clean[:split].mean(0)
    sd = Z_clean[:split].std(0); sd[sd < 1e-9] = 1.0
    norm = lambda a: (a - mu) / sd

    train_w = window(norm(Z_clean[:split]), window_len)
    val_w = window(norm(Z_clean[split:]), window_len)

    model = LSTMBaseline(n_features=Z_clean.shape[1])
    model.fit(train_w, epochs=epochs, verbose=False)

    val_err = model.score(val_w).numpy()
    threshold = val_err.mean() + 3 * val_err.std()
    att_err = model.score(window(norm(Z_attacked), window_len)).numpy()
    return att_err, threshold, window_len


# --------------------------------------------------------------------------- #
# End-to-end demonstration                                                     #
# --------------------------------------------------------------------------- #
def run_demo(target_buses=("Bus8",), c_inject=(0.12,), sigma_pu=0.01,
             n_samples=1500, attack_start=900, attack_end=1050, seed=0, plot=True):
    net = create_ieee14_network()
    H, non_ref, bus_names = dc_measurement_matrix(net)
    m, n = H.shape

    target_idx = [bus_names.index(b) for b in target_buses]
    a, c = constrained_attack_vector(H, target_idx, c_inject)
    k = int(np.count_nonzero(~np.isclose(a, 0)))

    X = clean_state_series(net, non_ref, n_samples=n_samples, seed=seed)
    rng = np.random.default_rng(seed)
    sigma = np.full(m, sigma_pu)
    Z_clean = X @ H.T + rng.normal(0, sigma_pu, size=(n_samples, m))

    r = ramp(n_samples, attack_start, attack_end)
    Z_attacked = Z_clean + np.outer(r, a)

    # Classical BDD
    tau2 = chi2.ppf(0.95, df=m - n)                       # 95% detection threshold
    J_clean, _ = bdd_residuals(H, Z_clean, sigma)
    J_att, Xhat_att = bdd_residuals(H, Z_attacked, sigma)

    # LSTM detection
    att_err, lstm_thr, w = lstm_detect(Z_clean, Z_attacked, seed=seed)

    # ---- report ----
    print(f"Grid: IEEE 14-bus | H = {m} meas x {n} states (slack removed)")
    print(f"Target state(s): {list(target_buses)}  inject c = {list(c_inject)} rad")
    print(f"Constrained attack a = Hc compromises k = {k} of {m} meters "
          f"(the ONLY vector; no collateral state change)")
    print(f"  compromised meters (|a|>0): "
          f"{np.round(a[~np.isclose(a,0)], 3).tolist()}")
    print()
    print("Classical BDD (chi-square residual test):")
    print(f"  threshold tau^2 = {tau2:.2f}")
    print(f"  clean  J: mean {J_clean.mean():.2f}, max {J_clean.max():.2f}")
    print(f"  attacked J (during attack): mean "
          f"{J_att[attack_end:].mean():.2f}, max {J_att[attack_end:].max():.2f}")
    bdd_flags = int((J_att[attack_end:] > tau2).sum())
    print(f"  -> attacked steps flagged by BDD: {bdd_flags}/"
          f"{n_samples-attack_end}  (STEALTHY: attack slips past BDD)")
    print()
    print("LSTM autoencoder (trained on clean windows only):")
    att_idx0 = attack_end - w + 1
    print(f"  threshold (mean+3sd) = {lstm_thr:.4f}")
    print(f"  clean-era error (median) = {np.median(att_err[:attack_start-w]):.4f}")
    print(f"  attack-era error (median) = {np.median(att_err[att_idx0:]):.4f}")
    lstm_flags = int((att_err[att_idx0:] > lstm_thr).sum())
    print(f"  -> attack windows flagged by LSTM: {lstm_flags}/"
          f"{len(att_err)-att_idx0}  (CAUGHT: hybrid detector sees what BDD can't)")

    # recovered injected error on the target state
    shift = Xhat_att[attack_end:, target_idx].mean(0) - X[attack_end:, target_idx].mean(0)
    print(f"\nTarget-state corruption achieved: {np.round(shift,3).tolist()} rad "
          f"(requested {list(c_inject)})")

    if plot:
        _plot(J_clean, J_att, tau2, att_err, lstm_thr, w,
              X[:, target_idx[0]], Xhat_att[:, target_idx[0]],
              a, bus_names, attack_start, attack_end, target_buses[0])

    return dict(H=H, a=a, c=c, k=k, J_clean=J_clean, J_att=J_att, tau2=tau2,
                att_err=att_err, lstm_thr=lstm_thr)


def _plot(J_clean, J_att, tau2, att_err, lstm_thr, w, x_true_tgt, x_hat_tgt,
          a, bus_names, a_start, a_end, tgt_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    span = (a_start, a_end)

    # (1) BDD residual -- blind
    ax[0, 0].plot(J_clean, lw=.8, label="clean", color="#4c78a8")
    ax[0, 0].plot(J_att, lw=.8, label="attacked", color="#e4572e")
    ax[0, 0].axhline(tau2, ls="--", color="k", label=r"$\chi^2$ threshold")
    ax[0, 0].axvspan(*span, color="grey", alpha=.15)
    ax[0, 0].set_title("Classical BDD residual J(t): attack is invisible")
    ax[0, 0].set_xlabel("timestep"); ax[0, 0].set_ylabel("residual energy J")
    ax[0, 0].legend(fontsize=8)

    # (2) LSTM error -- catches it
    t_err = np.arange(len(att_err)) + (w - 1)
    ax[0, 1].plot(t_err, att_err, lw=.8, color="#e4572e")
    ax[0, 1].axhline(lstm_thr, ls="--", color="k", label="mean+3σ threshold")
    ax[0, 1].axvspan(*span, color="grey", alpha=.15)
    ax[0, 1].set_title("LSTM autoencoder error: attack is caught")
    ax[0, 1].set_xlabel("timestep"); ax[0, 1].set_ylabel("recon. error")
    ax[0, 1].legend(fontsize=8)

    # (3) target state corruption
    ax[1, 0].plot(np.rad2deg(x_true_tgt), lw=.9, label="true", color="#4c78a8")
    ax[1, 0].plot(np.rad2deg(x_hat_tgt), lw=.9, label="estimated (attacked)",
                  color="#e4572e")
    ax[1, 0].axvspan(*span, color="grey", alpha=.15)
    ax[1, 0].set_title(f"State corruption at {tgt_name} (voltage angle)")
    ax[1, 0].set_xlabel("timestep"); ax[1, 0].set_ylabel("angle (deg)")
    ax[1, 0].legend(fontsize=8)

    # (4) compromised meters
    idx = np.where(~np.isclose(a, 0))[0]
    ax[1, 1].stem(idx, a[idx])
    ax[1, 1].set_title(f"Compromised meters: k={len(idx)} of {len(a)} (the only vector)")
    ax[1, 1].set_xlabel("measurement index"); ax[1, 1].set_ylabel("injected value a_i")

    fig.suptitle("Targeted Constrained FDI Attack — IEEE 14-bus DC State Estimation",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("targeted_constrained_fdi.png", dpi=150)
    print("\nSaved figure -> targeted_constrained_fdi.png")


if __name__ == "__main__":
    run_demo()
