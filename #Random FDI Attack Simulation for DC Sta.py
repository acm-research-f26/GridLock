#Random FDI Attack Simulation for DC State Estimation

import numpy as np

def simulate_clean_measurements(H, x_true, sigma, rng):
    noise = rng.normal(loc=0.0, scale=sigma, size=H.shape[0])
    return H @ x_true + noise

def make_stealthy_tampering(H, z_clean, c):
    attack = H @ c
    return z_clean + attack, attack, None

def dc_weighted_least_squares(H, z, sigma):
    w = 1.0 / np.square(sigma)
    W = np.diag(w)
    gain = H.T @ W @ H
    x_hat = np.linalg.solve(gain, H.T @ W @ z)
    residual = z - H @ x_hat
    J = residual.T @ W @ residual
    return x_hat, residual, J

def run_random_tampering_case(H, x_true, sigma, seed=14):
    rng = np.random.default_rng(seed)

    z_clean = simulate_clean_measurements(H, x_true, sigma, rng)
    x_clean, r_clean, j_clean = dc_weighted_least_squares(H, z_clean, sigma)

    # small stealthy shift
    c = 0.05 * rng.normal(size=H.shape[1])
    z_test, attack, _ = make_stealthy_tampering(H, z_clean, c)
    x_test, r_test, j_test = dc_weighted_least_squares(H, z_test, sigma)

    return {
        "state_shift": x_test - x_clean,
        "c_target": c,
        "clean_residual_energy": j_clean,
        "test_residual_energy": j_test,
    }

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, n = 15, 6
    H = rng.normal(size=(m, n))
    x_true = rng.normal(size=n)
    sigma = np.full(m, 0.01)

    res = run_random_tampering_case(H, x_true, sigma)

    print("Target state shift c:")
    print(res["c_target"])
    print("Actual state shift (should ~= c):")
    print(res["state_shift"])
    print(f"Clean J: {res['clean_residual_energy']:.6f}")
    print(f"Test  J: {res['test_residual_energy']:.6f}")