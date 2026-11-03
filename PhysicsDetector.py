import json
import numpy as np
import pandas as pd

from GridData import MEASUREMENT_NOISE, RAW_OUTPUT, noise_columns, add_measurement_noise as add_noise
graph = json.load(open(RAW_OUTPUT + ".report.json"))["graph"]

class PhysicsDetector:
    def __init__(self, graph):
        self.graph = graph
        self.N = graph["node_count"]

        self.p_cols = [[] for _ in range(self.N)]
        self.q_cols = [[] for _ in range(self.N)]

        for b in graph["branches"]:
            self.p_cols[b["source"]].append(b["feature_columns"][1])
            self.p_cols[b["target"]].append(b["reverse_feature_columns"][1])
            self.q_cols[b["source"]].append(b["feature_columns"][2])
            self.q_cols[b["target"]].append(b["reverse_feature_columns"][2])

        self.threshold = float("inf")
        self.win_threshold = float("inf")

    def score(self, frame):
        errors = []
        for i in range(self.N):
            dP_i = frame[f"Bus{i+1}_p_injection_mw"] - frame[self.p_cols[i]].sum(axis=1)
            dQ_i = frame[f"Bus{i+1}_q_injection_mvar"] - frame[self.q_cols[i]].sum(axis=1)
            errors.append(np.hypot(dP_i, dQ_i).to_numpy())

        bus_score = np.stack(errors, axis=1)
        row_score = bus_score.max(axis=1)

        return row_score, bus_score

    def fit_threshold(self, frame, q=0.99):
        row_score, _ = self.score(frame)
        self.threshold = np.quantile(row_score, q)

        return self.threshold

    def detect(self, frame):
        row_score, _ = self.score(frame)
        flags = row_score > self.threshold

        return flags, row_score

    def window_score(self, frame, ends, window=24):
        row_score, _ = self.score(frame)

        s = pd.Series(row_score, index=frame["timestep"].to_numpy())
        s = s.reindex(range(s.index.min(), s.index.max() + 1))

        win = s.rolling(window).max()

        return win.loc[ends].to_numpy()

    def fit_win_threshold(self, frame, ends, window=24, q=0.99):
        win_score = self.window_score(frame, ends, window)
        self.win_threshold = np.nanquantile(win_score, q)

        return self.win_threshold

    def detect_win(self, frame, ends, window=24):
        win_score = self.window_score(frame, ends, window)
        flags = win_score > self.win_threshold

        return flags, win_score

if __name__ == "__main__":
    det = PhysicsDetector(graph)
    print(sum(len(c) for c in det.p_cols))
    print(det.p_cols[0])

    raw = pd.read_csv("Dataset/grid_dataset_ieee39_raw.csv")
    valid = raw["measurements_valid"]
    frame = raw[valid]
    row, bus = det.score(frame)
    print(row.shape, bus.shape, row.max())

    det.fit_threshold(frame)
    fake = frame.iloc[[0]].copy()
    fake["Bus16_p_injection_mw"] += 50
    print(det.detect(fake))

    cols = noise_columns(graph)
    noisy = add_noise(raw, cols, rel=MEASUREMENT_NOISE)[valid]
    print(len(cols), f"columns at {MEASUREMENT_NOISE:.0%} noise")
    print(det.fit_threshold(noisy))

    for size in [5, 10, 20, 30, 50, 100]:
        attacked = noisy.copy()
        attacked["Bus16_p_injection_mw"] += size
        flags, _ = det.detect(attacked)
        print(size, flags.mean())

    ends = frame["timestep"].to_numpy()[23:]
    w = det.window_score(noisy, ends)
    print(w.shape, np.isnan(w).sum())

    print(det.fit_win_threshold(noisy, ends))
    for size in [5, 10, 20, 30, 50]:
        attacked = noisy.copy()
        attacked["Bus16_p_injection_mw"] += size
        flags, _ = det.detect_win(attacked, ends)
        print(size, flags.mean())