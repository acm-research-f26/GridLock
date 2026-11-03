import json
import numpy as np
import pandas as pd
graph = json.load(open("Dataset/grid_dataset_ieee39_raw.csv.report.json"))["graph"]

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

if __name__ == "__main__":
    det = PhysicsDetector(graph)
    print(sum(len(c) for c in det.p_cols))
    print(det.p_cols[0])

    frame = pd.read_csv("Dataset/grid_dataset_ieee39_raw.csv")
    frame = frame[frame["measurements_valid"]]
    row, bus = det.score(frame)
    print(row.shape, bus.shape, row.max())

    det.fit_threshold(frame)
    fake = frame.iloc[[0]].copy()
    fake["Bus16_p_injection_mw"] += 50
    print(det.detect(fake))