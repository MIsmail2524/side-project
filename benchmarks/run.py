"""Compare FlySentinel with classic detectors on security- and finance-style data.

Novelty setting: train on normal traffic only, test on normal traffic mixed with attacks.
All data is synthetic and generated locally, so results are reproducible offline.
Usage: python benchmarks/run.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flysentinel import FlySentinel  # noqa: E402


def correlated_clusters(rng, n, d, k):
    """Normal behavior: k service profiles, each with its own feature correlations."""
    centers = rng.normal(0, 4, (k, d))
    mixes = [rng.normal(0, 1, (d, d)) / np.sqrt(d) for _ in range(k)]
    lab = rng.integers(k, size=n)
    return np.stack([centers[c] + rng.normal(size=d) @ mixes[c] for c in lab]), centers, mixes


def network_intrusion(rng):
    d = 24
    normal, centers, mixes = correlated_clusters(rng, 12000, d, 6)
    n_att = 300
    # Port scan: one profile, but a few features pushed to extremes.
    scan = centers[rng.integers(6, size=n_att)] + rng.normal(size=(n_att, d))
    scan[:, :3] += rng.choice([-1, 1], (n_att, 3)) * 3.5
    # Exfiltration: every feature looks normal on its own, but the correlation structure is broken.
    exfil = centers[rng.integers(6, size=n_att)] + rng.normal(0, 1.0, (n_att, d))
    return normal, np.vstack([scan, exfil])


def card_fraud(rng):
    d = 30
    normal, centers, mixes = correlated_clusters(rng, 15000, d, 3)
    n_att = 250
    # Fraud: account takeover sits between customer segments and drifts on a few spend features.
    pick = rng.integers(3, size=(n_att, 2))
    fraud = (centers[pick[:, 0]] + centers[pick[:, 1]]) / 2 + rng.normal(0, 0.8, (n_att, d))
    fraud[:, -4:] += 2.0
    return normal, fraud


def sensor_health(rng):
    """ICU-style vitals: smooth low-dimensional manifold; anomalies leave the manifold."""
    n, d, latent = 12000, 16, 3
    basis = rng.normal(size=(latent, d))
    t = rng.normal(size=(n, latent))
    normal = np.tanh(t) @ basis + rng.normal(0, 0.15, (n, d))
    t2 = rng.normal(size=(300, latent))
    anomal = np.tanh(t2) @ basis + rng.normal(0, 0.15, (300, d))
    anomal[:, rng.choice(d, 4, replace=False)] += rng.normal(0, 1.2, (300, 4))
    return normal, anomal


DATASETS = {"network intrusion": network_intrusion, "card fraud": card_fraud,
            "patient vitals": sensor_health}


def models():
    return {
        "FlySentinel": lambda: FlySentinel(n_flies=8, seed=0),
        "IsolationForest": lambda: IsolationForest(n_estimators=200, random_state=0),
        "LOF (novelty)": lambda: LocalOutlierFactor(n_neighbors=35, novelty=True),
    }


def score(model, X):
    if isinstance(model, FlySentinel):
        return model.score_samples(X)
    return -model.score_samples(X)  # sklearn: higher = more normal


def main():
    rows = []
    for dname, make in DATASETS.items():
        rng = np.random.default_rng(42)
        normal, attacks = make(rng)
        rng.shuffle(normal)
        split = int(len(normal) * 0.7)
        train, test_norm = normal[:split], normal[split:]
        X_test = np.vstack([test_norm, attacks])
        y = np.r_[np.zeros(len(test_norm)), np.ones(len(attacks))]
        for mname, factory in models().items():
            m = factory()
            t0 = time.perf_counter()
            m.fit(train)
            t1 = time.perf_counter()
            s = score(m, X_test)
            t2 = time.perf_counter()
            rows.append((dname, mname, roc_auc_score(y, s), average_precision_score(y, s),
                         t1 - t0, len(X_test) / (t2 - t1)))

    print("| Dataset | Model | ROC-AUC | PR-AUC | Fit (s) | Score (samples/s) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:.2f} | {r[5]:,.0f} |")


if __name__ == "__main__":
    main()
