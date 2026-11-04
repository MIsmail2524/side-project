"""Command line: train on a CSV of normal rows, then score another CSV.

  flysentinel train normal.csv model.pkl
  flysentinel score model.pkl new.csv [--explain 3]
"""

import argparse
import csv
import pickle
import sys

import numpy as np

from .detector import FlySentinel


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    try:
        float(rows[0][0])
        header = None
    except ValueError:
        header, rows = rows[0], rows[1:]
    return header, np.array(rows, dtype=float)


def main(argv=None):
    p = argparse.ArgumentParser(prog="flysentinel")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("csv")
    t.add_argument("model")
    t.add_argument("--flies", type=int, default=8)
    t.add_argument("--contamination", type=float, default=0.01)
    t.add_argument("--decay", type=float, default=0.0)
    s = sub.add_parser("score")
    s.add_argument("model")
    s.add_argument("csv")
    s.add_argument("--explain", type=int, default=0, help="top-N features behind each alert")
    a = p.parse_args(argv)

    if a.cmd == "train":
        header, X = read_csv(a.csv)
        det = FlySentinel(n_flies=a.flies, contamination=a.contamination, decay=a.decay).fit(X)
        det.feature_names_ = header
        with open(a.model, "wb") as f:
            pickle.dump(det, f)
        print(f"trained on {len(X)} rows, {X.shape[1]} features; threshold={det.threshold_:.4f}")
        return

    with open(a.model, "rb") as f:
        det = pickle.load(f)  # only load model files you created yourself
    _, X = read_csv(a.csv)
    scores = det.score_samples(X)
    w = csv.writer(sys.stdout)
    w.writerow(["row", "score", "alert"] + (["why"] if a.explain else []))
    for i, sc in enumerate(scores):
        alert = int(sc > det.threshold_)
        row = [i, f"{sc:.5f}", alert]
        if a.explain:
            why = det.explain(X[i], det.feature_names_, a.explain) if alert else []
            row.append("; ".join(f"{n}:{v:.0%}" for n, v in why))
        w.writerow(row)


if __name__ == "__main__":
    main()
