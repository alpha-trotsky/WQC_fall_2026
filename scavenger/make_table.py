"""Precompute the batch sample-size table (run this before the competition).

    python -m scavenger.make_table --trials 2000 --mult 0.0

Writes tables/batch_table.json (used by the `bayes:n=table` strategy and the advisor)
and tables/batch_table.txt (human-readable).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from .sampling import batch_table, format_table
from .strategies import TABLE_PATH


def _job(args):
    seed, trials, n_max = args
    return batch_table(n_max=n_max, trials=trials, seed=seed, progress=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1200)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--mult", type=float, default=0.0,
                    help="value accuracy in the auctions at this multiple of its value in the guess")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse", action="store_true",
                    help="skip the simulation and re-score the saved raw results (e.g. with a new --mult)")
    a = ap.parse_args()
    raw_path = TABLE_PATH.replace(".json", "_raw.npz")
    if a.reuse:
        z = np.load(raw_path)
        res = {"pct": z["pct"], "score": z["score"]}
    else:
        per = int(np.ceil(a.trials / a.workers))
        with mp.get_context("fork").Pool(a.workers) as pool:
            parts = pool.map(_job, [(a.seed * 1000 + i, per, a.n_max) for i in range(a.workers)])
        res = {"pct": np.concatenate([p["pct"] for p in parts]),
               "score": np.vstack([p["score"] for p in parts])}
    text, best = format_table(res, a.mult)
    print(text)
    os.makedirs(os.path.dirname(TABLE_PATH), exist_ok=True)
    with open(TABLE_PATH, "w") as f:
        json.dump({"best_n": best, "auction_mult": a.mult, "trials": int(len(res["pct"])),
                   "mean_score_by_n": res["score"].mean(axis=0).tolist()}, f, indent=1)
    with open(TABLE_PATH.replace(".json", ".txt"), "w") as f:
        f.write(text + "\n")
    np.savez(raw_path, **res)
    print(f"\nwrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
