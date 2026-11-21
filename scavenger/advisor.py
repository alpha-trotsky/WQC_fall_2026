"""Live helper for competition day: type in what the real game shows you, get advice.

    python -m scavenger.advisor

Flow: percentile -> recommended n (batch table) -> type the samples you got ->
posterior + model bid -> after each auction type rank & payout -> updated bid ->
final guess. Opponent model settings via --opp-lo/--opp-hi.
"""
from __future__ import annotations

import argparse

import numpy as np

from .families import PriorConfig
from .auction import OpponentModel, auction_loglik, best_bid
from .posterior import Posterior
from .sampling import best_n_lookup, lookahead_decision
from .strategies import load_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=int, default=7)
    ap.add_argument("--opp-lo", type=float, default=0.0)
    ap.add_argument("--opp-hi", type=float, default=0.4)
    ap.add_argument("--sequential", action="store_true", help="also run the lookahead after each sample")
    ap.add_argument("--delta", type=float, default=1.0, help="sample grid: 1 = whole numbers, 1e-6 = real-valued")
    ap.add_argument("--tp", default="random", choices=["random", "cdf", "below", "mid"],
                    help="two-point percentile convention")
    ap.add_argument("--pos", default="0.01", help="max P(pool<0) of a setup; 'none' = no limit")
    ap.add_argument("--rank", default="low", choices=["low", "high"], help="auction rank counts from")
    a = ap.parse_args()
    prior = PriorConfig(delta=a.delta, twopoint_pct=a.tp,
                        positive_tol=None if a.pos == "none" else float(a.pos))
    rng = np.random.default_rng()
    opp = OpponentModel(a.opp_lo, a.opp_hi)
    k = int(input("percentile shown: "))
    try:
        print(f"batch table: buy n = {best_n_lookup(load_table(), k)}")
    except OSError:
        print("(no batch table yet: run python -m scavenger.make_table)")
    xs = np.array([float(v) for v in input("samples (space separated): ").split()])
    post = Posterior.fit(xs, k, prior, n_particles=12000, rng=rng)
    print(post.summary())
    if a.sequential:
        buy, info = lookahead_decision(post, xs, k, prior, rng)
        print(f"lookahead: buy more? {buy}   net gain by #extra: "
              + ", ".join(f"{j}: {v:+.1f}" for j, v in info["net"].items()))
    for i in range(3):
        b, _ = best_bid(post, a.room, opp, rng)
        print(f"\nAUCTION {i + 1}: model bid = {b}")
        mine = int(input("  bid you actually placed: ") or b)
        rank = int(input(f"  your rank as shown (1 = {a.rank}est bid): "))
        if a.rank == "high":
            rank = a.room + 1 - rank
        pay = float(input("  your payout: "))
        post.reweight(auction_loglik(post, a.room, opp, mine, rank, pay, rng))
        print(post.summary())
    g, ev = post.best_guess()
    print(f"\nFINAL GUESS: {g:.0f}")


if __name__ == "__main__":
    main()
