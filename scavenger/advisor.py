"""Live helper for competition day: type in what the real game shows you, get advice.

    python -m scavenger.advisor                     # 5 rounds, learns the room across rounds
    python -m scavenger.advisor --rank low --tp cdf # once you know how the game reports things

Per round: percentile + room size -> suggested n -> type your samples -> scenario summary
-> suggested bid each auction -> type the rank and payout you got -> final guess.
What it learns about the other bidders (timid / mixed / aggressive) carries into the next
round, because the rooms are reshuffled but the players are the same population.
"""
from __future__ import annotations

import argparse

import numpy as np

from .families import PriorConfig
from .sampling import best_n_lookup
from .strategies import BayesStrategy, load_table


def ask(prompt, cast=float, default=None):
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            return cast(s)
        except ValueError:
            print("  (enter a number)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--delta", type=float, default=1.0, help="sample grid: 1 = whole numbers, 1e-6 = real-valued")
    ap.add_argument("--tp", default="random", choices=["random", "cdf", "below", "mid"],
                    help="two-point percentile convention")
    ap.add_argument("--pos", default="0.01", help="max P(pool<0) of a setup; 'none' = no limit")
    ap.add_argument("--rank", default="auto", choices=["auto", "low", "high"],
                    help="auction rank counts from the lowest bid, highest, or infer it")
    a = ap.parse_args()
    prior = PriorConfig(delta=a.delta, twopoint_pct=a.tp, m_range=(1, 1e6),
                        positive_tol=None if a.pos == "none" else float(a.pos))
    strat = BayesStrategy(prior=prior, adapt=True, rank_from=a.rank, particles=12000)
    try:
        table = load_table()
    except OSError:
        table = None
    for r in range(a.rounds):
        print("\n" + "=" * 60 + f"\nROUND {r + 1}")
        k = ask("percentile shown: ", int)
        room = ask("players in the room [7]: ", int, 7)
        strat.start_round(k, room, np.random.default_rng())
        hint = f"table says {best_n_lookup(table, k)}; " if table else ""
        print(f"suggested n: 5 ({hint}full-game harness: 4-7 beat 2-3 in every room, 5 best on points; results/choose_n.txt)")
        xs = [float(v) for v in input("samples you got (space separated): ").split()]
        strat.observe_samples(xs)
        print(strat.post.summary())
        for i in range(3):
            b = strat.bid(i)
            w = dict(zip(strat.room_names, strat.room_weights()))
            print(f"\nAUCTION {i + 1}{' (counts DOUBLE)' if i == 2 else ''}:  suggested bid = {b}"
                  f"   (room looks: " + ", ".join(f"{n} {v:.0%}" for n, v in w.items()) + ")")
            mine = ask(f"  bid you placed [{b}]: ", int, b)
            rank = ask("  rank shown: ", int)
            pay = ask("  payout shown: ")
            strat.observe_auction(i, mine, rank, pay)
            q = strat.post.p_quantiles([0.1, 0.5, 0.9])
            print(f"  p now: median {q[1]:.0f}  (80% range {q[0]:.0f}-{q[2]:.0f})"
                  + (f"   rank counts from: {max(strat.rank_evidence, key=strat.rank_evidence.get)}est"
                     if a.rank == "auto" else ""))
        print(f"\nFINAL GUESS: {strat.guess():.0f}")


if __name__ == "__main__":
    main()
