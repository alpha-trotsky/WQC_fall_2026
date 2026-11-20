"""Play Scavenger in the terminal against bots.

    python -m scavenger.play                 # 5 rounds, buy all samples at once
    python -m scavenger.play --sequential    # buy samples one at a time
    python -m scavenger.play --hints         # show the estimator's view and suggestions

After each auction you see only your own payout and rank (rank 1 = lowest bid), as in
the real game. At the end of the round everything is revealed.
"""
from __future__ import annotations

import argparse

import numpy as np

from .auction import OpponentModel, auction_loglik, best_bid
from .game import BotConfig, GameConfig, Round
from .posterior import Posterior
from .strategies import load_table
from .sampling import best_n_lookup


def ask_num(prompt, cast=float, lo=None):
    while True:
        s = input(prompt).strip()
        try:
            v = cast(s)
            if lo is not None and v < lo:
                raise ValueError
            return v
        except ValueError:
            print("  enter a number" + (f" >= {lo}" if lo is not None else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sequential", action="store_true", help="buy samples one at a time")
    ap.add_argument("--hints", action="store_true", help="show posterior + recommended n/bid/guess")
    ap.add_argument("--bot-lo", type=float, default=0.0)
    ap.add_argument("--bot-hi", type=float, default=0.5)
    ap.add_argument("--bot-b", type=float, default=0.0)
    a = ap.parse_args()
    cfg = GameConfig(bots=BotConfig(lo_frac=a.bot_lo, hi_frac=a.bot_hi, add_b=a.bot_b))
    seed0 = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 10**9)
    opp = OpponentModel()
    try:
        table = load_table()
    except OSError:
        table = None
    placements = []
    for r in range(a.rounds):
        rnd = Round(cfg, seed0 + r)
        rng = np.random.default_rng(seed0 + r + 99)
        print("\n" + "=" * 60)
        print(f"ROUND {r + 1}/{a.rounds}   room of {rnd.room_size} players")
        print(f"The pool size is at the {rnd.percentile}th percentile of its distribution.")
        print("Sample cost: n samples cost sqrt(n)/12 of the pool  "
              "(1: 8.3%, 2: 11.8%, 3: 14.4%, 4: 16.7%, 9: 25%)")
        post = Posterior.fit([], rnd.percentile, rng=rng)
        if a.hints and table:
            print(f"  [hint] batch table says buy n = {best_n_lookup(table, rnd.percentile)}")
        if a.sequential:
            while True:
                n = len(rnd.hero_samples)
                ans = input(f"You have {n} samples (cost so far {np.sqrt(n) / 12:.1%}). Buy one more? [y/N] ")
                if ans.strip().lower() != "y":
                    break
                x = rnd.buy(1)[0]
                print(f"  sample: {x:g}")
                if a.hints:
                    post = Posterior.fit(rnd.hero_samples, rnd.percentile, rng=rng)
                    print("  [hint] " + post.summary().replace("\n", "\n  [hint] "))
        else:
            n = ask_num("How many samples do you buy? ", int, 0)
            xs = rnd.buy(min(n, cfg.max_samples))
            print("  samples:", ", ".join(f"{x:g}" for x in xs))
        if a.hints:
            post = Posterior.fit(rnd.hero_samples, rnd.percentile, rng=rng)
            print("  [hint] " + post.summary().replace("\n", "\n  [hint] "))
        for i in range(3):
            label = f"Auction {i + 1}/3" + (" (counts DOUBLE)" if i == 2 else "")
            if a.hints:
                b, _ = best_bid(post, rnd.room_size, opp, rng)
                print(f"  [hint] model bid = {b}")
            bid = ask_num(f"{label}: your bid? ", int, 0)
            res = rnd.auction(bid)
            print(f"  -> you were rank {res['hero_rank']} of {rnd.room_size} (1 = lowest bid), "
                  f"paid {res['hero_pay']:g}")
            if a.hints:
                post.reweight(auction_loglik(post, rnd.room_size, opp, bid, res["hero_rank"], res["hero_pay"], rng))
        if a.hints:
            g, ev = post.best_guess()
            print("  [hint] " + post.summary().replace("\n", "\n  [hint] "))
        guess = ask_num("Final guess for the pool size? ")
        out = rnd.finish(guess)
        placements.append(out["placement"])
        print("\n--- round result ---")
        print(f"true pool p = {out['p']:.2f}   from {out['truth']}")
        print(f"auction bids (you first): ")
        for i, au in enumerate(rnd.auctions):
            print(f"  A{i + 1}: bids {au['bids'].astype(int).tolist()}  pays {np.round(au['pay'], 1).tolist()}")
        print(f"your points: auctions {out['auction_pts']} (A3 x2), guess {out['guess_pts']:.1f}, "
              f"sample cost -{out['sample_cost']:.1f}  => total {out['total']:.1f} ({out['total_frac']:.1%} of p)")
        print(f"room totals: {np.round(out['all_totals'], 1).tolist()}")
        print(f"your rank this round: {out['rank']:g} of {rnd.room_size}  (placement {out['placement']:.3f})")
    print("\n" + "=" * 60)
    print(f"mean placement over {a.rounds} rounds: {np.mean(placements):.3f}  (lower is better)")


if __name__ == "__main__":
    main()
