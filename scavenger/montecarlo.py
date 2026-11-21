"""Monte Carlo harness: run strategies through many simulated rounds and compare.

Every strategy sees the SAME rounds (same hidden setups, same bots, same sample
stream), so differences between strategies are not just luck of the draw.

    python -m scavenger.montecarlo -s "bayes:n=1" "bayes:n=3" "bayes:n=table" "naive:n=3" -r 400

Bot behaviour (the simulated room) is set with --bot-lo/--bot-hi/--bot-b/--bot-noise.
The opponent model inside a bayes strategy is separate (opp_lo/opp_hi/opp_noise in
its spec), so you can test a strategy whose beliefs about the room are wrong.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from .families import SHAPE_PRESETS, PriorConfig
from .game import BotConfig, GameConfig, play_round
from .strategies import make_strategy


def _worker(args):
    spec, cfg, seeds = args
    strat = make_strategy(spec)
    return [play_round(strat, cfg, s) for s in seeds]


def run(specs, rounds=200, seed=0, cfg=None, workers=None):
    cfg = cfg or GameConfig()
    workers = workers or max(mp.cpu_count() - 0, 1)
    seeds = [seed * 1_000_003 + i for i in range(rounds)]
    chunks = [seeds[i::workers] for i in range(workers)]
    results = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for spec in specs:
            t = time.time()
            out = pool.map(_worker, [(spec, cfg, c) for c in chunks if c])
            res = [r for chunk in out for r in chunk]
            res.sort(key=lambda r: (r["p"], r["percentile"]))
            results[spec] = res
            print(f"  {spec}: {len(res)} rounds in {time.time() - t:.0f}s", flush=True)
    return results


def report(results):
    hdr = (f"{'strategy':<34} {'placement':>9} {'±':>5} {'P(1st)':>7} {'pts/p':>7} {'±':>6} "
           f"{'auct/p':>7} {'guess/p':>8} {'cost/p':>7} {'avg n':>6}")
    lines = [hdr, "-" * len(hdr)]
    for spec, res in results.items():
        pl = np.array([r["placement"] for r in res])
        tf = np.array([r["total_frac"] for r in res])
        p = np.array([r["p"] for r in res])
        auc = np.array([r["auction_pts"][0] + r["auction_pts"][1] + 2 * r["auction_pts"][2] for r in res]) / p
        gs = np.array([r["guess_pts"] for r in res]) / p
        cs = np.array([r["sample_cost"] for r in res]) / p
        n = np.array([r["n"] for r in res])
        first = np.mean([r["rank"] == 1 for r in res])
        se = lambda a: a.std() / np.sqrt(len(a))
        lines.append(f"{spec:<34} {pl.mean():9.3f} {se(pl):5.2f} {first:7.1%} {tf.mean():7.3f} {se(tf):6.3f} "
                     f"{auc.mean():7.3f} {gs.mean():8.3f} {cs.mean():7.3f} {n.mean():6.2f}")
    lines.append("placement: mean rank in room (1 = best, x0.875 in rooms of 7) - this is what the game scores.")
    lines.append("pts/p: round points as a fraction of the pool. ± = one standard error.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--strategies", nargs="+", default=["bayes:n=1", "bayes:n=3", "bayes:n=5", "naive:n=3"])
    ap.add_argument("-r", "--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bot-lo", type=float, default=0.0)
    ap.add_argument("--bot-hi", type=float, default=0.5)
    ap.add_argument("--bot-b", type=float, default=0.0)
    ap.add_argument("--bot-noise", type=float, default=0.35)
    ap.add_argument("--bot-mix", default="",
                    help="mixed room, e.g. '0.4:0:0.15,0.4:0.1:0.3,0.2:0.6:1.0' = weight:lo:hi per archetype")
    g = ap.add_argument_group("how the simulated game resolves the rules the packet leaves open")
    g.add_argument("--truth-delta", type=float, default=1.0, help="sample rounding grid (1e-6 = real-valued)")
    g.add_argument("--truth-tp", default="random", choices=["random", "cdf", "below", "mid"],
                   help="two-point percentile convention")
    g.add_argument("--truth-pos", default="0.01", help="max P(pool<0) of a setup; 'none' = no limit")
    g.add_argument("--truth-shapes", default="default", choices=["default", "wide", "narrow"])
    g.add_argument("--truth-mlo", type=float, default=50)
    g.add_argument("--truth-mhi", type=float, default=5000)
    g.add_argument("--truth-rank", default="low", choices=["low", "high"], help="auction rank counts from")
    a = ap.parse_args()
    truth = PriorConfig(delta=a.truth_delta, twopoint_pct=a.truth_tp,
                        positive_tol=None if a.truth_pos == "none" else float(a.truth_pos),
                        shapes=SHAPE_PRESETS[a.truth_shapes], m_range=(a.truth_mlo, a.truth_mhi))
    mix = tuple(tuple(float(v) for v in part.split(":")) for part in a.bot_mix.split(",") if part)
    cfg = GameConfig(truth_prior=truth, rank_from=a.truth_rank,
                     bots=BotConfig(lo_frac=a.bot_lo, hi_frac=a.bot_hi, add_b=a.bot_b, est_noise=a.bot_noise,
                                   mix=mix))
    res = run(a.strategies, a.rounds, a.seed, cfg, a.workers)
    print()
    print(report(res))


if __name__ == "__main__":
    main()
