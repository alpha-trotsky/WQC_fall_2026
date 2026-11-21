"""How many samples to buy.

Batch (buy n once): `batch_table` simulates, before the competition, the expected
final-guess score as a fraction of p for every n, split by the displayed percentile.
Net value(n) = E[guess pts / p] * (1 + auction_mult) - sqrt(n)/12. You read off the
best n for the percentile you are shown.

Sequential (buy one, look, decide): `lookahead_decision` estimates the gain from 1..k
more samples by simulating them from the current scenario bag and refitting.
"""
from __future__ import annotations

import numpy as np

from . import families as F
from .posterior import Posterior

PCT_BUCKETS = [(0, 5), (5, 15), (15, 35), (35, 65), (65, 85), (85, 95), (95, 101)]


def cost(n):
    return np.sqrt(n) / 12


def guess_frac(p_true, g):
    return max(p_true / 4 - abs(g - p_true), 0.0) / p_true


def batch_table(prior_truth=None, prior_est=None, n_max=10, trials=600, seed=0,
                n_particles=3000, delta=None, progress=True):
    """Monte Carlo of guess score/p for n = 0..n_max with common random numbers.

    Returns dict with arrays: pct [trials], score [trials, n_max+1].
    """
    prior_truth = prior_truth or F.PriorConfig()
    prior_est = prior_est or prior_truth
    delta = prior_truth.delta if delta is None else delta
    rng = np.random.default_rng(seed)
    pct = np.zeros(trials, int)
    score = np.zeros((trials, n_max + 1))
    for t in range(trials):
        fam, par = F.draw_truth(prior_truth, rng)
        q = rng.random()
        k, p = F.display_percentile(fam, par, q, prior_truth.twopoint_pct)
        xs = np.round(F.scalar_sample(fam, par, n_max, rng) / delta) * delta
        pct[t] = k
        for n in range(n_max + 1):
            post = Posterior.fit(xs[:n], k, prior_est, n_particles=n_particles, rng=rng)
            g, _ = post.best_guess()
            score[t, n] = guess_frac(p, g) if p > 0 else 0.0
        if progress and (t + 1) % 50 == 0:
            print(f"  {t + 1}/{trials} trials", flush=True)
    return {"pct": pct, "score": score}


def format_table(res, auction_mult=0.0):
    pct, score = res["pct"], res["score"]
    n_max = score.shape[1] - 1
    ns = np.arange(n_max + 1)
    net_cost = cost(ns)
    lines = []
    hdr = "percentile  " + " ".join(f"{'n=' + str(n):>7}" for n in ns) + "   best n"
    lines.append(f"Net value per round as % of p  (guess score x (1+{auction_mult}) - sqrt(n)/12)")
    lines.append(hdr)
    rows = [("all", np.ones(len(pct), bool))] + [
        (f"{a}-{min(b, 100)}", (pct >= a) & (pct < b)) for a, b in PCT_BUCKETS]
    best = {}
    for name, mask in rows:
        if mask.sum() == 0:
            continue
        v = score[mask].mean(axis=0) * (1 + auction_mult) - net_cost
        se = score[mask].std(axis=0) / np.sqrt(mask.sum())
        b = int(np.argmax(v))
        best[name] = b
        lines.append(f"{name:>10}  " + " ".join(f"{100 * x:7.2f}" for x in v)
                     + f"   {b}   (N={mask.sum()}, s.e.~{100 * se.mean():.2f})")
    lines.append("")
    lines.append("Raw guess score as % of p (no cost):")
    lines.append(hdr[:-9])
    for name, mask in rows:
        if mask.sum():
            lines.append(f"{name:>10}  " + " ".join(f"{100 * x:7.2f}" for x in score[mask].mean(axis=0)))
    return "\n".join(lines), best


def best_n_lookup(best, percentile):
    for a, b in PCT_BUCKETS:
        if a <= percentile < b:
            return best.get(f"{a}-{min(b, 100)}", best.get("all", 3))
    return best.get("all", 3)


def lookahead_decision(post, samples, percentile, prior, rng, max_ahead=3, K=48,
                       n_particles=2000, delta=None, auction_mult=0.0):
    """Should we buy another sample? Checks buying 1..max_ahead more (catches the case
    where one more sample cannot separate families but three can).

    Returns (buy: bool, info dict). Gains and costs are in points (same units as p).
    """
    prior = prior or F.PriorConfig()
    delta = prior.delta if delta is None else delta
    n = len(samples)
    _, base = post.best_guess()
    Ep = post.p_mean()
    idx = post.resample_idx(K, rng)
    gains = {}
    for k in range(1, max_ahead + 1):
        vals = []
        for i in idx:
            fam, par = post.particle(i)
            new = np.round(F.scalar_sample(fam, par, k, rng) / delta) * delta
            p2 = Posterior.fit(np.concatenate([samples, new]), percentile, prior,
                               n_particles=n_particles, rng=rng)
            vals.append(p2.best_guess()[1])
        gains[k] = (np.mean(vals) - base) * (1 + auction_mult)
    costs = {k: (cost(n + k) - cost(n)) * Ep for k in gains}
    net = {k: gains[k] - costs[k] for k in gains}
    buy = max(net.values()) > 0
    return buy, {"gain": gains, "cost": costs, "net": net, "E[p]": Ep}
