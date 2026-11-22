"""One-shot estimate of the pool size p from the percentile and your samples.

    python -m scavenger.estimate 60 980 1210 1105 1043 1170
                                 ^percentile  ^samples...

Prints family probabilities, the distribution of p (quantiles + a text histogram),
and the guess that maximises expected final-guess points.
Same rule switches as the advisor: --delta, --tp, --pos.
"""
from __future__ import annotations

import argparse

import numpy as np

from .families import PriorConfig
from .posterior import Posterior


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("percentile", type=int)
    ap.add_argument("samples", type=float, nargs="*")
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--tp", default="random", choices=["random", "cdf", "below", "mid"])
    ap.add_argument("--pos", default="0.01")
    ap.add_argument("--particles", type=int, default=12000)
    a = ap.parse_args()
    prior = PriorConfig(delta=a.delta, twopoint_pct=a.tp, m_range=(1, 1e6),
                        positive_tol=None if a.pos == "none" else float(a.pos))
    post = Posterior.fit(a.samples, a.percentile, prior, n_particles=a.particles,
                         rng=np.random.default_rng())

    fp = post.family_probs()
    print("family:  " + "  ".join(f"{f} {v:.0%}" for f, v in sorted(fp.items(), key=lambda kv: -kv[1])))
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    print("p quantiles:  " + "  ".join(f"{int(q * 100)}%: {v:.0f}" for q, v in zip(qs, post.p_quantiles(qs))))
    g, ev = post.best_guess()
    print(f"E[p] = {post.p_mean():.0f}    BEST GUESS = {g:.0f}  (expected guess points {ev:.0f})")

    lo, hi = post.p_quantiles([0.01, 0.99])
    edges = np.linspace(lo, hi, 21)
    h, _ = np.histogram(post.p, bins=edges, weights=post.w)
    print("\ndistribution of p:")
    for i, v in enumerate(h):
        print(f"  {edges[i]:9.0f} - {edges[i + 1]:9.0f}  {v:5.1%}  " + "#" * int(round(60 * v / max(h.max(), 1e-12))))


if __name__ == "__main__":
    main()
