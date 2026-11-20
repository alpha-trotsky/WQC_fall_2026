"""The five distribution families from the packet, plus the prior over their parameters.

Every family is parametrised by a base scale M (log-uniform prior) plus dimensionless
shape parameters, each uniform on a box. That makes the whole problem (almost)
scale-free: everything you score is proportional to p, so a decision table computed
in "fractions of p" holds whatever the actual pool size is.

    normal       mu = M,                sigma = s*M
    uniform      a = M(1-h),            b = M(1+h)
    shexp        m = M,                 1/lambda = e*M          (mean = M(1+e))
    twohump      mu1 = M(1-d), mu2 = M(1+d), sigma_i = s_i*M, weight w on hump 1
    twopoint     L = M(1-d1), H = M(1+d2), P(L) = q

The organisers have not published their parameter ranges, so the boxes below are
GUESSES. They live in one place (PriorConfig) so you can change them, and the
simulator can use a different PriorConfig for the "truth" than the estimator uses,
to test how badly a wrong prior hurts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr, ndtri

FAMILIES = ("normal", "uniform", "shexp", "twohump", "twopoint")

LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


@dataclass
class PriorConfig:
    """Prior over hidden setups. All shape ranges are (lo, hi) for a uniform prior."""

    m_range: tuple = (50.0, 5000.0)          # base scale M, log-uniform
    family_weights: dict = field(default_factory=lambda: {f: 1.0 for f in FAMILIES})
    shapes: dict = field(default_factory=lambda: {
        "normal": {"s": (0.05, 0.40)},
        "uniform": {"h": (0.05, 0.90)},
        "shexp": {"e": (0.05, 1.00)},
        "twohump": {"d": (0.10, 0.70), "s1": (0.03, 0.25), "s2": (0.03, 0.25), "w": (0.2, 0.8)},
        "twopoint": {"d1": (0.10, 0.80), "d2": (0.10, 1.50), "q": (0.2, 0.8)},
    })

    def log_m_bounds(self):
        return np.log(self.m_range[0]), np.log(self.m_range[1])

    def family_probs(self):
        w = np.array([self.family_weights.get(f, 0.0) for f in FAMILIES], float)
        return w / w.sum()

    def shape_names(self, fam):
        return list(self.shapes[fam].keys())


# ---------------------------------------------------------------------------
# Vectorised family maths. `par` is a dict of equal-length arrays: "M" plus shapes.
# ---------------------------------------------------------------------------

def natural(fam, par):
    M = par["M"]
    if fam == "normal":
        return {"mu": M, "sigma": par["s"] * M}
    if fam == "uniform":
        return {"a": M * (1 - par["h"]), "b": M * (1 + par["h"])}
    if fam == "shexp":
        return {"m": M, "scale": par["e"] * M}
    if fam == "twohump":
        return {"mu1": M * (1 - par["d"]), "mu2": M * (1 + par["d"]),
                "s1": par["s1"] * M, "s2": par["s2"] * M, "w": par["w"]}
    if fam == "twopoint":
        return {"L": M * (1 - par["d1"]), "H": M * (1 + par["d2"]), "q": par["q"]}
    raise ValueError(fam)


def _norm_logpdf(x, mu, sig):
    z = (x - mu) / sig
    return -0.5 * z * z - np.log(sig) - LOG_SQRT_2PI


def loglik_continuous(fam, par, x, delta):
    """Sum over samples x of log(f(x) * delta). x: 1-d array of observations.

    Returns array of shape [N]. Observations are treated as rounded to a grid of
    width `delta`, so a continuous density contributes f(x)*delta per sample.
    """
    nat = natural(fam, par)
    N = len(par["M"])
    x = np.asarray(x, float)[None, :]  # [1, n]
    if fam == "normal":
        lp = _norm_logpdf(x, nat["mu"][:, None], nat["sigma"][:, None])
    elif fam == "uniform":
        a, b = nat["a"][:, None], nat["b"][:, None]
        inside = (x >= a - delta / 2) & (x <= b + delta / 2)
        lp = np.where(inside, -np.log(b - a), -np.inf)
    elif fam == "shexp":
        m, sc = nat["m"][:, None], nat["scale"][:, None]
        # allow the half-cell of rounding below m
        lp = np.where(x >= m - delta / 2, -np.log(sc) - np.maximum(x - m, 0) / sc, -np.inf)
    elif fam == "twohump":
        w = nat["w"][:, None]
        l1 = np.log(w) + _norm_logpdf(x, nat["mu1"][:, None], nat["s1"][:, None])
        l2 = np.log1p(-w) + _norm_logpdf(x, nat["mu2"][:, None], nat["s2"][:, None])
        lp = np.logaddexp(l1, l2)
    else:
        raise ValueError(fam)
    if x.shape[1] == 0:
        return np.zeros(N)
    return lp.sum(axis=1) + x.shape[1] * np.log(delta)


def ppf(fam, par, q):
    """Quantile function, vectorised over particles (q same shape as par arrays)."""
    nat = natural(fam, par)
    q = np.clip(q, 1e-9, 1 - 1e-9)
    if fam == "normal":
        return nat["mu"] + nat["sigma"] * ndtri(q)
    if fam == "uniform":
        return nat["a"] + (nat["b"] - nat["a"]) * q
    if fam == "shexp":
        return nat["m"] - nat["scale"] * np.log1p(-q)
    if fam == "twopoint":
        return np.where(q < nat["q"], nat["L"], nat["H"])
    if fam == "twohump":
        mu1, mu2, s1, s2, w = (nat[k] for k in ("mu1", "mu2", "s1", "s2", "w"))
        lo = np.minimum(mu1 - 9 * s1, mu2 - 9 * s2)
        hi = np.maximum(mu1 + 9 * s1, mu2 + 9 * s2)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            c = w * ndtr((mid - mu1) / s1) + (1 - w) * ndtr((mid - mu2) / s2)
            below = c < q
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)
    raise ValueError(fam)


def sample(fam, par, size, rng):
    """Draw `size` samples for EACH particle -> array [N, size]."""
    N = len(par["M"])
    u = rng.random((N, size))
    rep = {k: np.repeat(v[:, None], size, axis=1).ravel() for k, v in par.items()}
    return ppf(fam, rep, u.ravel()).reshape(N, size)


def draw_prior(fam, prior: PriorConfig, N, rng):
    lo, hi = prior.log_m_bounds()
    par = {"M": np.exp(rng.uniform(lo, hi, N))}
    for k, (a, b) in prior.shapes[fam].items():
        par[k] = rng.uniform(a, b, N)
    return par


def draw_truth(prior: PriorConfig, rng):
    """One hidden setup: (family, params dict of scalars)."""
    fam = FAMILIES[rng.choice(len(FAMILIES), p=prior.family_probs())]
    par = {k: float(v[0]) for k, v in draw_prior(fam, prior, 1, rng).items()}
    return fam, par


def scalar_ppf(fam, par, q):
    return float(ppf(fam, {k: np.array([v]) for k, v in par.items()}, np.array([q]))[0])


def scalar_sample(fam, par, n, rng):
    return sample(fam, {k: np.array([v]) for k, v in par.items()}, n, rng)[0]


def describe(fam, par):
    nat = natural(fam, {k: np.array([v]) for k, v in par.items()})
    return fam + "(" + ", ".join(f"{k}={float(v[0]):.4g}" for k, v in nat.items()) + ")"
