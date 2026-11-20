"""The weighted bag of scenarios (particles).

A particle = (family, parameters, implied pool size p). Its weight is how believable it
is given everything seen so far. Samples set the weights (via importance sampling),
the percentile turns each particle's parameters into a p, and auction results
reweight the bag afterwards (see auction.py).

Fitting, per family:
  * continuous families: adaptive importance sampling in an unconstrained space
    (log M, logit of each shape). Start from a proposal centred on the data, then
    refit a Gaussian to the weighted particles a couple of times so the particles
    land where the posterior is, even when it is narrow (e.g. 10 uniform samples).
  * two-point: handled exactly. Samples are either all one value (it was L, or it
    was H), two values (L and H are pinned), or 3+ values (impossible).
Each family also gets its evidence Z_f = P(samples | family), which gives the
family probabilities.

Observations are treated as rounded to a grid of width `delta` (default 1, i.e.
integers). That is what makes "two identical samples" strong evidence for two-point.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from . import families as F
from .families import FAMILIES, PriorConfig


def percentile_interval(k):
    """Displayed percentile k (integer) -> interval of the true quantile in [0, 1]."""
    return max(k - 0.5, 0.0) / 100.0, min(k + 0.5, 100.0) / 100.0


def _log_logistic(u):
    # log density of the standard logistic: log s(1-s)
    return -np.abs(u) - 2 * np.log1p(np.exp(-np.abs(u)))


class Posterior:
    def __init__(self, groups, percentile, rng, n_data):
        # groups: list of dict(fam, par, logc) where sum(exp(logc)) = P(data | fam)
        self.percentile = percentile
        self.n_data = n_data
        self.groups = groups
        q_lo, q_hi = percentile_interval(percentile)
        ps, fams, logw = [], [], []
        for gi, g in enumerate(groups):
            N = len(g["logc"])
            q = rng.uniform(q_lo, q_hi, N)
            g["p"] = F.ppf(g["fam"], g["par"], q)
            ps.append(g["p"])
            fams.append(np.full(N, gi))
            logw.append(g["logc"] + np.log(g["prior_f"]))
        self.p = np.concatenate(ps)
        self.group_idx = np.concatenate(fams)
        self.local_idx = np.concatenate([np.arange(len(g["logc"])) for g in groups])
        lw = np.concatenate(logw)
        self.log_evidence = logsumexp(lw)
        self.logw = lw - self.log_evidence
        self._w_cache = None

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, samples, percentile, prior: PriorConfig | None = None,
            n_particles=6000, delta=1.0, rng=None, n_iter=3):
        prior = prior or PriorConfig()
        rng = rng if rng is not None else np.random.default_rng()
        x = np.asarray(samples, float)
        groups = []
        fprobs = prior.family_probs()
        for fam, pf in zip(FAMILIES, fprobs):
            if pf <= 0:
                continue
            if len(x) == 0:
                par = F.draw_prior(fam, prior, n_particles, rng)
                logc = np.full(n_particles, -np.log(n_particles))
            elif fam == "twopoint":
                par, logc = _fit_twopoint(x, prior, n_particles, delta, rng)
            else:
                par, logc = _fit_continuous(fam, x, prior, n_particles, delta, rng, n_iter)
            if par is None:
                continue
            groups.append({"fam": fam, "par": par, "logc": logc, "prior_f": pf})
        return cls(groups, percentile, rng, len(x))

    # ---------------------------------------------------------------- summaries
    @property
    def w(self):
        if self._w_cache is None:
            self._w_cache = np.exp(self.logw)
        return self._w_cache

    def ess(self):
        w = self.w
        return 1.0 / np.sum(w * w)

    def family_probs(self):
        out = {f: 0.0 for f in FAMILIES}
        for gi, g in enumerate(self.groups):
            out[g["fam"]] += float(self.w[self.group_idx == gi].sum())
        return out

    def p_mean(self):
        return float(np.sum(self.w * self.p))

    def p_quantiles(self, qs):
        order = np.argsort(self.p)
        cw = np.cumsum(self.w[order])
        return np.interp(np.atleast_1d(qs), cw - self.w[order] / 2, self.p[order])

    def resample_idx(self, K, rng):
        return rng.choice(len(self.p), size=K, p=self.w / self.w.sum())

    def resample_p(self, K, rng):
        return self.p[self.resample_idx(K, rng)]

    def particle(self, i):
        g = self.groups[self.group_idx[i]]
        j = self.local_idx[i]
        return g["fam"], {k: float(v[j]) for k, v in g["par"].items()}

    # ---------------------------------------------------------------- updating
    def reweight(self, loglik):
        """Multiply weights by exp(loglik) (one value per particle) and renormalise."""
        lw = self.logw + loglik
        self.logw = lw - logsumexp(lw)
        self._w_cache = None

    # ---------------------------------------------------------------- decisions
    def guess_score_curve(self, g, max_particles=20000, rng=None):
        """Expected final-guess points E[max(p/4 - |g-p|, 0)] for each candidate g."""
        p, w = self._subset(max_particles, rng)
        g = np.atleast_1d(g)[:, None]
        s = np.maximum(p[None, :] / 4 - np.abs(g - p[None, :]), 0)
        return s @ w

    def best_guess(self, n_grid=241, rng=None):
        """Guess maximising expected guess points. Returns (g, expected points)."""
        qs = np.linspace(0.005, 0.995, n_grid)
        cand = np.unique(self.p_quantiles(qs))
        vals = self.guess_score_curve(cand, rng=rng)
        i = int(np.argmax(vals))
        # local refine
        lo = cand[max(i - 1, 0)]
        hi = cand[min(i + 1, len(cand) - 1)]
        fine = np.linspace(lo, hi, 41)
        fv = self.guess_score_curve(fine, rng=rng)
        j = int(np.argmax(fv))
        return float(fine[j]), float(fv[j])

    def _subset(self, max_particles, rng):
        if len(self.p) <= max_particles:
            return self.p, self.w
        rng = rng if rng is not None else np.random.default_rng(0)
        idx = self.resample_idx(max_particles, rng)
        return self.p[idx], np.full(max_particles, 1.0 / max_particles)

    def summary(self):
        fp = self.family_probs()
        q = self.p_quantiles([0.05, 0.25, 0.5, 0.75, 0.95])
        g, ev = self.best_guess()
        lines = ["family probabilities: " + ", ".join(f"{f}={v:.3f}" for f, v in fp.items()),
                 "p quantiles 5/25/50/75/95%: " + ", ".join(f"{v:.1f}" for v in q),
                 f"E[p] = {self.p_mean():.1f}   best guess g = {g:.1f}   "
                 f"E[guess pts] = {ev:.1f} ({ev / self.p_mean():.1%} of E[p])",
                 f"effective particles: {self.ess():.0f} of {len(self.p)}"]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# continuous families: adaptive importance sampling
# ---------------------------------------------------------------------------

def _to_par(fam, V, prior):
    """Unconstrained vectors V [N, d] -> params dict. Column 0 = log M."""
    par = {"M": np.exp(V[:, 0])}
    for j, (k, (a, b)) in enumerate(prior.shapes[fam].items(), start=1):
        par[k] = a + (b - a) * expit(V[:, j])
    return par


def _log_prior_V(V, prior):
    lo, hi = prior.log_m_bounds()
    inside = (V[:, 0] >= lo) & (V[:, 0] <= hi)
    lp = np.where(inside, -np.log(hi - lo), -np.inf)
    return lp + _log_logistic(V[:, 1:]).sum(axis=1)


def _fit_continuous(fam, x, prior, N, delta, rng, n_iter):
    d = 1 + len(prior.shapes[fam])
    lo, hi = prior.log_m_bounds()
    c0 = np.log(max(np.median(np.abs(x)), 1e-9))
    s0 = 0.6

    def q0_sample(n):
        V = rng.logistic(size=(n, d))
        m = rng.normal(c0, s0, n)
        u = rng.random(n) < 0.15
        m[u] = rng.uniform(lo, hi, u.sum())
        V[:, 0] = m
        return V

    def q0_logpdf(V):
        m = V[:, 0]
        ln = -0.5 * ((m - c0) / s0) ** 2 - np.log(s0) - F.LOG_SQRT_2PI
        lu = np.where((m >= lo) & (m <= hi), -np.log(hi - lo), -np.inf)
        return (np.logaddexp(np.log(0.85) + ln, np.log(0.15) + lu)
                + _log_logistic(V[:, 1:]).sum(axis=1))

    def log_target(V):
        lp = _log_prior_V(V, prior)
        ok = np.isfinite(lp)
        out = np.full(len(V), -np.inf)
        if ok.any():
            out[ok] = lp[ok] + F.loglik_continuous(fam, _to_par(fam, V[ok], prior), x, delta)
        return out

    V = q0_sample(N)
    logq = q0_logpdf(V)
    for it in range(n_iter + 1):
        logw = log_target(V) - logq
        if it == n_iter or not np.isfinite(logw).any():
            break
        wn = np.exp(logw - logsumexp(logw))
        mean = wn @ V
        C = (V - mean).T @ ((V - mean) * wn[:, None])
        C = 0.5 * (C + C.T) + 1e-6 * np.eye(d)
        C *= 2.0  # inflate: heavier coverage than the fitted posterior
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.diag(np.diag(C)) + 1e-4 * np.eye(d)
            Lc = np.linalg.cholesky(C)
        n_g = int(0.8 * N)
        Vg = mean + rng.standard_normal((n_g, d)) @ Lc.T
        V = np.vstack([Vg, q0_sample(N - n_g)])
        # mixture density (deterministic-mixture weights)
        Ci = np.linalg.inv(C)
        dV = V - mean
        logdet = 2 * np.log(np.diag(Lc)).sum()
        lg = -0.5 * np.einsum("ij,jk,ik->i", dV, Ci, dV) - 0.5 * logdet - d * F.LOG_SQRT_2PI
        logq = np.logaddexp(np.log(0.8) + lg, np.log(0.2) + q0_logpdf(V))
    if not np.isfinite(logw).any():
        return None, None
    keep = np.isfinite(logw)
    return _to_par(fam, V[keep], prior), logw[keep] - np.log(N)


# ---------------------------------------------------------------------------
# two-point: exact handling of the discrete structure
# ---------------------------------------------------------------------------

def _distinct(x, delta):
    xs = np.sort(x)
    vals, counts = [xs[0]], [1]
    for v in xs[1:]:
        if abs(v - vals[-1]) <= delta / 2 + 1e-12:
            counts[-1] += 1
        else:
            vals.append(v)
            counts.append(1)
    return vals, counts


def _fit_twopoint(x, prior, N, delta, rng):
    vals, counts = _distinct(x, delta)
    if len(vals) >= 3:
        return None, None
    sh = prior.shapes["twopoint"]
    (d1a, d1b), (d2a, d2b), (qa, qb) = sh["d1"], sh["d2"], sh["q"]
    lo, hi = prior.log_m_bounds()
    log_rM, log_rd1, log_rd2 = np.log(hi - lo), np.log(d1b - d1a), np.log(d2b - d2a)

    def m_interval(cons):
        a, b = lo, hi
        for ca, cb in cons:
            a, b = max(a, ca), min(b, cb)
        return a, b

    def case(n_case, L=None, H=None, nL=0, nH=0):
        """Sample particles with L and/or H pinned to observed values."""
        cons = []
        if L is not None:  # d1 = 1 - L/M in [d1a, d1b]  ->  M in [L/(1-d1a), L/(1-d1b)]
            cons.append((np.log(L / (1 - d1a)), np.log(L / (1 - d1b)) if d1b < 1 else np.inf))
        if H is not None:  # d2 = H/M - 1 in [d2a, d2b]  ->  M in [H/(1+d2b), H/(1+d2a)]
            cons.append((np.log(H / (1 + d2b)), np.log(H / (1 + d2a))))
        a, b = m_interval(cons)
        if b <= a or n_case == 0:
            return None
        logM = rng.uniform(a, b, n_case)
        M = np.exp(logM)
        q = rng.uniform(qa, qb, n_case)
        par = {"M": M, "q": q}
        logc = -log_rM + np.log(b - a)  # prior(logM) / proposal(logM)
        if L is not None:
            par["d1"] = 1 - L / M
            logc = logc - log_rd1 - np.log(M) + np.log(delta)  # density of L, times cell
        else:
            par["d1"] = rng.uniform(d1a, d1b, n_case)
        if H is not None:
            par["d2"] = H / M - 1
            logc = logc - log_rd2 - np.log(M) + np.log(delta)
        else:
            par["d2"] = rng.uniform(d2a, d2b, n_case)
        logc = logc + nL * np.log(q) + nH * np.log1p(-q) - np.log(n_case)
        return par, logc

    if len(vals) == 2:
        outs = [case(N, L=vals[0], H=vals[1], nL=counts[0], nH=counts[1])]
    else:
        v, n = vals[0], counts[0]
        outs = [case(N // 2, L=v, nL=n), case(N - N // 2, H=v, nH=n)]
    outs = [o for o in outs if o is not None]
    if not outs:
        return None, None
    par = {k: np.concatenate([o[0][k] for o in outs]) for k in outs[0][0]}
    logc = np.concatenate([np.broadcast_to(o[1], o[0]["M"].shape) for o in outs])
    return par, logc
