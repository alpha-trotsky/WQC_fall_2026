"""Posterior over the 30 chips, and the expected value of VAULT, PUB and G5.

    t = VaultTracker()
    t.partial(1, 7)            # chip 1 partial reveal shows 7
    t.combo(2, 4, 1, 25)       # chip 2 is a combination chip: round(mean(T1, T25)) = 4
    t.true(1, 8)               # chip 1 true value is 8
    est = t.estimate()
    est["VAULT"].mean, est["VAULT"].sd

How it works
------------
Prior: 30 independent chips, each uniform on 1..10.

Evidence comes in three forms, all multiplied into the prior (Bayes):
  * a true value pins a chip:                      unary[i] = indicator(T_i = v)
  * a non-combination partial reveal showing d:   unary[i] *= P(shows d | T_i)   (model.reveal_likelihood)
  * a combination reveal of chip c over (a, b):   factor(T_a, T_b) = 1 if round((T_a+T_b)/2) = d
    (it says nothing about T_c itself, only about a and b -- which may be private or future chips)

Without combination chips the chips stay independent, so each one has its own posterior and
    E[VAULT] = sum_i E[T_i],   E[PUB] = sum_{i<=20} E[T_i],   E[G5] = sum_i P(T_i > 5).
Combination chips couple pairs of chips, so for the full distribution (and the s.d.) we compute the
exact distribution of each contract by variable elimination: add the chips one at a time, carrying a
table over (values of the chips still linked to unadded chips) x (running contract value). Chips
not linked to anything are summed out at once, so the table stays tiny (the combination graph has
at most 20 edges, so in practice at most 2-3 chips are ever "open"). A Gibbs sampler is the fallback
if a pathological graph would make that table too big.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import (CONTRACTS, N_CHIPS, N_PUBLIC, PUBLIC_IDS, Rules, combo_matrix, maker_wmax,
                    reveal_likelihood, type_posterior)

MAX_OPEN = 4            # largest number of open chips in the exact elimination (10^5 x 301 table)


class InconsistentEvidence(ValueError):
    pass


# ---------------------------------------------------------------- the inference engine

def _contrib(contract: str) -> np.ndarray:
    """c[i, x] = how much chip i (0-based) with true value x+1 adds to the contract."""
    vals = np.arange(1, 11)
    if contract == "VAULT":
        return np.tile(vals, (N_CHIPS, 1))
    if contract == "PUB":
        c = np.tile(vals, (N_CHIPS, 1))
        c[N_PUBLIC:] = 0
        return c
    if contract == "G5":
        return np.tile((vals > 5).astype(int), (N_CHIPS, 1))
    raise ValueError(contract)


def _elimination_order(n, nbrs):
    added, open_, order, worst = set(), [], [], 0
    remaining = set(range(n))
    while remaining:
        best = None
        for v in sorted(remaining):
            after = added | {v}
            new_open = [u for u in open_ + [v] if not nbrs[u] <= after]
            linked = sum(1 for u in nbrs[v] if u in added)
            key = (len(new_open), -linked, v)
            if best is None or key < best[0]:
                best = (key, v, new_open)
        _, v, new_open = best
        worst = max(worst, len(open_) + 1)
        order.append(v)
        added.add(v)
        remaining.discard(v)
        open_ = new_open
    return order, worst


def stat_distribution(unary: np.ndarray, pairs, contrib: np.ndarray) -> np.ndarray:
    """Exact P(sum_i contrib[i, T_i] = s) for s = 0..max, given unary factors and pairwise factors.

    unary: (n, 10) non-negative; pairs: list of (i, j, M) with M[x_i, x_j] (0-based ids);
    contrib: (n, 10) non-negative ints.
    """
    n = len(unary)
    nbrs = [set() for _ in range(n)]
    for i, j, _ in pairs:
        nbrs[i].add(j)
        nbrs[j].add(i)
    order, worst = _elimination_order(n, nbrs)
    if worst > MAX_OPEN:
        return _gibbs_distribution(unary, pairs, contrib)

    smax = int(contrib.max(axis=1).sum())
    T = np.zeros(smax + 1)
    T[0] = 1.0
    open_ = []                      # axis k of T (except the last) is chip open_[k]
    added = set()
    for v in order:
        f = len(open_)
        fac = np.ones((10,) * f + (10,))
        for i, j, M in pairs:
            if v not in (i, j):
                continue
            u = j if i == v else i
            if u not in added:
                continue            # factor is applied when the other end is added
            Mv = M if i == u else M.T               # Mv[x_u, x_v]
            shape = [1] * (f + 1)
            shape[open_.index(u)] = 10
            shape[-1] = 10
            fac = fac * Mv.reshape(shape)
        fac = fac * unary[v].reshape([1] * f + [10])
        newT = np.zeros(T.shape[:-1] + (10, smax + 1))
        for x in range(10):
            c = int(contrib[v, x])
            w = fac[..., x][..., None]
            if c:
                newT[..., x, c:] = T[..., :smax + 1 - c] * w
            else:
                newT[..., x, :] = T * w
        T = newT
        open_.append(v)
        added.add(v)
        # sum out chips that have no links to unadded chips
        for u in [u for u in open_ if nbrs[u] <= added][::-1]:
            T = T.sum(axis=open_.index(u))
            open_.remove(u)
        z = T.sum()
        if z <= 0:
            raise InconsistentEvidence("no assignment of chip values fits everything entered")
        T = T / z
    return T


def _gibbs_distribution(unary, pairs, contrib, chains=2000, burn=150, sweeps=300, seed=0):
    """Fallback for a densely linked combination graph: parallel-chain Gibbs sampling."""
    rng = np.random.default_rng(seed)
    n = len(unary)
    U = unary / unary.sum(axis=1, keepdims=True)
    by_chip = [[] for _ in range(n)]
    for i, j, M in pairs:
        by_chip[i].append((j, M))          # P(x_i | x_j) ∝ M[x_i, x_j]
        by_chip[j].append((i, M.T))

    def draw(p):
        cum = p.cumsum(axis=1)
        return (rng.random((len(p), 1)) * cum[:, -1:] > cum).sum(axis=1)

    X = np.stack([draw(np.tile(U[v], (chains, 1))) for v in range(n)], axis=1)
    smax = int(contrib.max(axis=1).sum())
    hist = np.zeros(smax + 1)
    for s in range(burn + sweeps):
        for v in range(n):
            p = np.tile(U[v], (chains, 1))
            for u, M in by_chip[v]:
                p = p * M[:, X[:, u]].T
            dead = p.sum(axis=1) == 0
            p[dead] = U[v]
            X[:, v] = draw(p)
        if s >= burn:
            ok = np.ones(chains, bool)
            for i, j, M in pairs:
                ok &= M[X[:, i], X[:, j]] > 0
            stat = contrib[np.arange(n), X].sum(axis=1)
            hist += np.bincount(stat[ok], minlength=smax + 1)
    if hist.sum() == 0:
        raise InconsistentEvidence("no assignment of chip values fits everything entered")
    return hist / hist.sum()


# ---------------------------------------------------------------- results

@dataclass
class ContractEstimate:
    name: str
    dist: np.ndarray                 # dist[s] = P(contract settles at s)

    @property
    def mean(self) -> float:
        return float(np.arange(len(self.dist)) @ self.dist)

    @property
    def sd(self) -> float:
        s = np.arange(len(self.dist))
        return float(np.sqrt(max((s * s) @ self.dist - self.mean ** 2, 0.0)))

    def quantile(self, q: float) -> int:
        return int(np.searchsorted(np.cumsum(self.dist), q - 1e-12))


@dataclass
class Estimate:
    contracts: dict[str, ContractEstimate]
    chip_means: np.ndarray           # E[T_i], index 0 = chip 1
    chip_g5: np.ndarray              # P(T_i > 5)
    wmax: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, k):
        return self.contracts[k]


# ---------------------------------------------------------------- the tracker

class VaultTracker:
    """Everything one team has been shown. Enter reveals as they arrive; call estimate()."""

    def __init__(self, rules: Rules = Rules()):
        self.rules = rules
        self.L = reveal_likelihood(rules)
        self.partials: dict[int, tuple] = {}    # id -> (d,) or (d, a, b)
        self.trues: dict[int, int] = {}
        self.log: list[tuple] = []               # for undo

    # -- input
    @staticmethod
    def _check(i, v=None):
        if not 1 <= i <= N_CHIPS:
            raise ValueError(f"chip id must be 1..{N_CHIPS}, got {i}")
        if v is not None and not 1 <= v <= 10:
            raise ValueError(f"chip value must be 1..10, got {v}")

    def partial(self, i: int, d: int):
        self._check(i, d)
        if i > N_PUBLIC:
            raise ValueError("only public chips (1..20) get partial reveals")
        self.log.append(("p", i, self.partials.get(i)))
        self.partials[i] = (d,)

    def combo(self, i: int, d: int, a: int, b: int):
        self._check(i, d)
        self._check(a)
        self._check(b)
        if i in (a, b):
            raise ValueError("a combination chip refers to two *other* chips")
        self.log.append(("p", i, self.partials.get(i)))
        self.partials[i] = (d, a, b)

    def true(self, i: int, v: int):
        self._check(i, v)
        self.log.append(("t", i, self.trues.get(i)))
        self.trues[i] = v

    def undo(self):
        if not self.log:
            return False
        kind, i, old = self.log.pop()
        store = self.partials if kind == "p" else self.trues
        if old is None:
            store.pop(i, None)
        else:
            store[i] = old
        return True

    # -- views
    def maker_view(self) -> "VaultTracker":
        """What the makers know, from a taker's information.

        Makers get exactly what takers got one interval earlier, and never see private chips. If the
        highest public chip whose true value takers know is k, makers know partial reveals of chips
        1..k and true values of chips 1..k-1.
        """
        m = VaultTracker(self.rules)
        k = max([i for i in self.trues if i <= N_PUBLIC], default=0)
        m.partials = {i: p for i, p in self.partials.items() if i <= k}
        m.trues = {i: v for i, v in self.trues.items() if i <= min(k - 1, N_PUBLIC)}
        return m

    # -- inference
    def evidence(self):
        unary = np.ones((N_CHIPS, 10))
        pairs = []
        for i, p in self.partials.items():
            if len(p) == 1:
                unary[i - 1] *= self.L[p[0] - 1]
            else:
                d, a, b = p
                pairs.append((a - 1, b - 1, combo_matrix(d, self.rules)))
        for i, v in self.trues.items():
            unary[i - 1] = 0.0
            unary[i - 1, v - 1] = 1.0
        # a pair naming the same chip twice is a constraint on that chip alone
        same = [(i, M) for i, j, M in pairs if i == j]
        for i, M in same:
            unary[i] *= np.diag(M)
        pairs = [p for p in pairs if p[0] != p[1]]
        if (unary.sum(axis=1) == 0).any():
            raise InconsistentEvidence("a chip has no value consistent with what was entered")
        return unary, pairs

    def estimate(self) -> Estimate:
        unary, pairs = self.evidence()
        contracts = {c: ContractEstimate(c, stat_distribution(unary, pairs, _contrib(c)))
                     for c in CONTRACTS}
        # per-chip posteriors: independent chips straight from the unary, linked chips exactly
        post = unary / unary.sum(axis=1, keepdims=True)
        linked = {i for i, j, _ in pairs} | {j for i, j, _ in pairs}
        for v in linked:
            c = np.zeros((N_CHIPS, 10), int)
            c[v] = np.arange(10)
            post[v] = stat_distribution(unary, pairs, c)[:10]
        vals = np.arange(1, 11)
        return Estimate(contracts, post @ vals, post[:, 5:].sum(axis=1),
                        maker_wmax([i for i in self.trues]))

    def chip_type_odds(self, i: int):
        p = self.partials.get(i)
        if p is None or len(p) != 1:
            return None
        return type_posterior(p[0], self.rules)
