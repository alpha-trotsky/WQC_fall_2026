"""Strategies the harness can pit against the bots.

Build one from a spec string, e.g.
    bayes:n=3                     fixed 3 samples, model-based bids, Bayes-optimal guess
    bayes:n=table                 n from the precomputed batch table (tables/batch_table.json)
    bayes:seq=1,n0=2              buy 2, then one at a time with lookahead
    bayes:n=3,bid=frac,frac=0.1   bid 10% of the median p estimate every auction
    bayes:n=3,bid=zero
    bayes:n=3,learn=0             don't reweight scenarios on auction results
    naive:n=3                     guess the sample mean, bid 20% of it
Options for bayes: n, seq, n0, bid (model|frac|zero), frac, learn, mult (auction
multiplier for the sample decision), opp_lo, opp_hi, opp_noise, risk, particles,
and the assumptions about unresolved rules (see README):
    delta=1        samples are whole numbers (use 1e-6 for real-valued samples)
    tp=random      two-point percentile convention: random | cdf | below | mid
    pos=0.01       max P(pool < 0) a setup may have; pos=none allows anything
    rank=auto      auction rank counts from the lowest bid (low), highest (high), or
                   auto = infer it from how well each reading explains the payouts
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.special import logsumexp

from . import families as F
from .auction import OpponentModel, auction_loglik, best_bid
from .posterior import Posterior
from .sampling import best_n_lookup, lookahead_decision

TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tables", "batch_table.json")


def load_table(path=TABLE_PATH):
    with open(path) as f:
        return json.load(f)["best_n"]


class BayesStrategy:
    def __init__(self, n=3, seq=False, n0=2, bid="model", frac=0.1, learn=True, mult=0.0,
                 opp=None, risk=0.0, particles=4000, prior=None, table=None,
                 rank_from="auto", name=None):
        self.n, self.sequential, self.n0 = n, seq, n0
        self.bid_mode, self.frac, self.learn, self.mult = bid, frac, learn, mult
        self.opp = opp or OpponentModel()
        self.risk, self.particles = risk, particles
        self.prior = prior or F.PriorConfig()
        self.delta = self.prior.delta
        self.rank_from = rank_from
        self.table = table
        self.name = name or "bayes"

    def start_round(self, percentile, room_size, rng):
        self.k, self.room, self.rng = percentile, room_size, rng
        self.samples = np.array([])
        self.post = None
        self.rank_evidence = {"low": 0.0, "high": 0.0}

    def _refit(self):
        self.post = Posterior.fit(self.samples, self.k, self.prior, n_particles=self.particles,
                                  rng=self.rng)

    def choose_n(self):
        if self.n == "table":
            return best_n_lookup(self.table, self.k)
        return int(self.n)

    def initial_n(self):
        return self.n0

    def observe_samples(self, xs):
        self.samples = np.concatenate([self.samples, xs])
        self._refit()

    def want_more(self):
        buy, self.last_lookahead = lookahead_decision(
            self.post, self.samples, self.k, self.prior, self.rng,
            auction_mult=self.mult, K=24, n_particles=1500)
        return buy

    def bid(self, i):
        if self.bid_mode == "zero":
            return 0
        if self.bid_mode == "frac":
            return int(round(self.frac * self.post.p_quantiles(0.5)[0]))
        b, _ = best_bid(self.post, self.room, self.opp, self.rng, risk_aversion=self.risk)
        return b

    def observe_auction(self, i, bid, rank, payout):
        if not self.learn:
            return
        if self.rank_from == "auto":
            # Try both readings of the displayed rank; keep a running log-evidence for each
            # and use whichever explains the auctions so far better.
            lls = {d: auction_loglik(self.post, self.room, self.opp, bid,
                                     rank if d == "low" else self.room + 1 - rank, payout, self.rng)
                   for d in ("low", "high")}
            for d, ll in lls.items():
                self.rank_evidence[d] += float(logsumexp(self.post.logw + ll))
            best = max(self.rank_evidence, key=self.rank_evidence.get)
            self.post.reweight(lls[best])
            return
        if self.rank_from == "high":
            rank = self.room + 1 - rank  # convert to 1 = lowest bid
        self.post.reweight(auction_loglik(self.post, self.room, self.opp, bid, rank, payout, self.rng))

    def guess(self):
        return self.post.best_guess()[0]


class NaiveStrategy:
    sequential = False

    def __init__(self, n=3, frac=0.2, name="naive"):
        self.n, self.frac, self.name = n, frac, name

    def start_round(self, percentile, room_size, rng):
        self.samples = []

    def choose_n(self):
        return self.n

    def observe_samples(self, xs):
        self.samples += list(xs)

    def bid(self, i):
        return int(round(self.frac * np.mean(self.samples))) if self.samples else 0

    def observe_auction(self, *a):
        pass

    def guess(self):
        return float(np.mean(self.samples)) if self.samples else 0.0


def _num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _tol(v):
    return None if v in ("none", "None") else float(v)


def make_strategy(spec):
    kind, _, rest = spec.partition(":")
    kw = {}
    for part in filter(None, rest.split(",")):
        k, _, v = part.partition("=")
        kw[k.strip()] = _num(v.strip())
    if kind == "naive":
        return NaiveStrategy(n=kw.get("n", 3), frac=kw.get("frac", 0.2), name=spec)
    if kind == "bayes":
        prior = F.PriorConfig(delta=float(kw.pop("delta", 1.0)), twopoint_pct=kw.pop("tp", "random"),
                              positive_tol=_tol(kw.pop("pos", 0.01)))
        opp = OpponentModel(lo_frac=kw.pop("opp_lo", 0.0), hi_frac=kw.pop("opp_hi", 0.4),
                            est_noise=kw.pop("opp_noise", 0.25))
        table = load_table() if kw.get("n") == "table" else None
        return BayesStrategy(
            n=kw.get("n", 3), seq=bool(kw.get("seq", 0)), n0=kw.get("n0", 2),
            bid=kw.get("bid", "model"), frac=kw.get("frac", 0.1), learn=bool(kw.get("learn", 1)),
            mult=kw.get("mult", 0.0), opp=opp, risk=kw.get("risk", 0.0),
            particles=kw.get("particles", 4000), table=table, prior=prior,
            rank_from=kw.get("rank", "auto"), name=spec)
    raise ValueError(f"unknown strategy kind {kind!r}")
