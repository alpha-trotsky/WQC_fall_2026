"""The Scavenger game engine: one round = hidden setup, info stage, 3 auctions, final guess.

Used both by the interactive game (play.py) and the Monte Carlo harness (montecarlo.py).
The "hero" is the one player that is you or your strategy; everyone else is a bot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import families as F
from .auction import payouts

AUCTION_WEIGHTS = (1, 1, 2)


@dataclass
class BotConfig:
    """Crude bots. Each bot buys n samples (uniform over n_choices) and gets an estimate of p
    p_hat = p * exp(N(0, est_noise / sqrt(n))). Every auction it bids
    p_hat * U(lo_frac, hi_frac) + add_b (rounded, >= 0). Its final guess is p_hat.
    """

    lo_frac: float = 0.0
    hi_frac: float = 0.5
    add_b: float = 0.0
    est_noise: float = 0.35
    n_choices: tuple = (1, 2, 3, 4)
    mix: tuple = ()
    """Optional mixed room: ((weight, lo_frac, hi_frac), ...). Each bot picks one archetype per
    round and keeps it for all three auctions. Overrides lo_frac/hi_frac when set."""


@dataclass
class GameConfig:
    truth_prior: F.PriorConfig = field(default_factory=F.PriorConfig)
    bots: BotConfig = field(default_factory=BotConfig)
    room_sizes: tuple = (6, 7)
    rank_from: str = "low"      # rank shown after an auction: "low" (1 = lowest bid) or "high"
    max_samples: int = 30


class Round:
    def __init__(self, cfg: GameConfig, seed):
        self.cfg = cfg
        ss = np.random.SeedSequence(seed)
        r_truth, r_bots, r_ties = (np.random.default_rng(s) for s in ss.spawn(3))
        self.rng_ties = r_ties
        self.fam, self.par = F.draw_truth(cfg.truth_prior, r_truth)
        self.q = r_truth.random()
        self.percentile, p = F.display_percentile(self.fam, self.par, self.q, cfg.truth_prior.twopoint_pct)
        self.p = max(p, 0.0)
        d = cfg.truth_prior.delta
        self._stream = np.round(F.scalar_sample(self.fam, self.par, cfg.max_samples, r_truth) / d) * d
        self.room_size = int(r_bots.choice(cfg.room_sizes))
        nb = self.room_size - 1
        b = cfg.bots
        self.bot_n = r_bots.choice(b.n_choices, nb)
        self.bot_est = self.p * np.exp(r_bots.normal(0, 1, nb) * b.est_noise / np.sqrt(self.bot_n))
        if b.mix:
            w = np.array([m[0] for m in b.mix], float)
            arch = r_bots.choice(len(b.mix), nb, p=w / w.sum())
            lo = np.array([b.mix[i][1] for i in arch])
            hi = np.array([b.mix[i][2] for i in arch])
        else:
            lo, hi = np.full(nb, b.lo_frac), np.full(nb, b.hi_frac)
        self.bot_bids = np.maximum(np.round(
            self.bot_est[None, :] * r_bots.uniform(lo, hi, (3, nb)) + b.add_b), 0)
        self.hero_samples = []
        self.auctions = []  # list of dicts

    # ---------------- info stage
    def buy(self, k):
        n0 = len(self.hero_samples)
        new = list(self._stream[n0:n0 + k])
        self.hero_samples += new
        return new

    # ---------------- auctions
    def auction(self, hero_bid):
        i = len(self.auctions)
        bids = np.concatenate([[max(int(hero_bid), 0)], self.bot_bids[i]])
        pay, ranks = payouts(bids, self.p, self.rng_ties)
        shown = ranks if self.cfg.rank_from == "low" else len(bids) + 1 - ranks
        res = {"bids": bids, "pay": pay, "ranks": shown,
               "hero_pay": float(pay[0]), "hero_rank": int(shown[0])}
        self.auctions.append(res)
        return res

    # ---------------- final
    def finish(self, hero_guess):
        p = self.p
        guesses = np.concatenate([[hero_guess], self.bot_est])
        g_pts = np.maximum(p / 4 - np.abs(guesses - p), 0)
        ns = np.concatenate([[len(self.hero_samples)], self.bot_n])
        auc = sum(w * a["pay"] for w, a in zip(AUCTION_WEIGHTS, self.auctions))
        total = auc + g_pts - p * np.sqrt(ns) / 12
        # rank 1 = best; ties share the mean rank
        order = (-total).argsort(kind="stable")
        ranks = np.empty(len(total))
        ranks[order] = np.arange(1, len(total) + 1)
        for v in np.unique(total):
            m = total == v
            ranks[m] = ranks[m].mean()
        mult = 0.875 if self.room_size == 7 else 1.0
        return {
            "p": p, "family": self.fam, "truth": F.describe(self.fam, self.par),
            "percentile": self.percentile, "room_size": self.room_size,
            "n": int(ns[0]), "auction_pts": [float(a["pay"][0]) for a in self.auctions],
            "guess": float(hero_guess), "guess_pts": float(g_pts[0]),
            "sample_cost": float(p * np.sqrt(ns[0]) / 12),
            "total": float(total[0]), "total_frac": float(total[0] / p) if p > 0 else 0.0,
            "rank": float(ranks[0]), "placement": float(ranks[0] * mult),
            "all_totals": total, "all_ranks": ranks,
        }


def play_round(strategy, cfg: GameConfig, seed):
    """Run one full round with `strategy` as the hero. Returns the finish() dict."""
    rnd = Round(cfg, seed)
    strategy.start_round(rnd.percentile, rnd.room_size, np.random.default_rng([seed, 7]))
    if getattr(strategy, "sequential", False):
        n0 = strategy.initial_n()
        strategy.observe_samples(rnd.buy(n0))
        while len(rnd.hero_samples) < cfg.max_samples and strategy.want_more():
            strategy.observe_samples(rnd.buy(1))
    else:
        n = min(strategy.choose_n(), cfg.max_samples)
        strategy.observe_samples(rnd.buy(n))
    for i in range(3):
        b = strategy.bid(i)
        res = rnd.auction(b)
        strategy.observe_auction(i, b, res["hero_rank"], res["hero_pay"])
    return rnd.finish(strategy.guess())
