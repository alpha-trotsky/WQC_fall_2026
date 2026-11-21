"""Auction mechanics, a crude opponent model, best-bid search, and learning from results.

Payout rule (packet, page 5): sort bids ascending.
  lowest three: w_i = min(b_i, R) where R is what is left of the pool, in order.
  highest:      R - |R - b|   (R after the lowest three)  ==  b if b <= R, else 2R - b.
  everyone else: 0.

ASSUMPTIONS (not stated in the packet, ask the organisers):
  * ties in bids are ordered at random;
  * the "rank" shown after an auction counts from the lowest bid (rank 1 = lowest).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def payouts(bids, p, rng=None):
    """Exact payouts for one auction. bids: list/array of length m. Returns (payouts, ranks).

    ranks are 1-based, ascending (1 = lowest bid).
    """
    bids = np.asarray(bids, float)
    m = len(bids)
    rng = rng if rng is not None else np.random.default_rng()
    order = np.lexsort((rng.random(m), bids))  # ascending, random tie-break
    out = np.zeros(m)
    R = float(p)
    n_low = min(3, m - 1) if m >= 2 else 1
    for pos in range(n_low):
        i = order[pos]
        w = min(bids[i], R)
        out[i] = w
        R -= w
    if m >= 2:
        top = order[-1]
        out[top] = R - abs(R - bids[top])
    ranks = np.empty(m, int)
    ranks[order] = np.arange(1, m + 1)
    return out, ranks


@dataclass
class OpponentModel:
    """Each opponent bids  p * exp(N(0, est_noise)) * U(lo_frac, hi_frac) + add_b.

    This is what OUR strategy believes about the room. The simulator's bots have their
    own (possibly different) settings, so you can test what happens when this is wrong.
    """

    lo_frac: float = 0.0
    hi_frac: float = 0.4
    est_noise: float = 0.25
    add_b: float = 0.0
    mix: tuple = ()  # optional ((weight, lo, hi), ...): each opponent picks an archetype

    def draw(self, p, n_opp, rng):
        """p: array [S] -> opponent bids [S, n_opp]."""
        p = np.asarray(p, float)[:, None]
        est = p * np.exp(rng.normal(0, self.est_noise, (len(p), n_opp)))
        if self.mix:
            w = np.array([m[0] for m in self.mix], float)
            arch = rng.choice(len(self.mix), est.shape, p=w / w.sum())
            lo = np.array([m[1] for m in self.mix])[arch]
            hi = np.array([m[2] for m in self.mix])[arch]
        else:
            lo, hi = self.lo_frac, self.hi_frac
        return np.maximum(np.round(est * rng.uniform(lo, hi, est.shape) + self.add_b), 0)


# Candidate rooms for the adaptive strategy: each opponent bids p * U(lo, hi) (fractions of p).
# A 3x3 grid from timid to aggressive, plus a mixed room of low bidders and a few top-chasers.
ROOM_MODELS = {f"U({lo:g}-{hi:g})": OpponentModel(lo, hi)
               for lo in (0.0, 0.15, 0.3) for hi in (0.4, 0.8, 1.2)}
ROOM_MODELS["mixed"] = OpponentModel(mix=((0.4, 0.0, 0.15), (0.4, 0.1, 0.3), (0.2, 0.6, 1.0)))


class OpponentMixture:
    """Several candidate rooms with weights: each simulated auction draws its room from them."""

    def __init__(self, models, weights):
        self.models = list(models)
        w = np.asarray(weights, float)
        self.weights = w / w.sum()

    def draw(self, p, n_opp, rng):
        p = np.asarray(p, float)
        which = rng.choice(len(self.models), len(p), p=self.weights)
        out = np.empty((len(p), n_opp))
        for i, m in enumerate(self.models):
            sel = which == i
            if sel.any():
                out[sel] = m.draw(p[sel], n_opp, rng)
        return out


def _our_payout(b, p, opp_sorted, cum_rem):
    """Vectorised payout and rank for our bid b against sorted opponent bids.

    b: [G, 1] or scalar; p: [S]; opp_sorted: [S, k]; cum_rem: [S, 4] = remaining pool after
    0..3 of the lowest opponents are paid. Returns payout [G, S], rank [G, S] (1-based).
    """
    k = opp_sorted.shape[1]
    b = np.atleast_2d(np.asarray(b, float)).reshape(-1, 1, 1)  # [G,1,1]
    below = (opp_sorted[None, :, :] < b).sum(axis=2)            # [G,S]
    b2 = b[:, :, 0]                                             # [G,1]
    pos = np.minimum(below, 3)
    rem = np.take_along_axis(cum_rem[None, :, :].repeat(len(b2), 0), pos[:, :, None], axis=2)[:, :, 0]
    low_pay = np.minimum(b2, rem)
    rem3 = cum_rem[None, :, 3]
    top_pay = rem3 - np.abs(rem3 - b2)
    pay = np.where(below <= 2, low_pay, np.where(below == k, top_pay, 0.0))
    return pay, below + 1


def _prep(p, opp):
    opp_sorted = np.sort(opp, axis=1)
    k = opp_sorted.shape[1]
    cs = np.zeros((len(p), 4))
    for j in range(1, 4):
        if j <= k:
            cs[:, j] = opp_sorted[:, j - 1]
    cum_rem = np.maximum(p[:, None] - np.cumsum(cs, axis=1), 0)
    return opp_sorted, cum_rem


def best_bid(post, room_size, model: OpponentModel, rng, n_sims=4000, n_grid=120,
             risk_aversion=0.0):
    """Pick the bid that maximises expected payout across scenarios and opponent draws.

    Objective = mean(payout) - risk_aversion * std(payout). Returns (bid, table) where
    table is (candidate bids, expected payout) for inspection.
    """
    p = post.resample_p(n_sims, rng)
    opp = model.draw(p, room_size - 1, rng)
    opp_sorted, cum_rem = _prep(p, opp)
    hi = np.quantile(p, 0.99) * 1.1
    cand = np.unique(np.round(np.concatenate([np.linspace(0, hi, n_grid),
                                              np.quantile(opp, np.linspace(0, 1, 40))])))
    pay, _ = _our_payout(cand, p, opp_sorted, cum_rem)
    obj = pay.mean(axis=1) - risk_aversion * pay.std(axis=1)
    i = int(np.argmax(obj))
    return int(cand[i]), (cand, pay.mean(axis=1))


def auction_loglik(post, room_size, model: OpponentModel, our_bid, rank, payout, rng,
                   n_sims=48, floor=0.02):
    """log P(observed rank & payout | each particle), via simulation of the opponents.

    Payout is matched with a Gaussian kernel (bandwidth ~3% of the bid, at least 1 point).
    `floor` keeps any particle from being killed outright, since the opponent model is crude.
    """
    p_all = post.p
    S = n_sims
    out = np.empty(len(p_all))
    h = max(1.0, 0.03 * max(our_bid, 1))
    chunk = 4000
    for s in range(0, len(p_all), chunk):
        p = np.repeat(p_all[s:s + chunk], S)
        opp = model.draw(p, room_size - 1, rng)
        opp_sorted, cum_rem = _prep(p, opp)
        pay, rk = _our_payout(our_bid, p, opp_sorted, cum_rem)
        match = (rk[0] == rank) * np.exp(-0.5 * ((pay[0] - payout) / h) ** 2)
        lik = match.reshape(-1, S).mean(axis=1)
        out[s:s + chunk] = lik
    out = (1 - floor) * out + floor * max(out.max(), 1e-300)
    return np.log(out)
