"""The chip model from the packet (pp. 10-12), as likelihoods.

Every chip has a true value T uniform on 1..10, independent of every other chip.
A partial reveal of a chip is one of four types:

    True        p=0.50   shows T
    Error       p=0.25   shows a value within 2 of T (uniform over the qualifying values in 1..10)
    False       p=0.15   shows a value independent of T (uniform 1..10)
    Combination p=0.10   shows round(mean(T_a, T_b)) of two *other* chips a, b, and names a and b

A non-combination reveal of value d gives a likelihood vector over T: P(shows d | T=t).
A combination reveal says nothing about the chip itself; it is a constraint on the pair (T_a, T_b).

The packet leaves three details open; each is a field of `Rules`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALUES = np.arange(1, 11)          # possible true values
N_CHIPS = 30
N_PUBLIC = 20                      # chips 1..20 are public, 21..30 private
PUBLIC_IDS = range(1, N_PUBLIC + 1)
PRIVATE_IDS = range(N_PUBLIC + 1, N_CHIPS + 1)

P_TRUE, P_ERROR, P_FALSE, P_COMBO = 0.50, 0.25, 0.15, 0.10

CONTRACTS = ("VAULT", "PUB", "G5")
TICK = {"VAULT": 0.25, "PUB": 0.25, "G5": 0.10}
BAND = {"VAULT": (30, 300), "PUB": (20, 200), "G5": (0, 30)}


@dataclass(frozen=True)
class Rules:
    # Can an Error chip show exactly its true value? "Within 2 ... in either direction" reads like an
    # offset of +-1 or +-2, so the default is no (qualifying values: 1 <= d <= 10, 1 <= |d - T| <= 2).
    error_can_equal: bool = False
    # Can a False chip show its true value by chance? The packet says both "incorrect" and
    # "independent of its true value"; those contradict. Default: independent (uniform 1..10).
    false_can_equal: bool = True
    # How a combination chip rounds a .5 mean: "half_up" (3.5 -> 4), "half_even" (Python round),
    # or "either" (accept floor or ceil -- the safe choice if you are unsure).
    combo_rounding: str = "half_up"


def _error_matrix(rules: Rules) -> np.ndarray:
    """E[d-1, t-1] = P(Error chip shows d | T=t)."""
    E = np.zeros((10, 10))
    for t in VALUES:
        q = [d for d in VALUES if abs(d - t) <= 2 and (rules.error_can_equal or d != t)]
        for d in q:
            E[d - 1, t - 1] = 1.0 / len(q)
    return E


def _false_matrix(rules: Rules) -> np.ndarray:
    if rules.false_can_equal:
        return np.full((10, 10), 0.1)
    F = np.full((10, 10), 1.0 / 9)
    np.fill_diagonal(F, 0.0)
    return F


def reveal_likelihood(rules: Rules) -> np.ndarray:
    """L[d-1, t-1] = P(partial reveal shows d and is not a combination chip | T=t)."""
    return P_TRUE * np.eye(10) + P_ERROR * _error_matrix(rules) + P_FALSE * _false_matrix(rules)


def type_posterior(d: int, rules: Rules) -> dict[str, float]:
    """P(type | shows d, not combination), with T integrated out under the uniform prior."""
    parts = {"true": P_TRUE * np.eye(10)[d - 1], "error": P_ERROR * _error_matrix(rules)[d - 1],
             "false": P_FALSE * _false_matrix(rules)[d - 1]}
    tot = {k: v.sum() for k, v in parts.items()}
    z = sum(tot.values())
    return {k: v / z for k, v in tot.items()}


def combo_round(s: np.ndarray, rules: Rules) -> np.ndarray:
    """Displayed value of a combination chip whose two referenced chips sum to s."""
    if rules.combo_rounding == "half_up":
        return (s + 1) // 2
    if rules.combo_rounding == "half_even":
        return np.round(s / 2).astype(int)
    raise ValueError("combo_round is not single-valued for rounding='either'")


def combo_matrix(d: int, rules: Rules) -> np.ndarray:
    """C[a-1, b-1] = 1 if a combination chip over chips with values (a, b) can show d."""
    s = VALUES[:, None] + VALUES[None, :]
    if rules.combo_rounding == "either":
        return ((s // 2 == d) | ((s + 1) // 2 == d)).astype(float)
    return (combo_round(s, rules) == d).astype(float)


def maker_wmax(known_true_ids) -> dict[str, float]:
    """Maximum unpenalised width for makers: the contract's s.d. given ONLY the true values makers know.

    The packet defines Wmax from "chip true value information available to the makers", so partial
    reveals are ignored: every chip whose true value is unknown is still uniform on 1..10
    (variance 8.25) and, for G5, a fair coin for T > 5 (variance 0.25).
    """
    known = set(known_true_ids)
    unknown_all = N_CHIPS - len(known)
    unknown_pub = N_PUBLIC - len(known & set(PUBLIC_IDS))
    return {"VAULT": float(np.sqrt(8.25 * unknown_all)),
            "PUB": float(np.sqrt(8.25 * unknown_pub)),
            "G5": float(np.sqrt(0.25 * unknown_all))}
