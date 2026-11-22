"""Simulate Vault rounds under the packet's rules, and check the tracker is calibrated.

    python -m vault.sim --demo                 # one round: EV path for taker and maker vs the settlement
    python -m vault.sim --rounds 300           # calibration: mean z and sd of z per interval (should be ~0, ~1)

Schedule (packet p. 12), interval n = 1..21 at time 30(n-1) s:
    takers: partial reveal of chip n (n <= 20) and true value of chip n-1 (n >= 2)
    makers: what takers got at interval n-1 (so nothing at 0:00, never chip 20's true value)
    takers only: true value of private chip 20+m at time 60m s, m = 1..10
"""
from __future__ import annotations

import argparse

import numpy as np

from .model import N_CHIPS, N_PUBLIC, P_COMBO, P_ERROR, P_FALSE, P_TRUE, Rules, combo_round
from .tracker import VaultTracker


def deal(rng, rules: Rules = Rules()):
    """Hidden state of one round: true values and the partial reveal of each public chip."""
    T = rng.integers(1, 11, N_CHIPS)
    reveals = {}
    for i in range(1, N_PUBLIC + 1):
        t = T[i - 1]
        kind = rng.choice(4, p=[P_TRUE, P_ERROR, P_FALSE, P_COMBO])
        if kind == 0:
            reveals[i] = ("p", int(t))
        elif kind == 1:
            q = [d for d in range(1, 11) if abs(d - t) <= 2 and (rules.error_can_equal or d != t)]
            reveals[i] = ("p", int(rng.choice(q)))
        elif kind == 2:
            q = [d for d in range(1, 11) if rules.false_can_equal or d != t]
            reveals[i] = ("p", int(rng.choice(q)))
        else:
            a, b = rng.choice([j for j in range(1, N_CHIPS + 1) if j != i], 2, replace=False)
            r = "half_up" if rules.combo_rounding == "either" else rules.combo_rounding
            d = int(combo_round(np.array(T[a - 1] + T[b - 1]), Rules(combo_rounding=r)))
            reveals[i] = ("c", d, int(a), int(b))
    return T, reveals


def schedule(T, reveals, role: str):
    """List of (time_s, event) for one role. event = ('p', id, d) | ('c', id, d, a, b) | ('t', id, v)."""
    ev = []
    lag = 0 if role == "taker" else 1
    for n in range(1, 22):
        src = n - lag                   # the taker interval whose information arrives now
        if src < 1:
            continue
        t = 30 * (n - 1)
        if src <= N_PUBLIC:
            r = reveals[src]
            ev.append((t, (r[0], src) + r[1:]))
        if 2 <= src <= N_PUBLIC + 1:
            ev.append((t, ("t", src - 1, int(T[src - 2]))))
    if role == "taker":
        for m in range(1, 11):
            ev.append((60 * m, ("t", N_PUBLIC + m, int(T[N_PUBLIC + m - 1]))))
    ev.sort(key=lambda e: e[0])
    return ev


def apply(tr: VaultTracker, e):
    if e[0] == "p":
        tr.partial(e[1], e[2])
    elif e[0] == "c":
        tr.combo(e[1], e[2], e[3], e[4])
    else:
        tr.true(e[1], e[2])


def settlement(T):
    return {"VAULT": int(T.sum()), "PUB": int(T[:N_PUBLIC].sum()), "G5": int((T > 5).sum())}


def run_round(rng, rules, role):
    """Estimate after each 30 s interval: list of (time, {contract: (mean, sd)})."""
    T, rev = deal(rng, rules)
    ev = schedule(T, rev, role)
    tr = VaultTracker(rules)
    out = []
    for t in range(0, 601, 30):
        for e in [e for tt, e in ev if tt == t]:
            apply(tr, e)
        est = tr.estimate()
        out.append((t, {c: (est[c].mean, est[c].sd) for c in est.contracts}))
    return settlement(T), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    rules = Rules()

    if args.demo:
        rng_t, rng_m = np.random.default_rng(args.seed), np.random.default_rng(args.seed)
        settle, tk = run_round(rng_t, rules, "taker")
        _, mk = run_round(rng_m, rules, "maker")
        print("settles at", settle)
        print(" time   taker VAULT        maker VAULT       taker PUB   maker PUB   taker G5  maker G5")
        for (t, a), (_, b) in zip(tk, mk):
            print(f"{t // 60:2d}:{t % 60:02d}  {a['VAULT'][0]:6.1f} ±{a['VAULT'][1]:4.1f}"
                  f"     {b['VAULT'][0]:6.1f} ±{b['VAULT'][1]:4.1f}"
                  f"     {a['PUB'][0]:6.1f}     {b['PUB'][0]:6.1f}"
                  f"      {a['G5'][0]:5.2f}     {b['G5'][0]:5.2f}")
        return

    for role in ("taker", "maker"):
        rng = np.random.default_rng(args.seed)
        z = {c: [] for c in ("VAULT", "PUB", "G5")}
        for _ in range(args.rounds):
            settle, path = run_round(rng, rules, role)
            for c in z:
                z[c].append([(settle[c] - m) / s if s > 0 else 0.0 for _, est in path
                             for m, s in [est[c]]])
        print(f"\n{role}: z = (settlement - EV) / sd per interval, over {args.rounds} rounds "
              "(calibrated: mean ~0, sd ~1)")
        print(" time " + "".join(f"{c:>16}" for c in z))
        for k, t in enumerate(range(0, 601, 30)):
            if t % 120 and t != 600:
                continue
            cells = []
            for c in z:
                col = np.array(z[c])[:, k]
                cells.append(f"{col.mean():+6.2f} / {col.std():4.2f}")
            print(f"{t // 60:2d}:{t % 60:02d} " + "".join(f"{x:>16}" for x in cells))


if __name__ == "__main__":
    main()
