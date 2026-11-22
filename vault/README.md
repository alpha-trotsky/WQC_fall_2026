# Vault — live EV tracker (packet pp. 10–13)

Tracks the expected settlement of the three tradeables as chips are revealed, for either role.

```bash
python -m vault.track --role taker      # or --role maker
python -m vault.sim --demo              # one simulated round: taker vs maker EV path vs settlement
python -m vault.sim --rounds 300        # calibration check (~1 min)
```

Type each reveal as it shows up (several per line with `;`):

| command | meaning |
|---|---|
| `p 3 7` | chip 3 partial reveal shows 7 |
| `c 3 6 12 27` | chip 3 is a combination chip: round(mean(chip 12, chip 27)) = 6 |
| `t 2 4` | chip 2's true value is 4 (public or private) |
| `chips` / `u` / `reset` / `q` | per-chip table / undo / new round / quit |

A taker's 30-second tick is typically `t 4 7; p 5 3` (plus `t 2m 9` for the private chip each minute).
A maker types the same thing 30 s later and never gets private chips.

**Output.** For each of VAULT, PUB, G5: EV, s.d., 5–95% range.
* **Maker** also sees `Wmax` (max unpenalised width) and the widest quote centred on EV that avoids the penalty.
* **Taker** also sees the **makers' EV**. A taker knows everything a maker knows (makers get the takers'
  information 30 s late, with no private chips), so the tool computes the makers' view exactly
  from your own entries, plus `edge = your EV − makers' EV` and the makers' `Wmax`.
  Makers should be quoting around their EV, so `edge` tells you which side to hit.

## How it works

**Prior.** 30 independent chips, each uniform on 1..10: VAULT starts at 165 ± 15.7, PUB at 110 ± 12.8, G5 at 15 ± 2.7.

**Each reveal is a likelihood** multiplied into the prior (Bayes' rule):

* **True value** `v` pins the chip: `P(T = v) = 1`.
* **Partial reveal showing `d`** (not combination): the chip type is hidden, so
  `P(shows d | T = t) = 0.5·[d = t] + 0.25·Error(d | t) + 0.15·False(d | t)`, where Error is uniform on the
  values within 2 of `t` (clipped to 1..10) and False is uniform on 1..10.
  Showing 6, for example, gives P(true chip) ≈ 0.56 and E[T] ≈ 5.92. In the middle, a partial reveal of `d`
  moves the chip ~85% of the way from 5.5 to `d`; at the ends the pull is weaker (shows 1 → 2.11, shows 10 → 8.89):

  | shown | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | E[T] | 2.11 | 2.65 | 3.23 | 4.20 | 5.08 | 5.92 | 6.80 | 7.77 | 8.35 | 8.89 |
* **Combination reveal** of chip `c` over `(a, b)` showing `d`: says **nothing about chip `c`** itself
  (its own value is still uniform until its true value comes). It is a hard constraint
  `round((T_a + T_b)/2) = d` linking chips `a` and `b`. Those can be private chips (21–30), the only way
  a maker learns anything about them, or future public chips, or chips already known. Once one of the two
  is known the other is pinned to 2 values (e.g. `T_a = 7, d = 5` → `T_b ∈ {2, 3}`).

**EV.** By linearity `E[VAULT] = Σ E[T_i]`, `E[PUB] = Σ_{i≤20} E[T_i]`, `E[G5] = Σ P(T_i > 5)`.
Combination chips make chips dependent, which matters for the s.d., so the tool computes the **exact
distribution** of each contract. It adds chips one at a time and keeps a table over (chips still linked
to chips not yet added) × (running contract value). Unlinked chips are summed out at once, so the table stays tiny. A Gibbs
sampler takes over if a strange combination graph links more than 4 chips at once (checked against exact: 166.0 vs 165.98).
One update takes a few ms.

**Maker Wmax** follows the packet: the contract's s.d. using **only true values makers know**
(partials ignored): `√(8.25 · #unknown)` for VAULT and PUB (over public chips), `√(0.25 · #unknown)` for G5.
It is *not* the posterior s.d.: partials make the posterior s.d. a little smaller.

**Who knows what** (interval `n` at `30(n−1)` s): takers get chip `n`'s partial reveal and chip `n−1`'s true value.
They also get private chip `20+m`'s true value at minute `m`. Makers get the takers' interval `n−1` (nothing at 0:00, and
never chip 20's true value). At 10:00 the taker knows the exact settlement. The maker still has
VAULT ± ~9 (10 private chips + chip 20).

## Checked

`python -m vault.sim --rounds 300` simulates rounds from the packet's rules. It feeds each role its own
information schedule and compares settlement to EV: z = (settle − EV)/sd has mean ≈ 0 and sd ≈ 1 for every
contract, at every time, for both roles. So the EVs are unbiased and the s.d. is right. Per-chip posteriors with
combinations were also checked against brute-force enumeration.

## Rules the packet leaves open (flags)

| question | default | flag |
|---|---|---|
| Can an Error chip show its exact true value? ("within 2 … in either direction") | no: off by 1 or 2 | `--error-can-equal` |
| False chip: "incorrect" or "independent"? (they contradict) | independent, uniform 1..10 | `--false-never-equal` |
| Combination rounding of x.5 | half up (3.5 → 4) | `--rounding half_even` / `either` |

The first two move any one chip's EV by at most ~0.12 (worst at shown 1 or 10). For rounding: if your entries become impossible
(the tool refuses a line with "no assignment fits"), check for a typo first, then restart with `--rounding either`.
Also assumed: the two chips a combination refers to can be any other chips, private ones included.
