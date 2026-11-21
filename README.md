# WQC Fall 2026 — Scavenger toolkit

Tools for the **Scavenger** game (packet pp. 5–7): a Bayesian estimator for the pool size `p`,
a playable simulator of the game, and a Monte Carlo harness to compare strategies.

```bash
pip install -r requirements.txt
python -m scavenger.play --hints                 # play 5 rounds against bots, with the model's suggestions
python -m scavenger.play --sequential            # same, buying samples one at a time
python -m scavenger.make_table --trials 2000     # (re)build the batch "how many samples" table
python -m scavenger.montecarlo -s "bayes:n=1" "bayes:n=3" "bayes:n=table" "naive:n=3" -r 400
python -m scavenger.advisor                      # competition-day helper: type in what the game shows you
```

## Layout

| file | what it does |
|---|---|
| `scavenger/families.py` | The 5 distribution families, vectorised pdf / quantile / sampling, and **the prior** (`PriorConfig`). |
| `scavenger/posterior.py` | The weighted bag of scenarios (particles): fit to samples, turn percentile into `p`, Bayes-optimal final guess, reweighting. |
| `scavenger/auction.py` | Exact payout rule, crude opponent model, best-bid search, and learning about `p` from your auction rank/payout. |
| `scavenger/sampling.py` | Batch table (pick `n` up front) and sequential lookahead (buy one more?). |
| `scavenger/game.py` | Game engine: hidden setup, bots, 3 auctions, final guess, round points and ranking. |
| `scavenger/strategies.py` | Strategies as spec strings (`bayes:n=3,bid=model`, `naive:n=3`, ...). |
| `scavenger/montecarlo.py` | Runs strategies on identical rounds and reports mean placement, points/p, etc. |
| `scavenger/play.py` | Interactive terminal game. |
| `scavenger/advisor.py` | Live helper for the real game. |
| `tables/batch_table.*` | Output of `make_table`: best `n` per percentile bucket. |

## How the estimator works

**Scenarios.** Each particle is one full guess at the hidden setup — a family, its parameters —
plus a weight. Every family is written as a base scale `M` (log-uniform prior) times dimensionless
shapes (uniform priors), so the problem is scale-free and a table computed "as a fraction of p"
holds at any pool size.

**Samples → weights.** For the four continuous families the bag is built by adaptive importance
sampling: propose around the data, fit a Gaussian to the weighted particles, re-propose (3 passes),
so the particles sit where the posterior is even when it is narrow. Two-point is handled exactly:
samples show 1 distinct value (it was `L` or it was `H`), 2 values (`L`,`H` pinned), or 3+ (ruled out).
Samples are treated as **rounded to integers** (`delta=1`), which is what makes two identical samples
strong evidence for two-point. Each family's evidence `P(samples | family)` gives the family probabilities.

**Percentile → p.** The displayed percentile `k` is rounded, so the true quantile is in
`[k-0.5, k+0.5]%` (a 1-point-wide window). Each particle gets `p = F⁻¹(q)` with `q` uniform in that
window. The percentile itself carries no information about the parameters (it is uniform whatever
they are), so it does not reweight anything.

**Final guess.** Maximise `E[max(p/4 − |g−p|, 0)]` over `g` across the bag (grid + refine). This is
not the mean or median: it is the centre of the densest window of half-width ≈ p/4.

**Auction updates.** After each auction, for every particle simulate the opponents under the
opponent model and ask how likely your observed (rank, payout) was; reweight. A floor keeps any
particle from being wiped out, because the opponent model is crude.

### The prior is a guess — the biggest risk

The organisers did not publish parameter ranges. Defaults in `PriorConfig` (`families.py`):
`M ∈ [50, 5000]` log-uniform; normal `σ/μ ∈ [0.05, 0.4]`; uniform half-width `∈ [0.05, 0.9]·M`;
shifted-exp mean excess `∈ [0.05, 1]·M`; two-hump separation `±[0.1, 0.7]·M`, hump σ `∈ [0.03, 0.25]·M`;
two-point `L = M(1−[0.1,0.8])`, `H = M(1+[0.1,1.5])`, `P(L) ∈ [0.2, 0.8]`.
With 1 sample the prior on the *shape* is all you have to turn the percentile into `p`, so this matters
most when `n` is small. `GameConfig.truth_prior` and the strategy's `prior` are separate, so you can test
robustness by simulating truths from a different prior than the estimator assumes.

## a) How many samples: batch table

`python -m scavenger.make_table` simulates, for each `n = 0..10` on the same hidden setups,
the Bayes-optimal guess score as a fraction of `p`, split by displayed percentile, and subtracts `√n/12`.
`--mult X` values accuracy in the auctions at `X` times its value in the final guess (tune with the
harness). Current output is in `tables/batch_table.txt`; `bayes:n=table` uses it.

### Converting to one-at-a-time buying (if the organisers allow it)

The packet only says you "buy n samples" for `√n/12 · p`. **Ask the organisers whether you can buy one,
look, then buy more.** If you can, nothing in the estimator changes — only the stopping rule:

1. Buy a starting number `n0` (the batch table's answer for your percentile minus 1 or 2 is a sensible start; `n0=2` is the default).
2. After each sample, run `lookahead_decision` (already implemented in `sampling.py`): draw scenarios
   from the bag, simulate 1, 2 and 3 more samples from each, refit, and compute the expected guess score.
   Gain(k) = that − the current expected score, times `(1 + mult)`.
3. Cost(k) = `(√(n+k) − √n)/12 · E[p]` (e.g. the 4th sample costs 2.3% of `p`).
4. Buy one more if `Gain(k) > Cost(k)` for *any* k ≤ 3 (checking k > 1 avoids stopping just before the
   sample that would separate, say, normal from uniform). Otherwise stop.
5. Total cost depends only on the final `n`, so buying one at a time never costs more than buying the same
   `n` at once — the only cost is the 60 s clock (each lookahead takes a few seconds; `K` and
   `n_particles` trade speed for accuracy).

Try it: `python -m scavenger.play --sequential --hints`, or in the harness `bayes:seq=1,n0=2`.

## b) The game simulator

`game.py` follows the packet: percentile shown, samples bought at `√n/12·p`, three auctions with the exact
payout rule, final guess scored `max(p/4 − |g−p|, 0)`, round total `r1 + r2 + 2·r3 + g − p√n/12`,
rank in a room of 6–7 (×0.875 in rooms of 7), ties share the mean rank.

**Bots are crude placeholders.** Each bot buys 1–4 samples, gets an estimate `p̂ = p·exp(N(0, 0.35/√n))`,
bids `p̂ · U(lo, hi) + b` every auction (defaults `lo=0, hi=0.5, b=0`; change with `--bot-lo/--bot-hi/--bot-b`)
and guesses `p̂`. Results depend heavily on these settings: in the default room the bots underbid so the
top slot is lucrative, which may not hold against real players. Re-run the harness under several bot
settings before trusting a conclusion.

## c) Monte Carlo comparison

`python -m scavenger.montecarlo -s <spec> <spec> ... -r <rounds>` plays every strategy on the same rounds
(same setups, bots, sample stream) and reports **mean placement** (what the game ranks on), P(1st),
points/p split into auctions / guess / sample cost, and average `n`. Specs are documented at the top of
`strategies.py`. Note that the strategy's opponent model (`opp_lo`, `opp_hi`, `opp_noise`) is separate
from the bots (`--bot-*`), so you can test what a wrong belief about the room costs.

## First results (default prior, default bots — treat as provisional)

Full-game harness, 200 rounds, identical rounds for every strategy:

| strategy | placement | P(1st) | pts/p | auctions/p | guess/p | cost/p |
|---|---|---|---|---|---|---|
| bayes:n=0 | 2.93 | 50% | 0.32 | 0.20 | 0.12 | 0.00 |
| bayes:n=1 | 2.06 | 65% | 1.13 | 1.08 | 0.14 | 0.08 |
| bayes:n=2 | 1.97 | 68% | 1.30 | 1.25 | 0.17 | 0.12 |
| bayes:n=3 | 1.88 | 69% | 1.34 | 1.31 | 0.17 | 0.14 |
| bayes:n=5 | **1.81** | 74% | 1.41 | 1.41 | 0.19 | 0.19 |
| bayes:n=7 | 1.87 | 73% | 1.38 | 1.40 | 0.20 | 0.22 |
| bayes:n=10 | 1.88 | 73% | 1.36 | 1.43 | 0.20 | 0.26 |
| naive:n=3 | 3.46 | 12% | 0.42 | 0.48 | 0.08 | 0.14 |

Takeaways:
* **For the final guess alone, samples barely pay.** With `--mult 0` the batch table says buy 0–2: going
  from 1 to 10 samples lifts the guess score from ~9% to ~18% of p, and that costs 17% more of p.
* **Samples pay through the auctions.** Knowing p well is what lets you take the top slot without
  overshooting. Against these bots, n ≈ 3–7 is a flat optimum (n=5 is best but within noise), and
  the auctions are worth several times the guess. The default bots underbid (≤ 0.5·p̂), so the top slot
  is very lucrative. Against tougher opponents this will shrink, so re-test with other `--bot-*` settings.
* The shipped table uses `--mult 1` (best n ≈ 3–4 overall). The per-percentile rows are noisy
  (s.e. ≈ 0.5–1.2% of p), so trust the `all` row more than individual buckets until you rerun with more trials.
* Sequential lookahead (`bayes:seq=1,n0=2,mult=3`) over-bought (avg n 7.7) and was not better than a fixed
  n=5 in a 60-round test. Its `mult` needs tuning, and it is slow (~seconds per decision).
* Learning from auction results helps: `learn=0` lost about 0.2 placement in an 80-round test.

## Recommended strategy (current best guess)

1. **Buy 3 samples.** 3–5 were all close in every room tested, 1 is clearly worse, and fewer samples win ties.
   If you can buy one at a time: 2, look, usually a 3rd; 4–5 only if the samples look confusing.
2. **Final guess:** the model's guess (maximises expected guess points, not the mean or median).
3. **Auctions: learn the room, then pick an end.** Bottom three get their bid, the top bidder gets
   R − |R − b| (R = p minus the three lowest bids), and everyone in between gets 0.
   In a timid room, go for the top and bid about R. In an aggressive room, bid low and stay in the
   bottom three. Use auction 1 to learn; auction 3 counts double. If you were top and got less than your bid,
   R = (payout + bid)/2 exactly.
4. **On the day:** `python -m scavenger.advisor`. It runs `bayes:n=3,adapt=1`: it scores 10 candidate rooms
   (a 3×3 grid of opponents bidding p·U(lo, hi), plus a mixed room) by how well each explains your
   ranks and payouts, bids against the blend, and carries that across rounds.

Why adapt: with **fixed** beliefs about the room, being wrong is very costly (placement, lower is better):

| room \ model believes | timid (0–50%) | aggressive (30–120%) | adaptive, 3 rooms | adaptive, 10 rooms |
|---|---|---|---|---|
| timid 0–50% | 1.98 | 3.69 | 1.89 | 2.22 |
| mixed | 2.71 | 3.80 | 2.47 | 2.62 |
| aggressive 30–120% | 4.39 | 1.41 | 1.35 | 1.27 |
| in-between 20–80% | 5.50 | – | 3.99 | 1.79 |
| off-grid 10–60% | – | – | – | 2.93 |
| fair share 5–25% | – | – | – | 0.98 |

The 10-room version gives up ~0.15–0.3 in the rooms the 3-room version already modelled, but removes
the disaster in rooms between them. The weak spot left is the 10–60% room (2.93).
All numbers are against bots; real players will differ. Scripts: `results/run_adaptive*.sh`.

## Unresolved rules (the packet doesn't say)

Each one is a setting, separately for the simulated game (`--truth-*` flags on `montecarlo`) and for
what a strategy assumes (spec options), so you can measure what a wrong guess costs.
Scripts and full output: `results/run_*.sh`, `results/sensitivity*.txt` (300 rounds each, n=3, same rounds within a block).

| question | default assumption | switch | cost of guessing wrong (mean placement) |
|---|---|---|---|
| Does the post-auction rank count from the lowest or highest bid? | **inferred** (`rank=auto`) | `rank=low\|high\|auto`, `--truth-rank` | Reading it backwards: 1.97 → **2.37**, the biggest effect found. `auto` scores each reading by how well it explains your payout and keeps the better one, at almost no cost (1.98–1.99 vs 1.97). On game day you will see which way it counts after one auction anyway (bid 0: you are the lowest bid). |
| Percentile of a two-point pool (discrete, so F jumps)? | `random`: a uniform point inside the jump | `tp=random\|cdf\|below\|mid`, `--truth-tp` | If the game uses `cdf` (`P(X≤p)`: the high value always shows **100**), assuming `random` costs 2.02 vs 1.91. If it uses `below` (low value always shows **0**), assuming `random` costs ~nothing (1.95 vs 1.94). Wrongly assuming `below` costs 1.97 → 2.10, wrongly assuming `cdf` ~nothing → keep `random` unless you see evidence. **Tell:** a displayed percentile of exactly 100 or 0 with samples that look two-valued. |
| Are samples whole numbers? | yes (`delta=1`: samples treated as rounded to integers) | `delta=1\|1e-6`, `--truth-delta` | No measurable difference either way (1.97–1.99). `delta=1` is the safer choice: if the samples are real numbers, it only merges values less than 0.5 apart. If they are integers and you assume real-valued, a repeated value from a narrow continuous distribution would be misread as two-point. |
| Can the pool be negative? | no setup has more than 1% of its mass below 0 (`pos=0.01`) | `pos=0.01\|none`, `--truth-pos` | No measurable difference (1.97 vs 1.96). Under the default ranges, few setups have much mass below 0, and the simulator clamps p at 0. If a pool really were negative, every paid bidder would get min(b, R) < 0, so bidding 0 would be the only safe bid. Unlikely to be intended. |

Still open, and the most important: **can samples be bought one at a time?** (see "Converting to one-at-a-time buying").
Is the pool size itself an integer? (Barely matters: the guess is scored on a scale of p/4.)
