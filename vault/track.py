"""Live Vault tracker: type in each reveal as it appears, get the EV of VAULT, PUB and G5.

    python -m vault.track --role taker
    python -m vault.track --role maker

Commands (several per line, separated by ';'):
    p ID VAL         partial reveal of chip ID shows VAL                 e.g.  p 3 7
    c ID VAL A B     chip ID is a combination chip: round(mean(A, B)) = VAL   c 3 6 12 27
    t ID VAL         true value of chip ID is VAL (public or private)    e.g.  t 2 4
    chips            per-chip table (what you entered, E[value], P(>5), odds the partial is true)
    u                undo the last entry
    reset            start a new round
    q                quit

A taker's round, every 30 s:  t <previous chip> <value>; p <new chip> <shown>   and every minute
also  t <20+m> <value>  for the private chip. A maker types the same, 30 s later, with no private chips.
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from .model import BAND, CONTRACTS, N_CHIPS, TICK, Rules, maker_wmax
from .tracker import InconsistentEvidence, VaultTracker


def tick_round(x, c, how):
    k = x / TICK[c]
    k = math.ceil(k - 1e-9) if how == "up" else math.floor(k + 1e-9)
    lo, hi = BAND[c]
    return min(max(k * TICK[c], lo), hi)


def report(tr: VaultTracker, role: str) -> str:
    est = tr.estimate()
    lines = []
    if role == "maker":
        w = maker_wmax(tr.trues)
        lines.append(f"{'':6}{'EV':>9}{'sd':>7}{'5%-95%':>13}{'Wmax':>7}   widest unpenalised quote")
        for c in CONTRACTS:
            e = est[c]
            bid, ask = tick_round(e.mean - w[c] / 2, c, "up"), tick_round(e.mean + w[c] / 2, c, "down")
            lines.append(f"{c:6}{e.mean:9.2f}{e.sd:7.2f}{f'[{e.quantile(.05)}, {e.quantile(.95)}]':>13}"
                         f"{w[c]:7.2f}   {bid:g} @ {ask:g}")
    else:
        mk = tr.maker_view()
        me = mk.estimate()
        w = maker_wmax(mk.trues)
        lines.append(f"{'':6}{'EV':>9}{'sd':>7}{'5%-95%':>13}   {'makers EV':>9}{'edge':>8}{'maker Wmax':>12}")
        for c in CONTRACTS:
            e, m = est[c], me[c]
            lines.append(f"{c:6}{e.mean:9.2f}{e.sd:7.2f}{f'[{e.quantile(.05)}, {e.quantile(.95)}]':>13}"
                         f"   {m.mean:9.2f}{e.mean - m.mean:+8.2f}{w[c]:12.2f}")
    return "\n".join(lines)


def chips_table(tr: VaultTracker) -> str:
    est = tr.estimate()
    out = [" id  entered                 E[value]  P(>5)   P(partial is true/error/false)"]
    for i in range(1, N_CHIPS + 1):
        p, t = tr.partials.get(i), tr.trues.get(i)
        what = []
        if p is not None:
            what.append(f"shows {p[0]}" if len(p) == 1 else f"combo {p[0]}=avg({p[1]},{p[2]})")
        if t is not None:
            what.append(f"true {t}")
        odds = tr.chip_type_odds(i)
        o = "" if odds is None or t is not None else \
            f"{odds['true']:.2f} / {odds['error']:.2f} / {odds['false']:.2f}"
        out.append(f"{i:3d}  {', '.join(what) or '-':22}{est.chip_means[i - 1]:8.2f}"
                   f"{est.chip_g5[i - 1]:7.2f}   {o}")
    return "\n".join(out)


def run_command(tr: VaultTracker, cmd: str) -> str | None:
    parts = cmd.split()
    if not parts:
        return None
    op, args = parts[0].lower(), [int(x) for x in parts[1:]]
    if op == "p" and len(args) == 2:
        tr.partial(*args)
    elif op == "c" and len(args) == 4:
        tr.combo(*args)
    elif op == "t" and len(args) == 2:
        tr.true(*args)
    elif op in ("u", "undo") and not args:
        if not tr.undo():
            return "nothing to undo"
    else:
        raise ValueError(f"don't understand '{cmd}' (p ID VAL | c ID VAL A B | t ID VAL | u | chips)")
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["taker", "maker"], required=True)
    ap.add_argument("--error-can-equal", action="store_true",
                    help="an Error chip may show its true value (default: it is off by 1 or 2)")
    ap.add_argument("--false-never-equal", action="store_true",
                    help="a False chip never shows its true value (default: independent, uniform 1..10)")
    ap.add_argument("--rounding", default="half_up", choices=["half_up", "half_even", "either"],
                    help="how a combination chip rounds x.5 (default half_up; 'either' if unsure)")
    args = ap.parse_args()
    rules = Rules(error_can_equal=args.error_can_equal, false_can_equal=not args.false_never_equal,
                  combo_rounding=args.rounding)

    tr = VaultTracker(rules)
    print(__doc__.split("Commands")[1].split("A taker")[0].rstrip())
    print()
    print(report(tr, args.role))
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if line in ("q", "quit", "exit"):
            break
        if line == "reset":
            tr = VaultTracker(rules)
        elif line == "chips":
            print(chips_table(tr))
            continue
        else:
            n_before = len(tr.log)
            try:
                for cmd in line.split(";"):
                    msg = run_command(tr, cmd)
                    if msg:
                        print(msg)
                tr.estimate()
            except InconsistentEvidence as e:
                while len(tr.log) > n_before:
                    tr.undo()
                print(f"!! {e} -- line not applied. Typo? If a combination chip is involved, "
                      "the rounding rule may differ: restart with --rounding either.")
                continue
            except ValueError as e:
                while len(tr.log) > n_before:
                    tr.undo()
                print(f"!! {e} -- line not applied")
                continue
        print(report(tr, args.role))


if __name__ == "__main__":
    main()
