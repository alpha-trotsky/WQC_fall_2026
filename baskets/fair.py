"""Baskets live fair-value tracker.

Type the first letter of the basket an event hit plus how much that basket moved,
and the fair values of all three baskets are recomputed and redrawn.

    g 4     GRID event, GRID went up $4       -> CURE, SAFE each +$2
    c -3    CURE event, CURE went down $3     -> GRID, SAFE each -$1.50
    s2.5    SAFE event (space optional)
    u       undo last input
    r       reset everything to 400
    q       quit

Why "others move half": from the packet's settlement formulas, an event of Q shares in
  GRID moves GRID Q/50 and CURE, SAFE Q/100 each;
  CURE moves CURE Q/100 and GRID, SAFE Q/200 each;  (same for SAFE)
so whichever basket an event hits, the other two should move half as much.
"""
import os
import re
import sys

BASKETS = {"g": "GRID", "c": "CURE", "s": "SAFE"}
# shares of event volume per $1 move of the event basket itself
VOL_PER_DOLLAR = {"g": 50.0, "c": 100.0, "s": 100.0}


def fair_values(events):
    """events: list of (letter, move). Returns (indices, baskets) dicts."""
    q = {"g": 0.0, "c": 0.0, "s": 0.0}
    for letter, move in events:
        q[letter] += move * VOL_PER_DOLLAR[letter]
    tek = 200 + (2 * q["g"] + q["c"]) / 200
    nrg = 200 + (2 * q["g"] + q["s"]) / 200
    med = 200 + (q["c"] + q["s"]) / 200
    idx = {"TEK": tek, "NRG": nrg, "MED": med}
    bsk = {"GRID": tek + nrg, "CURE": tek + med, "SAFE": nrg + med}
    return idx, bsk


def implied(letter, move):
    """Implied move of every basket from one event."""
    return {name: (move if l == letter else move / 2) for l, name in BASKETS.items()}


def render(events, msg):
    os.system("cls" if os.name == "nt" else "clear")
    idx, bsk = fair_values(events)
    print("=" * 44)
    print("  BASKETS FAIR VALUE")
    print("=" * 44)
    for name, v in bsk.items():
        print(f"  {name:5s} {v:9.2f}   ({v - 400:+.2f})")
    print("-" * 44)
    print("  " + "   ".join(f"{k} {v:.2f}" for k, v in idx.items()))
    print("-" * 44)
    if events:
        letter, move = events[-1]
        imp = implied(letter, move)
        print(f"  LAST: {BASKETS[letter]} {move:+.2f}  ->  " +
              "  ".join(f"{n} {d:+.2f}" for n, d in imp.items() if n != BASKETS[letter]))
        print("  history: " + ", ".join(f"{l.upper()}{m:+g}" for l, m in events[-8:]))
    else:
        print("  no events yet")
    if msg:
        print(f"  ! {msg}")
    print("-" * 44)
    print("  g/c/s <move>   u=undo  r=reset  q=quit")


def main():
    events, msg = [], ""
    pat = re.compile(r"^\s*([gcs])\s*([+-]?\d*\.?\d+)\s*$", re.I)
    while True:
        render(events, msg)
        msg = ""
        try:
            line = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line in ("q", "quit", "exit"):
            return
        if line == "u":
            if events:
                events.pop()
            continue
        if line == "r":
            events.clear()
            continue
        m = pat.match(line)
        if not m:
            msg = f"didn't understand '{line}' (try 'g 4' or 'c -2.5')"
            continue
        events.append((m.group(1), float(m.group(2))))


if __name__ == "__main__":
    sys.exit(main())
