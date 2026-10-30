"""Audit of the characterization failures quoted in the supplementary
material (Characterization Failures).

1. Every synthesis log left in the approxlg_syn_* work directories: how many
   carry ABC's "non-driven nets" warning (a torn input netlist), and the
   failed designs (no mapped netlist) by cause and width.
2. The designs whose emitted adder carries a zero-width constant (truncation
   and OR-based adder leave only the top bit exact): their NMED, and how often
   the shared sampler draws one.

    python scripts/audit_synth_failures.py [--tmp DIR] [--skip-logs]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from approxlg.arch import Builder, Space                       # noqa: E402
from approxlg.simulate import error_metrics, make_stimulus, simulate  # noqa: E402

ZERO_WIDTH = re.compile(r"(?<![0-9])0'[bdh]")
WIDTH = re.compile(r"input \[(\d+):0\] a")


def audit_logs(base: str) -> None:
    recent = time.time() - 20 * 60          # skip runs still being written
    n_logs = n_torn = 0
    total = Counter()
    failed = defaultdict(Counter)
    seen = set()
    with os.scandir(base) as it:
        work = [e.path for e in it
                if e.is_dir() and e.name.startswith("approxlg_syn_")]
    for w in work:
        with os.scandir(w) as it:
            for e in it:
                try:
                    if os.stat(e.path).st_mtime > recent:
                        continue
                    head = open(os.path.join(e.path, "src.v"),
                                errors="replace").read(300)
                except OSError:
                    continue
                m = WIDTH.search(head)
                n = int(m.group(1)) + 1 if m else 0
                total[n] += 1
                try:
                    log = open(os.path.join(e.path, "yosys.log"),
                               errors="replace").read()
                    n_logs += 1
                except OSError:
                    log = ""
                if "non-driven nets" in log:
                    n_torn += 1
                if os.path.exists(os.path.join(e.path, "map.json")) \
                        or e.name in seen:
                    continue
                seen.add(e.name)
                last = log.strip().splitlines()[-1] if log.strip() else ""
                if "Illegal integer constant size of zero" in log:
                    k = "zero-width constant"
                elif "non-driven nets" in log:
                    k = "torn ABC input"
                elif last.endswith("input.blif'.."):
                    k = "stopped inside ABC (time limit)"
                elif not log:
                    k = "no log (tool did not start)"
                else:
                    k = "other"
                failed[n][k] += 1
    print(f"{n_logs} synthesis logs in {len(work)} work directories; "
          f"{n_torn} carry ABC's 'non-driven nets' warning")
    for n in sorted(total):
        if n:
            print(f"  {n:2d}x{n:<2d}: {total[n]} runs, failed designs "
                  f"{dict(failed[n])}")


def audit_zero_width(samples: int = 300, draws: int = 20000) -> None:
    rng = np.random.default_rng(3)
    for n, signed in ((8, False), (16, False), (16, True), (32, False)):
        sp = Space(n, signed=signed)
        b = Builder(sp)
        stim = make_stimulus(n, 1 << 16, gpu=False, seed=0, signed=signed)
        nmed = []
        for _ in range(samples):
            g = sp.random(rng)
            t = int(rng.integers(0, sp.max_t + 1))
            g[0], g[1], g[3] = t, sp.ncol - t, sp.ncol - t
            nl = b.build(sp.canonical(g))
            if ZERO_WIDTH.search(nl.to_verilog()):
                nmed.append(error_metrics(simulate(nl, stim), stim)["NMED"])
        hits = sum(bool(ZERO_WIDTH.search(b.build(sp.random(rng)).to_verilog()))
                   for _ in range(draws))
        print(f"  {n}x{n} {'signed' if signed else 'unsigned'}: zero-width "
              f"designs have NMED >= {min(nmed):.3g} (median "
              f"{np.median(nmed):.3g}); the sampler draws one "
              f"{100 * hits / draws:.2f}% of the time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default=tempfile.gettempdir())
    ap.add_argument("--skip-logs", action="store_true")
    args = ap.parse_args()
    if not args.skip_logs:
        audit_logs(args.tmp)
    audit_zero_width()


if __name__ == "__main__":
    main()
