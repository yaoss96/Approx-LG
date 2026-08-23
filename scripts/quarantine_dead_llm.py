"""Separate search runs whose LLM endpoint was down from the ones that ran.

A guided run whose every guidance call failed is not a run of the guided
algorithm.  The fail-closed operators quietly refill from the standard
stochastic ones, so the record that lands in ``search_*.jsonl`` is labelled
``approxlg`` but the search that produced it was plain NSGA-II.  Averaging
those into the guided arm silently dilutes the effect towards zero, which is
exactly what happened to the seed-extension campaign of 2026-08-20: seeds
10..29 were recorded as Approx-LG with ``calls=0, failures=70..127``.

The rule is a failure *rate*, not a count, because a run legitimately retries.
Runs above the threshold are moved to ``search_<tag>_dead.jsonl``.  They are
kept rather than deleted: an unplanned outage over 20 paired seeds is the
cleanest evidence available that guidance failure degrades a run to NSGA-II
instead of below it, which is the honest form of the fail-closed claim.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

MAX_FAIL_RATE = 0.25


def fail_rate(rec: dict) -> float:
    llm = rec.get("llm")
    if not llm:
        return 0.0
    c, f = llm.get("calls", 0), llm.get("failures", 0)
    return f / max(1, c + f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=MAX_FAIL_RATE)
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the files; without it, only report")
    args = ap.parse_args()

    total_dead = 0
    for path in sorted(glob.glob(os.path.join(RESULTS, "search_*.jsonl"))):
        if path.endswith("_dead.jsonl"):
            continue
        live, dead = [], []
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            (dead if fail_rate(rec) > args.rate else live).append((rec, line))
        if not dead:
            continue
        total_dead += len(dead)
        name = os.path.basename(path)
        print(f"{name}: {len(dead)} dead of {len(dead)+len(live)}")
        for rec, _ in dead:
            llm = rec.get("llm") or {}
            print(f"    {rec['algo']:22s} seed {rec['seed']:2d} "
                  f"hv={rec['hv']:.4f} calls={llm.get('calls')} "
                  f"fail={llm.get('failures')}")
        if not args.apply:
            continue
        dead_path = path.replace(".jsonl", "_dead.jsonl")
        with open(dead_path, "a") as fh:
            for _, line in dead:
                fh.write(line if line.endswith("\n") else line + "\n")
        with open(path, "w") as fh:
            for _, line in live:
                fh.write(line if line.endswith("\n") else line + "\n")
        print(f"    -> {os.path.basename(dead_path)}")

    print(f"\n{total_dead} runs above a {args.rate:.0%} guidance-failure rate"
          + ("" if args.apply else "  (dry run, pass --apply)"))


if __name__ == "__main__":
    main()
