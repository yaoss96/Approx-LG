"""Does the guided arm still win when the budget is wall-clock, not evaluations?

The paper counts EDA evaluations, which is the right currency only if the
evaluation dominates the run.  On this open flow it does not: one
characterisation costs a fraction of a second while guidance costs minutes, so
an evaluation budget quietly hands the guided arm extra wall-clock that the
unguided arms could have spent on more evaluations.

This re-scores every run against a *time* axis instead.  For a run of arm A the
elapsed time after k evaluations is modelled as

    t(k) = k * c + g(k),

with c the pooled per-characterisation cost measured over every timed run and
g(k) the guidance time that arm had spent by then, taken as its measured total
scaled by k/B.  The hypervolume trace is then re-indexed onto t, and the arms
are compared at matched times.  Both terms are measured, not assumed; what the
model supplies is the interpolation between recorded points.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_stats import paired, wilcoxon_exact             # noqa: E402


def load(tags):
    by = defaultdict(list)
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                by[d["algo"]].append(d)
    return by


def per_circuit(by):
    n = t = 0
    for runs in by.values():
        for r in runs:
            if "t_synth" in r and r.get("n_synth", 0) > 0:
                n += r["n_synth"]
                t += r["t_synth"]
    return t / n if n else float("nan")


def time_trace(run, c, budget):
    """(time, hypervolume) for one run under the wall-clock model."""
    g = run.get("t_llm", (run.get("llm") or {}).get("seconds", 0.0)) or 0.0
    ts, hs = [], []
    for k, h in run["hv_trace"]:
        ts.append(k * c + g * min(k / budget, 1.0))
        hs.append(h)
    return np.array(ts), np.array(hs)


def hv_at(run, c, budget, t):
    ts, hs = time_trace(run, c, budget)
    if not len(ts) or t < ts[0]:
        return 0.0
    return float(hs[np.searchsorted(ts, t, side="right") - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["n16u"])
    ap.add_argument("--budget", type=int, default=500)
    ap.add_argument("--arms", nargs="+",
                    default=["approxlg", "approxlg_sham", "nsga2_rf", "nsga2"])
    ap.add_argument("--ref", default="approxlg")
    ap.add_argument("--long-tag", default="n16ulong",
                    help="campaign of the same arms at a larger evaluation "
                         "budget, for the equal-wall-clock comparison")
    ap.add_argument("--long-budget", type=int, default=1070)
    args = ap.parse_args()

    by = load(args.tags)
    allby = load(["n8", "n8L", "n8abl", "n8p", "n16u", "n16s"])
    c = per_circuit(allby)
    print(f"pooled cost of one characterisation: {c:.3f} s")

    arms = [a for a in args.arms if a in by]
    # the time at which the reference arm exhausts its evaluation budget
    tmax = float(np.median([time_trace(r, c, args.budget)[0][-1]
                            for r in by[args.ref]]))
    print(f"the guided arm reaches its evaluation budget at {tmax:.0f} s; "
          f"comparisons below are at that wall clock\n")

    print(f"  {'arm':16s} {'runs':>4s} {'HV @ equal time':>16s} "
          f"{'evals used':>11s}")
    scores = {}
    for a in arms:
        vals, evs = {}, []
        for r in by[a]:
            vals[r["seed"]] = hv_at(r, c, args.budget, tmax)
            ts, _ = time_trace(r, c, args.budget)
            k = np.searchsorted(ts, tmax, side="right")
            evs.append(r["hv_trace"][min(k, len(r["hv_trace"]) - 1)][0])
        scores[a] = vals
        print(f"  {a:16s} {len(vals):4d} {np.mean(list(vals.values())):16.4f} "
              f"{np.mean(evs):11.0f}")

    print()
    out = {"per_circuit_s": c, "t_match_s": tmax, "arms": {}}
    for a in arms:
        if a == args.ref:
            continue
        r = paired(scores, args.ref, a)
        out["arms"][a] = r
        print(f"  {args.ref} vs {a:16s} n={r['n']:2d} {r['delta']:+6.2f}% "
              f"CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] p={r['p']:.4f}")

    # ---- the comparison that actually answers the question ---------------
    # Capping the baselines at the guided arm's evaluation budget is not equal
    # time: the guidance seconds would have bought them more evaluations.  The
    # long campaign gives them those evaluations.
    long = load([args.long_tag]) if args.long_tag else {}
    if long:
        gtime = np.mean([r.get("t_llm", (r.get("llm") or {}).get("seconds",
                                                                 0.0)) or 0.0
                         for r in by[args.ref]])
        target = float(np.mean([r["hv"] for r in by[args.ref]]))
        print(f"\nagainst the same arms run to {args.long_budget} evaluations, "
              f"the budget {tmax:.0f}s buys at {c:.3f}s each:")
        sc2 = {args.ref: {r["seed"]: r["hv"] for r in by[args.ref]}}
        out["equal_time"] = {"guidance_s": gtime, "target_hv": target,
                             "arms": {}}
        for a, runs in long.items():
            sc2[a] = {r["seed"]: hv_at(r, c, args.long_budget, tmax)
                      for r in runs}
            if len(set(sc2[args.ref]) & set(sc2[a])) < 6:
                continue
            r = paired(sc2, args.ref, a)
            # how many evaluations that arm needs to match the guided arm, and
            # therefore how expensive a characterisation has to be before the
            # guidance time is worth paying
            reach = []
            for run in runs:
                for k, h in run["hv_trace"]:
                    if h >= target:
                        reach.append(k)
                        break
            ev = float(np.median(reach)) if reach else float("nan")
            break_even = gtime / (ev - args.budget) if ev > args.budget \
                else float("nan")
            out["equal_time"]["arms"][a] = {
                **r, "reached": len(reach), "runs": len(runs),
                "median_evals_to_match": ev, "break_even_s": break_even}
            print(f"  {args.ref} vs {a:18s} n={r['n']:2d} {r['delta']:+6.2f}% "
                  f"CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] p={r['p']:.4f}"
                  f"   matches at {ev:.0f} evals -> break-even "
                  f"{break_even:.2f}s/characterisation")

    with open(os.path.join(TABS, f"walltime_{args.tags[0]}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(TABS, f"walltime_{args.tags[0]}.json"))


if __name__ == "__main__":
    main()
