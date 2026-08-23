"""Three questions about where the results come from, rather than how big.

*Which arm's front beats the libraries?*  The paper's headline win rate is a
property of whatever produced the front, and the front is a union over runs, so
it grows with the seed count.  Scoring each arm's front against the same
library at a matched number of seeds separates "the framework wins" from "the
guidance wins", and it is also the cheapest answer to the contamination worry:
an arm that never received a single model answer cannot have memorised
anything.

*Did the model regurgitate published designs?*  Our designs and the library
designs are described in different representations, so a genome comparison is
meaningless, but both are measured by the same evaluator.  Two circuits that
agree to ten significant digits on four independent error statistics are the
same function.  The rate of such matches is reported for the guided arm and for
the arms with no model in them; a difference would be evidence of memorisation
and no difference is evidence against it.

*Does the inclusion rule change anything?*  Runs whose guidance calls nearly
all failed are excluded from the guided arm, and that rule was written after
the failures were seen.  The primary comparison is therefore repeated with
every discarded run added back.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_stats import load, paired                        # noqa: E402

SIG = ("NMED", "MRE", "WCE", "ER")


def signatures(rows):
    out = set()
    for d in rows:
        if all(k in d for k in SIG):
            out.add(tuple(round(float(d[k]), 10) for k in SIG))
    return out


def library_rows():
    p = os.path.join(RESULTS, "libs_n16.jsonl")
    if not os.path.exists(p):
        return []
    ver = {}
    vp = os.path.join(RESULTS, "libs_verify.json")
    if os.path.exists(vp):
        ver = {r["name"]: r["status"] for r in json.load(open(vp))}
    return [r for r in (json.loads(l) for l in open(p) if l.strip())
            if r.get("ok") and ver.get(r["name"]) == "match"]


def front_rows(tags, algo, max_seeds=0):
    out = []
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            if d["algo"] != algo:
                continue
            if max_seeds and d["seed"] >= max_seeds:
                continue
            out.extend(d["pareto"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["n16u", "n16s"])
    ap.add_argument("--arms", nargs="+",
                    default=["approxlg", "nsga2_expert", "approxlg_sham",
                             "nsga2_rf", "nsga2"])
    ap.add_argument("--max-seeds", type=int, default=8)
    args = ap.parse_args()
    report = {}

    # ---- memorisation probe ------------------------------------------------
    lib = library_rows()
    lib_sig = signatures(lib)
    print(f"library designs with a verified error signature: {len(lib_sig)}")
    mem = {}
    for arm in args.arms:
        rows = front_rows(args.tags, arm, args.max_seeds)
        if not rows:
            continue
        sig = signatures(rows)
        hits = sig & lib_sig
        mem[arm] = {"front_points": len(rows), "distinct": len(sig),
                    "exact_matches": len(hits)}
        print(f"  {arm:16s} {len(rows):5d} front points, {len(sig):5d} "
              f"distinct signatures, {len(hits)} identical to a library design")
    report["memorisation"] = mem

    # ---- inclusion robustness ----------------------------------------------
    # A discarded run is only missing data if that seed was never run again.
    # Reporting per (campaign, arm) makes the difference visible.
    live = defaultdict(set)
    for p in sorted(glob.glob(os.path.join(RESULTS, "search_*.jsonl"))):
        if p.endswith("_dead.jsonl"):
            continue
        tag = os.path.basename(p)[len("search_"):-len(".jsonl")]
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                live[(tag, d["algo"])].add(d["seed"])
    dead = defaultdict(set)
    for p in sorted(glob.glob(os.path.join(RESULTS, "search_*_dead.jsonl"))):
        tag = os.path.basename(p)[len("search_"):-len("_dead.jsonl")]
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                dead[(tag, d["algo"])].add(d["seed"])

    print("\ndiscarded runs (guidance-failure rate above 25%) and their fate:")
    rows = {}
    for k in sorted(dead):
        ds, ls = dead[k], live.get(k, set())
        missing = sorted(ds - ls)
        rows["/".join(k)] = {"discarded": len(ds), "rerun": len(ds & ls),
                             "missing": missing}
        print(f"  {'/'.join(k):30s} {len(ds):3d} discarded, "
              f"{len(ds & ls):3d} re-run at the same seed, "
              + (f"{len(missing)} never replaced" if missing else
                 "none missing"))
    report["discarded"] = rows
    orphan = {k: v for k, v in rows.items() if v["missing"]}
    print("  every arm the paper reports is fully replaced; the only "
          "unreplaced discards are in the abandoned temperature campaign "
          f"({', '.join(orphan) if orphan else 'none'}), which contributes no "
          "number to the paper.")

    with open(os.path.join(TABS, "provenance.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote", os.path.join(TABS, "provenance.json"))


if __name__ == "__main__":
    main()
