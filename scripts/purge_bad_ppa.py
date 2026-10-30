"""Find characterisations that are physically impossible and remove them.

The retry logic in ``Synthesizer.run`` protects against a synthesis that
*fails*.  It does not protect against one that returns a number: under heavy
process contention a run can come back with an area and a delay parsed from an
incomplete log, marked ``ok``, and be cached forever.  One such record survived
this campaign -- 39 cells reported as 19.68 um^2 and 39 ps, where a fresh
synthesis of the same genome gives 369.74 um^2 and 440 ps -- and it doubled the
hypervolume of the run that found it.

Mapped area rises monotonically and tightly with the pre-synthesis cell count,
so the detector is per-cell-count: flag any design whose area falls below half
the 0.5th percentile of designs with the same cell count.  On 190k records that
rule fires exactly once, which is the point: it is a corruption detector, not a
filter on the search.

Removing the design record is not enough, because the PPA cache is keyed on the
emitted Verilog and would serve the same wrong answer again, so the cache entry
goes too.  Runs whose stored Pareto front contains a purged design are listed
and must be re-run; the script does not delete them silently.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

from approxlg.arch import Builder, Space                     # noqa: E402

FRACTION = 0.5       # of the 0.5th percentile for the same cell count
MIN_GROUP = 20       # cell counts rarer than this are not judged


def tag_of(path: str) -> str:
    return os.path.basename(path).split("designs_")[1].rsplit(".jsonl", 1)[0]


def space_of(tag: str, g) -> Space:
    """The space a record was drawn from: the width from the genome length,
    the signedness from the tag (n16s, n16smoo, n16sfill, ...)."""
    signed = re.match(r"n\d+s", tag) is not None
    for n in (8, 16, 32):
        sp = Space(n, signed=signed)
        if sp.length == len(g):
            return sp
    raise SystemExit(f"no space of length {len(g)} for {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="purge only these tags, e.g. while campaigns on "
                         "the other tags are still appending to their files")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RESULTS, "designs_*.jsonl")))
    recs = {}
    groups = defaultdict(list)
    for p in files:
        rows = []
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # blank, or a fragment left by a process stopped mid-append
                continue
        recs[p] = rows
        tag = tag_of(p)
        for d in rows:
            groups[(tag, int(d.get("cells", 0)))].append(d["area"])

    floor = {k: FRACTION * np.percentile(v, 0.5)
             for k, v in groups.items() if len(v) >= MIN_GROUP}

    purge = defaultdict(list)
    n_all = 0
    for p, rows in recs.items():
        tag = tag_of(p)
        for d in rows:
            n_all += 1
            k = (tag, int(d.get("cells", 0)))
            if k in floor and d["area"] < floor[k]:
                purge[p].append(d)

    total = sum(len(v) for v in purge.values())
    print(f"{n_all} characterisations scanned, {total} physically impossible")
    # the paper quotes these, so they are written even on a dry run
    tabs = os.path.join(ROOT, "paper", "tabs")
    if os.path.isdir(tabs):
        with open(os.path.join(tabs, "charcheck.json"), "w") as fh:
            json.dump({"scanned": n_all, "flagged": total,
                       "examples": [{"cells": int(d["cells"]),
                                     "area": d["area"], "delay": d["delay"]}
                                    for ds in purge.values() for d in ds]},
                      fh, indent=1)
    for p, ds in purge.items():
        for d in ds:
            print(f"  {os.path.basename(p)}: cells={int(d['cells'])} "
                  f"area={d['area']:.2f} delay={d['delay']:.4f} "
                  f"NMED={d['NMED']:.3e}")
    if not total:
        return

    # which runs used them
    keys = {tuple(d["g"]) for ds in purge.values() for d in ds}
    hit = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "search_*.jsonl"))):
        if p.endswith("_dead.jsonl"):
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if any(tuple(d["g"]) in keys for d in r["pareto"]):
                hit.append((os.path.basename(p), r["algo"], r["seed"]))
    print(f"\n{len(hit)} run(s) have one on their recorded front and must be "
          f"re-run after the purge:")
    for h in hit:
        print("   ", h)

    if not args.apply:
        print("\ndry run; pass --apply")
        return

    for p, ds in purge.items():
        if args.only and tag_of(p) not in args.only:
            print(f"left {len(ds)} in {os.path.basename(p)} (not in --only)")
            continue
        drop = {id(d) for d in ds}
        keep = [d for d in recs[p] if id(d) not in drop]
        with open(p, "w") as fh:
            for d in keep:
                fh.write(json.dumps(d) + "\n")
        print(f"purged {len(ds)} from {os.path.basename(p)}")

        # the synthesis cache is keyed on sha1(verilog + top)[:16]
        cache_path = os.path.join(RESULTS, f"ppa_cache_{tag_of(p)}.json")
        if not os.path.exists(cache_path):
            continue
        cache = json.load(open(cache_path))
        n_before = len(cache)
        for d in ds:
            g = np.array(d["g"], dtype=np.int32)
            sp = space_of(tag_of(p), g)
            v = Builder(sp).build(sp.canonical(g)).to_verilog()
            h = hashlib.sha1((v + "mul").encode()).hexdigest()[:16]
            cache.pop(h, None)
        json.dump(cache, open(cache_path, "w"))
        print(f"  dropped {n_before - len(cache)} cache entries from "
              f"{os.path.basename(cache_path)}")


if __name__ == "__main__":
    main()
