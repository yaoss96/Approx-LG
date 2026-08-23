"""Is the library win rate an artefact of one synthesis operating point?

Every area and delay in the paper comes from a single delay-driven mapping with
area recovery and no required-time constraint.  Approximate multipliers differ
in how much slack they have, so an ordering established at one operating point
need not hold at another, and the headline win rate is exactly such an
ordering.

ABC's required-time option has no effect in this build -- area and delay come
back bit-identical from 0.3ns to 5ns -- so the sweep varies the mapping
strategy instead, from delay-driven with no area recovery to a purely
area-oriented mapper.  On the exact 16x16 multiplier those five points span
1526 to 2305 um^2 and 0.94 to 1.44 ns, so they are genuinely different
operating points.  Both sides of every iso-accuracy pair are re-synthesised at
each and the comparison recomputed.  Accuracy does not depend
on the recipe, so the *pairing* is fixed once, under the default recipe; what
moves is which member of each pair is cheaper.  That is the question being
asked: not whether the matching is stable, but whether the verdict is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approxlg.arch import Builder, Space                     # noqa: E402
from approxlg.synth import Synthesizer                       # noqa: E402
from analyze_sota import iso_accuracy, load_front             # noqa: E402
from run_libs import discover                                 # noqa: E402

MAP_LABEL = {
    "default": "delay-driven, area recovery (as reported)",
    "delay": "delay-driven, no area recovery",
    "balanced": "delay-driven, half area recovery",
    "area": "area mode",
    "amap": "area-oriented mapper",
}

RECIPES = ["default", "delay", "balanced", "area", "amap"]


def verified_libs():
    ver = {r["name"]: r["status"]
           for r in json.load(open(os.path.join(RESULTS, "libs_verify.json")))}
    out = []
    for line in open(os.path.join(RESULTS, "libs_n16.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok") and ver.get(r["name"]) == "match" \
                and 1e-12 <= r["NMED"] <= 0.10:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    libs = verified_libs()
    src = {d["name"]: d for d in discover(16)}
    front = load_front(["n16u", "n16s"], ("approxlg",), {"n16s"},
                       max_seeds=20)
    for d in front:
        d["ADP"] = d["area"] * d["delay"]

    # fix the pairing under the default recipe
    pairs = []
    for r in libs:
        sub = [d for d in front if bool(d.get("signed")) == bool(r["signed"])]
        m = iso_accuracy(sub, r, lambda d: d["ADP"], ("NMED", "MRE"))
        if m is not None and r["name"] in src:
            pairs.append((r, m))
    print(f"{len(pairs)} iso-accuracy pairs carried through the sweep")

    rows = []
    for recipe in RECIPES:
        syn = Synthesizer(cache=os.path.join(
            RESULTS, f"ppa_sweep_{recipe}.json"),
            n=16, power=False, recipe=recipe)
        sp = {False: Space(16), True: Space(16, signed=True)}
        b = {k: Builder(v) for k, v in sp.items()}

        def ours(m):
            g = np.array(m["g"], dtype=np.int32)
            s = bool(m.get("signed"))
            v = b[s].build(sp[s].canonical(g)).to_verilog()
            return syn.run(v)

        def theirs(r):
            d = src[r["name"]]
            return syn.run_files(d["files"], d["top"], d.get("incdirs", ()),
                                 *(r["ports"][0], r["ports"][1]))

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            mine = list(ex.map(ours, [m for _, m in pairs]))
            lib = list(ex.map(theirs, [r for r, _ in pairs]))
        syn.save_cache()

        ok = [(a, c) for a, c in zip(mine, lib) if a.ok and c.ok]
        wins = sum(1 for a, c in ok if a.area * a.delay < c.area * c.delay)
        red = [100 * (1 - (a.area * a.delay) / (c.area * c.delay))
               for a, c in ok]
        lab = MAP_LABEL[recipe]
        rows.append({"recipe": recipe, "label": lab, "pairs": len(ok),
                     "wins": wins, "rate": 100 * wins / max(len(ok), 1),
                     "median_reduction": float(np.median(red)) if red else 0.0})
        print(f"  {lab:24s} {wins:3d}/{len(ok):<3d} = {rows[-1]['rate']:5.1f}%"
              f"   median ADP reduction {rows[-1]['median_reduction']:5.1f}%")

    with open(os.path.join(TABS, "delaysweep.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(TABS, "delaysweep.tex"), "w") as fh:
        fh.write("% generated by scripts/analyze_delay_sweep.py\n")
        for r in rows:
            fh.write(f"{r['label']} & {r['pairs']} & {r['wins']} & "
                     f"{r['rate']:.1f}\\% & {r['median_reduction']:.1f}\\% "
                     f"\\\\\n")
    print("\nwrote", os.path.join(TABS, "delaysweep.tex"))


if __name__ == "__main__":
    main()
