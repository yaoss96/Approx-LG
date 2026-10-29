"""Which parts of the template produce the trade-offs?

Every circuit characterised at a width -- by any arm, any seed -- is in the
design cache of that width.  The non-dominated set of the whole cache is the
best front the flow has seen; restricting the cache to genomes that do not use
one mechanism and recomputing the front shows what that mechanism is worth on
its own terms.  This answers whether the added design variables produce
trade-offs that are otherwise unattainable, without any new synthesis.

    python scripts/analyze_mechanisms.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tcad", "tabs")
sys.path.insert(0, ROOT)

from approxlg.arch import Space                               # noqa: E402
from approxlg.metrics import hypervolume                       # noqa: E402


def nondominated_mask(F):
    """Non-dominated rows of an (n, 3) array, all objectives minimized.

    Same result as approxlg.metrics.nondominated_mask (identical rows do not
    dominate each other), in O(n log n) instead of O(n^2): in lexicographic
    order every dominator of a point comes before it, so a point is dominated
    iff the 2-D staircase of (f1, f2) over the points already seen holds one
    at or below it that is not the point itself.  The quadratic version takes
    most of an hour on the ~170k designs of all 8x8 campaigns."""
    import bisect
    F = np.asarray(F, dtype=float)
    n = len(F)
    keep = np.zeros(n, dtype=bool)
    order = np.lexsort((F[:, 2], F[:, 1], F[:, 0]))
    xs, ys, rows = [], [], []          # staircase: f1 ascending, f2 descending
    for i in order:
        q = F[i]
        j = bisect.bisect_right(xs, q[1]) - 1
        if j >= 0 and ys[j] <= q[2] and not np.array_equal(F[rows[j]], q):
            continue                    # dominated
        keep[i] = True
        if j >= 0 and xs[j] == q[1] and ys[j] == q[2]:
            continue                    # identical (f1, f2) already stored
        k = bisect.bisect_left(xs, q[1])
        e = k
        while e < len(xs) and ys[e] >= q[2]:
            e += 1                      # staircase points q covers in 2-D
        xs[k:e], ys[k:e], rows[k:e] = [q[1]], [q[2]], [i]
    return keep

def _caches(prefix, skip=()):
    """Every design cache of one width and alphabet, whatever the campaign."""
    import glob
    out = []
    for p in sorted(glob.glob(os.path.join(RESULTS, f"{prefix}*.jsonl"))):
        f = os.path.basename(p)
        if not any(s in f for s in skip):
            out.append(f)
    return out


# every campaign of the base alphabet: the NMED and MRE campaigns and every
# model setting; the extended-alphabet MRE campaign (n8mrex) decodes
# differently, and the signed 16-bit caches belong to another template
CASES = {"8": (8, _caches("designs_n8", skip=("mrex",)), "exact_n8.json"),
         "16u": (16, _caches("designs_n16u"), "exact_n16u.json")}

RESTRICT = [
    ("full template", lambda t, a, c, k, apx: True),
    ("without truncation ($T=0$)", lambda t, a, c, k, apx: t == 0),
    ("without compensation ($c=0$)", lambda t, a, c, k, apx: c == 0),
    ("without OR-based adder ($k=0$)", lambda t, a, c, k, apx: k == 0),
    ("without approximate cells", lambda t, a, c, k, apx: not apx),
    ("compressors only ($T=k=0$)", lambda t, a, c, k, apx: t == 0 and k == 0),
]


def load(n, files):
    sp = Space(n)
    seen, rows = set(), []
    for f in files:
        p = os.path.join(RESULTS, f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("area", 0) > 1e8:
                continue
            g = sp.canonical(np.array(d["g"], dtype=np.int32))
            key = g.tobytes()
            if key in seen:
                continue
            seen.add(key)
            t, a, c, k = (int(x) for x in g[:4])
            slots = g[sp.n_head:].reshape(sp.nstage, sp.ncol)[:, t:t + a]
            rows.append((t, a, c if t > 0 else 0, k, bool((slots != 0).any()),
                         d["NMED"], d["area"], d["delay"], d.get("MRE", 0.0)))
    return rows


def main():
    os.makedirs(TABS, exist_ok=True)
    out, lines = {}, []
    for w, (n, files, exf) in CASES.items():
        ex = json.load(open(os.path.join(RESULTS, exf)))
        lo = np.array([math.log10(1e-6), 0.0, 0.0])
        hi = np.array([math.log10(0.02), ex["area"], ex["delay"]])
        rows = load(n, files)
        print(f"n={w}: {len(rows)} distinct characterised designs")
        res = {}
        full_front = None
        for name, keep in RESTRICT:
            sub = [r for r in rows if keep(*r[:5])]
            F = np.array([[math.log10(max(r[5], 1e-7)), r[6], r[7]] for r in sub])
            Fn = (F - lo) / (hi - lo)
            inbox = np.all(Fn <= 1.0, axis=1)
            Fb = Fn[inbox]
            nd = nondominated_mask(Fb) if len(Fb) else np.array([], bool)
            hv = hypervolume(Fb[nd], np.ones(3)) if len(Fb) else 0.0
            res[name] = {"designs": len(sub), "front": int(nd.sum()), "hv": hv}
            if name == "full template":
                full_front = [r for r, k in zip([r for r, b in zip(sub, inbox) if b], nd) if k]
            print(f"  {name:34s} designs {len(sub):6d} front {int(nd.sum()):4d} HV {hv:.4f}")
        # how the full front uses each mechanism
        use = {"T>0": np.mean([r[0] > 0 for r in full_front]),
               "c>0": np.mean([r[2] > 0 for r in full_front]),
               "k>0": np.mean([r[3] > 0 for r in full_front]),
               "approx cells": np.mean([r[4] for r in full_front])}
        print("  front usage:", {k: f"{100 * v:.0f}%" for k, v in use.items()})
        out[w] = {"restrict": res, "usage": use, "n_designs": len(rows)}
    with open(os.path.join(TABS, "mechanisms.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(TABS, "mechanisms.tex"), "w") as fh:
        for name, _ in RESTRICT:
            cells = []
            for w in CASES:
                r = out[w]["restrict"][name]
                base = out[w]["restrict"]["full template"]["hv"]
                cells.append(f"{r['hv']:.4f} & ${100 * (r['hv'] / base - 1):+.0f}$\\%")
            fh.write(f"{name} & " + " & ".join(cells) + r" \\" + "\n")
    print("wrote", os.path.join(TABS, "mechanisms.tex"))


if __name__ == "__main__":
    main()
