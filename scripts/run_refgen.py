"""Generate and characterise the compressor-tree reference designs.

* OPACT (DATE'22): the allocation ILP in its three modes (CoOpt, AreaOpt,
  ErrOpt), swept over its parameter and over the number of reduction stages,
  each allocation then given its optimised connection order;
* Esposito et al. (TCAS-I'18): 1StepFull, 2StepsFull, 1StepTrunc, 2StepsTrunc;
* Wang et al. (ISCAS'20): the AM strategy over its one parameter, the length
  L of the exact part (AM8-5 and AM16-10 are the published points).

Every design is built by :mod:`approxlg.ctree`, simulated with the stimulus
of the search campaigns at that width and synthesised with the same recipe,
and written in the record format of ``run_libs.py`` so the published-design
comparison can read it unchanged.

    python scripts/run_refgen.py --n 8
    python scripts/run_refgen.py --n 16 --gpu --nvec 4194304
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approxlg import ctree, opact                           # noqa: E402
from approxlg.simulate import error_metrics, make_stimulus, simulate  # noqa
from approxlg.synth import Synthesizer                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def alloc_key(al: ctree.Allocation) -> str:
    return json.dumps([sorted((int(j), c) for j, c in st.items())
                       for st in al.stages] + [al.trunc])


def opact_family(n: int, stages, n_pts: int, tl: float, log):
    """Allocations of all three OPACT modes over their parameter ranges."""
    out = {}
    for s in stages:
        # the two ends of the trade-off at this stage count
        lo_area = opact.allocate(n, s, "co", 0.0, time_limit=tl)
        if lo_area is None:
            log(f"  s={s}: infeasible")
            continue
        lo_err = opact.allocate(n, s, "err", 1e9, time_limit=tl)
        if lo_err is None:
            # at 32 bits the error-optimal end can exceed the solver's time
            # limit; without it the sweep has no range, so the stage is
            # skipped rather than guessed
            log(f"  s={s}: no error-optimal allocation within {tl:.0f}s; "
                f"stage skipped")
            continue
        d_hi, d_lo = lo_area.meta["D"], lo_err.meta["D"]
        a_lo, a_hi = lo_area.meta["area_est"], lo_err.meta["area_est"]
        log(f"  s={s}: D in [{d_lo:.4g}, {d_hi:.4g}], "
            f"area in [{a_lo:.1f}, {a_hi:.1f}]")
        cands = [lo_area, lo_err]
        # an all-exact tree has D = 0; start the AreaOpt sweep at the error
        # of a single approximate 3:2 compressor in column 0 instead
        d_start = max(d_lo, 1 / 64)
        if d_hi > d_start:
            for d in np.geomspace(d_start, d_hi, n_pts)[:-1]:         # AreaOpt
                cands.append(opact.allocate(n, s, "area", float(d), time_limit=tl))
        for a in np.linspace(a_lo, a_hi, n_pts)[1:-1]:                # ErrOpt
            cands.append(opact.allocate(n, s, "err", float(a), time_limit=tl))
        slope = (a_hi - a_lo) / max(d_hi - d_lo, 1e-12)
        for w in slope * np.geomspace(1e-2, 1e2, n_pts):              # CoOpt
            cands.append(opact.allocate(n, s, "co", float(w), time_limit=tl))
        for al in cands:
            if al is not None:
                out.setdefault(alloc_key(al), al)
    return list(out.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nvec", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--points", type=int, default=10,
                    help="parameter values per OPACT mode and stage count")
    ap.add_argument("--extra-stages", type=int, default=2)
    ap.add_argument("--order-time", type=float, default=60.0)
    ap.add_argument("--ilp-time", type=float, default=60.0)
    ap.add_argument("--skip-opact", action="store_true")
    args = ap.parse_args()
    n = args.n
    out_path = os.path.join(RESULTS, f"refgen_n{n}.jsonl")
    logf = open(os.path.join(RESULTS, f"log_refgen_n{n}.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            if line.strip():
                done.add(json.loads(line)["key"])

    designs = []          # (lib, name, allocation, extra)
    for steps in (1, 2):
        for tr in (False, True):
            al = ctree.esposito(n, steps, trunc=tr)
            designs.append(("Esposito", al.name, al, {}))
    for L in range(max(2, math.ceil(n / 2) - 2), 2 * n - 2):
        # the strategy reaches its target height only for exact parts long
        # enough; shorter ones leave a column it cannot reduce, and are skipped
        try:
            al = ctree.wang(n, L)
            ctree.build(al)
            designs.append(("Wang", al.name, al, {"L": L}))
        except ctree.TreeError as e:
            log(f"  Wang L={L}: not realisable ({e})")

    if not args.skip_opact:
        s0 = opact.min_stages(n)
        stages = list(range(s0, s0 + 1 + args.extra_stages))
        log(f"OPACT n={n}: stage counts {stages}")
        t0 = time.time()
        fam = opact_family(n, stages, args.points, args.ilp_time, log)
        log(f"OPACT n={n}: {len(fam)} distinct allocations "
            f"({time.time() - t0:.0f}s)")
        for al in fam:
            if alloc_key(al) in done:
                continue
            t1 = time.time()
            perms, info = opact.order(al, time_limit=args.order_time)
            al.perms = perms
            extra = dict(al.meta)
            extra["order"] = info
            extra["order_seconds"] = time.time() - t1
            designs.append(("OPACT", al.name, al, extra))

    todo = [d for d in designs if alloc_key(d[2]) not in done]
    log(f"n={n}: {len(designs)} designs, {len(todo)} to characterise")
    if not todo:
        return

    stim = make_stimulus(n, args.nvec or None, gpu=args.gpu, seed=0)
    syn = Synthesizer(cache=os.path.join(RESULTS, f"ppa_cache_refgen_n{n}.json"),
                      n=n, power_vectors=1 << 14)
    nls = [ctree.build(al) for _, _, al, _ in todo]
    errs = [error_metrics(simulate(nl, stim), stim) for nl in nls]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ppas = list(ex.map(lambda nl: syn.run(nl.to_verilog()), nls))
    syn.save_cache()
    with open(out_path, "a") as fh:
        for (lib, name, al, extra), nl, er, ppa in zip(todo, nls, errs, ppas):
            if not ppa.ok:
                log(f"  !! {name}: synthesis failed")
                continue
            rec = {"lib": lib, "name": name, "key": alloc_key(al), "n": n,
                   "signed": False, "encoding": "unsigned", "ok": True,
                   "area": ppa.area, "delay": ppa.delay, "power": ppa.power,
                   "gates": ppa.gates, "ADP": ppa.adp, "PDP": ppa.pdp,
                   "cells": nl.n_cells, "approx_cells": nl.n_approx_cells,
                   "stages": len(al.stages), "trunc": al.trunc,
                   **{k: float(v) for k, v in er.items()}, **extra}
            fh.write(json.dumps(rec) + "\n")
            log(f"  {lib:8s} {name:28s} NMED={er['NMED']:.3e} "
                f"MRE={er['MRE']:.4f} area={ppa.area:7.1f} "
                f"delay={ppa.delay:.3f} PDP={ppa.pdp:7.2f}")


if __name__ == "__main__":
    main()
