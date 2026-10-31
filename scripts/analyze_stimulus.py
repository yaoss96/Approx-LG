"""Out-of-sample check of the error estimates at 16x16 and 32x32.

Every design of a width was simulated on one fixed random stimulus (seed 0),
so a search that selects on it could keep designs whose error that sample
happens to underestimate.  This re-simulates the final fronts of the arms the
paper's comparisons rest on with an independent stimulus of the same size
(seed 1) and reports the change of NMED and MRE per design, every run's
hypervolume recomputed with the new errors (area and delay unchanged), and
the paper's comparisons recomputed on those hypervolumes.

    python scripts/analyze_stimulus.py
Writes paper/tcad/tabs/stimulus.json, which analyze_tcad.py turns into macros.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import analyze_tcad as A                                       # noqa: E402
from approxlg.arch import Builder, Space                      # noqa: E402
from approxlg.metrics import hypervolume                      # noqa: E402
from approxlg.simulate import Evaluator, error_metrics, simulate  # noqa: E402

ARMS = ("approxlg", "approxlg_sham", "nsga2", "smsemoa", "mobo", "nsga2_rf")
WIDTHS = (("16u", 16, False, 1 << 22, "exact_n16u.json"),
          ("16s", 16, True, 1 << 22, "exact_n16s.json"),
          ("32u", 32, False, 1 << 20, "exact_n32u.json"))
FLOOR, LO, HI = 1e-7, 1e-6, 0.02          # as in engine.py and run_search.py


def run_hv(front, nmed, ex):
    lo = np.array([math.log10(LO), 0.0, 0.0])
    hi = np.array([math.log10(HI), ex["area"], ex["delay"]])
    F = np.array([[math.log10(max(e, FLOOR)), d["area"], d["delay"]]
                  for d, e in zip(front, nmed)])
    return hypervolume((F - lo) / (hi - lo), np.ones(3))


def main():
    runs = A.load()
    out = {}
    for w, n, signed, nvec, exf in WIDTHS:
        t0 = time.time()
        ex = json.load(open(os.path.join(A.RESULTS, exf)))
        sp = Space(n, signed=signed)
        b = Builder(sp)
        ev = Evaluator(sp, nvec=nvec, gpu=True, seed=1)
        new = {}
        for a in ARMS:
            for r in runs[w].get(a, {}).values():
                for d in r["pareto"]:
                    k = tuple(d["g"])
                    if k not in new:
                        nl = b.build(sp.canonical(np.array(d["g"], dtype=np.int32)))
                        er = error_metrics(simulate(nl, ev.stim), ev.stim)
                        new[k] = (float(er["NMED"]), float(er["MRE"]),
                                  float(d["NMED"]), float(d["MRE"]))
        v = np.array(list(new.values()))
        ok = v[:, 2] > 0
        dn = v[ok, 0] / v[ok, 2] - 1
        okm = v[:, 3] > 0
        dm = v[okm, 1] / v[okm, 3] - 1
        hv_old, hv_new, check = {}, {}, []
        for a in ARMS:
            for s, r in runs[w].get(a, {}).items():
                fr = r["pareto"]
                hv_old.setdefault(a, {})[s] = r["hv"]
                hv_new.setdefault(a, {})[s] = run_hv(
                    fr, [new[tuple(d["g"])][0] for d in fr], ex)
                check.append(abs(run_hv(fr, [d["NMED"] for d in fr], ex) - r["hv"]))
        comp = {}
        for y in ARMS[1:]:
            if y in hv_new:
                po = A.paired(hv_old, "approxlg", y)
                pn = A.paired(hv_new, "approxlg", y)
                comp[y] = {"old": po["delta"], "new": pn["delta"],
                           "new_wins": f"{pn['wins']}/{pn['npairs']}",
                           "new_p": pn["p"]}
        mean_old = {a: float(np.mean(list(h.values()))) for a, h in hv_old.items()}
        mean_new = {a: float(np.mean(list(h.values()))) for a, h in hv_new.items()}
        rank_old = sorted(mean_old, key=mean_old.get, reverse=True)
        rank_new = sorted(mean_new, key=mean_new.get, reverse=True)
        out[w] = {
            "designs": len(new),
            "nmed_median_abs_pct": float(100 * np.median(np.abs(dn))),
            "nmed_p95_abs_pct": float(100 * np.percentile(np.abs(dn), 95)),
            "nmed_mean_pct": float(100 * np.mean(dn)),
            "mre_median_abs_pct": float(100 * np.median(np.abs(dm))),
            "mre_mean_pct": float(100 * np.mean(dm)),
            "hv_change_pct": {a: 100 * (mean_new[a] / mean_old[a] - 1)
                              for a in mean_old},
            "hv_recompute_max_err": float(max(check)),
            "rank_old": rank_old, "rank_new": rank_new,
            "approxlg_vs": comp,
        }
        print(f"{w}: {len(new)} designs re-simulated in {time.time() - t0:.0f} s; "
              f"NMED change median |{out[w]['nmed_median_abs_pct']:.3f}|%, "
              f"mean {out[w]['nmed_mean_pct']:+.3f}%; MRE median "
              f"|{out[w]['mre_median_abs_pct']:.3f}|%; HV recompute error "
              f"{out[w]['hv_recompute_max_err']:.2e}")
        print(f"   ranking kept: {rank_old == rank_new}  "
              + "  ".join(f"vs {A.LABEL[y]} {c['old']:+.2f}->{c['new']:+.2f}"
                          for y, c in comp.items()))
    with open(os.path.join(A.TABS, "stimulus.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
