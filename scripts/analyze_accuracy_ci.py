"""How precise are the 16x16 accuracy numbers, and which of them are estimates?

At 8x8 the input space is enumerated, so NMED, MRE and the worst case are
exact.  At 16x16 they come from 2^22 uniformly drawn operand pairs out of
2^32, and three different things follow from that:

* **NMED** is a mean of a bounded quantity and concentrates well, so the
  sampling error should be small -- but "should be" is not a measurement.
* **MRE** divides by the exact product, so it is dominated by the smallest
  operands, which uniform sampling covers worst.  Its sampling error is the
  one that matters, because every iso-accuracy comparison in the paper
  constrains MRE as well as NMED.
* **The worst case** is a maximum.  A maximum over a sample is a *lower
  bound* on the maximum over the space and nothing else; no amount of
  sampling makes it an estimate.  This script measures the first two and
  demonstrates the third rather than papering over it.

Each design is re-evaluated under several independent stimulus draws.  The
spread across draws is the sampling error, reported relative to the value
itself so it can be compared against the iso-accuracy margins the paper
relies on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approxlg.arch import Builder, Space                     # noqa: E402
from approxlg.simulate import (Evaluator, error_metrics,     # noqa: E402
                               simulate)
from analyze_sota import iso_accuracy, load_front             # noqa: E402


def _library_and_front():
    """The verified 16x16 library designs and the pooled guided front."""
    vp = os.path.join(RESULTS, "libs_verify.json")
    lp = os.path.join(RESULTS, "libs_n16.jsonl")
    if not (os.path.exists(vp) and os.path.exists(lp)):
        return [], []
    ver = {r["name"]: r["status"] for r in json.load(open(vp))}
    libs = []
    for line in open(lp):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("ok") and ver.get(r["name"]) == "match" \
                and 1e-12 <= r["NMED"] <= 0.10:
            libs.append(r)
    fr = load_front(["n16u", "n16s"], ("approxlg",), {"n16s"}, max_seeds=20)
    return libs, fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", type=int, default=40)
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--nvec", type=int, default=1 << 22)
    ap.add_argument("--gpu", action="store_true", default=True)
    args = ap.parse_args()

    front = load_front(["n16u"], ("approxlg",), set(), max_seeds=20)
    # spread the sample over the error range rather than clustering where the
    # front is densest
    front.sort(key=lambda d: d["NMED"])
    idx = np.unique(np.linspace(0, len(front) - 1, args.designs).astype(int))
    picks = [front[i] for i in idx]
    print(f"{len(picks)} designs spanning NMED "
          f"{picks[0]['NMED']:.2e} to {picks[-1]['NMED']:.2e}, "
          f"{args.draws} independent stimulus draws of 2^"
          f"{int(np.log2(args.nvec))}")

    sp = Space(16)
    b = Builder(sp)
    evs = [Evaluator(sp, nvec=args.nvec, gpu=args.gpu, seed=1000 + i)
           for i in range(args.draws)]

    rows = []
    for d in picks:
        nl = b.build(sp.canonical(np.array(d["g"], dtype=np.int32)))
        vals = [error_metrics(simulate(nl, e.stim), e.stim) for e in evs]
        row = {"NMED_ref": d["NMED"]}
        for k in ("NMED", "MRE", "WCE"):
            v = np.array([x[k] for x in vals], float)
            row[k] = {"mean": float(v.mean()),
                      "rsd": float(v.std(ddof=1) / v.mean()) if v.mean() else 0.0,
                      "min": float(v.min()), "max": float(v.max())}
        rows.append(row)

    def summarise(metric):
        r = np.array([x[metric]["rsd"] for x in rows]) * 100
        return r.mean(), np.median(r), r.max()

    print(f"\n  {'metric':6s} {'mean rel. s.d.':>15s} {'median':>8s} "
          f"{'worst design':>13s}")
    out = {"designs": len(rows), "draws": args.draws, "nvec": args.nvec}
    for m in ("NMED", "MRE", "WCE"):
        mu, med, mx = summarise(m)
        out[m] = {"mean_rsd_pct": mu, "median_rsd_pct": med, "max_rsd_pct": mx}
        print(f"  {m:6s} {mu:14.3f}% {med:7.3f}% {mx:12.3f}%")

    # the worst case is a lower bound: show how much it still grows with the
    # sample, which is the honest way to say it has not converged
    grow = []
    for x in rows:
        lo, hi = x["WCE"]["min"], x["WCE"]["max"]
        if lo > 0:
            grow.append(100 * (hi / lo - 1))
    if grow:
        out["wce_spread_pct"] = {"mean": float(np.mean(grow)),
                                 "max": float(np.max(grow))}
        print(f"\n  worst-case error varies by {np.mean(grow):.1f}% on average "
              f"(up to {np.max(grow):.1f}%) between draws of the same design,")
        print("  so it is reported as a sampled lower bound, not an estimate.")

    # ---- what that noise does to the headline win rate -------------------
    # Every iso-accuracy comparison requires our design to be at least as good
    # on MRE as well as NMED, and MRE is the noisy one.  Rather than assert
    # the noise is harmless, perturb both sides by the measured amount and
    # recount, so the reported rate carries an interval.
    rsd = out["MRE"]["mean_rsd_pct"] / 100.0
    libs, front_all = _library_and_front()
    if libs and front_all:
        rng = np.random.default_rng(7)
        rates = []
        for _ in range(400):
            f2 = []
            for d in front_all:
                e = dict(d)
                e["MRE"] = d["MRE"] * float(rng.lognormal(0.0, rsd))
                e["ADP"] = d["area"] * d["delay"]
                f2.append(e)
            wins = tot = 0
            for r in libs:
                r2 = dict(r)
                r2["MRE"] = r["MRE"] * float(rng.lognormal(0.0, rsd))
                sub = [d for d in f2
                       if bool(d.get("signed")) == bool(r2["signed"])]
                m = iso_accuracy(sub, r2, lambda d: d["ADP"], ("NMED", "MRE"))
                tot += 1
                if m is not None and m["ADP"] < r2["area"] * r2["delay"]:
                    wins += 1
            rates.append(100 * wins / max(tot, 1))
        rates = np.array(rates)
        out["winrate_under_mre_noise"] = {
            "median": float(np.median(rates)),
            "lo": float(np.quantile(rates, 0.025)),
            "hi": float(np.quantile(rates, 0.975))}
        print("  library ADP win rate under the measured MRE noise: "
              f"{np.median(rates):.1f}% "
              f"[{np.quantile(rates, 0.025):.1f}, "
              f"{np.quantile(rates, 0.975):.1f}]")

    with open(os.path.join(TABS, "accuracy_ci.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote", os.path.join(TABS, "accuracy_ci.json"))


if __name__ == "__main__":
    main()
