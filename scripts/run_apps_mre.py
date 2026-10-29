"""Applications at matched MRE: does retargeting the error objective pay off?

For each MRE target the cheapest design (in power x delay) with MRE at or
below it is taken from three sources -- the MRE-retargeted Approx-LG front,
the NMED-objective Approx-LG front and the OPACT family -- and all three run
the image and LeNet-5 workloads of ``run_apps.py`` through a 256x256 product
table built by exhaustive simulation of the design's own netlist.

    python scripts/run_apps_mre.py --cnn --epochs 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approxlg import ctree                                    # noqa: E402
from approxlg.arch import Builder, Space                      # noqa: E402
from approxlg.simulate import make_stimulus, simulate         # noqa: E402
from run_apps import cnn_eval, image_quality, load_images     # noqa: E402

STIM = make_stimulus(8, None)


def lut_of(nl):
    """256x256 product table, indexed [a, b], from exhaustive simulation."""
    prod = simulate(nl, STIM)
    a, b = STIM.a.astype(np.int64), STIM.b.astype(np.int64)
    lut = np.zeros((256, 256), dtype=np.int64)
    lut[a, b] = prod
    return lut


def front(tag, algo, max_seeds):
    pts = []
    for line in open(os.path.join(RESULTS, f"search_{tag}.jsonl")):
        d = json.loads(line)
        if d["algo"] == algo and d["seed"] < max_seeds:
            pts.extend(d["pareto"])
    for p in pts:
        p["PDP"] = p["power"] * p["delay"]
    return pts


def opact_family():
    out = []
    for line in open(os.path.join(RESULTS, "refgen_n8.jsonl")):
        d = json.loads(line)
        if d["lib"] == "OPACT":
            out.append(d)
    return out


def rebuild_opact(rec):
    """Recreate an OPACT netlist from its stored allocation key."""
    from approxlg import opact
    stages_trunc = json.loads(rec["key"])
    stages, trunc = stages_trunc[:-1], stages_trunc[-1]
    al = ctree.Allocation(n=8, stages=[{int(j): c for j, c in st}
                                       for st in stages], trunc=trunc)
    al.perms, _ = opact.order(al, time_limit=30)
    return ctree.build(al)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", type=float,
                    default=[1e-3, 3e-3, 1e-2, 2e-2])
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()

    # genomes of the extended campaign use style 21; with the extension on,
    # every base-alphabet genome still decodes to the same circuit
    if os.environ.get("APPROXLG_STYLES", "").lower() != "ext":
        raise SystemExit("run with APPROXLG_STYLES=ext")
    sp = Space(8)
    b = Builder(sp)
    mre_front = front("n8mre", "approxlg", 12)
    mrex_front = (front("n8mrex", "approxlg", 12)
                  if os.path.exists(os.path.join(RESULTS, "search_n8mrex.jsonl"))
                  else [])
    nmed_front = front("n8L", "approxlg", 20)
    opx = opact_family()
    rows = []
    for t in args.targets:
        for src, pool in (("Approx-LG (MRE objective, +E)", mrex_front),
                          ("Approx-LG (MRE objective)", mre_front),
                          ("Approx-LG (NMED objective)", nmed_front),
                          ("OPACT", opx)):
            cand = [p for p in pool if 0 < p["MRE"] <= t]
            if not cand:
                continue
            best = min(cand, key=lambda p: p["PDP"])
            nl = (b.build(np.array(best["g"], dtype=np.int32)) if "g" in best
                  else rebuild_opact(best))
            rows.append({"target": t, "source": src, "NMED": best["NMED"],
                         "MRE": best["MRE"], "PDP": best["PDP"],
                         "area": best["area"], "delay": best["delay"],
                         "lut": lut_of(nl)})
    exact = lut_of(b.build(np.zeros(sp.length, dtype=np.int32)))
    images = load_images()
    for r in rows:
        r.update(image_quality(exact, r["lut"], images))
        print(f"MRE<={r['target']:.0e} {r['source']:28s} MRE={r['MRE']:.4f} "
              f"NMED={r['NMED']:.2e} PDP={r['PDP']:6.2f}  blend "
              f"{r['blend_psnr']:6.2f}dB smooth {r['smooth_psnr']:6.2f}dB")
    if args.cnn:
        names = [f"{r['source']}@{r['target']}" for r in rows]
        accs = cnn_eval([None] + [r["lut"] for r in rows], ["exact"] + names,
                        epochs=args.epochs)
        for r, nm in zip(rows, names):
            r["cnn_acc"] = accs.get(nm)
        rows_exact = accs.get("exact")
        print("exact top-1", rows_exact)
        # the exact network is retrained on every call (GPU training is not
        # bit-reproducible), so its accuracy is stored beside the rows
        with open(os.path.join(RESULTS, "apps_mre_exact.json"), "w") as fh:
            json.dump({"cnn_acc": rows_exact}, fh)
        for r in rows:
            print(f"  {r['source']:28s} MRE<={r['target']:.0e} top-1 "
                  f"{r['cnn_acc']}")
    for r in rows:
        r.pop("lut")
    with open(os.path.join(RESULTS, "apps_mre.json"), "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    print("wrote results/apps_mre.json")


if __name__ == "__main__":
    main()
