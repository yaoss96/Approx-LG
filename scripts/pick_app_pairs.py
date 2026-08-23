"""Pick iso-cost design pairs (reference method vs Approx-LG) for the
application study: for each selected reference design we take the Approx-LG
design whose power-delay product does not exceed it and whose NMED is lowest."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def front(tags, algos):
    pts = []
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["algo"] in algos:
                    pts.extend(d["pareto"])
    for d in pts:
        d["PDP"] = d["power"] * d["delay"]
        d["ADP"] = d["area"] * d["delay"]
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", nargs="+", default=["n8L", "n8"])
    ap.add_argument("--ours-algos", nargs="+", default=["approxlg"])
    ap.add_argument("--refs", nargs="+", default=["n8"])
    ap.add_argument("--ref-algos", nargs="+", default=["eamva", "biased_ga"])
    ap.add_argument("--pdp", nargs="+", type=float, default=[50, 40, 30, 22])
    ap.add_argument("--metrics", nargs="+", default=["NMED", "MRE"])
    ap.add_argument("--npairs", type=int, default=4)
    ap.add_argument("--match", choices=["cost", "accuracy"], default="cost")
    ap.add_argument("--nmed-lo", type=float, default=1e-5)
    ap.add_argument("--nmed-hi", type=float, default=1e-2)
    ap.add_argument("--out", default=os.path.join(RESULTS, "app_designs.json"))
    args = ap.parse_args()

    ours = front(args.ours, set(args.ours_algos))
    refs = front(args.refs, set(args.ref_algos))
    # neutral selection: reference points evenly spaced in log-NMED over the
    # useful accuracy range, the same rule used for the tables of the paper
    from approxlg.metrics import nondominated_mask
    pool = [d for d in refs if args.nmed_lo <= d["NMED"] <= args.nmed_hi]
    if pool:
        F = np.array([[np.log10(d["NMED"]), d["area"], d["delay"]] for d in pool])
        pool = [d for d, k in zip(pool, nondominated_mask(F)) if k]
        pool.sort(key=lambda d: d["NMED"])
    picks = [pool[int(round(i))]
             for i in np.linspace(0, len(pool) - 1, args.npairs)] if pool else []
    out = []
    used = set()
    for i, r in enumerate(picks):
        k = tuple(r["g"])
        if k in used:
            continue
        used.add(k)
        out.append({"name": f"REF-{len(out)//2+1}", "g": r["g"], "NMED": r["NMED"],
                    "MRE": r.get("MRE"), "area": r["area"], "delay": r["delay"],
                    "power": r["power"], "PDP": r["PDP"], "ADP": r["ADP"],
                    "kind": "reference"})
        mine = [d for d in ours if d["NMED"] > 0]
        if args.match == "cost":
            mine = [d for d in mine if d["PDP"] <= r["PDP"] * (1 + 1e-9)]
            key = lambda d: d["NMED"]
        else:
            for met in args.metrics:
                lim = r.get(met)
                if lim:
                    mine = [d for d in mine
                            if d.get(met, 9e9) <= lim * (1 + 1e-9)]
            key = lambda d: d["PDP"]
        if mine:
            m = min(mine, key=key)
            out.append({"name": f"ALG-{len(out)//2+1}", "g": m["g"],
                        "NMED": m["NMED"], "MRE": m.get("MRE"),
                        "area": m["area"], "delay": m["delay"],
                        "power": m["power"], "PDP": m["PDP"], "ADP": m["ADP"],
                        "kind": "approxlg"})
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for d in out:
        print(f"{d['name']:7s} {d['kind']:9s} NMED={d['NMED']:.3e} "
              f"PDP={d['PDP']:6.2f} ADP={d['ADP']:7.1f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
