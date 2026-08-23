"""Select a slice of the Approx-LG front for the application study."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["n8L", "n8"])
    ap.add_argument("--algos", nargs="+", default=["approxlg"])
    ap.add_argument("--targets", nargs="+", type=float,
                    default=[1e-5, 5e-5, 2e-4, 1e-3, 3e-3, 1e-2])
    ap.add_argument("--out", default=os.path.join(RESULTS, "app_designs.json"))
    args = ap.parse_args()

    pts = []
    for tag in args.tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["algo"] in args.algos:
                    pts.extend(d["pareto"])
    if not pts:
        print("no designs")
        return
    for d in pts:
        d["PDP"] = d["power"] * d["delay"]
        d["ADP"] = d["area"] * d["delay"]

    out, seen = [], set()
    for i, t in enumerate(args.targets):
        cand = [d for d in pts if 0 < d["NMED"] <= t]
        if not cand:
            continue
        best = min(cand, key=lambda d: d["PDP"])
        key = tuple(best["g"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": f"ALG-{len(out)+1}", "g": best["g"],
                    "NMED": best["NMED"], "MRE": best.get("MRE"),
                    "area": best["area"], "delay": best["delay"],
                    "power": best["power"], "ADP": best["ADP"],
                    "PDP": best["PDP"], "target": t})
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    for d in out:
        print(f"{d['name']:7s} NMED={d['NMED']:.3e} area={d['area']:7.1f} "
              f"delay={d['delay']:.3f} PDP={d['PDP']:6.2f} ADP={d['ADP']:7.1f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
