"""Pick a matched-accuracy published design for every Approx-LG design that
enters the application table.

The application study is only meaningful against a reference of comparable
accuracy, so for each Approx-LG point we take the verified third-party
8x8 design whose NMED is closest to it in log space, preferring the cheapest
one when several are equally close.  Only designs that reproduce their own RTL
bit-exactly and behave as plain unsigned multipliers are eligible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", default=os.path.join(RESULTS, "app_designs.json"))
    ap.add_argument("--libs", default=os.path.join(RESULTS, "libs_n8.jsonl"))
    ap.add_argument("--verify",
                    default=os.path.join(RESULTS, "libs_verify_n8.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "app_refs.json"))
    ap.add_argument("--nmed-max", type=float, default=0.10)
    args = ap.parse_args()

    ours = json.load(open(args.apps))
    ok = {r["name"] for r in json.load(open(args.verify))
          if r["status"] == "match"}

    refs = []
    with open(args.libs) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("ok") or r["name"] not in ok:
                continue
            if r.get("signed") or not (0 < r["NMED"] <= args.nmed_max):
                continue
            r["PDP"] = r["power"] * r["delay"]
            r["ADP"] = r["area"] * r["delay"]
            refs.append(r)
    if not refs:
        print("no eligible reference designs")
        return

    out, used = [], set()
    for d in ours:
        cand = [r for r in refs if r["name"] not in used]
        if not cand:
            break
        best = min(cand, key=lambda r: (abs(math.log10(r["NMED"])
                                            - math.log10(d["NMED"])),
                                        r["PDP"]))
        used.add(best["name"])
        out.append({"name": best["name"], "lib": best["lib"],
                    "match": d["name"], "NMED": best["NMED"],
                    "MRE": best.get("MRE"), "area": best["area"],
                    "delay": best["delay"], "power": best["power"],
                    "ADP": best["ADP"], "PDP": best["PDP"],
                    "json": os.path.join(RESULTS, "libjson",
                                         f"{best['name']}.json"),
                    "top": best.get("top", best["name"])})
        print(f"{d['name']:7s} NMED={d['NMED']:.3e}  ->  "
              f"{best['name']:18s} ({best['lib']}) NMED={best['NMED']:.3e} "
              f"PDP={best['PDP']:.1f}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
