"""Serial cost of one characterisation, per width.

The campaign logs time synthesis as wall clock of a pool of W concurrent
processes, which is the right number for the runs as they were made but
hides what one characterisation costs on its own.  This script re-synthesises
a random sample of characterised designs of each width one at a time, cache
bypassed, on an otherwise idle machine, and reports the serial seconds per
design (synthesis plus the power estimate, exactly as the search pays them).

    python scripts/time_serial.py --widths 8 16 32 --samples 40
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

from approxlg.arch import Builder, Space                    # noqa: E402
from approxlg.synth import Synthesizer                      # noqa: E402

TAG = {8: "n8", 16: "n16u", 32: "n32u"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--batch", type=int, default=96,
                    help="designs for the parallel-throughput measurement")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = {}
    for n in args.widths:
        p = os.path.join(RESULTS, f"designs_{TAG[n]}.jsonl")
        if not os.path.exists(p):
            print(f"n={n}: no designs")
            continue
        recs = [json.loads(l) for l in open(p) if l.strip()]
        random.Random(1).shuffle(recs)
        sp = Space(n)
        syn = Synthesizer(n=n)                      # no cache file
        b = Builder(sp)
        ts = []
        for r in recs[: args.samples]:
            v = b.build(np.array(r["g"], dtype=np.int32)).to_verilog()
            t0 = time.time()
            ppa = syn._synthesize(v, "mul", f"t{len(ts)}")
            dt = time.time() - t0
            if ppa.ok:
                ts.append(dt)
        ts = np.array(ts)
        out[n] = {"n": len(ts), "mean": float(ts.mean()),
                  "median": float(np.median(ts)), "p90": float(np.percentile(ts, 90))}
        print(f"n={n}: {len(ts)} designs, serial s/characterisation mean "
              f"{ts.mean():.2f} median {np.median(ts):.2f} p90 "
              f"{np.percentile(ts, 90):.2f}")
        # the same designs again through the campaigns' pool of W processes:
        # wall clock of the batch over its size is the amortised cost a run pays
        from concurrent.futures import ThreadPoolExecutor
        batch = [b.build(np.array(r["g"], dtype=np.int32)).to_verilog()
                 for r in recs[args.samples: args.samples + args.batch]]
        syn2 = Synthesizer(n=n)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(lambda iv: syn2._synthesize(iv[1], "mul",
                                                          f"p{iv[0]}"),
                              enumerate(batch)))
        dt = time.time() - t0
        okn = sum(r.ok for r in res)
        out[n]["parallel_workers"] = args.workers
        out[n]["parallel_s_per_circuit"] = dt / max(okn, 1)
        print(f"n={n}: {okn} designs on {args.workers} workers, amortised "
              f"{dt / max(okn, 1):.3f} s/characterisation")
    with open(os.path.join(RESULTS, "time_serial.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
