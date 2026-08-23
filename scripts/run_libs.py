"""Characterise the published RTL of AM-Lib and MulApprox in *our* flow.

Every third-party design is (a) synthesised with the same Yosys/ABC/Nangate45
recipe used for the generated designs and (b) simulated with the same
packed-bit gate-level kernel, so the resulting accuracy/PPA points are directly
comparable with the Approx-LG Pareto front.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approxlg.netjson import GateNetlist, to_gate_json      # noqa: E402
from approxlg.simulate import _unpack, make_stimulus        # noqa: E402
from approxlg.synth import Synthesizer                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
RESULTS = os.path.join(ROOT, "results")
AMLIB = os.path.join(REPO, "AM-Lib-main", "AM-Lib-main")
MULAPPROX = os.path.join(REPO, "MulApprox-main", "MulApprox-main", "verilog")
EVOAPPROX = os.path.join(REPO, "evoapproxlib", "multiplers")

SKIP = {"rad4_16bit"}          # sequential design, outside the combinational scope


def discover(n: int = 16):
    out = []
    for src in sorted(glob.glob(os.path.join(AMLIB, "**", "src"),
                                recursive=True)):
        files = sorted(glob.glob(os.path.join(src, "*.v")))
        if not files:
            continue
        rel = os.path.relpath(os.path.dirname(src), AMLIB).replace("\\", "/")
        out.append(dict(lib="AM-Lib", name=rel.replace("/", "_"), top="top",
                        files=files, incdirs=[src]))
    for f in sorted(glob.glob(os.path.join(MULAPPROX, "**", "*.v"),
                              recursive=True)):
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in SKIP:
            continue
        out.append(dict(lib="MulApprox", name=stem, top=stem, files=[f],
                        incdirs=[os.path.dirname(f)]))
    # EvoApproxLib: the same circuit appears in several per-objective Pareto
    # sub-sets, so deduplicate by module name.  The *_pdk45.v variants are
    # already mapped to a specific technology and would bypass our synthesis
    # recipe, so only the generic sources are taken.
    seen = set()
    for sub in (f"{n}x{n}_unsigned", f"{n}x{n}_signed"):
        for f in sorted(glob.glob(os.path.join(EVOAPPROX, sub, "*", "*.v"))):
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem.endswith("_pdk45") or stem in seen:
                continue
            seen.add(stem)
            out.append(dict(lib="EvoApproxLib", name=stem, top=stem, files=[f],
                            incdirs=[os.path.dirname(f)]))
    return out


def pick_ports(gn: GateNetlist, n: int):
    """Accept only a plain n x n integer multiplier interface.

    Designs with a different signature (the floating-point mantissa/exponent
    multipliers of AM-Lib, for instance) are outside the scope of this
    comparison and are reported as skipped rather than silently mis-evaluated.
    """
    ins, outs = [], []
    for nm, p in gn.ports.items():
        (ins if p["direction"] == "input" else outs).append((nm, len(p["bits"])))
    outs = sorted(outs, key=lambda x: -x[1])
    if len(ins) != 2 or len(outs) != 1:
        return None
    if any(w < n for _, w in ins) or outs[0][1] < 2 * n - 1:
        return None
    return ins[0][0], ins[1][0], outs[0][0]


def evaluate(gn: GateNetlist, stim, n: int, a_p, b_p, p_p):
    xp = stim.xp
    zero = xp.zeros(stim.nvec // 64, dtype=xp.uint64)
    in_bits = {}
    for port, bits in ((a_p, stim.abits), (b_p, stim.bbits)):
        for j, netid in enumerate(gn.ports[port]["bits"]):
            in_bits[netid] = bits[j] if j < n else zero
    env = gn.evaluate(in_bits, stim.nvec, xp)
    pbits = gn.ports[p_p]["bits"]
    prod = xp.zeros(stim.nvec, dtype=xp.int64)
    for j, netid in enumerate(pbits):
        if netid in env:
            prod += _unpack(env[netid], stim.nvec, xp).astype(xp.int64) << j
    w = len(pbits)
    exact = stim.exact
    maxp = float((2 ** n - 1) ** 2)

    def metrics(pr, ex, mp):
        err = pr - ex
        ae = xp.abs(err)
        nz = ex != 0
        return dict(MED=float(ae.mean()), NMED=float(ae.mean()) / mp,
                    WCE=float(ae.max()), ER=float((err != 0).mean()),
                    MRE=float((ae[nz] / xp.abs(ex[nz])).mean()),
                    RMSE=float(((err.astype(xp.float64) ** 2).mean()) ** 0.5))

    # The output encoding of a third-party design is not documented in a
    # machine-readable way, so all three encodings actually used in the two
    # libraries are tried and the one that fits the design is reported.
    sa = stim.a - ((stim.a >> (n - 1)) << n)
    sb = stim.b - ((stim.b >> (n - 1)) << n)
    sp_ = prod - ((prod >> (w - 1)) << w)
    smax = float(2 ** (2 * n - 2))
    cands = [("unsigned", metrics(prod, exact, maxp)),
             ("twos", metrics(sp_, sa * sb, smax)),
             ("ones", metrics(sp_ + (sp_ < 0), sa * sb, smax))]
    name, m = min(cands, key=lambda kv: kv[1]["NMED"])
    m["encoding"] = name
    m["signed"] = name != "unsigned"
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--nvec", type=int, default=1 << 22)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    out_path = args.out or os.path.join(RESULTS, f"libs_n{args.n}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["name"])

    syn = Synthesizer(cache=os.path.join(RESULTS, "ppa_cache_libs.json"),
                      n=args.n, power_vectors=1 << 14)
    stim = make_stimulus(args.n, args.nvec, gpu=args.gpu, seed=7)
    wd = os.path.join(RESULTS, "libjson")
    os.makedirs(wd, exist_ok=True)

    designs = discover(args.n)
    print(f"{len(designs)} designs discovered; yosys={syn.yosys}")
    for d in designs:
        if d["name"] in done:
            continue
        rec = dict(lib=d["lib"], name=d["name"], top=d["top"])
        try:
            js = os.path.join(wd, f"{d['name']}.json")
            if not os.path.exists(js):
                to_gate_json(d["files"], d["top"], js, syn.yosys, d["incdirs"])
            gn = GateNetlist(js, d["top"])
            ports = pick_ports(gn, args.n)
            if ports is None:
                raise ValueError("cannot identify operand/product ports")
            rec["ports"] = ports
            ppa = syn.run_files(d["files"], d["top"], d["incdirs"],
                                a_port=ports[0], b_port=ports[1])
            rec.update(area=ppa.area, delay=ppa.delay, power=ppa.power,
                       gates=ppa.gates, ADP=ppa.adp, PDP=ppa.pdp)
            rec.update(evaluate(gn, stim, args.n, *ports))
            rec["ok"] = True
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=1)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{rec['lib']:10s} {rec['name']:16s} "
              + (f"area={rec.get('area', float('nan')):8.1f} "
                 f"delay={rec.get('delay', float('nan')):6.3f} "
                 f"NMED={rec.get('NMED', float('nan')):.3e} "
                 f"signed={rec.get('signed')}" if rec["ok"]
                 else f"FAILED {rec.get('error')}"))
    syn.save_cache()


if __name__ == "__main__":
    main()
