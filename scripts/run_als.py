"""Characterize the circuits of the ALS baseline in our flow.

run_tcad_als.sh runs ALSRAC (public code of SJTU-ECTL/ALSRAC, built in WSL)
from the exact multiplier of the template under NMED bounds, and ALSRAC
writes the current circuit as it goes, mapped to its own mcnc.genlib.  Each
of them is unmapped by ABC, read back by Yosys with its ports merged
into buses, and then synthesized and simulated exactly like the library
designs (scripts/run_libs.py).  Appends to results/als_n<N>.jsonl; files
already recorded are skipped, and a circuit whose netlist was already
characterized is recorded as a duplicate only; the rounds, CPU time and
circuits of every run go to results/als_runs_n<N>.json.  --prep writes the
input of the runs, the template's exact multiplier as BLIF
(runs/exact<N>.blif).

    source env.sh
    python scripts/run_als.py --prep --n 8      # then run_tcad_als.sh
    python scripts/run_als.py --n 8
    python scripts/run_als.py --n 16 --gpu
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approxlg.netjson import GateNetlist, to_gate_json      # noqa: E402
from approxlg.simulate import _unpack, make_stimulus        # noqa: E402
from approxlg.synth import Synthesizer                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
# ~/ALSRAC inside WSL; run_tcad_als.sh writes runs/n<N>/nmed_<bound>/
ALSRAC = "D:/eda/home/ALSRAC"
# <model>_<ALSRAC's NMED estimate>_<mcnc area>_<mcnc delay>.blif
NAME = re.compile(r"_(?P<err>[^_]+)_(?P<area>[^_]+)_(?P<delay>[^_]+)\.blif$")


def unmap(blif: str, out: str, abc: str, genlib: str) -> bool:
    """ALSRAC's circuits are mapped to mcnc.genlib; turn every gate back into
    its sum of products.  (Strashing to an AIG instead splits the XOR gates,
    and the exact circuit then synthesizes 12% larger in our flow; unmapped
    it is 3% smaller than the template's own exact multiplier.)"""
    cmd = f"read_genlib {genlib}; read_blif {blif}; unmap; write_blif {out}"
    subprocess.run([abc, "-q", cmd], capture_output=True, text=True,
                   timeout=600)
    return os.path.exists(out)


def prep(n: int, runs: str, yosys: str) -> None:
    """The exact multiplier of the template, as the logic network ALSRAC reads."""
    from approxlg.arch import Builder, Space, exact_genome
    sp = Space(n, signed=False)
    os.makedirs(runs, exist_ok=True)
    v = os.path.join(runs, f"exact{n}.v").replace("\\", "/")
    with open(v, "w") as fh:
        fh.write(Builder(sp).build(exact_genome(sp)).to_verilog())
    ys = os.path.join(runs, f"to_blif{n}.ys")
    with open(ys, "w") as fh:
        fh.write(f"read_verilog {v}\nsynth -flatten -top mul\nopt_clean -purge\n"
                 f"write_blif {v[:-2]}.blif\n")
    subprocess.run([yosys, "-q", ys], capture_output=True, text=True,
                   timeout=600)
    print(f"wrote {v[:-2]}.blif")


def to_verilog(blif: str, out: str, yosys: str) -> bool:
    ys = out[:-2] + ".ys"
    with open(ys, "w") as fh:
        fh.write(f"read_blif -wideports {blif}\nhierarchy -auto-top\n"
                 f"rename -top mul\nopt_clean -purge\n"
                 f"write_verilog -noattr {out}\n")
    subprocess.run([yosys, "-q", ys], capture_output=True, text=True,
                   timeout=600)
    return os.path.exists(out)


def metrics(gn: GateNetlist, stim, n: int) -> dict:
    """Unsigned error metrics, as in simulate.error_metrics."""
    xp = stim.xp
    zero = xp.zeros(stim.nvec // 64, dtype=xp.uint64)
    in_bits = {}
    for port, bits in (("a", stim.abits), ("b", stim.bbits)):
        for j, netid in enumerate(gn.ports[port]["bits"]):
            in_bits[netid] = bits[j] if j < n else zero
    env = gn.evaluate(in_bits, stim.nvec, xp)
    prod = xp.zeros(stim.nvec, dtype=xp.int64)
    for j, netid in enumerate(gn.ports["p"]["bits"]):
        if netid in env:
            prod += _unpack(env[netid], stim.nvec, xp).astype(xp.int64) << j
    err = prod - stim.exact
    ae = xp.abs(err)
    nz = stim.exact != 0
    maxp = float((2 ** n - 1) ** 2)
    return dict(MED=float(ae.mean()), NMED=float(ae.mean()) / maxp,
                WCE=float(ae.max()), ER=float((err != 0).mean()),
                MRE=float((ae[nz] / xp.abs(stim.exact[nz])).mean()))


def run_stats(runs: str, n: int) -> dict:
    """Rounds, last error, CPU time and circuits written of every run, from
    ALSRAC's own logs (runs/n<N>/nmed_<bound>.log), which stay in WSL."""
    out = {}
    for log in sorted(glob.glob(os.path.join(runs, f"n{n}", "nmed_*.log"))):
        bound = os.path.basename(log)[5:-4]
        txt = open(log, errors="replace").read()
        errs = re.findall(r"current error = (\S+)", txt)
        cpu = re.findall(r"^time = (\d+)", txt, flags=re.M)
        out[bound] = dict(
            rounds=len(re.findall(r"^-+ round \d+ -+$", txt, flags=re.M)),
            last_error=float(errs[-1]) if errs else 0.0,
            cpu_s=int(cpu[-1]) / 1e6 if cpu else 0.0,     # clock() in us
            written=len(glob.glob(os.path.join(runs, f"n{n}", f"nmed_{bound}",
                                               "*.blif"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--nvec", type=int, default=1 << 22)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--alsrac", default=ALSRAC)
    ap.add_argument("--prep", action="store_true")
    args = ap.parse_args()
    n = args.n
    if args.prep:
        prep(n, os.path.join(args.alsrac, "runs"),
             Synthesizer(cache=None, n=n).yosys)
        return

    out_path = os.path.join(RESULTS, f"als_n{n}.jsonl")
    done, seen = set(), set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    if d.get("ok"):              # failures are retried
                        done.add(d["file"])
                    if d.get("ok") and not d.get("dup"):
                        seen.add(d["hash"])

    syn = Synthesizer(cache=os.path.join(RESULTS, "ppa_cache_als.json"),
                      n=n, power_vectors=1 << 14)
    abc = os.path.join(os.path.dirname(syn.yosys),
                       "yosys-abc" + os.path.splitext(syn.yosys)[1])
    genlib = os.path.join(args.alsrac, "data", "library", "mcnc.genlib")
    stim = make_stimulus(n, args.nvec, gpu=args.gpu, seed=7)
    wd = os.path.join(RESULTS, "als_work", f"n{n}")
    os.makedirs(wd, exist_ok=True)

    runs = os.path.join(args.alsrac, "runs")
    files = sorted(glob.glob(os.path.join(runs, f"n{n}", "nmed_*", "*.blif")))
    print(f"{len(files)} circuits, {len(done)} already recorded")
    for f in files:
        rel = os.path.relpath(f, runs).replace("\\", "/")
        if rel in done or time.time() - os.path.getmtime(f) < 120:
            continue                         # recorded, or still being written
        m = NAME.search(os.path.basename(f))
        rec = dict(lib="ALSRAC", file=rel,
                   bound=float(os.path.basename(os.path.dirname(f))[5:]),
                   als_err=float(m["err"]) if m else None)
        stem = rel[:-5].replace("/", "_")
        try:
            aig = os.path.join(wd, stem + "_aig.blif")
            v = os.path.join(wd, stem + ".v")
            if not (unmap(f, aig, abc, genlib) and to_verilog(aig, v, syn.yosys)):
                raise RuntimeError("conversion failed")
            with open(v) as fh:
                body = re.sub(r"/\*.*?\*/", "", fh.read(), flags=re.S)
            rec["hash"] = hashlib.sha1(body.encode()).hexdigest()[:16]
            if rec["hash"] in seen:
                rec.update(ok=True, dup=True)
            else:
                js = os.path.join(wd, stem + ".json")
                if not to_gate_json([v], "mul", js, syn.yosys):
                    raise RuntimeError("gate-level export failed")
                gn = GateNetlist(js, "mul")
                ppa = syn.run_files([v], "mul", a_port="a", b_port="b")
                if not ppa.ok:
                    raise RuntimeError("synthesis failed")
                rec.update(area=ppa.area, delay=ppa.delay, power=ppa.power,
                           gates=ppa.gates, ADP=ppa.adp, PDP=ppa.pdp)
                rec.update(metrics(gn, stim, n))
                rec["ok"] = True
                seen.add(rec["hash"])
        except Exception as e:
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc(limit=1)
        with open(out_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["ok"] and not rec.get("dup"):
            print(f"{rel:52s} NMED={rec['NMED']:.3e} (ALSRAC {rec['als_err']:.3e}) "
                  f"area={rec['area']:8.1f} delay={rec['delay']:.3f}")
    syn.save_cache()
    stats = run_stats(runs, n)
    with open(os.path.join(RESULTS, f"als_runs_n{n}.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(stats)


if __name__ == "__main__":
    main()
