"""How much do glitches change the energy comparison?

The power model of the paper simulates the mapped netlist at zero delay, so
it counts one transition per net per input change and misses glitches.  This
script re-synthesises a sample of 16x16 unsigned designs -- points of the
Approx-LG front, every verified library design and a spread of the
reproduced compressor-tree designs -- and simulates each mapped netlist again
under a unit-delay model: from the steady state of the previous operand pair,
every gate re-evaluates from its inputs one time step earlier until the
netlist settles, and every change of every net is counted.  Unequal path
depths into a reconvergent gate then produce the glitches a zero-delay model
cannot see.  Dynamic power is recomputed from those counts with the same
capacitances, supply and frequency, and the iso-accuracy power--delay
comparison is repeated with it.

    python scripts/analyze_glitch.py
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tcad", "tabs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approxlg import ctree                                    # noqa: E402
from approxlg.arch import Builder, Space                      # noqa: E402
from approxlg.power import OUTPUT_LOAD_FF, PowerEstimator     # noqa: E402
from approxlg.synth import DEFAULT_LIB, VDD, Synthesizer, abc_script  # noqa
from run_libs import discover                                 # noqa: E402

WORK = os.path.join(RESULTS, "glitch_work")
FREQ = 2e8


def synth_json(files, top, tag, syn, incdirs=()):
    """Map with the paper's recipe and keep the mapped JSON."""
    d = os.path.join(WORK, tag)
    os.makedirs(d, exist_ok=True)
    js = os.path.join(d, "map.json").replace("\\", "/")
    if os.path.exists(js):
        return js
    inc = " ".join("-I" + os.path.abspath(i).replace("\\", "/")
                   for i in incdirs)
    reads = "\n".join("read_verilog " + inc + " " +
                      os.path.abspath(f).replace("\\", "/") for f in files)
    ys = os.path.join(d, "run.ys")
    with open(ys, "w") as fh:
        fh.write(f"""{reads}
hierarchy -check -top {top}
synth -top {top} -flatten
opt -purge
abc -liberty {syn.lib} -script {abc_script()}
setundef -zero
opt_clean -purge
write_json {js}
""")
    subprocess.run([syn.yosys, "-q", ys], capture_output=True, text=True,
                   timeout=1800)
    return js if os.path.exists(js) else None


def pack(bits):
    return np.packbits(np.ascontiguousarray(bits.astype(np.uint8)),
                       bitorder="little").view(np.uint64)


def popcount(w):
    return int(np.bitwise_count(w).sum())


def unit_delay(pe: PowerEstimator, json_path, top, ports, n, npairs=4096,
               seed=7):
    """(zero-delay dynamic uW, unit-delay dynamic uW, leakage uW)."""
    j = json.load(open(json_path))
    mods = j["modules"]
    mod = mods.get(top) or mods.get("\\" + top)
    a_port, b_port = ports[0], ports[1]
    rng = np.random.default_rng(seed)
    hi = 1 << n
    xa, xb = rng.integers(0, hi, npairs), rng.integers(0, hi, npairs)
    ya, yb = rng.integers(0, hi, npairs), rng.integers(0, hi, npairs)
    words = npairs // 64
    ZERO = np.zeros(words, dtype=np.uint64)
    ONE = np.full(words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)

    cells = []
    for c in mod["cells"].values():
        ct = c["type"].lstrip("\\")
        lc = pe.cells.get(ct)
        if lc is None:
            continue
        fns = pe.functions(ct)
        ins = [(pin, c["connections"].get(pin, [None])[0]) for pin in lc.inputs]
        outs = [(opin, c["connections"][opin][0]) for opin in fns
                if c["connections"].get(opin)]
        cells.append((ct, ins, outs, fns))

    def inputs(va, vb):
        env = {"0": ZERO, "1": ONE, "x": ZERO, "z": ZERO}
        for pname, v in ((a_port, va), (b_port, vb)):
            for i, net in enumerate(mod["ports"][pname]["bits"]):
                env[net] = pack((v >> i) & 1) if i < n else ZERO
        return env

    def step(env):
        new = dict(env)
        for ct, ins, outs, fns in cells:
            loc = {pin: (env.get(net, ZERO) if net is not None else ZERO)
                   for pin, net in ins}
            for opin, net in outs:
                new[net] = eval(fns[opin], {}, {"env": loc, "ZERO": ZERO,
                                                "ONE": ONE})
        return new

    def settle(env, count=None):
        for _ in range(400):
            new = step(env)
            changed = False
            for k, v in new.items():
                if k in env and not np.array_equal(v, env[k]):
                    changed = True
                    if count is not None:
                        count[k] = count.get(k, 0) + popcount(v ^ env[k])
                elif k not in env:
                    changed = True
            env = new
            if not changed:
                return env
        return env

    old = settle(inputs(xa, xb))                    # steady state of x
    new_zero = settle(inputs(ya, yb))               # steady state of y
    # unit delay: start from x's state with the inputs switched to y
    start = dict(old)
    start.update({k: v for k, v in inputs(ya, yb).items()})
    count = {}
    for k in start:                                  # input toggles
        if k in old:
            count[k] = popcount(start[k] ^ old[k])
    settle(start, count)

    cap, leak = {}, 0.0
    for c in mod["cells"].values():
        ct = c["type"].lstrip("\\")
        lc = pe.cells.get(ct)
        if lc is None:
            continue
        leak += lc.leakage
        for pin in lc.inputs:
            conn = c["connections"].get(pin)
            if conn:
                cap[conn[0]] = cap.get(conn[0], 0.0) + lc.pins[pin].capacitance
    for p in mod["ports"].values():
        if p["direction"] == "output":
            for net in p["bits"]:
                cap[net] = cap.get(net, 0.0) + OUTPUT_LOAD_FF

    def dyn(toggles):
        e = sum(cap.get(k, 0.0) * t for k, t in toggles.items())
        return 0.5 * e * 1e-15 * VDD ** 2 * FREQ / npairs * 1e6
    zero = {k: popcount(new_zero[k] ^ old[k]) for k in cap if k in old}
    return dyn(zero), dyn(count), leak * 1e-3


def pick_front(n_per=2):
    """Points of the 16x16 unsigned guided front, the cheapest in PDP in
    each half-decade of NMED."""
    pts = []
    for line in open(os.path.join(RESULTS, "search_n16u.jsonl")):
        d = json.loads(line)
        if d["algo"] == "approxlg":
            pts.extend(d["pareto"])
    out = []
    edges = np.logspace(-6, math.log10(0.02), 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [p for p in pts if lo <= p["NMED"] < hi]
        if sub:
            out.append(min(sub, key=lambda p: p["power"] * p["delay"]))
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    syn = Synthesizer(n=16, power=False)
    pe = PowerEstimator(syn.lib, cache=os.path.join(os.path.dirname(syn.lib),
                                                     "nangate45.json"))
    rows = []
    # Approx-LG front points
    sp = Space(16)
    b = Builder(sp)
    for i, p in enumerate(pick_front()):
        v = b.build(np.array(p["g"], dtype=np.int32)).to_verilog()
        src = os.path.join(WORK, f"alg{i}.v")
        open(src, "w").write(v)
        js = synth_json([src], "mul", f"alg{i}", syn)
        rows.append(("Approx-LG", f"ALG16-{i}", js, "mul", ("a", "b"), p))
    # library designs (verified, unsigned, in the window)
    ver = {r["name"]: r["status"] for r in
           json.load(open(os.path.join(RESULTS, "libs_verify.json")))}
    libs = {}
    for l in open(os.path.join(RESULTS, "libs_n16.jsonl")):
        d = json.loads(l)
        if (d.get("ok") and not d.get("signed") and ver.get(d["name"]) == "match"
                and 1e-7 <= d["NMED"] <= 0.02):
            libs[d["name"]] = d
    for d in discover(16):
        r = libs.get(d["name"])
        if r is None:
            continue
        js = synth_json(d["files"], d["top"], "lib_" + d["name"], syn,
                        d.get("incdirs", ()))
        rows.append((r["lib"], r["name"], js, d["top"],
                     tuple(r["ports"][:2]), r))
    # reproduced compressor trees: a spread of the families
    refs = [json.loads(l) for l in open(os.path.join(RESULTS,
                                                     "refgen_n16.jsonl"))]
    refs = [r for r in refs if 1e-7 <= r["NMED"] <= 0.02]
    refs.sort(key=lambda r: r["NMED"])
    keep = refs[::max(1, len(refs) // 16)]
    from approxlg import opact
    for r in keep:
        st = json.loads(r["key"])
        al = ctree.Allocation(n=16, stages=[{int(j): c for j, c in s}
                                            for s in st[:-1]], trunc=st[-1])
        if r["lib"] == "Wang":
            al = ctree.wang(16, r["L"])
        if r["lib"] == "OPACT":
            al.perms, _ = opact.order(al, time_limit=30)
        src = os.path.join(WORK, f"ref_{r['name']}.v".replace("+", "p"))
        open(src, "w").write(ctree.build(al).to_verilog())
        js = synth_json([src], "mul", "ref_" + r["name"].replace("+", "p"), syn)
        rows.append((r["lib"], r["name"], js, "mul", ("a", "b"), r))

    out = []
    for lib, name, js, top, ports, rec in rows:
        if js is None:
            print("!! no netlist for", name)
            continue
        z, u, lk = unit_delay(pe, js, top, ports, 16)
        out.append(dict(lib=lib, name=name, NMED=rec["NMED"],
                        MRE=rec.get("MRE"), delay=rec["delay"],
                        area=rec["area"], dyn_zero=z, dyn_unit=u, leak=lk,
                        glitch=u / z if z > 0 else float("nan")))
        print(f"{lib:12s} {name:24s} NMED {rec['NMED']:.2e}  dyn zero "
              f"{z:7.1f} unit {u:7.1f} uW  x{u / z:5.2f}")
    with open(os.path.join(TABS, "glitch.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # glitch factor per family and the iso-accuracy PDP check with it
    fam = {}
    for r in out:
        fam.setdefault(r["lib"], []).append(r["glitch"])
    for k, v in fam.items():
        print(f"glitch factor {k:12s} n={len(v):2d} median {np.median(v):.2f} "
              f"range {min(v):.2f}-{max(v):.2f}")
    ours = [r for r in out if r["lib"] == "Approx-LG"]
    ref = [r for r in out if r["lib"] != "Approx-LG"]
    for model in ("zero", "unit"):
        wins = 0
        tot = 0
        for r in ref:
            cand = [o for o in ours if o["NMED"] <= r["NMED"] and
                    (o["MRE"] or 0) <= (r["MRE"] or float("inf"))]
            if not cand:
                continue
            tot += 1
            pdp = lambda x: (x[f"dyn_{model}"] + x["leak"]) * x["delay"]
            if min(pdp(o) for o in cand) < pdp(r):
                wins += 1
        print(f"iso-accuracy PDP wins ({model}-delay power, sampled front): "
              f"{wins}/{tot}")


def iso_phase(k_best: int = 4):
    """The comparison done properly: for every sampled reference, the k
    cheapest designs (zero-delay PDP) of the whole pooled 16x16 front that
    are at least as accurate -- on NMED alone, and on NMED and MRE -- are
    re-characterised under unit delay, and the best of them is compared with
    the reference under each power model."""
    syn = Synthesizer(n=16, power=False)
    pe = PowerEstimator(syn.lib, cache=os.path.join(os.path.dirname(syn.lib),
                                                     "nangate45.json"))
    rows = json.load(open(os.path.join(TABS, "glitch.json")))
    refs = [r for r in rows if r["lib"] != "Approx-LG"]
    pts = []
    for line in open(os.path.join(RESULTS, "search_n16u.jsonl")):
        d = json.loads(line)
        if d["algo"] == "approxlg":
            pts.extend(d["pareto"])
    for p in pts:
        p["PDP0"] = p["power"] * p["delay"]
    sp = Space(16)
    b = Builder(sp)
    cache = {}

    def unit(p):
        key = bytes(np.array(p["g"], dtype=np.int8))
        if key not in cache:
            tag = "iso_" + key.hex()[:24]
            src = os.path.join(WORK, tag + ".v")
            open(src, "w").write(b.build(np.array(p["g"], dtype=np.int32))
                                 .to_verilog())
            js = synth_json([src], "mul", tag, syn)
            z, u, lk = unit_delay(pe, js, "mul", ("a", "b"), 16)
            cache[key] = (z, u, lk)
        return cache[key]

    out = []
    for r in refs:
        res = {"lib": r["lib"], "name": r["name"]}
        for match in ("N", "NM"):
            cand = [p for p in pts if p["NMED"] <= r["NMED"] and
                    (match == "N" or p["MRE"] <= (r["MRE"] or 1e9))]
            cand.sort(key=lambda p: p["PDP0"])
            best0 = best1 = None
            for p in cand[:k_best]:
                z, u, lk = unit(p)
                e0 = (z + lk) * p["delay"]
                e1 = (u + lk) * p["delay"]
                best0 = e0 if best0 is None else min(best0, e0)
                best1 = e1 if best1 is None else min(best1, e1)
            r0 = (r["dyn_zero"] + r["leak"]) * r["delay"]
            r1 = (r["dyn_unit"] + r["leak"]) * r["delay"]
            res[match] = None if best0 is None else {
                "win_zero": bool(best0 < r0), "win_unit": bool(best1 < r1),
                "gain_zero": 100 * (1 - best0 / r0),
                "gain_unit": 100 * (1 - best1 / r1)}
        out.append(res)
        print(r["lib"], r["name"], {m: res[m] for m in ("N", "NM")})
    with open(os.path.join(TABS, "glitch_iso.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for match in ("N", "NM"):
        for lib in sorted({r["lib"] for r in out}):
            sub = [r[match] for r in out if r["lib"] == lib and r[match]]
            if not sub:
                continue
            wz = sum(s["win_zero"] for s in sub)
            wu = sum(s["win_unit"] for s in sub)
            gz = np.median([s["gain_zero"] for s in sub])
            gu = np.median([s["gain_unit"] for s in sub])
            print(f"{match:2s} {lib:12s} PDP wins zero-delay {wz}/{len(sub)} "
                  f"(median {gz:+.0f}%), unit-delay {wu}/{len(sub)} "
                  f"(median {gu:+.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--iso":
        iso_phase()
    else:
        main()
