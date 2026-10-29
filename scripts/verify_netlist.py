"""Cross-check the functional model against the *mapped* netlist.

Each sampled design is synthesised with the paper's recipe onto Nangate45,
and the mapped netlist is evaluated with the Liberty functions of its cells
(the evaluator of the power model) on random operand pairs; every product
must equal the packed-bit model's.  Complements scripts/verify_rtl.py, which
checks the emitted RTL with Icarus Verilog.

    python scripts/verify_netlist.py --n 32 --designs 4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from approxlg.arch import Builder, Space                    # noqa: E402
from approxlg.power import PowerEstimator, PowerStimulus    # noqa: E402
from approxlg.simulate import make_stimulus, simulate       # noqa: E402
from approxlg.synth import Synthesizer                      # noqa: E402
from analyze_glitch import synth_json                       # noqa: E402

WORK = os.path.join(ROOT, "results", "netlist_check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--designs", type=int, default=4)
    ap.add_argument("--vectors", type=int, default=4096)
    args = ap.parse_args()
    n = args.n
    os.makedirs(WORK, exist_ok=True)
    sp = Space(n)
    b = Builder(sp)
    syn = Synthesizer(n=n, power=False)
    pe = PowerEstimator(syn.lib, cache=os.path.join(os.path.dirname(syn.lib),
                                                     "nangate45.json"))
    pst = PowerStimulus(n, args.vectors, seed=11)
    stim = make_stimulus(n, args.vectors, seed=0)
    # drive the model with exactly the operands of the netlist stimulus
    stim.a, stim.b = pst.a.copy(), pst.b.copy()
    from approxlg.simulate import _pack
    stim.abits = {j: _pack((stim.a >> j) & 1, np) for j in range(n)}
    stim.bbits = {i: _pack((stim.b >> i) & 1, np) for i in range(n)}
    rng = np.random.default_rng(5)
    ok = 0
    for d in range(args.designs):
        g = sp.random(rng) if d else np.zeros(sp.length, dtype=np.int32)
        nl = b.build(g)
        src = os.path.join(WORK, f"n{n}_d{d}.v")
        open(src, "w").write(nl.to_verilog())
        js = synth_json([src], "mul", f"n{n}_d{d}", syn)
        mod, env = pe.simulate(js, "mul", pst)
        mapped = pe.product(mod, env, pst).view(np.uint64)
        model = np.asarray(simulate(nl, stim)).astype(np.uint64)
        bad = int((mapped != model).sum())
        print(f"n={n} design {d}: T={g[0]} A={g[1]} k={g[3]} -> "
              f"{'OK' if bad == 0 else f'FAIL({bad})'}")
        ok += bad == 0
    print(f"{ok}/{args.designs} mapped netlists bit-exact")


if __name__ == "__main__":
    main()
