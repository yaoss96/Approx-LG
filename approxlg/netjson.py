"""Evaluate *any* Verilog multiplier with the same bit-parallel engine.

Third-party designs (AM-Lib, MulApprox) are mapped by Yosys onto a small set of
simple gate primitives, exported as JSON and then simulated here with the same
packed-bit kernel used for the generated designs.  Using one simulator for our
designs and for the baselines removes any doubt about metric consistency, and
it is orders of magnitude faster than event-driven RTL simulation.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .simulate import Stimulus, _pack, _unpack, xp_of

GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX,NMUX,AOI3,OAI3,AOI4,OAI4"

YS = """
{reads}
hierarchy -check -top {top}
proc; opt -purge
flatten
synth -top {top} -flatten
opt -purge
techmap
abc -g {gates}
opt_clean -purge
write_json {out}
"""


def to_gate_json(files: Sequence[str], top: str, out: str, yosys: str,
                 incdirs: Sequence[str] = ()) -> bool:
    reads = "\n".join(
        "read_verilog " + " ".join(f"-I{os.path.abspath(d)}" for d in incdirs) +
        f" {os.path.abspath(f)}".replace("\\", "/") for f in files)
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    ys = os.path.join(d, f"{top}_json.ys")
    with open(ys, "w") as fh:
        fh.write(YS.format(reads=reads.replace("\\", "/"), top=top,
                           gates=GATES, out=out.replace("\\", "/")))
    r = subprocess.run([yosys, "-q", "-l", ys + ".log", ys],
                       capture_output=True, text=True, timeout=1200)
    return os.path.exists(out)


# --------------------------------------------------------------------------
_EVAL = {
    "$_NOT_":    lambda i: ~i[0],
    "$_AND_":    lambda i: i[0] & i[1],
    "$_NAND_":   lambda i: ~(i[0] & i[1]),
    "$_OR_":     lambda i: i[0] | i[1],
    "$_NOR_":    lambda i: ~(i[0] | i[1]),
    "$_XOR_":    lambda i: i[0] ^ i[1],
    "$_XNOR_":   lambda i: ~(i[0] ^ i[1]),
    "$_ANDNOT_": lambda i: i[0] & ~i[1],
    "$_ORNOT_":  lambda i: i[0] | ~i[1],
    "$_MUX_":    lambda i: (i[0] & ~i[2]) | (i[1] & i[2]),
    "$_NMUX_":   lambda i: ~((i[0] & ~i[2]) | (i[1] & i[2])),
    "$_AOI3_":   lambda i: ~((i[0] & i[1]) | i[2]),
    "$_OAI3_":   lambda i: ~((i[0] | i[1]) & i[2]),
    "$_AOI4_":   lambda i: ~((i[0] & i[1]) | (i[2] & i[3])),
    "$_OAI4_":   lambda i: ~((i[0] | i[1]) & (i[2] | i[3])),
    "$_BUF_":    lambda i: i[0],
}
_IN_ORDER = {
    "$_NOT_": ["A"], "$_BUF_": ["A"],
    "$_MUX_": ["A", "B", "S"], "$_NMUX_": ["A", "B", "S"],
    "$_AOI3_": ["A", "B", "C"], "$_OAI3_": ["A", "B", "C"],
    "$_AOI4_": ["A", "B", "C", "D"], "$_OAI4_": ["A", "B", "C", "D"],
}


class GateNetlist:
    def __init__(self, path: str, top: str):
        with open(path) as fh:
            j = json.load(fh)
        mod = j["modules"][top] if top in j["modules"] else \
            j["modules"]["\\" + top]
        self.ports = mod["ports"]
        self.cells = mod["cells"]
        self.top = top

    def port_names(self) -> Dict[str, dict]:
        return self.ports

    def evaluate(self, in_bits: Dict[str, List], nvec: int, xp) -> Dict[int, object]:
        """in_bits maps net-id -> packed array for every primary-input bit."""
        env: Dict[object, object] = dict(in_bits)
        one = xp.full(nvec // 64, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=xp.uint64)
        zero = xp.zeros(nvec // 64, dtype=xp.uint64)
        env["0"] = zero
        env["1"] = one
        env["x"] = zero
        env["z"] = zero

        pending = list(self.cells.values())
        guard = 0
        while pending and guard < 200:
            guard += 1
            nxt = []
            for c in pending:
                t = c["type"]
                fn = _EVAL.get(t)
                if fn is None:
                    raise ValueError(f"unsupported cell {t}")
                order = _IN_ORDER.get(t, ["A", "B"])
                try:
                    ins = [env[c["connections"][p][0]] for p in order]
                except KeyError:
                    nxt.append(c)
                    continue
                y = c["connections"].get("Y") or c["connections"].get("Q")
                env[y[0]] = fn(ins)
            if len(nxt) == len(pending):
                raise RuntimeError("combinational loop or missing driver")
            pending = nxt
        return env


def eval_verilog_design(files: Sequence[str], top: str, n: int, stim: Stimulus,
                        yosys: str, a_port: str, b_port: str, p_port: str,
                        workdir: str, incdirs: Sequence[str] = (),
                        signed: bool = False) -> Optional[np.ndarray]:
    """Return the product vector produced by a third-party design."""
    out = os.path.join(workdir, f"{top}.json")
    if not os.path.exists(out):
        if not to_gate_json(files, top, out, yosys, incdirs):
            return None
    gn = GateNetlist(out, top)
    xp = stim.xp
    pa = gn.ports[a_port]["bits"]
    pb = gn.ports[b_port]["bits"]
    pp = gn.ports[p_port]["bits"]
    in_bits = {}
    for j, netid in enumerate(pa):
        in_bits[netid] = stim.abits[j] if j < n else stim.abits[n - 1] * 0
    for j, netid in enumerate(pb):
        in_bits[netid] = stim.bbits[j] if j < n else stim.bbits[n - 1] * 0
    env = gn.evaluate(in_bits, stim.nvec, xp)
    prod = xp.zeros(stim.nvec, dtype=xp.int64)
    for j, netid in enumerate(pp):
        if netid in env:
            prod += _unpack(env[netid], stim.nvec, xp).astype(xp.int64) << j
    return prod
