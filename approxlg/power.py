"""Post-mapping, activity-based power estimation.

The mapped netlist produced by ABC is simulated gate by gate with the Liberty
Boolean functions over a sequence of uniformly random operand pairs.  The
toggle rate of every net is measured, the load capacitance of every net is
summed from the Liberty input-pin capacitances of its sinks, and

    P_dyn  = 0.5 * V^2 * f * sum_nets C_load(net) * alpha(net)
    P_leak = sum_cells cell_leakage_power

This is the textbook switching-activity model; it is applied identically to the
generated designs and to the third-party reference designs, so all reported
power / PDP numbers are produced by one estimator.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .liberty import LibCell, compile_function, parse_liberty

VDD = 1.1
DEFAULT_FREQ = 2e8
OUTPUT_LOAD_FF = 4.0          # fF on every primary output


class PowerStimulus:
    """A random operand sequence held as packed bit-vectors."""

    def __init__(self, n: int, nvec: int = 1 << 14, seed: int = 12345):
        rng = np.random.default_rng(seed)
        self.n = n
        self.nvec = int(nvec) // 64 * 64
        a = rng.integers(0, 1 << n, size=self.nvec, dtype=np.int64)
        b = rng.integers(0, 1 << n, size=self.nvec, dtype=np.int64)
        self.a, self.b = a, b
        self.abits = [_pack((a >> j) & 1) for j in range(n)]
        self.bbits = [_pack((b >> j) & 1) for j in range(n)]
        self.words = self.nvec // 64


def _pack(bits) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(bits.astype(np.uint8)),
                       bitorder="little").view(np.uint64)


def _shift1(w: np.ndarray) -> np.ndarray:
    """Shift the packed bit-stream right by one vector position."""
    out = w >> np.uint64(1)
    out[:-1] |= (w[1:] << np.uint64(63))
    return out


def toggle_rate(w: np.ndarray, nvec: int) -> float:
    x = w ^ _shift1(w)
    x[-1] &= np.uint64((1 << 63) - 1)          # last vector has no successor
    return float(np.bitwise_count(x).sum()) / (nvec - 1)


@dataclass
class PowerReport:
    dynamic_uw: float
    leakage_uw: float
    total_uw: float
    nets: int
    cells: int


class PowerEstimator:
    def __init__(self, lib_path: str, cache: Optional[str] = None,
                 freq: float = DEFAULT_FREQ, vdd: float = VDD):
        self.cells: Dict[str, LibCell] = parse_liberty(lib_path, cache)
        self.freq = freq
        self.vdd = vdd
        self._fn_cache: Dict[str, Dict[str, str]] = {}

    def functions(self, cell_type: str) -> Dict[str, str]:
        if cell_type not in self._fn_cache:
            c = self.cells[cell_type]
            self._fn_cache[cell_type] = {
                p.name: compile_function(p.function)
                for p in c.pins.values() if p.direction == "output" and p.function}
        return self._fn_cache[cell_type]

    # ------------------------------------------------------------------
    def simulate(self, json_path: str, top: str, stim: PowerStimulus,
                 a_port: str = "a", b_port: str = "b"):
        """Return (module, net -> packed value) for the mapped netlist."""
        try:
            with open(json_path) as fh:
                j = json.load(fh)
        except Exception:
            return None, None
        mods = j.get("modules", {})
        mod = mods.get(top) or mods.get("\\" + top)
        if mod is None:
            return None, None
        env = self._eval(mod, stim, a_port, b_port)
        return mod, env

    def product(self, mod, env, stim: PowerStimulus, p_port: str = "p"):
        from .simulate import _unpack
        out = np.zeros(stim.nvec, dtype=np.int64)
        for i, net in enumerate(mod["ports"][p_port]["bits"]):
            w = env.get(net)
            if w is not None:
                out += _unpack(w, stim.nvec, np).astype(np.int64) << i
        return out

    def estimate(self, json_path: str, top: str, stim: PowerStimulus,
                 a_port: str = "a", b_port: str = "b") -> Optional[PowerReport]:
        mod, env = self.simulate(json_path, top, stim, a_port, b_port)
        if mod is None:
            return None
        return self._power(mod, env, stim)

    def _eval(self, mod, stim: PowerStimulus, a_port: str, b_port: str):
        words = stim.words
        ZERO = np.zeros(words, dtype=np.uint64)
        ONE = np.full(words, np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        env: Dict[object, np.ndarray] = {"0": ZERO, "1": ONE,
                                         "x": ZERO, "z": ZERO}
        for pname, bits in ((a_port, stim.abits), (b_port, stim.bbits)):
            p = mod["ports"].get(pname)
            if p is None:
                return env
            for i, net in enumerate(p["bits"]):
                env[net] = bits[i] if i < len(bits) else ZERO

        cells = list(mod["cells"].values())
        # ---- levelised evaluation -------------------------------------
        pending = cells
        guard = 0
        while pending and guard < 400:
            guard += 1
            nxt = []
            for c in pending:
                ct = c["type"].lstrip("\\")
                lc = self.cells.get(ct)
                if lc is None:
                    continue
                fns = self.functions(ct)
                ins = {}
                ready = True
                for pin in lc.inputs:
                    conn = c["connections"].get(pin)
                    if conn is None:
                        ins[pin] = ZERO
                        continue
                    net = conn[0]
                    if net not in env:
                        ready = False
                        break
                    ins[pin] = env[net]
                if not ready:
                    nxt.append(c)
                    continue
                for opin, expr in fns.items():
                    conn = c["connections"].get(opin)
                    if not conn:
                        continue
                    env[conn[0]] = eval(expr, {},
                                        {"env": ins, "ZERO": ZERO, "ONE": ONE})
            if len(nxt) == len(pending):
                break
            pending = nxt
        return env

    def _power(self, mod, env, stim: PowerStimulus) -> PowerReport:
        cells = list(mod["cells"].values())
        # ---- load capacitance per net ---------------------------------
        cap: Dict[object, float] = {}
        leak = 0.0
        for c in cells:
            ct = c["type"].lstrip("\\")
            lc = self.cells.get(ct)
            if lc is None:
                continue
            leak += lc.leakage
            for pin in lc.inputs:
                conn = c["connections"].get(pin)
                if conn:
                    cap[conn[0]] = cap.get(conn[0], 0.0) + lc.pins[pin].capacitance
        for pname, p in mod["ports"].items():
            if p["direction"] == "output":
                for net in p["bits"]:
                    cap[net] = cap.get(net, 0.0) + OUTPUT_LOAD_FF

        # ---- dynamic power ---------------------------------------------
        dyn = 0.0
        counted = 0
        for net, cl in cap.items():
            w = env.get(net)
            if w is None or cl <= 0:
                continue
            a = toggle_rate(w, stim.nvec)
            dyn += 0.5 * (cl * 1e-15) * self.vdd ** 2 * self.freq * a
            counted += 1
        return PowerReport(dynamic_uw=dyn * 1e6, leakage_uw=leak * 1e-3,
                           total_uw=dyn * 1e6 + leak * 1e-3,
                           nets=counted, cells=len(cells))
