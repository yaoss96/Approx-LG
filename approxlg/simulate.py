"""Bit-parallel functional simulation of generated multipliers.

Every wire of the netlist is held as a *bit-packed* ``uint64`` vector, so one
elementwise NumPy/CuPy operation evaluates a gate for 64 input vectors at a
time.  For an 8x8 multiplier the whole 65 536-point input space fits in 1024
words per wire, which makes an exhaustive, bit-accurate error characterisation
cost well under a millisecond.  For 16x16 the same kernel runs on the GPU over
a Monte-Carlo sample (or the full space in chunks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .arch import Builder, Netlist, Space

try:                                                       # pragma: no cover
    import cupy as _cp
    _HAS_GPU = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:                                          # pragma: no cover
    _cp, _HAS_GPU = None, False


def xp_of(gpu: bool):
    return _cp if (gpu and _HAS_GPU) else np


def has_gpu() -> bool:
    return _HAS_GPU


# --------------------------------------------------------------------------
# Stimulus
# --------------------------------------------------------------------------
@dataclass
class Stimulus:
    n: int
    a: "np.ndarray"          # int64 operand values
    b: "np.ndarray"
    abits: Dict[int, "np.ndarray"]   # packed uint64 per operand bit
    bbits: Dict[int, "np.ndarray"]
    nvec: int
    exhaustive: bool
    xp: object

    signed: bool = False

    @property
    def sa(self):
        return self.a - ((self.a >> (self.n - 1)) << self.n)

    @property
    def sb(self):
        return self.b - ((self.b >> (self.n - 1)) << self.n)

    @property
    def exact(self):
        if self.signed:
            return self.sa * self.sb
        return self.a.astype(np.int64) * self.b.astype(np.int64)


def _pack(bits, xp):
    """bool vector -> packed uint64 vector (little bit order)."""
    b = xp.ascontiguousarray(bits.astype(xp.uint8))
    return xp.packbits(b, bitorder="little").view(xp.uint64)


def _unpack(word, nvec, xp):
    b = word.view(xp.uint8)
    if xp is np:
        return np.unpackbits(b, bitorder="little", count=nvec).astype(np.uint8)
    return xp.unpackbits(b, bitorder="little")[:nvec].astype(xp.uint8)


def make_stimulus(n: int, nvec: int | None = None, gpu: bool = False,
                  seed: int = 0, signed: bool = False) -> Stimulus:
    xp = xp_of(gpu)
    full = 1 << (2 * n)
    if nvec is None or nvec >= full:
        exhaustive = True
        nvec = full
        idx = xp.arange(nvec, dtype=xp.int64)
        a = idx >> n
        b = idx & ((1 << n) - 1)
    else:
        exhaustive = False
        nvec = int(nvec) // 64 * 64
        if xp is np:
            rng = np.random.default_rng(seed)
            a = rng.integers(0, 1 << n, size=nvec, dtype=np.int64)
            b = rng.integers(0, 1 << n, size=nvec, dtype=np.int64)
        else:
            rs = _cp.random.RandomState(seed)
            a = rs.randint(0, 1 << n, size=nvec, dtype=xp.int64)
            b = rs.randint(0, 1 << n, size=nvec, dtype=xp.int64)
    abits = {j: _pack((a >> j) & 1, xp) for j in range(n)}
    bbits = {i: _pack((b >> i) & 1, xp) for i in range(n)}
    return Stimulus(n, a, b, abits, bbits, nvec, exhaustive, xp, signed)


# --------------------------------------------------------------------------
# Netlist evaluation
# --------------------------------------------------------------------------
_EVAL_GLOBALS: dict = {}


def _ONES(stim, xp):
    return xp.full(stim.nvec // 64, xp.uint64(0xFFFFFFFFFFFFFFFF),
                   dtype=xp.uint64)


def simulate(nl: Netlist, stim: Stimulus):
    """Return the approximate product as an int64 vector."""
    xp = stim.xp
    env: Dict[str, object] = {}
    for st in nl.stmts:
        if st.kind == "and":
            j = int(st.ins[0][2:-1]); i = int(st.ins[1][2:-1])
            env[st.outs[0]] = stim.abits[j] & stim.bbits[i]
        elif st.kind == "nand":
            j = int(st.ins[0][2:-1]); i = int(st.ins[1][2:-1])
            env[st.outs[0]] = ~(stim.abits[j] & stim.bbits[i])
        elif st.kind == "const":
            env[st.outs[0]] = _ONES(stim, xp)
        elif st.kind == "or":
            acc = env[st.ins[0]]
            for w in st.ins[1:]:
                acc = acc | env[w]
            env[st.outs[0]] = acc
        else:
            local = {nm: env[w] for nm, w in zip("abcd", st.ins)}
            for o, (_, expr) in zip(st.outs, st.cell.outs):
                env[o] = eval(expr, _EVAL_GLOBALS, local)  # noqa: S307

    n, m = nl.n, 2 * nl.n
    lo = nl.trunc + nl.cpa_k
    hi_w = m - lo
    zero = xp.zeros(stim.nvec, dtype=xp.uint8)

    def bit(row, j):
        w = row.get(j)
        return zero if w is None else _unpack(env[w], stim.nvec, xp)

    prod = xp.zeros(stim.nvec, dtype=xp.int64)
    # OR-based (LOA) part
    for j in range(nl.trunc, lo):
        prod += (bit(nl.row_a, j) | bit(nl.row_b, j)).astype(xp.int64) << j
    # exact carry-propagate part (wraps at hi_w bits, exactly like the RTL)
    hi = xp.zeros(stim.nvec, dtype=xp.int64)
    for j in range(lo, m):
        hi += bit(nl.row_a, j).astype(xp.int64) << (j - lo)
        hi += bit(nl.row_b, j).astype(xp.int64) << (j - lo)
    if nl.cpa_k > 0:
        hi += (bit(nl.row_a, lo - 1) & bit(nl.row_b, lo - 1)).astype(xp.int64)
    hi += nl.const_add >> lo if lo else nl.const_add
    hi &= (1 << hi_w) - 1
    prod += hi << lo
    return prod & ((1 << m) - 1)


# --------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------
def error_metrics(approx, stim: Stimulus) -> Dict[str, float]:
    xp = stim.xp
    n = stim.n
    m = 2 * n
    exact = stim.exact
    approx = approx.astype(xp.int64)
    if stim.signed:                       # interpret the product as two's compl.
        approx = approx - ((approx >> (m - 1)) << m)
        maxp = float(2 ** (m - 2))
    else:
        maxp = float((2 ** n - 1) ** 2)
    err = approx - exact
    aerr = xp.abs(err)
    nz = exact != 0
    med = float(aerr.mean())
    res = {
        "MED": med,
        "NMED": med / maxp,
        "WCE": float(aerr.max()),
        "NWCE": float(aerr.max()) / maxp,
        "ER": float((err != 0).mean()),
        "MRE": float((aerr[nz] / xp.abs(exact[nz])).mean()),
        "MSE": float((err.astype(xp.float64) ** 2).mean()),
        "MED_signed": float(err.mean()),
    }
    res["RMSE"] = res["MSE"] ** 0.5
    res["NRMSE"] = res["RMSE"] / maxp
    return res


class Evaluator:
    """Caches stimulus + builder; evaluates a genome's error metrics."""

    def __init__(self, space: Space, nvec: int | None = None, gpu: bool = False,
                 seed: int = 0):
        self.space = space
        self.builder = Builder(space)
        self.stim = make_stimulus(space.n, nvec, gpu, seed,
                                  getattr(space, "signed", False))

    def evaluate(self, g) -> Tuple[Dict[str, float], Netlist]:
        nl = self.builder.build(g)
        prod = simulate(nl, self.stim)
        return error_metrics(prod, self.stim), nl
