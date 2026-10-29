"""OPACT: optimisation of the approximate compressor tree (Xiao, Zhuo and Qian,
DATE'22), re-implemented from the paper as a baseline.

Two stages, as in the paper:

1. *Compressor allocation* (Section III-A).  An ILP chooses, for every stage i
   and column j, the numbers h, f, p, q of exact 2:2, exact 3:2, approximate
   3:2 and approximate 4:2 compressors (the approximate ones are those of
   Esposito et al. with equally weighted outputs).  The bit counts of the next
   bit matrix follow (16)-(17), every column must end with at most two bits
   (18), and the left-most column hosts no exact compressor (12).  The
   objective is one of the three modes of the paper: CoOpt minimises
   alpha*H + beta*F + gamma*P + delta*Q + w*D, AreaOpt bounds D and minimises
   the area term, ErrOpt bounds the area term and minimises D, where D is the
   rough MED estimate (10)-(11) with every input bit at probability 1/4.

2. *Connection order* (Section III-B).  Given the allocation, decide which
   bit of each column drives which compressor pin so as to minimise the
   improved MED estimate, in which the probability of every bit is propagated
   from stage to stage through the compressors' output probabilities
   (Example 1, (21)-(22)).  OPACT writes this as one MIP.  We solve that MIP
   with SCIP where it is small enough to close (8x8) and otherwise optimise
   the same objective by a greedy assignment followed by pairwise swaps
   (``order``); :func:`order_mip` is the exact model and the two are compared
   in ``scripts/run_opact.py``.

The paper used Gurobi; the formulation does not depend on the solver.  The
area constants are the Nangate45 cell areas of each compressor's gate
implementation, the counterpart of the paper's per-compressor Design Compiler
areas on the same library.
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .ctree import (ESP_CELLS, Allocation, initial_heights, kind_cell,
                    kind_inputs)
from .cells import CELL_BY_NAME

# Nangate45 X1 areas (um^2) of the gate implementation of each compressor:
# HA_X1; FA_X1; approx 3:2 = AOI21+INV and OR2; approx 4:2 = 2x(AOI211+INV).
AREA = {"HA": 2.660, "FA": 4.256, "E32": 2.660, "E42": 3.724}
# mean error of the approximate compressors at input probability 1/4, (5)-(6)
MED_P = {"E32": 1 / 64, "E42": 7 / 128}
KINDS = ("HA", "FA", "E32", "E42")


# ---------------------------------------------------------------------------
# Stage 1: compressor allocation
# ---------------------------------------------------------------------------
def allocate(n: int, s: int, mode: str = "co", param: float = 1.0,
             time_limit: float = 120.0, solver: str = "scip"
             ) -> Optional[Allocation]:
    """Solve the allocation ILP; returns None if infeasible."""
    K = 2 * n - 1
    V0 = initial_heights(n)
    if solver == "gurobi":                                   # pragma: no cover
        return _allocate_gurobi(n, s, mode, param, time_limit)
    from pyscipopt import Model, quicksum

    m = Model("opact_alloc")
    m.hideOutput()
    m.setParam("limits/time", time_limit)
    x = {}
    for i in range(s):
        for j in range(K):
            for k in KINDS:
                ub = 0 if (j == K - 1 and k in ("HA", "FA")) else None  # (12)
                x[i, j, k] = m.addVar(vtype="I", lb=0, ub=ub,
                                      name=f"{k}_{i}_{j}")
    V = {}
    for i in range(1, s + 1):
        for j in range(K):
            V[i, j] = m.addVar(vtype="I", lb=0,
                               ub=(2 if i == s else None),            # (18)
                               name=f"V_{i}_{j}")

    def Vij(i, j):
        return V0[j] if i == 0 else V[i, j]

    for i in range(s):
        for j in range(K):
            h, f, p, q = (x[i, j, k] for k in KINDS)
            m.addCons(2 * h + 3 * f + 3 * p + 4 * q <= Vij(i, j))     # (15)
            carry = (x[i, j - 1, "HA"] + x[i, j - 1, "FA"]) if j else 0
            m.addCons(V[i + 1, j] ==
                      Vij(i, j) - (h + 2 * f + p + 2 * q) + carry)   # (16)-(17)
    area = quicksum(AREA[k] * x[i, j, k] for i in range(s) for j in range(K)
                    for k in KINDS)
    # Above 16 bits the column weights 2^j reach 2^62, beyond what the
    # solver's tolerances can handle: ErrOpt then finds no solution at all,
    # not even the exact tree.  The error is therefore modelled in units of
    # 2^(2n-2) there (the meaning of `param` and of meta["D"] is unchanged);
    # at 8 and 16 bits the model is exactly the one of the paper.
    scale = 1.0 if n <= 16 else 2.0 ** -(2 * n - 2)
    D = quicksum((2 ** j * scale) * (MED_P["E32"] * x[i, j, "E32"] +
                                     MED_P["E42"] * x[i, j, "E42"])
                 for i in range(s) for j in range(K))
    if mode == "co":                                          # CoOpt, (7)
        m.setObjective(area + (param / scale) * D, "minimize")
    elif mode == "area":                                      # AreaOpt
        m.addCons(D <= param * scale)
        m.setObjective(area, "minimize")
    elif mode == "err":                                       # ErrOpt
        m.addCons(area <= param)
        m.setObjective(D, "minimize")
    else:
        raise ValueError(mode)
    m.optimize()
    if m.getNSols() == 0:
        return None
    sol = m.getBestSol()
    alloc = Allocation(n=n, name=f"OPACT{n}-s{s}-{mode}-{param:.4g}")
    for i in range(s):
        st: Dict[int, List[str]] = {}
        for j in range(K):
            comps = []
            for k in KINDS:
                comps += [k] * int(round(m.getSolVal(sol, x[i, j, k])))
            if comps:
                st[j] = comps
        alloc.stages.append(st)
    alloc.meta = {"status": m.getStatus(),
                  "D": float(m.getSolVal(sol, D)) / scale,
                  "area_est": float(m.getSolVal(sol, area)), "s": s,
                  "mode": mode, "param": param}
    return alloc


def _allocate_gurobi(n, s, mode, param, time_limit):          # pragma: no cover
    raise NotImplementedError(
        "gurobipy needs a valid licence on this machine; use solver='scip'")


def min_stages(n: int) -> int:
    """Fewest stages in which the allocation ILP is feasible."""
    for s in range(1, 12):
        if allocate(n, s, "co", 1e-9, time_limit=30) is not None:
            return s
    raise RuntimeError("no feasible stage count")


# ---------------------------------------------------------------------------
# Stage 2: connection order under the probability-propagation model
# ---------------------------------------------------------------------------
def _multilinear(cell_outs, n_in):
    """Output probabilities and the expected error of a cell as functions of
    its input probabilities, by enumeration of the truth table (exact under
    the independence assumption the paper makes)."""
    import numpy as np

    pats = list(itertools.product((0, 1), repeat=n_in))

    def evaluate(expr, bits):
        # numpy booleans give ~ & | ^ their logical meaning
        env = {nm: np.bool_(b) for nm, b in zip("abcdef", bits)}
        return bool(eval(expr, {}, env))  # noqa: S307 - trusted literals
    tables = [[evaluate(e, b) for b in pats] for _, e in cell_outs]
    return pats, tables


class _CellModel:
    def __init__(self, kind: str):
        cell = kind_cell(kind)
        self.kind = kind
        self.n_in = cell.n_in
        self.weights = [w for w, _ in cell.outs]
        self.pats, self.tables = _multilinear(cell.outs, cell.n_in)
        # error distance of every input pattern (exact count - approx value)
        self.err = []
        for b, *vals in zip(self.pats, *self.tables):
            exact = sum(b)
            approx = sum(w * int(v) for w, v in zip(self.weights, vals)
                         if w == 1)
            approx += sum(2 * int(v) for w, v in zip(self.weights, vals)
                          if w == 2)
            self.err.append(exact - approx)

    def _pr(self, probs):
        out = []
        for b in self.pats:
            pr = 1.0
            for x, p in zip(b, probs):
                pr *= p if x else (1.0 - p)
            out.append(pr)
        return out

    def outputs(self, probs) -> List[float]:
        w = self._pr(probs)
        return [sum(pp for pp, v in zip(w, tab) if v) for tab in self.tables]

    def med(self, probs) -> float:
        w = self._pr(probs)
        return sum(pp * e for pp, e in zip(w, self.err))


_MODELS: Dict[str, _CellModel] = {}


def _model(kind):
    if kind not in _MODELS:
        _MODELS[kind] = _CellModel(kind)
    return _MODELS[kind]


def evaluate_order(alloc: Allocation, perms) -> Tuple[float, list]:
    """Improved MED estimate of an allocation under a connection order.

    Bits are tracked in the same source order :func:`approxlg.ctree.build`
    uses, so ``perms`` can be handed to the builder unchanged."""
    n = alloc.n
    K = 2 * n - 1
    cols = [[0.25] * h for h in initial_heights(n)]
    total = 0.0
    for s, stage in enumerate(alloc.stages):
        carry_ha = [[] for _ in range(K)]
        carry_fa = [[] for _ in range(K)]
        sum_ha = [[] for _ in range(K)]
        sum_fa = [[] for _ in range(K)]
        apx = [[] for _ in range(K)]
        thru = [[] for _ in range(K)]
        for j in range(K):
            bits = cols[j]
            perm = perms[s].get(j) if perms and s < len(perms) else None
            pins = [bits[r] for r in perm] if perm is not None else list(bits)
            pos = 0
            for k in stage.get(j, []):
                mdl = _model(k)
                pr = pins[pos:pos + mdl.n_in]
                pos += mdl.n_in
                outs = mdl.outputs(pr)
                if k in ("HA", "FA"):
                    (sum_ha if k == "HA" else sum_fa)[j].append(outs[0])
                    if j + 1 < K:
                        (carry_ha if k == "HA" else carry_fa)[j + 1].append(outs[1])
                else:
                    total += (2 ** j) * mdl.med(pr)
                    apx[j].extend(outs)
            thru[j].extend(pins[pos:])
        cols = [carry_ha[j] + carry_fa[j] + sum_ha[j] + sum_fa[j] + apx[j] +
                thru[j] for j in range(K)]
    return total, cols


def _identity(alloc):
    perms = []
    hs = [initial_heights(alloc.n)]
    from .ctree import next_heights
    for st in alloc.stages:
        perms.append({j: list(range(hs[-1][j])) for j in range(len(hs[-1]))})
        hs.append(next_heights(hs[-1], st))
    return perms


def _bit_probs(alloc, perms, upto):
    """Probabilities of the bits of BM_upto under ``perms`` for earlier
    stages."""
    sub = Allocation(n=alloc.n, stages=alloc.stages[:upto])
    _, cols = evaluate_order(sub, perms[:upto])
    return cols


def greedy_order(alloc: Allocation):
    """Lowest-probability bits to approximate compressors, stage by stage."""
    perms = _identity(alloc)
    for s in range(1, len(alloc.stages)):
        cols = _bit_probs(alloc, perms, s)
        for j, comps in alloc.stages[s].items():
            probs = cols[j]
            order = sorted(range(len(probs)), key=lambda r: probs[r])
            # pin slots: approximate pins first in the bit ranking, then exact
            # compressor pins, then pass-through; mapped back to pin order
            pin_kind = []
            for k in comps:
                pin_kind += [k] * kind_inputs(k)
            pin_kind += ["THRU"] * (len(probs) - len(pin_kind))
            rank = {"E42": 0, "E32": 1, "FA": 2, "HA": 3, "THRU": 4}
            slots = sorted(range(len(pin_kind)), key=lambda c: rank[pin_kind[c]])
            perm = [0] * len(probs)
            for slot, r in zip(slots, order):
                perm[slot] = r
            perms[s][j] = perm
    return perms


def order(alloc: Allocation, time_limit: float = 60.0, seed: int = 0):
    """Greedy start, then first-improvement pairwise swaps of the bits of a
    column, scored by the improved MED estimate of the whole tree."""
    perms = greedy_order(alloc)
    best, _ = evaluate_order(alloc, perms)
    start = best
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < time_limit:
        improved = False
        for s in range(1, len(alloc.stages)):
            for j in sorted(perms[s]):
                perm = perms[s][j]
                L = len(perm)
                for a in range(L):
                    for b in range(a + 1, L):
                        perm[a], perm[b] = perm[b], perm[a]
                        val, _ = evaluate_order(alloc, perms)
                        if val < best - 1e-12:
                            best = val
                            improved = True
                        else:
                            perm[a], perm[b] = perm[b], perm[a]
                if time.time() - t0 > time_limit:
                    break
    return perms, {"greedy": start, "final": best,
                   "identity": evaluate_order(alloc, _identity(alloc))[0],
                   "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# The exact MIP of Section III-B (SCIP), for the instances it can close
# ---------------------------------------------------------------------------
def order_mip(alloc: Allocation, time_limit: float = 600.0,
              start=None):
    """Minimise the improved MED estimate over all connection orders.

    Variables: a binary b[s,j,r,c] per (bit r, pin c) of every column of
    every stage after the first (stage-0 bits are all partial products of
    probability 1/4, so their order is immaterial), the pin probabilities
    z = sum_r y_r b_rc linearised exactly through McCormick products, and the
    bit probabilities y given by the output-probability polynomials of the
    compressor that produced them.  Returns (perms, objective, status)."""
    from pyscipopt import Model, quicksum

    n = alloc.n
    K = 2 * n - 1
    S = len(alloc.stages)
    m = Model("opact_order")
    m.hideOutput()
    m.setParam("limits/time", time_limit)

    # symbolic bit lists per stage: entries are (kind, producer index, output
    # index) or a constant 0.25 for stage 0
    def poly(mdl, out_idx, pin_vals):
        """Output probability as a polynomial in the pin variables."""
        terms = []
        for b, v in zip(mdl.pats, mdl.tables[out_idx]):
            if not v:
                continue
            t = 1
            for x, z in zip(b, pin_vals):
                t = t * (z if x else (1 - z))
            terms.append(t)
        return quicksum(terms) if terms else 0

    def med_poly(mdl, pin_vals):
        terms = []
        for b, e in zip(mdl.pats, mdl.err):
            if e == 0:
                continue
            t = e
            for x, z in zip(b, pin_vals):
                t = t * (z if x else (1 - z))
            terms.append(t)
        return quicksum(terms) if terms else 0

    cols = [[0.25] * h for h in initial_heights(n)]       # stage-0 bit values
    obj_terms = []
    bvars = {}
    for s, stage in enumerate(alloc.stages):
        carry_ha = [[] for _ in range(K)]
        carry_fa = [[] for _ in range(K)]
        sum_ha = [[] for _ in range(K)]
        sum_fa = [[] for _ in range(K)]
        apx = [[] for _ in range(K)]
        thru = [[] for _ in range(K)]
        for j in range(K):
            bits = cols[j]
            L = len(bits)
            if s == 0 or L <= 1:
                pins = list(bits)
            else:
                b = {(r, c): m.addVar(vtype="B", name=f"b_{s}_{j}_{r}_{c}")
                     for r in range(L) for c in range(L)}
                for r in range(L):
                    m.addCons(quicksum(b[r, c] for c in range(L)) == 1)  # (19)
                for c in range(L):
                    m.addCons(quicksum(b[r, c] for r in range(L)) == 1)  # (20)
                bvars[s, j] = (b, L)
                pins = []
                for c in range(L):
                    z = m.addVar(lb=0, ub=1, name=f"z_{s}_{j}_{c}")
                    prods = []
                    for r in range(L):
                        y = bits[r]
                        if isinstance(y, float):
                            prods.append(y * b[r, c])
                        else:
                            w = m.addVar(lb=0, ub=1)
                            m.addCons(w <= b[r, c])
                            m.addCons(w <= y)
                            m.addCons(w >= y - (1 - b[r, c]))
                            prods.append(w)
                    m.addCons(z == quicksum(prods))                       # (22)
                    pins.append(z)
            pos = 0
            for k in stage.get(j, []):
                mdl = _model(k)
                pv = pins[pos:pos + mdl.n_in]
                pos += mdl.n_in
                outs = []
                for oi in range(len(mdl.tables)):
                    if all(isinstance(v, float) for v in pv):
                        outs.append(mdl.outputs(pv)[oi])
                    else:
                        y = m.addVar(lb=0, ub=1)
                        m.addCons(y == poly(mdl, oi, pv))                 # (21)
                        outs.append(y)
                if k in ("HA", "FA"):
                    (sum_ha if k == "HA" else sum_fa)[j].append(outs[0])
                    if j + 1 < K:
                        (carry_ha if k == "HA" else carry_fa)[j + 1].append(outs[1])
                else:
                    if all(isinstance(v, float) for v in pv):
                        obj_terms.append((2 ** j) * mdl.med(pv))
                    else:
                        obj_terms.append((2 ** j) * med_poly(mdl, pv))
                    apx[j].extend(outs)
            thru[j].extend(pins[pos:])
        cols = [carry_ha[j] + carry_fa[j] + sum_ha[j] + sum_fa[j] + apx[j] +
                thru[j] for j in range(K)]
    t = m.addVar(lb=0, name="obj")
    m.addCons(t >= quicksum(obj_terms))
    m.setObjective(t, "minimize")
    m.optimize()
    status = m.getStatus()
    if m.getNSols() == 0:
        return None, float("nan"), status
    sol = m.getBestSol()
    perms = _identity(alloc)
    for (s, j), (b, L) in bvars.items():
        perm = [0] * L
        for c in range(L):
            for r in range(L):
                if m.getSolVal(sol, b[r, c]) > 0.5:
                    perm[c] = r
        perms[s][j] = perm
    return perms, float(m.getSolVal(sol, t)), status
