"""Compressor-tree multipliers of the reference methods.

OPACT (Xiao, Zhuo and Qian, DATE'22), the 1StepFull/2StepsFull multipliers of
Esposito et al. (TCAS-I'18) and the AM multipliers of Wang et al. (ISCAS'20)
are not points of the Approx-LG template.  Their approximate compressors have
*equally weighted* outputs -- a j-input compressor returns ceil(j/2) bits of
the same weight and no carry -- and each method decides how many exact and
approximate compressors go into every column of every reduction stage, and in
the case of OPACT also which bit drives which pin.  This module turns such an
allocation into the same :class:`Netlist` the template decodes to, so the
reference designs go through the identical simulator and synthesis flow.

Reduction follows the stage model all three papers use: the compressors of
stage ``i`` consume bits of bit matrix ``BM_i`` only, sums and approximate
outputs stay in their column of ``BM_{i+1}``, and the carry of an exact
compressor at column ``j`` enters column ``j+1`` of ``BM_{i+1}``.

Bit order inside a column of ``BM_{i+1}`` is the source order of OPACT
(Section III-B1): carries of exact 2:2 and 3:2 compressors from column j-1,
sums of exact 2:2 and 3:2 compressors, outputs of approximate compressors, and
bits passed through untouched.  Pins are ordered by compressor type in the
order the allocation lists them, pass-through ("dummy") pins last.  A
connection order is a permutation from pins to bits; the identity is the
natural order a hand-written netlist would use.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .arch import Netlist, Stmt
from .cells import CELL_BY_NAME, Cell, _mk

# ---------------------------------------------------------------------------
# Approximate compressors with equally weighted outputs (Esposito et al.,
# Section III).  Inputs a..f are p0..p5 of the paper; "+" there is OR and
# juxtaposition AND.
# ---------------------------------------------------------------------------
ESP_CELLS: Dict[str, Cell] = {
    # (6): S_APP = {p0 + p1}
    "E21": _mk("E21", "ESP", 2, [(1, "a|b")], note="approx 2/1"),
    # (13): S_APP = {p0p1 + p2, p0 + p1}
    "E32": _mk("E32", "ESP", 3, [(1, "(a&b)|c"), (1, "a|b")],
               note="approx 3/2"),
    # (17): S_APP = {p0p1 + p2 + p3, p2p3 + p0 + p1}
    "E42": _mk("E42", "ESP", 4, [(1, "(a&b)|c|d"), (1, "(c&d)|a|b")],
               note="approx 4/2"),
    # (21): S_APP = {p0p1 + p2 + p3, p0 + p1, p2p3 + p4}
    "E53": _mk("E53", "ESP", 5, [(1, "(a&b)|c|d"), (1, "a|b"), (1, "(c&d)|e")],
               note="approx 5/3"),
    # (25): S_APP = {p0p1 + p2 + p3, p2p3 + p4 + p5, p4p5 + p0 + p1}
    "E63": _mk("E63", "ESP", 6, [(1, "(a&b)|c|d"), (1, "(c&d)|e|f"),
                                 (1, "(e&f)|a|b")], note="approx 6/3"),
}
APPROX_BY_INPUTS = {2: "E21", 3: "E32", 4: "E42", 5: "E53", 6: "E63"}
EXACT_KINDS = ("HA", "FA")


def esposito_parts(j: int) -> List[str]:
    """A j-input approximate compressor as the sub-compressors of Table VI."""
    if j < 2:
        return []
    if j <= 6:
        return [APPROX_BY_INPUTS[j]]
    q, r = divmod(j, 4)
    if r == 0:
        return ["E42"] * q
    if r == 1:
        return ["E42"] * (q - 1) + ["E53"]
    if r == 2:
        return ["E42"] * (q - 1) + ["E63"]
    return ["E32"] + ["E42"] * q


def kind_inputs(kind: str) -> int:
    if kind == "HA":
        return 2
    if kind == "FA":
        return 3
    return ESP_CELLS[kind].n_in


def kind_cell(kind: str) -> Cell:
    return CELL_BY_NAME[kind] if kind in EXACT_KINDS else ESP_CELLS[kind]


# ---------------------------------------------------------------------------
# Allocation and decoding
# ---------------------------------------------------------------------------
@dataclass
class Allocation:
    """``stages[i][j]`` lists the compressors of column j in stage i, in pin
    order; ``perms[i][j][c]`` (optional) is the bit index driving pin c.

    ``ripple[i]`` marks a stage whose exact compressors may also consume the
    carries produced in the same stage by the column to their right, which is
    how the exact part of Wang et al. is drawn (their Fig. 3).  Those carries
    join the end of the column's bit list."""
    n: int
    stages: List[Dict[int, List[str]]] = field(default_factory=list)
    perms: Optional[List[Dict[int, List[int]]]] = None
    ripple: Optional[List[bool]] = None
    name: str = ""
    trunc: int = 0          # partial products of columns < trunc not formed

    @property
    def ncol(self) -> int:
        return 2 * self.n - 1


def initial_heights(n: int, trunc: int = 0) -> List[int]:
    return [0 if j < trunc else min(j + 1, 2 * n - 1 - j)
            for j in range(2 * n - 1)]


def next_heights(h: Sequence[int], stage: Dict[int, List[str]]) -> List[int]:
    ncol = len(h)
    out = list(h)
    for j, comps in stage.items():
        for k in comps:
            nin = kind_inputs(k)
            if k in EXACT_KINDS:
                out[j] -= nin - 1
                if j + 1 < ncol:
                    out[j + 1] += 1
            else:
                out[j] -= nin - len(ESP_CELLS[k].outs)
    return out


def heights_after(alloc: Allocation) -> List[List[int]]:
    hs = [initial_heights(alloc.n, alloc.trunc)]
    for st in alloc.stages:
        hs.append(next_heights(hs[-1], st))
    return hs


class TreeError(ValueError):
    pass


def build(alloc: Allocation) -> Netlist:
    """Decode an allocation into a netlist (unsigned AND array, exact CPA)."""
    n = alloc.n
    ncol = 2 * n - 1
    nl = Netlist(n=n, trunc=alloc.trunc, cpa_k=0, signed=False)
    uid = [0]

    def new(pfx):
        uid[0] += 1
        return f"{pfx}{uid[0]}"

    cols: List[List[str]] = [[] for _ in range(ncol)]
    for i in range(n):
        for j in range(n):
            if i + j < alloc.trunc:
                continue
            w = new("pp")
            nl.stmts.append(Stmt("and", [w], [f"a[{j}]", f"b[{i}]"]))
            cols[i + j].append(w)

    for s, stage in enumerate(alloc.stages):
        ripple = bool(alloc.ripple and s < len(alloc.ripple) and alloc.ripple[s])
        # per column, the six source groups of the next bit matrix
        carry_ha = [[] for _ in range(ncol)]
        carry_fa = [[] for _ in range(ncol)]
        sum_ha = [[] for _ in range(ncol)]
        sum_fa = [[] for _ in range(ncol)]
        apx = [[] for _ in range(ncol)]
        thru = [[] for _ in range(ncol)]
        for j in range(ncol):
            bits = cols[j]
            if ripple:
                # same-stage carries from column j-1 are inputs here too
                bits = bits + carry_ha[j] + carry_fa[j]
                carry_ha[j], carry_fa[j] = [], []
            comps = stage.get(j, [])
            need = sum(kind_inputs(k) for k in comps)
            if need > len(bits):
                raise TreeError(f"stage {s} column {j}: {need} pins, "
                                f"{len(bits)} bits")
            perm = None
            if alloc.perms is not None and s < len(alloc.perms):
                perm = alloc.perms[s].get(j)
            if perm is not None:
                if sorted(perm) != list(range(len(bits))):
                    raise TreeError(f"stage {s} column {j}: bad permutation")
                pins = [bits[r] for r in perm]
            else:
                pins = list(bits)
            pos = 0
            for k in comps:
                cell = kind_cell(k)
                ins = pins[pos:pos + cell.n_in]
                pos += cell.n_in
                outs = []
                for w, _ in cell.outs:
                    wire = new("w")
                    outs.append(wire)
                    if k in EXACT_KINDS:
                        if w == 1:
                            (sum_ha if k == "HA" else sum_fa)[j].append(wire)
                        elif j + 1 < ncol:
                            (carry_ha if k == "HA" else carry_fa)[j + 1].append(wire)
                    else:
                        apx[j].append(wire)
                nl.stmts.append(Stmt("cell", outs, ins, cell))
                nl.n_cells += 1
                if not cell.exact:
                    nl.n_approx_cells += 1
            thru[j].extend(pins[pos:])
        cols = [carry_ha[j] + carry_fa[j] + sum_ha[j] + sum_fa[j] + apx[j] +
                thru[j] for j in range(ncol)]

    for j in range(ncol):
        if len(cols[j]) > 2:
            raise TreeError(f"column {j} still holds {len(cols[j])} bits")
        if cols[j]:
            nl.row_a[j] = cols[j][0]
        if len(cols[j]) > 1:
            nl.row_b[j] = cols[j][1]
    return nl


# ---------------------------------------------------------------------------
# Exact Dadda completion (the exact stages that follow the approximate ones)
# ---------------------------------------------------------------------------
DADDA = [2, 3, 4, 6, 9, 13, 19, 28, 42, 63]


def dadda_stage(h: Sequence[int], target: int) -> Dict[int, List[str]]:
    """One Dadda stage to ``target`` under the stage model of this module:
    carries produced at column j-1 count towards column j's next height."""
    ncol = len(h)
    stage: Dict[int, List[str]] = {}
    cin = 0
    for j in range(ncol):
        excess = h[j] + cin - target
        fa = ha = 0
        if excess > 0:
            fa = excess // 2
            ha = excess - 2 * fa
            # never ask for more pins than the column has
            while 3 * fa + 2 * ha > h[j] and fa > 0:
                fa -= 1
                ha += 2
            if 3 * fa + 2 * ha > h[j]:
                raise TreeError(f"Dadda stage to {target}: column {j} "
                                f"cannot be reduced")
        comps = ["HA"] * ha + ["FA"] * fa
        if comps:
            stage[j] = comps
        cin = fa + ha
    return stage


def dadda_complete(alloc: Allocation) -> Allocation:
    """Append exact Dadda stages until every column holds at most two bits."""
    h = heights_after(alloc)[-1]
    while max(h) > 2:
        target = max(d for d in DADDA if d < max(h))
        st = dadda_stage(h, target)
        alloc.stages.append(st)
        h = next_heights(h, st)
    return alloc


def dadda_next(h: int) -> int:
    return max(d for d in DADDA if d < h)


# ---------------------------------------------------------------------------
# Esposito et al., Algorithm 1 (one reduction step with approximate
# compressors in the LSP and, where needed, in the MSP)
# ---------------------------------------------------------------------------
def _exact_reduction(h: int) -> int:
    """Largest reduction exact 3:2/2:2 compressors achieve within h bits."""
    return 2 * (h // 3) + (1 if h % 3 == 2 else 0)


def esposito_step(h: Sequence[int], n: int, hmax_next: int
                  ) -> Dict[int, List[str]]:
    """One reduction step of Algorithm 1.

    The LSP (columns 1..n) is compressed as much as possible by approximate
    compressors.  An MSP column uses exact compressors, and adds the smallest
    approximate compressor that still meets ``hmax_next`` only where the
    carries exact compressors would send left exceed what the next column can
    absorb, which is the case analysis of (36)-(41): C* is the number of
    exact compressors the column needs, C** the number of carries its left
    neighbour can take.  The printed closed forms are garbled in our copy,
    so the rule is implemented from that description; the reproduced
    multipliers match the published error figures (see the paper's Table of
    reproductions).
    """
    K = len(h)
    T = hmax_next
    hh = list(h) + [0, 0]
    FA = [0] * K
    HA = [0] * K
    J = [0] * K
    C = [0] * (K + 1)
    for j in range(n):                                    # LSP, k = 1..n
        if hh[j] > 2:
            J[j] = hh[j]

    def best_exact(hj, cmax):
        """(fa, ha) of largest reduction with at most cmax carries."""
        best = (0, 0)
        for fa in range(min(hj // 3, cmax) + 1):
            ha = min((hj - 3 * fa) // 2, cmax - fa, 1)
            if 2 * fa + ha > 2 * best[0] + best[1]:
                best = (fa, ha)
        return best

    # carries each MSP column can absorb with exact compressors only, from
    # the top column down (the top column must not produce a carry)
    cap = [0] * (K + 1)
    cap[K] = 0
    for j in range(K - 1, n - 1, -1):
        fa, ha = best_exact(hh[j], cap[j + 1])
        cap[j] = max(0, T - hh[j] + 2 * fa + ha)

    for j in range(n, K):                                 # MSP
        cprev = C[j - 1]
        need = max(0, hh[j] + cprev - T)                  # reduction needed
        if need == 0:
            continue
        cstar = math.ceil(need / 2)                       # (36): C*
        fa, ha = best_exact(hh[j], cap[j + 1])            # C** bound
        if 2 * fa + ha >= need:                           # exact suffices
            fa, ha = need // 2, need % 2
            if fa + ha > cap[j + 1] or 3 * fa + 2 * ha > hh[j]:
                fa, ha = best_exact(hh[j], cap[j + 1])
            FA[j], HA[j] = fa, ha
        else:                                 # approximate compressors needed
            fa = min(cap[j + 1], hh[j] // 3, need // 2)
            rest = need - 2 * fa
            jj = 2 * rest                                 # (41)
            if jj == 2 and hh[j] - 3 * fa >= 3:
                jj = 3
            while jj > hh[j] - 3 * fa and fa > 0:
                # not enough bits left: trade a full adder for approximation
                fa -= 1
                rest += 2
                jj = 2 * rest
            if jj > hh[j] - 3 * fa:
                raise TreeError(f"Esposito step to {T}: column {j}")
            FA[j], J[j] = fa, jj
        C[j] = FA[j] + HA[j]
    stage: Dict[int, List[str]] = {}
    for j in range(K):
        comps = ["HA"] * HA[j] + ["FA"] * FA[j]
        if J[j] >= 2:
            comps.append(APPROX_BY_INPUTS.get(J[j], f"J{J[j]}"))
        if comps:
            stage[j] = comps
    return stage


def _expand_composites(stage: Dict[int, List[str]]) -> Dict[int, List[str]]:
    out = {}
    for j, comps in stage.items():
        lst = []
        for k in comps:
            if k.startswith("J"):
                lst.extend(esposito_parts(int(k[1:])))
            else:
                lst.append(k)
        out[j] = lst
    return out


def _hnext_target(hmax: int) -> int:
    """h_max,next = ceil(h_max/2): the bound of Section IV taken with
    equality, which gives the sequences 8->4->2 and 12->6->3 the paper
    describes and reproduces its 16x16 error figures (a Dadda-level target
    of 9 there halves the published MED)."""
    return max(2, math.ceil(hmax / 2))


def esposito(n: int, steps: int, trunc: bool = False) -> Allocation:
    """1StepFull/2StepsFull (steps=1/2), unsigned; with ``trunc`` the n-1
    least significant columns are not formed (1StepTrunc/2StepsTrunc)."""
    kind = ("1Step" if steps == 1 else "2Steps") + ("Trunc" if trunc else "Full")
    alloc = Allocation(n=n, name=f"{kind}{n}", trunc=(n - 1 if trunc else 0))
    h = initial_heights(n, alloc.trunc)
    for _ in range(steps):
        hmax = max(h)
        if hmax <= 2:
            break
        target = _hnext_target(hmax)
        st = _expand_composites(esposito_step(h, n, target))
        alloc.stages.append(st)
        h = next_heights(h, st)
    return dadda_complete(alloc)


# ---------------------------------------------------------------------------
# Wang et al., Algorithm 1 (optimised universal compression strategy): the L
# most significant columns of the PPM are exact, everything below them is
# compressed by approximate compressors only, in two stages.
# ---------------------------------------------------------------------------
def wang(n: int, L: Optional[int] = None) -> Allocation:
    L = (math.ceil(n / 2) + 1) if L is None else L
    K = 2 * n - 1
    m = K - L                              # columns 0..m-1 are approximate
    h0 = math.ceil(n / 4)
    alloc = Allocation(n=n, name=f"AM{n}-{L}", ripple=[True, True])
    h = initial_heights(n)
    # stage 1: approximate part to ceil(h/2); exact part with as many full
    # adders as the column, including the carries from its right, can feed
    st1: Dict[int, List[str]] = {}
    c1 = [0] * (K + 1)
    for j in range(K):
        if j < m:
            if h[j] > 2:
                st1[j] = esposito_parts(h[j])
        else:
            cprev = c1[j - 1] if j > 0 else 0
            fa = (h[j] + cprev) // 3                                     # (4)
            c1[j] = fa
            if fa:
                st1[j] = ["FA"] * fa
    alloc.stages.append(st1)
    h1 = next_heights(h, st1)
    # stage 2: approximate part again; exact part down to h0 = ceil(n/4)
    st2: Dict[int, List[str]] = {}
    c2 = [0] * (K + 1)
    for j in range(K):
        if j < m:
            if h1[j] > 2:
                st2[j] = esposito_parts(h1[j])
        else:
            cprev = c2[j - 1] if j > 0 else 0
            c = max(0, math.ceil((h1[j] + cprev - h0) / 2))              # (5)
            fa = max(0, (h1[j] + cprev - h0) // 2)                       # (6)
            ha = c - fa                                                  # (7)
            while 3 * fa + 2 * ha > h1[j] + cprev and fa > 0:
                fa -= 1
                ha += 2
            c2[j] = fa + ha
            comps = ["HA"] * ha + ["FA"] * fa
            if comps:
                st2[j] = comps
    alloc.stages.append(st2)
    return dadda_complete(alloc)
