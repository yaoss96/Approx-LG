"""Approximate compressor cell library for Approx-LG.

Every cell is described by a list of (weight, boolean-expression) output pairs.
The expression string is written in a syntax that is simultaneously valid

  * Python/NumPy  (bit-wise ops on boolean ndarrays), and
  * Verilog-2001  (bit-wise ops on 1-bit wires),

so a single string drives both the functional simulator and the RTL emitter.
Inputs are named ``a``, ``b``, ``c``, ``d`` (in decreasing significance of the
"input order" used by the input-order optimisation of Wen et al.).

Cell families
-------------
``HA``  2 inputs -> outputs at weights (1, 2)
``FA``  3 inputs -> outputs at weights (1, 2)
``C42`` 4 inputs -> outputs at weights (1, 2)      (approximate 4:2 counters)
``ORC`` k inputs -> single output at weight 1      (OR-compression, LOA style)

The exact 4:2 counter is *not* a cell: it is built by the reduction engine from
two exact full adders, exactly as in a Dadda/Wallace tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Cell:
    name: str
    family: str                      # 'HA' | 'FA' | 'C42' | 'ORC'
    n_in: int
    outs: Tuple[Tuple[int, str], ...]  # (weight, expression)
    exact: bool = False
    note: str = ""

    @property
    def n_out(self) -> int:
        return len(self.outs)


def _mk(name, family, n_in, outs, exact=False, note=""):
    return Cell(name, family, n_in, tuple(outs), exact, note)


# --------------------------------------------------------------------------
# Half-adder (2:2) cells.  HA_A1..HA_A4 follow Table I of Devi et al. (ISQED'25)
# --------------------------------------------------------------------------
HA_CELLS: List[Cell] = [
    _mk("HA",    "HA", 2, [(1, "a^b"), (2, "a&b")], exact=True, note="exact half adder"),
    _mk("HA_A1", "HA", 2, [(1, "a|b"), (2, "a&b")], note="AC_HA_1"),
    _mk("HA_A2", "HA", 2, [(1, "a^b"), (2, "~(a^b)")], note="AC_HA_2"),
    _mk("HA_A3", "HA", 2, [(1, "b"),   (2, "a")], note="AC_HA_3"),
    _mk("HA_A4", "HA", 2, [(1, "a|b")], note="AC_HA_4 (carry dropped)"),
    _mk("HA_P1", "HA", 2, [(1, "a^b")], note="biased +, carry dropped"),
]

# --------------------------------------------------------------------------
# Full-adder (3:2) cells.
#   FA        exact
#   FA_A1..A5 follow Table I of Devi et al. (AC_FA_1,2,4,5,6)
#   FA_P*/FA_N* are the biased cells in the spirit of Wen et al. (ISCAS'25):
#             FA_P* over-estimate (mostly positive approximation error),
#             FA_N* under-estimate.  The bias is measured, not asserted --
#             see :func:`cell_statistics`.
# --------------------------------------------------------------------------
FA_CELLS: List[Cell] = [
    _mk("FA",    "FA", 3, [(1, "a^b^c"), (2, "(a&b)|(c&(a^b))")], exact=True,
        note="exact full adder"),
    _mk("FA_A1", "FA", 3, [(1, "(~(b|c))^a^c"),
                           (2, "~(((~(b|c))^a^c)|(~(b|c)))")], note="AC_FA_1"),
    _mk("FA_A2", "FA", 3, [(1, "(b|c)^a"), (2, "b")], note="AC_FA_2"),
    _mk("FA_A3", "FA", 3, [(1, "a&b&c"), (2, "a|b|c")], note="AC_FA_4"),
    _mk("FA_A4", "FA", 3, [(1, "a|b|c"), (2, "a&b&c")], note="AC_FA_5"),
    _mk("FA_A5", "FA", 3, [(1, "b|c"), (2, "a&(b|c)")], note="AC_FA_6"),
    _mk("FA_P1", "FA", 3, [(1, "a^b^c"), (2, "a&b")], note="biased +, exact sum"),
    _mk("FA_P2", "FA", 3, [(1, "a^b^c"), (2, "a&b&c")], note="biased +, exact sum"),
    _mk("FA_N1", "FA", 3, [(1, "a^b^c"), (2, "a|b")], note="biased -, exact sum"),
    _mk("FA_N2", "FA", 3, [(1, "a|b|c"), (2, "(a&b)|(c&(a^b))")],
        note="biased -, exact carry"),
    _mk("FA_B1", "FA", 3, [(1, "(a^b)&(~c)"), (2, "(a&b)|c")], note="balanced"),
    _mk("FA_OR", "FA", 3, [(1, "a|b|c")], note="OR compression, carry dropped"),
]

# --------------------------------------------------------------------------
# Approximate 4:2 counter cells (a,b,c,d -> weight-1 sum, weight-2 carry).
# The exact counter would need a second weight-2 output; dropping it is the
# source of approximation, which is why these cells are so area-efficient.
# --------------------------------------------------------------------------
C42_CELLS: List[Cell] = [
    _mk("C42_1", "C42", 4, [(1, "a^b^c^d"), (2, "(a&b)|(c&d)")],
        note="XOR sum, OR-of-AND carry"),
    _mk("C42_2", "C42", 4, [(1, "(a|b)^(c|d)"), (2, "(a|b)&(c|d)")],
        note="OR-reduced then exact HA"),
    _mk("C42_3", "C42", 4, [(1, "a|b|c|d"), (2, "(a&b)|(c&d)")],
        note="biased +"),
    _mk("C42_4", "C42", 4, [(1, "(a^b)|(c^d)"), (2, "(a&b)|(c&d)")],
        note="balanced"),
    _mk("C42_5", "C42", 4, [(1, "a^b^c^d"),
                            (2, "(a&b)|(a&c)|(a&d)|(b&c)|(b&d)|(c&d)")],
        note="exact-sum, majority carry (biased +)"),
    _mk("C42_6", "C42", 4, [(1, "~((a|b)^(c|d))"), (2, "(a|b)&(c|d)")],
        note="biased +"),
    _mk("C42_7", "C42", 4, [(1, "(a|b)&(~(c&d))"), (2, "c|d")],
        note="biased -"),
    _mk("C42_8", "C42", 4, [(1, "b|d"), (2, "a|c")], note="aggressive, biased -"),
    _mk("C42_9", "C42", 4, [(1, "a|b|c|d"), (2, "(a|b)&(c|d)")], note="biased -"),
    _mk("C42_10", "C42", 4, [(1, "(a^b)|(c^d)"), (2, "(a|b)&(c|d)")],
        note="biased -"),
    _mk("C42_OR", "C42", 4, [(1, "a|b|c|d")], note="OR compression, carry dropped"),
]

ALL_CELLS: List[Cell] = HA_CELLS + FA_CELLS + C42_CELLS
CELL_BY_NAME: Dict[str, Cell] = {c.name: c for c in ALL_CELLS}


# --------------------------------------------------------------------------
# Slot alphabet used by the genome.
#
# A gene value selects *how a slot compresses a column*: either the exact
# Dadda behaviour (FA/HA), or an approximate 3:2 cell paired with an
# approximate 2:2 cell, or an approximate 4:2 cell.  Every gene therefore
# names a (fa_cell, ha_cell, c42_cell) triple; entries may be None.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SlotMode:
    name: str
    fa: str | None
    ha: str | None
    c42: str | None = None
    exact: bool = False


SLOT_MODES: List[SlotMode] = [
    SlotMode("EXACT",  "FA",     "HA",    None,      exact=True),
    SlotMode("A1",     "FA_A1",  "HA_A2", None),
    SlotMode("A2",     "FA_A2",  "HA_A3", None),
    SlotMode("A3",     "FA_A3",  "HA_A1", None),
    SlotMode("A4",     "FA_A4",  "HA_P1", None),
    SlotMode("A5",     "FA_A5",  "HA_A1", None),
    SlotMode("P1",     "FA_P1",  "HA_P1", None),
    SlotMode("P2",     "FA_P2",  "HA_P1", None),
    SlotMode("N1",     "FA_N1",  "HA_A1", None),
    SlotMode("N2",     "FA_N2",  "HA_A2", None),
    SlotMode("B1",     "FA_B1",  "HA",    None),
    SlotMode("K1",     "FA",     "HA",    "C42_1"),
    SlotMode("K2",     "FA_A2",  "HA_A3", "C42_2"),
    SlotMode("K3",     "FA_P1",  "HA_P1", "C42_3"),
    SlotMode("K4",     "FA",     "HA",    "C42_4"),
    SlotMode("K5",     "FA_P2",  "HA_P1", "C42_5"),
    SlotMode("K6",     "FA_A4",  "HA_A1", "C42_6"),
    SlotMode("K7",     "FA_N1",  "HA_A2", "C42_9"),
    SlotMode("K8",     "FA_A3",  "HA_A4", "C42_8"),
    SlotMode("K9",     "FA_N2",  "HA_A1", "C42_10"),
    SlotMode("OR",     "FA_OR",  "HA_A4", "C42_OR"),
]

# Opt-in extension (APPROXLG_STYLES=ext): the carry-free compressors of
# Esposito et al., whose outputs all keep the weight of their inputs, as a
# 22nd style.  It is appended so that every genome of the paper's campaigns
# decodes unchanged, and it is off by default because a larger alphabet
# changes the random stream of every sampler and mutation; only the campaign
# that studies it sets the variable.
if os.environ.get("APPROXLG_STYLES", "").lower() == "ext":
    ESP_EXT = [
        _mk("E21", "HA", 2, [(1, "a|b")], note="Esposito 2/1"),
        _mk("E32", "FA", 3, [(1, "(a&b)|c"), (1, "a|b")], note="Esposito 3/2"),
        _mk("E42", "C42", 4, [(1, "(a&b)|c|d"), (1, "(c&d)|a|b")],
            note="Esposito 4/2"),
    ]
    ALL_CELLS = ALL_CELLS + ESP_EXT
    CELL_BY_NAME.update({c.name: c for c in ESP_EXT})
    SLOT_MODES.append(SlotMode("E", "E32", "E21", "E42"))
N_SLOT_MODES = len(SLOT_MODES)
SLOT_MODE_BY_NAME = {m.name: i for i, m in enumerate(SLOT_MODES)}


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def eval_expr(expr: str, env: Dict[str, "np.ndarray"], xp=np):
    """Evaluate a cell expression on boolean arrays."""
    return eval(expr, {}, env)  # noqa: S307 - trusted literals


def cell_statistics(cell: Cell) -> Dict[str, float]:
    """Exhaustive error characterisation of one cell (uniform inputs).

    Returns the mean/abs-mean/max of ED = exact_value - approx_value and the
    error rate, where ``exact_value`` is the population count of the inputs.
    """
    n = cell.n_in
    pats = np.arange(2 ** n, dtype=np.uint8)
    env = {}
    for i, nm in enumerate("abcd"[:n]):
        env[nm] = ((pats >> (n - 1 - i)) & 1).astype(bool)
    exact_val = np.zeros(2 ** n, dtype=np.int32)
    for nm in "abcd"[:n]:
        exact_val += env[nm].astype(np.int32)
    approx_val = np.zeros(2 ** n, dtype=np.int32)
    for w, e in cell.outs:
        approx_val += w * np.asarray(eval_expr(e, dict(env)), dtype=bool).astype(np.int32)
    ed = exact_val - approx_val
    return {
        "mean_ed": float(ed.mean()),
        "mean_abs_ed": float(np.abs(ed).mean()),
        "max_abs_ed": int(np.abs(ed).max()),
        "error_rate": float((ed != 0).mean()),
    }


def bias_symbol(mean_ed: float, tol: float = 1e-9) -> str:
    """'-' = over-estimating cell (ED<0), '+' = under-estimating, '±' = balanced."""
    if mean_ed > tol:
        return "+"
    if mean_ed < -tol:
        return "-"
    return "0"


def cell_table() -> List[Dict]:
    rows = []
    for c in ALL_CELLS:
        st = cell_statistics(c)
        rows.append(dict(name=c.name, family=c.family, exact=c.exact,
                         expr=" ; ".join(f"w{w}: {e}" for w, e in c.outs),
                         bias=bias_symbol(st["mean_ed"]), **st))
    return rows


if __name__ == "__main__":  # pragma: no cover
    import json
    print(json.dumps(cell_table(), indent=1))
