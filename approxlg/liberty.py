"""Minimal Liberty (.lib) reader: cell area, leakage, pin capacitance and the
Boolean function of every output pin.

Only the subset needed for post-mapping power estimation is parsed, but the
brace matching is exact so nested groups (timing arcs, LUT tables) are skipped
safely.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Pin:
    name: str
    direction: str = "input"
    capacitance: float = 0.0
    function: Optional[str] = None


@dataclass
class LibCell:
    name: str
    area: float = 0.0
    leakage: float = 0.0          # nW
    pins: Dict[str, Pin] = field(default_factory=dict)

    @property
    def inputs(self) -> List[str]:
        return [p.name for p in self.pins.values() if p.direction == "input"]

    @property
    def outputs(self) -> List[str]:
        return [p.name for p in self.pins.values() if p.direction == "output"]


def _groups(text: str, start: int = 0):
    """Yield (kind, name, body_slice) for the groups at the current level."""
    i, n = start, len(text)
    while i < n:
        m = re.compile(r"(\w+)\s*\(([^)]*)\)\s*\{").search(text, i)
        if not m:
            return
        depth, j = 1, m.end()
        while j < n and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        yield m.group(1), m.group(2).strip(), text[m.end():j - 1]
        i = j


def parse_liberty(path: str, cache: Optional[str] = None) -> Dict[str, LibCell]:
    if cache and os.path.exists(cache):
        with open(cache) as fh:
            raw = json.load(fh)
        out = {}
        for nm, d in raw.items():
            c = LibCell(nm, d["area"], d["leakage"])
            c.pins = {k: Pin(**v) for k, v in d["pins"].items()}
            out[nm] = c
        return out

    txt = open(path, errors="replace").read()
    cells: Dict[str, LibCell] = {}
    for kind, name, body in _groups(txt):
        if kind != "library":
            continue
        for k2, nm, cbody in _groups(body):
            if k2 != "cell":
                continue
            c = LibCell(nm)
            m = re.search(r"\barea\s*:\s*([0-9.eE+-]+)", cbody)
            if m:
                c.area = float(m.group(1))
            m = re.search(r"\bcell_leakage_power\s*:\s*([0-9.eE+-]+)", cbody)
            if m:
                c.leakage = float(m.group(1))
            for k3, pn, pbody in _groups(cbody):
                if k3 != "pin":
                    continue
                p = Pin(pn.strip('"'))
                m = re.search(r"\bdirection\s*:\s*(\w+)", pbody)
                if m:
                    p.direction = m.group(1)
                m = re.search(r"\bcapacitance\s*:\s*([0-9.eE+-]+)", pbody)
                if m:
                    p.capacitance = float(m.group(1))
                m = re.search(r'\bfunction\s*:\s*"([^"]*)"', pbody)
                if m:
                    p.function = m.group(1)
                c.pins[p.name] = p
            cells[nm] = c
        break
    if cache:
        with open(cache, "w") as fh:
            json.dump({nm: {"area": c.area, "leakage": c.leakage,
                            "pins": {k: v.__dict__ for k, v in c.pins.items()}}
                       for nm, c in cells.items()}, fh)
    return cells


# --------------------------------------------------------------------------
# Boolean function compiler:  Liberty syntax -> Python expression
# --------------------------------------------------------------------------
_TOK = re.compile(r"\s*(\(|\)|\+|\||\^|\*|&|!|'|[A-Za-z_]\w*|1|0)")


class _Parser:
    def __init__(self, s: str):
        self.toks = []
        i = 0
        while i < len(s):
            m = _TOK.match(s, i)
            if not m:
                i += 1
                continue
            self.toks.append(m.group(1))
            i = m.end()
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> str:
        e = self.or_expr()
        return e

    def or_expr(self) -> str:
        e = self.xor_expr()
        while self.peek() in ("+", "|"):
            self.next()
            e = f"({e} | {self.xor_expr()})"
        return e

    def xor_expr(self) -> str:
        e = self.and_expr()
        while self.peek() == "^":
            self.next()
            e = f"({e} ^ {self.and_expr()})"
        return e

    def and_expr(self) -> str:
        e = self.unary()
        while True:
            t = self.peek()
            if t in ("*", "&"):
                self.next()
                e = f"({e} & {self.unary()})"
            elif t is not None and (t == "(" or re.match(r"^[A-Za-z_01]", t)):
                e = f"({e} & {self.unary()})"
            else:
                return e

    def unary(self) -> str:
        t = self.next()
        if t == "!":
            return f"(~{self.unary()})"
        if t == "(":
            e = self.or_expr()
            if self.peek() == ")":
                self.next()
            return self._postfix(f"({e})")
        if t in ("0", "1"):
            return self._postfix("ZERO" if t == "0" else "ONE")
        return self._postfix(f"env['{t}']")

    def _postfix(self, e: str) -> str:
        while self.peek() == "'":
            self.next()
            e = f"(~{e})"
        return e


def compile_function(expr: str) -> str:
    """Return a Python expression string over ``env[...]`` bit vectors."""
    return _Parser(expr).parse()
