"""Thin, fail-closed LLM client plus the three Approx-LG guidance operators.

The client is deliberately defensive: every operator validates and repairs the
model's answer, and silently falls back to the corresponding stochastic
operator of plain NSGA-II when the answer cannot be parsed.  An unusable answer
therefore costs latency rather than correctness, and the variation operators
that remain are exactly NSGA-II's.

That fallback is a safety property, not a guarantee about the result.  A run in
which *every* call fails is a run of plain NSGA-II under a different random
stream, and must not be reported as a run of the guided algorithm, so the
client fails fast: after ``outage_after`` consecutive failures it raises
``LLMOutage`` and the caller abandons the run instead of recording a diluted
one.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import requests

from . import engine as _engine
from .arch import Space
from .cells import (SLOT_MODES, SLOT_MODE_BY_NAME, N_SLOT_MODES,
                    cell_statistics, CELL_BY_NAME, bias_symbol)


def _metric() -> str:
    """Name of the error objective (NMED unless the search is retargeted)."""
    return _engine.ERROR_METRIC

API_URL = os.environ.get("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DEFAULT_MODEL = os.environ.get("APPROXLG_LLM", "deepseek-v4-flash")


class LLMOutage(RuntimeError):
    """Raised when the endpoint has failed often enough that the run would
    silently degenerate into plain NSGA-II."""


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    sham: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    accepted: int = 0
    proposed: int = 0


class LLM:
    """Client for the guidance operators.

    ``sham=True`` turns it into the placebo control: every operator is asked
    its question, walks its own code path and gets nothing back, so the run
    consumes the guided loop's extra random draws and its top-up behaviour
    while receiving none of the model's content.  Plain NSGA-II does not
    control for that -- an unplanned outage showed the two differ by about a
    percent of hypervolume on its own -- so the sham arm, not NSGA-II, is the
    comparison the guidance has to beat.  No request is sent, which keeps the
    control free of the endpoint's own latency and variability.
    """

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.8,
                 api_key: Optional[str] = None, max_retries: int = 3,
                 log_path: Optional[str] = None,
                 fixed_temperature: bool = False,
                 outage_after: int = 6, sham: bool = False,
                 thinking: bool = False):
        # The endpoint serves a reasoning model.  Extended thinking is off by
        # default because it costs ~10x the latency per call for guidance this
        # small, but the paper's negative result about the elicited prior has
        # to be checked against the stronger configuration, so it is a knob.
        self.thinking = thinking
        self.sham = sham
        self.model = model
        self.temperature = temperature
        # Operators pass their own temperature per call.  A sensitivity sweep
        # needs one knob that overrides all of them, otherwise most calls
        # ignore the sweep.
        self.fixed_temperature = fixed_temperature
        self.key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.max_retries = max_retries
        self.stats = LLMStats()
        self.log_path = log_path
        self.outage_after = outage_after
        self.consecutive_failures = 0
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self.sham or bool(self.key)

    def chat(self, system: str, user: str, max_tokens: int = 2048,
             temperature: Optional[float] = None) -> Optional[str]:
        if self.sham:
            # counted separately: a sham answer is neither a success nor the
            # kind of failure that should trip the outage guard
            with self._lock:
                self.stats.sham += 1
            return None
        if not self.key:
            return None
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": (self.temperature
                            if temperature is None or self.fixed_temperature
                            else temperature),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if self.thinking else "disabled"},
        }
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = requests.post(API_URL, timeout=180,
                                  headers={"Authorization": f"Bearer {self.key}",
                                           "Content-Type": "application/json"},
                                  json=payload)
                dt = time.time() - t0
                if r.status_code != 200:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                d = r.json()
                txt = d["choices"][0]["message"]["content"]
                u = d.get("usage", {})
                with self._lock:
                    self.stats.calls += 1
                    self.consecutive_failures = 0
                    self.stats.seconds += dt
                    self.stats.prompt_tokens += u.get("prompt_tokens", 0)
                    self.stats.completion_tokens += u.get("completion_tokens", 0)
                if self.log_path:
                    with self._lock, open(self.log_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"sys": system[:200], "user": user,
                                             "resp": txt, "dt": dt}) + "\n")
                return txt
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        with self._lock:
            self.stats.failures += 1
            self.consecutive_failures += 1
            dead = self.consecutive_failures >= self.outage_after
        if dead:
            raise LLMOutage(
                f"{self.consecutive_failures} consecutive failed calls to "
                f"{self.model}; abandoning the run rather than recording a "
                f"guided run that received no guidance")
        return None


# --------------------------------------------------------------------------
# Prompt material
# --------------------------------------------------------------------------
def slot_mode_doc() -> str:
    lines = []
    for i, m in enumerate(SLOT_MODES):
        parts = []
        for role, nm in (("3:2", m.fa), ("2:2", m.ha), ("4:2", m.c42)):
            if nm:
                st = cell_statistics(CELL_BY_NAME[nm])
                parts.append(f"{role}={nm}(bias {bias_symbol(st['mean_ed'])},"
                             f"|ED|={st['mean_abs_ed']:.2f})")
        lines.append(f"  {i:2d} {m.name:<6} " + " ".join(parts) +
                     ("  [EXACT]" if m.exact else ""))
    return "\n".join(lines)


SYSTEM = """You are an expert VLSI arithmetic-circuit designer assisting a \
multi-objective design-space exploration loop for approximate multipliers.
You reason about how partial-product truncation, heterogeneous approximate \
compressors and approximate final adders trade error against area and delay, \
and you exploit error cancellation between positively and negatively biased \
compressors.
Answer with a single JSON object and nothing else."""


def space_doc(sp: Space) -> str:
    return f"""Design template: unsigned {sp.n}x{sp.n} multiplier, AND-array \
partial products, Dadda reduction with {sp.nstage} stages (column-height \
targets {sp.targets}), {sp.ncol} columns numbered 0 (LSB) to {sp.ncol-1} (MSB), \
final carry-propagate adder.

A design point is described by:
  "T"       : number of truncated LSB columns, 0..{sp.max_t}.  Their partial
              products are not generated at all -- the cheapest but coarsest
              approximation.
  "A"       : number of approximate columns immediately above T, 0..{sp.ncol}-T.
              Columns >= T+A always use exact full/half adders.
  "comp"    : constant compensation of the mean truncation error,
              0 = none, 1 = 0.5x, 2 = 1.0x, 3 = 1.5x.  It is free in hardware.
  "cpa_k"   : number of LSBs of the final adder replaced by an OR-based (LOA)
              adder, 0..A.
  "columns" : {{"<column index>": "<style>"}} for the columns in [T, T+A).
              A style may also be given per reduction stage as a list of
              {sp.nstage} style names, e.g. ["K5","P1","EXACT","EXACT"].

Compressor styles (name -> cells and their measured error bias, where the
error distance is ED = exact - approximate, so '+' means the cell
under-estimates, '-' means it over-estimates and '0' means it is unbiased):
{{modes}}

Design guidance that usually holds:
  * raising T by one roughly halves the area of the low columns but multiplies
    the mean error by about 4; comp>0 removes about half of that mean error for
    free, so a truncated design should almost always set comp=2.
  * pairing a '+' biased style with a '-' biased style in the same or adjacent
    columns cancels error while keeping both cells cheap.
  * OR-like styles (OR, K8, A3, A4) are only safe in the two or three columns
    directly above T; near the MSB they destroy the worst-case error.
  * cpa_k trades a large delay reduction for a modest error increase, because
    it breaks the carry chain of the final adder.
""" + (_MRE_NOTE if _metric() == "MRE" else "")


_MRE_NOTE = """
The error objective of this search is the mean RELATIVE error (MRE), averaged
over all operand pairs with a non-zero exact product.  MRE is dominated by
small products, so truncation, the OR-based adder bits and aggressive cells in
the low columns cost far more MRE than they cost NMED; reaching a low MRE
usually needs T near 0 and approximation confined to a few columns.
"""


# --------------------------------------------------------------------------
# Symbolic design-point representation exchanged with the LLM.
#
# A raw 64-integer vector is a poor interface for a language model: it is long,
# positional, and hides the circuit meaning.  We therefore expose a design
# point as a column-wise record naming the compressor style of each
# approximate column, which is exactly how a designer would describe it.
# --------------------------------------------------------------------------
def genome_to_spec(sp: Space, g: np.ndarray) -> Dict:
    g = sp.canonical(g)
    t, a, comp, k = (int(x) for x in g[:4])
    slots = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
    cols: Dict[str, object] = {}
    for j in range(t, min(t + a, sp.ncol)):
        names = [SLOT_MODES[int(slots[s, j])].name for s in range(sp.nstage)]
        cols[str(j)] = names[0] if len(set(names)) == 1 else names
    return {"T": t, "A": a, "comp": comp, "cpa_k": k, "columns": cols}


def spec_to_genome(sp: Space, d: Dict) -> Optional[np.ndarray]:
    if not isinstance(d, dict):
        return None
    try:
        g = np.zeros(sp.length, dtype=np.int32)
        g[0] = int(np.clip(int(d.get("T", 0)), 0, sp.max_t))
        g[1] = int(np.clip(int(d.get("A", 0)), 0, sp.ncol - int(g[0])))
        g[2] = int(np.clip(int(d.get("comp", 0)), 0, 3))
        g[3] = int(np.clip(int(d.get("cpa_k", 0)), 0, int(g[1])))
    except Exception:
        return None
    slots = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
    cols = d.get("columns") or {}
    if isinstance(cols, dict):
        items = cols.items()
    elif isinstance(cols, list):
        items = ((str(int(g[0]) + i), v) for i, v in enumerate(cols))
    else:
        items = ()
    for jk, v in items:
        try:
            j = int(jk)
        except Exception:
            continue
        if not (0 <= j < sp.ncol):
            continue
        names = v if isinstance(v, list) else [v] * sp.nstage
        for s in range(sp.nstage):
            nm = names[s] if s < len(names) else names[-1]
            idx = SLOT_MODE_BY_NAME.get(str(nm).strip().upper())
            if idx is not None:
                slots[s, j] = idx
    g[sp.n_head:] = slots.reshape(-1)
    return sp.canonical(g)


def _clip_genome(sp: Space, vals: Sequence) -> Optional[np.ndarray]:
    try:
        arr = np.array([int(v) for v in vals], dtype=np.int32)
    except Exception:
        return None
    if arr.size < 4:
        return None
    if arr.size < sp.length:
        arr = np.concatenate([arr, np.zeros(sp.length - arr.size, dtype=np.int32)])
    arr = arr[:sp.length]
    arr[0] = np.clip(arr[0], 0, sp.max_t)
    arr[1] = np.clip(arr[1], 0, sp.ncol - int(arr[0]))
    arr[2] = np.clip(arr[2], 0, 3)
    arr[3] = np.clip(arr[3], 0, int(arr[1]))
    arr[4:] = np.clip(arr[4:], 0, N_SLOT_MODES - 1)
    return sp.canonical(arr)


def parse_json(txt: Optional[str]) -> Optional[dict]:
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# --------------------------------------------------------------------------
# Operator 1 -- Pareto-conditioned candidate injection
# --------------------------------------------------------------------------
def format_archive(sp: Space, archive: List[dict], limit: int = 14) -> str:
    if not archive:
        return "(empty)"
    sel = archive
    if len(sel) > limit:                    # keep the spread, thin the middle
        idx = np.linspace(0, len(sel) - 1, limit).round().astype(int)
        sel = [sel[i] for i in dict.fromkeys(idx.tolist())]
    rows = []
    for r in sel:
        spec = genome_to_spec(sp, r["g"])
        spec[_metric()] = float(f"{r.get('err', r['NMED']):.3e}")
        spec["area"] = round(r["area"], 1)
        spec["delay"] = round(r["delay"], 3)
        rows.append("  " + json.dumps(spec))
    return "\n".join(rows)


ANSWER_FMT = ('{"designs": [{"T": int, "A": int, "comp": int, "cpa_k": int, '
              '"columns": {"<col>": "<style>"}}, ...]}')

DECADES = [(1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2)]
MRE_DECADES = [(1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1)]


def decades() -> List[tuple]:
    """The four useful error decades of the active error metric."""
    return MRE_DECADES if _metric() == "MRE" else DECADES


def coverage_note(archive: List[dict]) -> str:
    """Tell the model where the front is thin, in circuit-designer terms."""
    lines, gaps = [], []
    met = _metric()
    for lo, hi in decades():
        sel = [r for r in archive if lo <= r.get("err", r["NMED"]) < hi]
        if sel:
            best = min(sel, key=lambda r: r["area"] * r["delay"])
            lines.append(f"  {met} in [{lo:.0e},{hi:.0e}): {len(sel):2d} design(s), "
                         f"best area*delay = {best['area']*best['delay']:.1f} "
                         f"(area {best['area']:.1f}, delay {best['delay']:.3f})")
        else:
            lines.append(f"  {met} in [{lo:.0e},{hi:.0e}): EMPTY")
            gaps.append(f"[{lo:.0e},{hi:.0e})")
    txt = "Coverage of the current front:\n" + "\n".join(lines)
    if gaps:
        txt += ("\nThese error ranges are not covered at all and are the most "
                "valuable to hit: " + ", ".join(gaps))
    return txt


def _decode(sp: Space, d: Optional[dict], k: int, key: str = "designs"
            ) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if not d:
        return out
    items = d.get(key) or d.get("designs") or d.get("candidates") or \
        d.get("children") or []
    for c in items[:k]:
        g = spec_to_genome(sp, c) if isinstance(c, dict) else _clip_genome(sp, c)
        if g is not None:
            out.append(g)
    return out


def propose_candidates(llm: LLM, sp: Space, archive: List[dict], k: int,
                       history_note: str = "") -> List[np.ndarray]:
    user = (space_doc(sp).replace("{modes}", slot_mode_doc()) +
            f"""
Current non-dominated archive (all three objectives are minimised: {_metric()}, area \
in um^2, critical-path delay in ns):
{format_archive(sp, archive)}

{coverage_note(archive)}
{history_note}

Task: propose {k} NEW designs that are likely to be non-dominated.  Spend most \
of the proposals on the error ranges flagged as empty or thin, and use the rest \
to undercut the best area*delay listed above at an error level it already \
covers.  Do not repeat a design that is already listed, and make each proposal \
differ from the others in more than one field.
Return JSON: {ANSWER_FMT} with exactly {k} entries.""")
    out = _decode(sp, parse_json(llm.chat(SYSTEM, user, max_tokens=6000)), k)
    llm.stats.proposed += k
    return out


def _warm_targets() -> List[str]:
    return (["1e-4", "1e-3", "1e-2", "5e-2"] if _metric() == "MRE"
            else ["1e-5", "1e-4", "1e-3", "1e-2"])


def warm_start(llm: LLM, sp: Space, k: int) -> List[np.ndarray]:
    """Generation-0 population proposed from circuit knowledge alone."""
    user = (space_doc(sp).replace("{modes}", slot_mode_doc()) +
            f"""
No design has been characterised yet.

Task: propose {k} starting designs that already look like good approximate \
multipliers, covering the whole useful accuracy range: roughly one quarter with \
{_metric()} near {_warm_targets()[0]}, one quarter near {_warm_targets()[1]}, one quarter near {_warm_targets()[2]} and one quarter \
near {_warm_targets()[3]}.  Use truncation, compensation, the LOA final adder and heterogeneous \
biased compressors together rather than one mechanism alone.
Return JSON: {ANSWER_FMT} with exactly {k} entries.""")
    out = _decode(sp, parse_json(llm.chat(SYSTEM, user, max_tokens=6000)), k)
    llm.stats.proposed += k
    return out


# --------------------------------------------------------------------------
# Operator 2 -- mutation guidance
# --------------------------------------------------------------------------
def guide_mutation(llm: LLM, sp: Space, parents: List[dict], n_child: int
                   ) -> List[np.ndarray]:
    user = (space_doc(sp).replace("{modes}", slot_mode_doc()) +
            f"""
Parents selected by NSGA-II, with the objectives they achieved:
{format_archive(sp, parents, limit=len(parents))}

Task: for each parent in order produce exactly one MUTATED child.  Change only \
a small number of fields, and choose the change so that the child *dominates* \
its parent if possible -- cheaper at the same error, or more accurate at the \
same cost.  Useful moves: raise T by one and set comp=2; replace a '-' biased \
style by a '+' biased style in the column that dominates the mean error, or the \
converse; increase cpa_k to shorten the carry chain; pull a style back to EXACT \
in the most significant approximate column when the error is too large.
Return JSON: {ANSWER_FMT.replace('"designs"', '"children"')} with exactly \
{n_child} children, in the same order as the parents.""")
    out = _decode(sp, parse_json(llm.chat(SYSTEM, user, max_tokens=6000)),
                  n_child, key="children")
    llm.stats.proposed += n_child
    return out


# --------------------------------------------------------------------------
# Operator 2b -- elicited mutation prior
#
# Instead of steering a handful of children, the model is asked once per few
# generations for a *proposal distribution*: which compressor styles are worth
# trying at which distance above the truncation point.  That prior then biases
# every mutation the EA performs, so one cheap call influences the whole
# generation rather than eight individuals.
# --------------------------------------------------------------------------
OFFSET_KEYS = ["0", "1", "2", "3", "4+"]


@dataclass
class ModePrior:
    by_offset: Dict[str, List[int]] = field(default_factory=dict)
    epsilon: float = 0.25          # probability of ignoring the prior

    def key(self, offset: int) -> str:
        return OFFSET_KEYS[min(offset, 4)] if offset >= 0 else "4+"

    def sample(self, rng, offset: int) -> int:
        pool = self.by_offset.get(self.key(offset))
        if not pool or rng.random() < self.epsilon:
            return int(rng.integers(0, N_SLOT_MODES))
        # geometric preference for the head of the ranked list
        i = min(int(rng.geometric(0.45)) - 1, len(pool) - 1)
        return int(pool[i])


def elicit_prior(llm: LLM, sp: Space, archive: List[dict],
                 gen: int) -> Optional[ModePrior]:
    user = (space_doc(sp).replace("{modes}", slot_mode_doc()) +
            f"""
Non-dominated designs found so far (generation {gen}):
{format_archive(sp, archive, limit=12)}

{coverage_note(archive)}

Task: give a *proposal distribution* for the evolutionary operator.  For a
column that sits d positions above the truncation point (d = 0 is the first
approximate column, "4+" means five or more positions above it), list the
compressor styles that are worth trying there, most promising first, three to
six names per entry.  Base the ranking on the error weight of that column
(2^(T+d)), on the error bias of each style and on the styles that already
appear in the good designs above.
Return JSON: {{"prior": {{"0": ["<style>", ...], "1": [...], "2": [...], \
"3": [...], "4+": [...]}}}}""")
    d = parse_json(llm.chat(SYSTEM, user, max_tokens=2048, temperature=0.4))
    if not d:
        return None
    pri = d.get("prior") or d
    by: Dict[str, List[int]] = {}
    for k in OFFSET_KEYS:
        v = pri.get(k) or pri.get(k.rstrip("+"))
        if not isinstance(v, list):
            continue
        idx = [SLOT_MODE_BY_NAME[str(x).strip().upper()] for x in v
               if str(x).strip().upper() in SLOT_MODE_BY_NAME]
        if idx:
            by[k] = idx
    return ModePrior(by) if by else None


# --------------------------------------------------------------------------
# Operator 3 -- search-space refinement
# --------------------------------------------------------------------------
@dataclass
class Restriction:
    t_lo: int = 0
    t_hi: int = 10 ** 9
    a_lo: int = 0
    a_hi: int = 10 ** 9
    k_lo: int = 0
    k_hi: int = 10 ** 9
    modes: Optional[List[int]] = None

    def apply(self, sp: Space, g: np.ndarray) -> np.ndarray:
        g = np.array(g, dtype=np.int32, copy=True)
        g[0] = int(np.clip(g[0], self.t_lo, min(self.t_hi, sp.max_t)))
        g[1] = int(np.clip(g[1], self.a_lo, min(self.a_hi, sp.ncol - int(g[0]))))
        g[3] = int(np.clip(g[3], self.k_lo, min(self.k_hi, int(g[1]))))
        if self.modes:
            allowed = np.array(sorted(set(self.modes) | {0}), dtype=np.int32)
            sl = g[sp.n_head:]
            bad = ~np.isin(sl, allowed)
            if bad.any():
                sl[bad] = allowed[np.abs(allowed[None, :] -
                                         sl[bad][:, None]).argmin(axis=1)]
        return sp.canonical(g)


def refine_space(llm: LLM, sp: Space, archive: List[dict], gen: int,
                 budget_left: int) -> Optional[Restriction]:
    user = (space_doc(sp).replace("{modes}", slot_mode_doc()) +
            f"""
Generation {gen}; {budget_left} circuit evaluations remain.
Non-dominated archive so far:
{format_archive(sp, archive, limit=16)}

Task: restrict the search space so the remaining budget is spent where good \
trade-offs actually live.  Give inclusive bounds for T, A and cpa_k, and the \
subset of slot-style indices worth keeping (style 0 = EXACT is always kept).
Keep the ranges wide enough to preserve the whole observed front (do not \
collapse it to a single point).
Return JSON: {{"T": [lo, hi], "A": [lo, hi], "cpa_k": [lo, hi], \
"modes": [ints]}}""")
    d = parse_json(llm.chat(SYSTEM, user, max_tokens=1024, temperature=0.3))
    if not d:
        return None
    def rng(key, dflt_hi):
        v = d.get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                lo, hi = int(v[0]), int(v[1])
                if lo <= hi:
                    return max(0, lo), min(dflt_hi, hi)
            except Exception:
                pass
        return 0, dflt_hi
    t_lo, t_hi = rng("T", sp.max_t)
    a_lo, a_hi = rng("A", sp.ncol)
    k_lo, k_hi = rng("cpa_k", sp.ncol)
    modes = d.get("modes")
    if isinstance(modes, list):
        modes = [int(m) for m in modes if isinstance(m, (int, float))
                 and 0 <= int(m) < N_SLOT_MODES]
        modes = modes or None
    else:
        modes = None
    if t_hi - t_lo < 2:                      # never collapse the front
        t_lo, t_hi = max(0, t_lo - 1), min(sp.max_t, t_hi + 1)
    return Restriction(t_lo, t_hi, a_lo, a_hi, k_lo, k_hi, modes)


# --------------------------------------------------------------------------
# LLM-only search (baseline): iterative, feedback-driven, no EA machinery
# --------------------------------------------------------------------------
def llm_only_batch(llm: LLM, sp: Space, archive: List[dict], k: int,
                   evaluated: int, budget: int) -> List[np.ndarray]:
    user = (space_doc(sp).replace("{modes}", slot_mode_doc()) +
            f"""
You are performing the whole design-space exploration yourself.  {evaluated} of \
{budget} circuit evaluations have been used.
Best non-dominated designs found so far:
{format_archive(sp, archive, limit=16)}

Task: choose the next batch of {k} designs to evaluate so that the final \
non-dominated set covers the accuracy/area/delay trade-off as widely and as \
deeply as possible.
Return JSON: {ANSWER_FMT} with exactly {k} entries.""")
    out = _decode(sp, parse_json(llm.chat(SYSTEM, user, max_tokens=8000)), k)
    llm.stats.proposed += k
    return out
