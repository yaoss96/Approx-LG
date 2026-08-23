"""Design-point evaluation pipeline and EDA-budget bookkeeping.

Every search algorithm sees exactly the same interface and the same cost model:
one *new* design point costs one unit of the EDA budget, regardless of which
algorithm asked for it.  Points already characterised (by any algorithm, in any
run) are served from the on-disk cache and are free, which is what makes a
budget-matched comparison meaningful.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .arch import Builder, Space
from .metrics import nondominated_mask
from .simulate import Evaluator
from .synth import PPA, Synthesizer

OBJECTIVES = ("log10(NMED)", "area", "delay")
NMED_FLOOR = 1e-7


class BudgetExhausted(Exception):
    pass


@dataclass
class Record:
    g: np.ndarray
    NMED: float
    MED: float
    WCE: float
    ER: float
    MRE: float
    MSE: float
    area: float
    delay: float
    power: float
    gates: int
    cells: int
    approx_cells: int
    eval_index: int = -1

    def obj(self) -> np.ndarray:
        """Objective vector (all minimised).

        The error axis is logarithmic: an approximate multiplier is useful over
        four to five decades of NMED, so a linear error axis would collapse the
        whole accurate region into a few percent of the objective range and
        distort both the crowding distance and the hypervolume.  Dominance is
        unaffected because log is strictly monotone."""
        return np.array([math.log10(max(self.NMED, NMED_FLOOR)),
                         self.area, self.delay], dtype=float)

    def to_json(self) -> Dict:
        d = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
             for k, v in self.__dict__.items() if k != "g"}
        d["g"] = [int(x) for x in self.g]
        return d


class Engine:
    """Evaluates genomes: exhaustive error simulation + logic synthesis."""

    def __init__(self, space: Space, synth: Synthesizer,
                 evaluator: Optional[Evaluator] = None,
                 nworkers: int = 16, cache_file: Optional[str] = None):
        self.space = space
        self.synth = synth
        self.ev = evaluator or Evaluator(space)
        self.builder = Builder(space)
        self.nworkers = nworkers
        self.cache: Dict[bytes, Record] = {}
        self.cache_file = cache_file
        self.n_eval = 0                 # budget consumed (new points only)
        self.n_hit = 0
        self.n_fail = 0
        # wall-clock accounting.  Total run time is not comparable across
        # algorithms once the on-disk cache is warm, so the EDA cost is timed
        # where it is actually paid and reported per *characterisation*.
        self.n_synth = 0
        self.t_sim = 0.0
        self.t_synth = 0.0
        self._lock = threading.Lock()
        if cache_file and os.path.exists(cache_file):
            self._load(cache_file)

    # ------------------------------------------------------------------
    def key(self, g: np.ndarray) -> str:
        """Structural key: the SHA-1 of the emitted netlist.

        Different genomes can decode to the *same* circuit (for instance when a
        style is assigned to a column that holds a single partial product).
        Keying on the netlist makes every algorithm pay for a circuit exactly
        once, so the budget really counts distinct EDA characterisations."""
        return hashlib.sha1(
            self.builder.build(g).to_verilog().encode()).hexdigest()

    def _load(self, path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                g = np.array(d.pop("g"), dtype=np.int32)
                self.cache[self.key(g)] = Record(g=g, **d)

    def _append(self, rec: Record):
        if not self.cache_file:
            return
        with open(self.cache_file, "a") as fh:
            fh.write(json.dumps(rec.to_json()) + "\n")

    # ------------------------------------------------------------------
    def start_run(self):
        """Begin a new search run: the budget counter and the per-run set of
        visited design points are reset, while the *persistent* PPA cache is
        kept.  A design point counts against the budget the first time a run
        asks for it, whether or not another run has characterised it before;
        this is what keeps the budget-matched comparison honest."""
        self.n_eval = 0
        self.n_fail = 0
        self.n_synth = 0
        self.t_sim = 0.0
        self.t_synth = 0.0
        self.run_seen = set()

    def evaluate(self, genomes: Sequence[np.ndarray], budget: Optional[int] = None,
                 used: int = 0) -> List[Record]:
        """Evaluate a batch; returns the records of the points served."""
        if not hasattr(self, "run_seen"):
            self.run_seen = set()
        keys = [self.key(g) for g in genomes]
        todo, todo_keys = [], []
        served = []
        for g, k in zip(genomes, keys):
            if k in self.run_seen:
                served.append(k)
                continue
            if budget is not None and self.n_eval >= budget:
                continue
            self.run_seen.add(k)
            self.n_eval += 1
            served.append(k)
            if k not in self.cache:
                todo.append(self.space.canonical(g))
                todo_keys.append(k)

        if todo:
            nls = [self.builder.build(g) for g in todo]
            errs = []
            t0 = time.time()
            for nl in nls:
                from .simulate import error_metrics, simulate
                errs.append(error_metrics(simulate(nl, self.ev.stim), self.ev.stim))
            self.t_sim += time.time() - t0
            vs = [nl.to_verilog() for nl in nls]
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=self.nworkers) as ex:
                ppas = list(ex.map(lambda v: self.synth.run(v), vs))
            self.t_synth += time.time() - t0
            self.n_synth += len(vs)
            for g, k, nl, er, ppa in zip(todo, todo_keys, nls, errs, ppas):
                if not ppa.ok:
                    # a design that could not be characterised is dropped, not
                    # recorded: keeping it would poison every later run
                    self.n_fail += 1
                    self.run_seen.discard(k)
                    self.n_eval -= 1
                    continue
                with self._lock:
                    rec = Record(g=g, NMED=er["NMED"], MED=er["MED"],
                                 WCE=er["WCE"], ER=er["ER"], MRE=er["MRE"],
                                 MSE=er["MSE"], area=ppa.area, delay=ppa.delay,
                                 power=ppa.power, gates=ppa.gates,
                                 cells=nl.n_cells, approx_cells=nl.n_approx_cells,
                                 eval_index=self.n_eval)
                    self.cache[k] = rec
                    self._append(rec)
            self.synth.save_cache()

        out = []
        for k in served:
            r = self.cache.get(k)
            if r is not None:
                out.append(r)
        return out

    # ------------------------------------------------------------------
    def all_records(self) -> List[Record]:
        return list(self.cache.values())


def pareto_of(records: Sequence[Record]) -> List[Record]:
    if not records:
        return []
    F = np.array([r.obj() for r in records])
    keep = nondominated_mask(F)
    sel = [r for r, k in zip(records, keep) if k]
    sel.sort(key=lambda r: r.NMED)
    return sel


def archive_dicts(records: Sequence[Record]) -> List[Dict]:
    return [{"g": r.g, "NMED": r.NMED, "area": r.area, "delay": r.delay}
            for r in records]
