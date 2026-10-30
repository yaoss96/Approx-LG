"""Reproductions of the two GA/EA-based reference methods, and the harness
that characterises the published RTL of AM-Lib and MulApprox in our flow.

* :func:`run_eamva`  -- "Design of Hardware-Efficient Inexact Multiplier Using
  Evolutionary Algorithm Factored by Multi-variate Approximation" (ISQED'25):
  NSGA-II over (truncated bits, approximated bits, compressor configuration)
  with delay / area / power / error-rate objectives.

* :func:`run_biased_ga` -- "Biased Compressor Based Approximate Multiplier
  Design Using Genetic Algorithm" (ISCAS'25): design-space pruning by an
  approximation upper bound, a single-objective GA on
  ``a*P/P_exact + b*|EM-EM_req|/EM_req`` over heterogeneous biased
  compressors, followed by a greedy input-order adjustment.

Neither paper ships code, so both are re-implemented inside our template using
the cell definitions and algorithm descriptions given in the papers.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .arch import Space
from .cells import SLOT_MODE_BY_NAME
from .engine import Engine, Record
from .metrics import crowding_distance, fast_nondominated_sort
from .search import (Tracker, environmental_selection, eval_batch,
                     tournament)

# Cell pools of the two reference works, expressed in our slot alphabet.
EAMVA_MODES = [SLOT_MODE_BY_NAME[m] for m in
               ("EXACT", "A1", "A2", "A3", "A4", "A5", "K1", "K2", "K4")]
BIASED_MODES = [SLOT_MODE_BY_NAME[m] for m in
                ("EXACT", "P1", "P2", "N1", "N2", "B1", "K3", "K5", "K7", "K9")]


def _restrict(sp: Space, g: np.ndarray, modes: Sequence[int],
              t_hi: Optional[int] = None, allow_comp: bool = False,
              allow_cpa: bool = False) -> np.ndarray:
    g = np.array(g, dtype=np.int32, copy=True)
    if t_hi is not None:
        g[0] = int(np.clip(g[0], 0, t_hi))
    if not allow_comp:
        g[2] = 0
    if not allow_cpa:
        g[3] = 0
    allowed = np.array(sorted(set(modes)), dtype=np.int32)
    sl = g[sp.n_head:]
    bad = ~np.isin(sl, allowed)
    if bad.any():
        sl[bad] = allowed[np.random.randint(0, len(allowed), int(bad.sum()))]
    return sp.canonical(g)


def _rand(sp: Space, rng, modes, t_hi=None, allow_comp=False, allow_cpa=False):
    g = sp.random(rng)
    return _restrict(sp, g, modes, t_hi, allow_comp, allow_cpa)


def _mut(sp: Space, g, rng, modes, t_hi=None, allow_comp=False,
         allow_cpa=False, pm=None):
    from .search import mutate
    return _restrict(sp, mutate(sp, g, rng, pm), modes, t_hi, allow_comp,
                     allow_cpa)


def _xover(sp: Space, a, b, rng, modes, **kw):
    from .search import crossover
    return _restrict(sp, crossover(sp, a, b, rng), modes, **kw)


# --------------------------------------------------------------------------
# EA-MVA (ISQED'25)
# --------------------------------------------------------------------------
def run_eamva(eng: Engine, sp: Space, budget: int, seed: int, tracker: Tracker,
              pop: int = 32, pc: float = 0.9, pm: float = 0.1, **kw) -> Dict:
    """NSGA-II on (TBits, ABits, compressor pool); objectives delay/area/power/ER."""
    rng = np.random.default_rng(seed)

    def objs(r: Record) -> np.ndarray:
        return np.array([r.delay, r.area, r.power, r.ER], dtype=float)

    P = eng.evaluate([_rand(sp, rng, EAMVA_MODES) for _ in range(pop)],
                     budget=budget, used=eng.n_eval)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    while eng.n_eval < budget and P:
        F = np.array([objs(r) for r in P])
        fronts = fast_nondominated_sort(F)
        rank = np.zeros(len(P), dtype=int)
        crowd = np.zeros(len(P))
        for ri, fr in enumerate(fronts):
            rank[fr] = ri
            crowd[fr] = crowding_distance(F[fr])
        kids = []
        while len(kids) < pop:
            a = tournament(rng, rank, crowd)
            b = tournament(rng, rank, crowd)
            c = (_xover(sp, P[a].g, P[b].g, rng, EAMVA_MODES)
                 if rng.random() < pc else np.array(P[a].g, copy=True))
            kids.append(_mut(sp, c, rng, EAMVA_MODES, pm=pm))
        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        pool = P + Q
        Fp = np.array([objs(r) for r in pool])
        fronts = fast_nondominated_sort(Fp)
        nxt: List[Record] = []
        for fr in fronts:
            if len(nxt) + len(fr) <= pop:
                nxt.extend(pool[i] for i in fr)
            else:
                cd = crowding_distance(Fp[fr])
                for i in np.argsort(-cd)[: pop - len(nxt)]:
                    nxt.append(pool[fr[i]])
                break
        P = nxt
    return {}


# --------------------------------------------------------------------------
# Biased-compressor GA (ISCAS'25)
# --------------------------------------------------------------------------
def approximation_upper_bound(eng: Engine, sp: Space, em_req: float,
                              metric: str = "NMED", budget: int = 10 ** 9
                              ) -> int:
    """Algorithm 1 of Wen et al.: grow the approximate window column by column
    with the *lowest-error* approximate style until the error metric exceeds
    the expected value; the last feasible column is the upper bound.

    The bound needs only the error of each probe, so a probe that is not
    already characterised is simulated rather than synthesised, and charged to
    the budget as a characterisation would be.  At 32x32 these near-exact
    probes are the one kind of circuit ABC cannot map in the flow's time limit
    (the third does not finish in 10 minutes, while BC-GA's own designs take
    seconds), and a failed probe would end the search for the bound.  At 8
    and 16 bits every probe synthesises, so the bound and the run are those
    of a synthesised probe."""
    from .simulate import error_metrics, simulate
    best_mode = SLOT_MODE_BY_NAME["N1"]        # exact sum, under-estimating carry
    m = 0
    for i in range(1, sp.ncol):
        g = np.zeros(sp.length, dtype=np.int32)
        g[1] = i
        sl = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
        sl[:, :i] = best_mode
        g[sp.n_head:] = sl.reshape(-1)
        k = eng.key(g)
        if k not in eng.run_seen:
            if eng.n_eval >= budget:
                break
            eng.run_seen.add(k)
            eng.n_eval += 1
        r = eng.cache.get(k)
        if r is not None:
            err = getattr(r, metric)
        else:
            nl = eng.builder.build(sp.canonical(g))
            err = error_metrics(simulate(nl, eng.ev.stim), eng.ev.stim)[metric]
        if err < em_req:
            m = i
        else:
            break
    return m


def run_biased_ga(eng: Engine, sp: Space, budget: int, seed: int,
                  tracker: Tracker, pop: int = 24,
                  em_targets: Sequence[float] = (2e-4, 5e-4, 1e-3, 2e-3,
                                                 5e-3, 1e-2, 2e-2),
                  a_coef: float = 1.0, b_coef: float = 1.0,
                  input_order_opt: bool = True, **kw) -> Dict:
    rng = np.random.default_rng(seed)
    exact = eng.evaluate([np.zeros(sp.length, dtype=np.int32)])[0]
    per = max(8, budget // max(1, len(em_targets)))
    for em_req in em_targets:
        start = eng.n_eval
        cap = min(budget, start + per)
        m = approximation_upper_bound(eng, sp, em_req, budget=cap)
        if eng.n_eval >= cap:
            break

        def fitness(r: Record) -> float:
            return (a_coef * r.power / max(exact.power, 1e-12) +
                    b_coef * abs(r.NMED - em_req) / em_req)

        def sample():
            g = np.zeros(sp.length, dtype=np.int32)
            g[0] = 0
            g[1] = m
            sl = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
            for j in range(m):
                sl[:, j] = rng.choice(BIASED_MODES, size=sp.nstage)
            g[sp.n_head:] = sl.reshape(-1)
            return sp.canonical(g)

        cur = eng.evaluate([sample() for _ in range(pop)], budget=cap,
                           used=eng.n_eval)
        tracker.add(cur)
        tracker.snapshot(eng.n_eval)
        while eng.n_eval < cap and cur:
            fit = np.array([fitness(r) for r in cur])
            order = np.argsort(fit)
            elite = [cur[i] for i in order[: max(2, pop // 6)]]
            kids = []
            while len(kids) < pop:
                i = order[min(int(abs(rng.normal(0, max(len(cur) / 3, 1)))),
                              len(cur) - 1)]
                j = order[min(int(abs(rng.normal(0, max(len(cur) / 3, 1)))),
                              len(cur) - 1)]
                c = _xover(sp, cur[i].g, cur[j].g, rng, BIASED_MODES, t_hi=0)
                kids.append(_mut(sp, c, rng, BIASED_MODES, t_hi=0))
            before = eng.n_eval
            kr = eng.evaluate(kids, budget=cap)
            tracker.add(kr)
            tracker.snapshot(eng.n_eval)
            if not kr or eng.n_eval == before:
                break
            pool = elite + kr
            cur = [pool[i] for i in np.argsort([fitness(r) for r in pool])[:pop]]
        # Algorithm 2: greedy input-order adjustment of the best individual
        if input_order_opt and cur and eng.n_eval < budget:
            best = min(cur, key=fitness)
            _input_order_greedy(eng, sp, best.g, tracker, budget)
    return {}


def _input_order_greedy(eng: Engine, sp: Space, g: np.ndarray, tracker: Tracker,
                        budget: int, trials: int = 12) -> None:
    """Greedy search over the per-slot input ordering (Algorithm 2)."""
    from .simulate import error_metrics, simulate
    perm = np.zeros((sp.nstage, sp.ncol), dtype=np.int32)
    builder = eng.builder
    stim = eng.ev.stim
    base = error_metrics(simulate(builder.build(g, perm), stim), stim)["NMED"]
    slots = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
    cand = [(s, j) for s in range(sp.nstage) for j in range(sp.ncol)
            if slots[s, j] != 0]
    for (s, j) in cand[:trials]:
        best_code, best_err = 0, base
        for code in (1, 2):
            perm[s, j] = code
            e = error_metrics(simulate(builder.build(g, perm), stim), stim)["NMED"]
            if e < best_err:
                best_code, best_err = code, e
        perm[s, j] = best_code
        base = best_err
    # the reordered design is a real circuit: characterise it once
    nl = builder.build(g, perm)
    if eng.n_eval < budget:
        er = error_metrics(simulate(nl, stim), stim)
        ppa = eng.synth.run(nl.to_verilog())
        if not ppa.ok:              # a failed characterisation is not a design
            return
        eng.n_eval += 1
        rec = Record(g=g, NMED=er["NMED"], MED=er["MED"], WCE=er["WCE"],
                     ER=er["ER"], MRE=er["MRE"], MSE=er["MSE"],
                     area=ppa.area, delay=ppa.delay, power=ppa.power,
                     gates=ppa.gates, cells=nl.n_cells,
                     approx_cells=nl.n_approx_cells, eval_index=eng.n_eval)
        tracker.records.append(rec)
