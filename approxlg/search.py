"""Search algorithms compared in the paper.

All algorithms share the same design space, the same evaluator and the same
EDA budget (number of *newly characterised* design points).

    random     uniform random sampling
    ga         single-objective GA on a weighted scalarisation (the fitness
               shape used by GA-based approximate-multiplier works)
    nsga2      textbook NSGA-II
    llm_only   iterative LLM batch proposal with Pareto feedback, no EA
    nsga2_expert    NSGA-II + a fixed designer prior over compressor styles
    nsga2_adaptive  NSGA-II + an online credit-assignment prior over the same
                    channel; both are the no-LLM controls for the guidance
    nsga2_llmprior  NSGA-II + the elicited prior and nothing else, the LLM arm
                    of that three-way control
    approxlg   NSGA-II + LLM candidate injection + LLM mutation guidance +
               LLM search-space refinement  (this work)

``approxlg`` accepts flags to disable each of the three operators, which is how
the ablation study is produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .arch import Space
from .cells import N_SLOT_MODES
from .engine import Engine, Record, archive_dicts, pareto_of
from .llm import (LLM, DECADES, ModePrior, Restriction, decades, elicit_prior,
                  guide_mutation, llm_only_batch, propose_candidates,
                  refine_space, warm_start)
from .metrics import crowding_distance, fast_nondominated_sort, hypervolume


# --------------------------------------------------------------------------
@dataclass
class SearchResult:
    name: str
    seed: int
    records: List[Record] = field(default_factory=list)
    hv_trace: List[tuple] = field(default_factory=list)   # (n_eval, hv)
    llm_stats: Dict = field(default_factory=dict)
    seconds: float = 0.0
    accepted_llm: int = 0
    proposed_llm: int = 0

    def pareto(self):
        return pareto_of(self.records)


class Tracker:
    """Records evaluation order and the hypervolume trajectory."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray,
                 ref: Optional[np.ndarray] = None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.ref = np.ones(len(self.lo)) if ref is None else ref
        self.records: List[Record] = []
        self.trace: List[tuple] = []
        self._seen = set()

    def add(self, recs: Sequence[Record]):
        for r in recs:
            k = r.g.tobytes()
            if k in self._seen:
                continue
            self._seen.add(k)
            self.records.append(r)

    def norm_front(self) -> np.ndarray:
        F = np.array([r.obj() for r in self.records])
        return (F - self.lo) / np.where(self.hi - self.lo > 0,
                                        self.hi - self.lo, 1.0)

    def hv(self) -> float:
        if not self.records:
            return 0.0
        return hypervolume(self.norm_front(), self.ref)

    def snapshot(self, n_eval: int):
        self.trace.append((n_eval, self.hv()))


# --------------------------------------------------------------------------
# genetic operators
# --------------------------------------------------------------------------
def crossover(sp: Space, p1: np.ndarray, p2: np.ndarray, rng) -> np.ndarray:
    c = p1.copy()
    for i in range(sp.n_head):
        if rng.random() < 0.5:
            c[i] = p2[i]
    s1 = p1[sp.n_head:].reshape(sp.nstage, sp.ncol)
    s2 = p2[sp.n_head:].reshape(sp.nstage, sp.ncol)
    cut = rng.integers(0, sp.ncol + 1)
    cs = np.concatenate([s1[:, :cut], s2[:, cut:]], axis=1)
    c[sp.n_head:] = cs.reshape(-1)
    return sp.canonical(c)


def mutate(sp: Space, g: np.ndarray, rng, pm: Optional[float] = None,
           restr: Optional[Restriction] = None,
           prior: Optional[ModePrior] = None) -> np.ndarray:
    g = g.copy()
    pm = pm if pm is not None else 1.0 / sp.length
    if rng.random() < 0.25:
        g[0] = int(np.clip(g[0] + rng.integers(-1, 2), 0, sp.max_t))
    if rng.random() < 0.25:
        g[1] = int(np.clip(g[1] + rng.integers(-2, 3), 0, sp.ncol - int(g[0])))
    if rng.random() < 0.15:
        g[2] = int(rng.integers(0, 4))
    if rng.random() < 0.20:
        g[3] = int(np.clip(g[3] + rng.integers(-1, 2), 0, int(g[1])))
    slots = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
    m = rng.random(slots.shape) < max(pm, 2.0 / slots.size)
    idx = np.argwhere(m)
    if prior is None:
        if len(idx):
            slots[m] = rng.integers(0, N_SLOT_MODES, size=len(idx))
    else:
        t = int(g[0])
        for s_, j_ in idx:
            slots[s_, j_] = prior.sample(rng, int(j_) - t)
    g[sp.n_head:] = slots.reshape(-1)
    g = sp.canonical(g)
    if restr is not None:
        g = restr.apply(sp, g)
    return g


def tournament(rng, rank: np.ndarray, crowd: np.ndarray, k: int = 2) -> int:
    idx = rng.integers(0, len(rank), size=k)
    best = idx[0]
    for i in idx[1:]:
        if rank[i] < rank[best] or (rank[i] == rank[best] and crowd[i] > crowd[best]):
            best = i
    return int(best)


def environmental_selection(recs: List[Record], n: int) -> List[Record]:
    F = np.array([r.obj() for r in recs])
    fronts = fast_nondominated_sort(F)
    out: List[Record] = []
    for fr in fronts:
        if len(out) + len(fr) <= n:
            out.extend(recs[i] for i in fr)
        else:
            cd = crowding_distance(F[fr])
            order = np.argsort(-cd)
            for i in order[: n - len(out)]:
                out.append(recs[fr[i]])
            break
        if len(out) >= n:
            break
    return out



def decade_elites(records: List[Record], k: int) -> List[Record]:
    """One elite per error decade: the cheapest design (area*delay) whose NMED
    falls in that decade.  These are the points an LLM can most usefully try to
    dominate."""
    out: List[Record] = []
    for lo, hi in decades():
        sel = [r for r in records if lo <= r.err < hi]
        if sel:
            out.append(min(sel, key=lambda r: r.area * r.delay))
        if len(out) >= k:
            break
    return out[:k]


def _ranks(recs: List[Record]):
    F = np.array([r.obj() for r in recs])
    fronts = fast_nondominated_sort(F)
    rank = np.zeros(len(recs), dtype=int)
    crowd = np.zeros(len(recs))
    for ri, fr in enumerate(fronts):
        rank[fr] = ri
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd



# --------------------------------------------------------------------------
def eval_batch(eng: Engine, sp: Space, kids: List[np.ndarray], budget: int,
               rng, tracker: "Tracker", pop: int) -> List[Record]:
    """Evaluate one generation.

    Because the budget counts *distinct circuits*, a converged population can
    propose a whole generation of already-characterised designs and consume
    nothing.  When that happens the generation is topped up with random
    immigrants so that every algorithm always spends its budget.
    """
    before = eng.n_eval
    Q = eng.evaluate(kids, budget=budget)
    if eng.n_eval == before and eng.n_eval < budget:
        Q = Q + eng.evaluate([sp.random(rng) for _ in range(pop)], budget=budget)
    tracker.add(Q)
    tracker.snapshot(eng.n_eval)
    return Q


def _init_pop(sp: Space, rng, n: int) -> List[np.ndarray]:
    return [sp.random(rng) for _ in range(n)]


def run_random(eng: Engine, sp: Space, budget: int, seed: int, tracker: Tracker,
               pop: int = 32, **kw) -> None:
    rng = np.random.default_rng(seed)
    while eng.n_eval < budget:
        before = eng.n_eval
        recs = eng.evaluate([sp.random(rng) for _ in range(pop)], budget=budget)
        tracker.add(recs)
        tracker.snapshot(eng.n_eval)
        if eng.n_eval == before:
            break


def run_ga(eng: Engine, sp: Space, budget: int, seed: int, tracker: Tracker,
           pop: int = 32, weights=(1 / 3, 1 / 3, 1 / 3), norm=None, **kw) -> None:
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    lo, hi = tracker.lo, tracker.hi
    span = np.where(hi - lo > 0, hi - lo, 1.0)

    def scal(r):
        return float((((r.obj() - lo) / span) * w).sum())
    parents = _init_pop(sp, rng, pop)
    recs = eng.evaluate(parents, budget=budget, used=eng.n_eval)
    tracker.add(recs)
    tracker.snapshot(eng.n_eval)
    cur = recs
    while eng.n_eval < budget and cur:
        fit = np.array([scal(r) for r in cur])
        order = np.argsort(fit)
        elite = [cur[i] for i in order[: max(2, pop // 8)]]
        kids = []
        while len(kids) < pop:
            i = order[min(int(abs(rng.normal(0, len(cur) / 3))), len(cur) - 1)]
            j = order[min(int(abs(rng.normal(0, len(cur) / 3))), len(cur) - 1)]
            kids.append(mutate(sp, crossover(sp, cur[i].g, cur[j].g, rng), rng))
        before = eng.n_eval
        kr = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not kr or eng.n_eval == before:
            break
        pool = elite + kr
        fit = np.array([scal(r) for r in pool])
        cur = [pool[i] for i in np.argsort(fit)[:pop]]


def run_nsga2(eng: Engine, sp: Space, budget: int, seed: int, tracker: Tracker,
              pop: int = 32, **kw) -> None:
    rng = np.random.default_rng(seed)
    P = eng.evaluate(_init_pop(sp, rng, pop), budget=budget, used=eng.n_eval)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    while eng.n_eval < budget and P:
        rank, crowd = _ranks(P)
        kids = []
        while len(kids) < pop:
            a = tournament(rng, rank, crowd)
            b = tournament(rng, rank, crowd)
            kids.append(mutate(sp, crossover(sp, P[a].g, P[b].g, rng), rng))
        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        P = environmental_selection(P + Q, pop)


def run_llm_only(eng: Engine, sp: Space, budget: int, seed: int, tracker: Tracker,
                 pop: int = 32, llm: Optional[LLM] = None, **kw) -> None:
    rng = np.random.default_rng(seed)
    init = warm_start(llm, sp, pop) if llm else []
    init += _init_pop(sp, rng, max(0, pop - len(init)))
    P = eng.evaluate(init, budget=budget)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    while eng.n_eval < budget:
        arch = archive_dicts(pareto_of(tracker.records))
        cand = llm_only_batch(llm, sp, arch, pop, eng.n_eval, budget)
        if len(cand) < pop:                      # fail-closed top-up
            cand += [sp.random(rng) for _ in range(pop - len(cand))]
        before = eng.n_eval
        R = eval_batch(eng, sp, cand, budget, rng, tracker, pop)
        if not R or eng.n_eval == before:
            break


def run_nsga2_prior(eng: Engine, sp: Space, budget: int, seed: int,
                    tracker: Tracker, pop: int = 32, kind: str = "expert",
                    prior_every: int = 3, **kw) -> Dict:
    """NSGA-II whose mutation draws slot genes from a non-LLM proposal
    distribution.

    This is the control for the ablation's one significant operator.  Every
    other part of the loop -- sampler, crossover, mutation rates, epsilon, the
    geometric rank draw, environmental selection -- is the one Approx-LG uses,
    so the difference between this and the guided run is only the provenance of
    the ranked style lists.  ``kind`` selects the fixed designer prior, the online
    credit-assignment one, or the elicited one; the last is the arm that
    isolates what the language model contributes to this one channel and
    nothing else.
    """
    from .priors import AdaptivePrior, expert_prior

    rng = np.random.default_rng(seed)
    prior: Optional[ModePrior] = expert_prior() if kind == "expert" else None
    adaptive = (AdaptivePrior(sp.n_head, sp.nstage, sp.ncol)
                if kind == "adaptive" else None)
    llm: Optional[LLM] = kw.get("llm") if kind == "llm" else None
    n_updates = 0

    P = eng.evaluate(_init_pop(sp, rng, pop), budget=budget, used=eng.n_eval)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    gen = 0
    while eng.n_eval < budget and P:
        gen += 1
        rank, crowd = _ranks(P)
        kids = []
        while len(kids) < pop:
            a = tournament(rng, rank, crowd)
            b = tournament(rng, rank, crowd)
            kids.append(mutate(sp, crossover(sp, P[a].g, P[b].g, rng), rng,
                               prior=prior))
        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        P = environmental_selection(P + Q, pop)
        if (gen - 1) % prior_every:
            continue
        pr = None
        if adaptive is not None:
            pr = adaptive.update([r.g for r in tracker.records],
                                 [r.g for r in pareto_of(tracker.records)])
        elif llm is not None:
            pr = elicit_prior(llm, sp,
                              archive_dicts(pareto_of(tracker.records)), gen)
        if pr is not None:
            prior = pr
            n_updates += 1
    return {"prior_updates": n_updates}


def run_approxlg(eng: Engine, sp: Space, budget: int, seed: int, tracker: Tracker,
                 pop: int = 32, llm: Optional[LLM] = None,
                 use_inject: bool = True, use_mutguide: bool = True,
                 use_refine: bool = True, use_warm: bool = True,
                 use_prior: bool = True,
                 inject_every: int = 2,
                 n_inject: int = 6, n_mutguide: int = 6, refine_every: int = 5,
                 prior_every: int = 3, warm_frac: float = 0.5,
                 survivor: str = "nsga2", **kw) -> Dict:
    """The guided loop.  ``survivor`` picks the environmental selection the
    operators sit on: NSGA-II's rank-and-crowding (the paper's default) or
    SMS-EMOA's hypervolume-contribution reduction, which is how the guidance
    is tested on top of the stronger unguided optimizer."""
    rng = np.random.default_rng(seed)
    if survivor == "sms":
        from .moo import _norm, _sms_reduce

        def select(pool, n):
            pool = list({r.g.tobytes(): r for r in pool}.values())
            return _sms_reduce(pool, _norm(tracker, pool), n, np.ones(3))
    else:
        select = environmental_selection
    restr: Optional[Restriction] = None
    accepted_keys: set = set()
    proposed = 0
    llm_keys: set = set()          # design points first proposed by the LLM
    seen_keys: set = set()
    n_restrict = 0
    prior: Optional[ModePrior] = None

    # generation 0: half the population comes from circuit knowledge
    init: List[np.ndarray] = []
    n_warm = int(round(pop * warm_frac))
    if use_warm and llm and n_warm:
        init = warm_start(llm, sp, n_warm)
        proposed += n_warm
        llm_keys |= {g.tobytes() for g in init}
    init += _init_pop(sp, rng, pop - len(init))
    P = eng.evaluate(init, budget=budget)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    gen = 0
    while eng.n_eval < budget and P:
        gen += 1
        rank, crowd = _ranks(P)
        kids: List[np.ndarray] = []
        llm_tag: List[bool] = []

        # --- operator 1: Pareto-conditioned candidate injection -----------
        if use_inject and llm and gen % inject_every == 1:
            arch = archive_dicts(pareto_of(tracker.records))
            note = (f"Budget: {eng.n_eval}/{budget} evaluations used; "
                    f"generation {gen}.")
            cand = propose_candidates(llm, sp, arch, n_inject, note)
            proposed += n_inject
            for c in cand[:n_inject]:
                kids.append(restr.apply(sp, c) if restr else c)
                llm_tag.append(True)

        # --- operator 2: mutation guidance --------------------------------
        if use_mutguide and llm and len(kids) < pop:
            n_sel = min(n_mutguide, pop - len(kids))
            sel = decade_elites(tracker.records, n_sel)
            sel += [P[tournament(rng, rank, crowd)]
                    for _ in range(n_sel - len(sel))]
            ch = guide_mutation(llm, sp, archive_dicts(sel), len(sel))
            proposed += len(sel)
            for c in ch:
                if len(kids) >= pop:
                    break
                kids.append(restr.apply(sp, c) if restr else c)
                llm_tag.append(True)

        # --- classic variation fills the rest -----------------------------
        while len(kids) < pop:
            a = tournament(rng, rank, crowd)
            b = tournament(rng, rank, crowd)
            kids.append(mutate(sp, crossover(sp, P[a].g, P[b].g, rng), rng,
                               restr=restr, prior=prior))
            llm_tag.append(False)

        seen_keys |= {r.g.tobytes() for r in tracker.records}
        for g, tag in zip(kids, llm_tag):
            if tag and g.tobytes() not in seen_keys:
                llm_keys.add(g.tobytes())
        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        P = select(P + Q, pop)

        # LLM children that ever survived environmental selection
        accepted_keys |= (llm_keys & {r.g.tobytes() for r in P})

        # --- operator 2b: refresh the elicited mutation prior --------------
        # ``use_mutguide`` switches off two different things at once, the
        # guided children above and the proposal distribution here, so a
        # separate flag is needed to tell which of them the operator's effect
        # belongs to.
        if use_prior and use_mutguide and llm and (gen - 1) % prior_every == 0:
            pr = elicit_prior(llm, sp, archive_dicts(pareto_of(tracker.records)),
                              gen)
            if pr is not None:
                prior = pr

        # --- operator 3: search-space refinement --------------------------
        if use_refine and llm and gen % refine_every == 0:
            arch = archive_dicts(pareto_of(tracker.records))
            r = refine_space(llm, sp, arch, gen, budget - eng.n_eval)
            if r is not None:
                restr = r
                n_restrict += 1
    pf_keys = {r.g.tobytes() for r in pareto_of(tracker.records)}
    return {"accepted": len(accepted_keys), "proposed": proposed,
            "llm_points": len(llm_keys),
            "llm_on_front": len(llm_keys & pf_keys),
            "front": len(pf_keys), "restrictions": n_restrict}


ALGORITHMS: Dict[str, Callable] = {
    "random": run_random,
    "ga": run_ga,
    "nsga2": run_nsga2,
    "llm_only": run_llm_only,
    "approxlg": run_approxlg,
    "nsga2_expert": lambda *a, **k: run_nsga2_prior(*a, **{**k, "kind": "expert"}),
    "nsga2_adaptive": lambda *a, **k: run_nsga2_prior(*a, **{**k,
                                                             "kind": "adaptive"}),
    "nsga2_llmprior": lambda *a, **k: run_nsga2_prior(*a, **{**k, "kind": "llm"}),
}


def run(name: str, eng: Engine, sp: Space, budget: int, seed: int,
        lo: np.ndarray, hi: np.ndarray, **kw) -> SearchResult:
    """Convenience wrapper; ``scripts/run_search.py`` drives the algorithms
    directly so that it can also record the per-run bookkeeping."""
    tracker = Tracker(lo, hi)
    eng.start_run()
    t0 = time.time()
    extra = ALGORITHMS[name](eng, sp, budget, seed, tracker,
                             norm=hi - lo, **kw) or {}
    dt = time.time() - t0
    res = SearchResult(name=name, seed=seed, records=tracker.records,
                       hv_trace=tracker.trace, seconds=dt,
                       accepted_llm=extra.get("accepted", 0),
                       proposed_llm=extra.get("proposed", 0))
    llm = kw.get("llm")
    if llm is not None:
        res.llm_stats = dict(llm.stats.__dict__)
    return res
