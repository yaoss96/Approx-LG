"""Surrogate-assisted NSGA-II, the standard answer to an expensive evaluation.

When a design point costs a simulation plus a synthesis, the textbook response
is not to sample better by hand but to learn a cheap model of the objectives
and spend the real budget only on candidates the model likes.  That is the
direct competitor to injecting external knowledge with a language model: both
try to buy the same thing, fewer true evaluations, and a paper that proposes
one owes the reader the other at the same budget.

The loop here is the usual one.  Every truly evaluated point trains a random
forest per objective; each generation the standard variation operators produce
a pool ``screen`` times larger than the population, the forests rank it, and
only the promising front is sent to the EDA flow.  A fraction ``epsilon`` of
each generation is filled at random regardless of the model, so a confidently
wrong surrogate cannot lock the search into one region.

Encoding: the header genes as they are, plus, for the active slots, the
measured error statistics of the assigned cells.  Those statistics are exactly
what the elicited and the written priors reason over, so the surrogate is
given the same circuit knowledge rather than being handicapped to raw indices.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .arch import Space
from .cells import CELL_BY_NAME, N_SLOT_MODES, SLOT_MODES, cell_statistics
from .engine import Engine, pareto_of
from .metrics import fast_nondominated_sort


def _style_table() -> np.ndarray:
    """(n_styles, 2): mean |ED| and signed mean ED, averaged over the roles a
    style defines."""
    out = np.zeros((N_SLOT_MODES, 2))
    for i, m in enumerate(SLOT_MODES):
        a, s = [], []
        for nm in (m.fa, m.ha, m.c42):
            if nm:
                st = cell_statistics(CELL_BY_NAME[nm])
                a.append(st["mean_abs_ed"])
                s.append(st["mean_ed"])
        out[i] = (float(np.mean(a)), float(np.mean(s)))
    return out


STYLE = _style_table()


def features(sp: Space, genomes) -> np.ndarray:
    """Header genes, per-style counts inside the approximate window, and the
    error statistics of the assigned cells weighted by column significance."""
    rows = []
    for g in genomes:
        g = np.asarray(g, dtype=np.int32)
        t, a, c, k = (int(g[0]), int(g[1]), int(g[2]), int(g[3]))
        slots = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
        hi = min(t + a, sp.ncol)
        win = slots[:, t:hi] if hi > t else slots[:, :0]
        counts = np.bincount(win.reshape(-1), minlength=N_SLOT_MODES) \
            if win.size else np.zeros(N_SLOT_MODES)
        if win.size:
            absed = STYLE[win.reshape(-1), 0]
            sgned = STYLE[win.reshape(-1), 1]
            # a cell's contribution scales with the weight of its column
            w = np.repeat(2.0 ** np.arange(hi - t), sp.nstage)
            wsum = w.sum() or 1.0
            stats = [absed.mean(), sgned.mean(), absed.std(),
                     float((absed * w).sum() / wsum),
                     float((sgned * w).sum() / wsum)]
        else:
            stats = [0.0] * 5
        rows.append([t, a, c, k, hi - t] + list(counts) + stats)
    return np.asarray(rows, dtype=float)


def run_nsga2_rf(eng: Engine, sp: Space, budget: int, seed: int, tracker,
                 pop: int = 32, screen: int = 16, epsilon: float = 0.15,
                 n_trees: int = 120, **kw) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    from .search import (crossover, environmental_selection, eval_batch,
                         _init_pop, _ranks, mutate, tournament)

    rng = np.random.default_rng(seed)
    P = eng.evaluate(_init_pop(sp, rng, pop), budget=budget, used=eng.n_eval)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    n_model = 0
    while eng.n_eval < budget and P:
        rank, crowd = _ranks(P)
        pool = []
        while len(pool) < pop * screen:
            i = tournament(rng, rank, crowd)
            j = tournament(rng, rank, crowd)
            pool.append(mutate(sp, crossover(sp, P[i].g, P[j].g, rng), rng))

        seen = tracker.records
        kids: List[np.ndarray] = []
        if len(seen) >= 3 * pop:
            X = features(sp, [r.g for r in seen])
            Y = np.array([r.obj() for r in seen])
            pred = np.empty((len(pool), Y.shape[1]))
            Xp = features(sp, pool)
            for c in range(Y.shape[1]):
                rf = RandomForestRegressor(n_estimators=n_trees,
                                           min_samples_leaf=2, n_jobs=-1,
                                           random_state=seed + c)
                rf.fit(X, Y[:, c])
                pred[:, c] = rf.predict(Xp)
            n_model += 1
            n_keep = int(round(pop * (1 - epsilon)))
            for front in fast_nondominated_sort(pred):
                for idx in front:
                    if len(kids) >= n_keep:
                        break
                    kids.append(pool[int(idx)])
                if len(kids) >= n_keep:
                    break
        # the exploration remainder, and the whole generation before the
        # surrogate has enough data to be worth fitting
        while len(kids) < pop:
            kids.append(pool[int(rng.integers(0, len(pool)))])

        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        P = environmental_selection(P + Q, pop)
    return {"surrogate_fits": n_model}
