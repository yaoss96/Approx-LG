"""Stronger same-space optimisers: SMS-EMOA and batch Bayesian optimisation.

Both search the Approx-LG template with the variation operators, initial
sampler and EDA budget every other arm uses, so they differ from NSGA-II only
in how candidates are selected:

* ``smsemoa`` -- the S-metric selection EMOA of Beume, Naujoks and Emmerich
  (EJOR 2007) in its (mu+lambda) form: a generation of offspring is evaluated
  as one batch and the merged population is reduced by repeatedly discarding,
  from the worst non-dominated front, the point that contributes least
  hypervolume.  Contributions are measured in the same normalised box the
  paper scores with, so its selection is aligned with the indicator reported.

* ``mobo`` -- multi-objective Bayesian optimisation in the spirit of AMG (Li et
  al., ICFPT'23): one Gaussian process per objective on the circuit features
  the surrogate arm uses, expected hypervolume improvement estimated by Monte
  Carlo over the scoring box, and a batch of the population size chosen
  greedily with the Kriging-believer rule, so the parallel EDA flow is used
  as fully as by the evolutionary arms.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .arch import Space
from .engine import Engine, Record
from .metrics import fast_nondominated_sort, hypervolume, nondominated_mask


def _norm(tracker, recs: List[Record]) -> np.ndarray:
    F = np.array([r.obj() for r in recs], dtype=float)
    return (F - tracker.lo) / np.where(tracker.hi - tracker.lo > 0,
                                       tracker.hi - tracker.lo, 1.0)


# ---------------------------------------------------------------------------
# SMS-EMOA
# ---------------------------------------------------------------------------
def _hv_contributions(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    total = hypervolume(F, ref)
    out = np.empty(len(F))
    for i in range(len(F)):
        out[i] = total - hypervolume(np.delete(F, i, axis=0), ref)
    return out


def _sms_reduce(recs: List[Record], F: np.ndarray, mu: int,
                ref: np.ndarray) -> List[Record]:
    idx = list(range(len(recs)))
    fronts = fast_nondominated_sort(F)
    keep: List[int] = []
    for fr in fronts:
        fr = list(fr)
        if len(keep) + len(fr) <= mu:
            keep.extend(fr)
            continue
        # the critical front: drop the least contributor until it fits
        while len(keep) + len(fr) > mu:
            Fc = F[fr]
            # points outside the box contribute nothing and go first
            out = np.where(np.any(Fc > ref, axis=1))[0]
            if len(out):
                # the farthest outside first
                far = out[np.argmax(np.max(Fc[out] - ref, axis=1))]
                fr.pop(int(far))
                continue
            c = _hv_contributions(Fc, ref)
            fr.pop(int(np.argmin(c)))
        keep.extend(fr)
        break
    return [recs[i] for i in keep]


def run_smsemoa(eng: Engine, sp: Space, budget: int, seed: int, tracker,
                pop: int = 32, **kw) -> Dict:
    from .search import _init_pop, crossover, eval_batch, mutate

    rng = np.random.default_rng(seed)
    ref = np.ones(3)
    P = eng.evaluate(_init_pop(sp, rng, pop), budget=budget, used=eng.n_eval)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    while eng.n_eval < budget and P:
        kids = []
        while len(kids) < pop:
            a, b = rng.integers(0, len(P), size=2)      # uniform parent choice
            kids.append(mutate(sp, crossover(sp, P[a].g, P[b].g, rng), rng))
        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        pool = list({r.g.tobytes(): r for r in P + Q}.values())
        P = _sms_reduce(pool, _norm(tracker, pool), pop, ref)
    return {}


# ---------------------------------------------------------------------------
# Batch MOBO with Monte-Carlo EHVI
# ---------------------------------------------------------------------------
class _MCBox:
    """Quasi-uniform points of the scoring box; the hypervolume a candidate
    would add is the box volume times the share of still-undominated points
    it dominates."""

    def __init__(self, n_pts: int, seed: int):
        rng = np.random.default_rng(10_000 + seed)
        self.U = rng.random((n_pts, 3))

    def undominated(self, F: np.ndarray) -> np.ndarray:
        if F.size == 0:
            return self.U
        dom = np.zeros(len(self.U), dtype=bool)
        for f in F:
            dom |= np.all(self.U >= f, axis=1)
        return self.U[~dom]


def _fit_gps(X: np.ndarray, Y: np.ndarray, seed: int, max_train: int = 400):
    """One GP per objective on standardised features: a Matern-5/2 kernel
    with one length scale (an ARD kernel over ~30 features multiplied the
    fitting time by orders of magnitude for no measurable gain here) and a
    learned noise term.  Training is capped at the ``max_train`` most recent
    points, which bounds the cubic cost late in a run."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                  WhiteKernel)
    if len(X) > max_train:
        X, Y = X[-max_train:], Y[-max_train:]
    gps = []
    for c in range(Y.shape[1]):
        k = (ConstantKernel(1.0, (1e-2, 1e3)) *
             Matern(length_scale=3.0, nu=2.5, length_scale_bounds=(1e-2, 1e3))
             + WhiteKernel(1e-3, (1e-6, 1e-1)))
        gp = GaussianProcessRegressor(k, normalize_y=True,
                                      n_restarts_optimizer=0,
                                      random_state=seed + c)
        gp.fit(X, Y[:, c])
        gps.append(gp)
    return gps


def run_mobo(eng: Engine, sp: Space, budget: int, seed: int, tracker,
             pop: int = 32, pool_mult: int = 8, n_mc: int = 16,
             n_box: int = 3000, explore: float = 0.1, **kw) -> Dict:
    from .search import _init_pop, _ranks, crossover, eval_batch, mutate, \
        tournament
    from .surrogate import features

    rng = np.random.default_rng(seed)
    box = _MCBox(n_box, seed)
    P = eng.evaluate(_init_pop(sp, rng, pop), budget=budget, used=eng.n_eval)
    tracker.add(P)
    tracker.snapshot(eng.n_eval)
    n_fit = 0
    while eng.n_eval < budget and P:
        seen = tracker.records
        Fn = _norm(tracker, seen)
        X = features(sp, [r.g for r in seen])
        mu_x = X.mean(0)
        sd_x = X.std(0) + 1e-9
        gps = _fit_gps((X - mu_x) / sd_x, Fn, seed + 7 * n_fit)
        n_fit += 1

        # candidate pool: offspring of the current non-dominated set plus
        # fresh samples, minus what has been evaluated already
        rank, crowd = _ranks(P)
        seen_keys = {r.g.tobytes() for r in seen}
        pool, keys = [], set()
        tries = 0
        while len(pool) < pop * pool_mult and tries < pop * pool_mult * 4:
            tries += 1
            if rng.random() < explore:
                g = sp.random(rng)
            else:
                a = tournament(rng, rank, crowd)
                b = tournament(rng, rank, crowd)
                g = mutate(sp, crossover(sp, P[a].g, P[b].g, rng), rng)
            k = g.tobytes()
            if k in seen_keys or k in keys:
                continue
            keys.add(k)
            pool.append(g)
        if not pool:
            break
        Xp = (features(sp, pool) - mu_x) / sd_x
        mean = np.empty((len(pool), 3))
        std = np.empty((len(pool), 3))
        for c, gp in enumerate(gps):
            mean[:, c], std[:, c] = gp.predict(Xp, return_std=True)
        # Monte-Carlo samples of every candidate's objective vector
        Z = rng.standard_normal((n_mc, len(pool), 3))
        S = mean[None] + std[None] * Z                      # (mc, pool, 3)

        front = Fn[nondominated_mask(Fn)]
        U = box.undominated(front[np.all(front <= 1.0, axis=1)])
        chosen: List[int] = []
        avail = np.ones(len(pool), dtype=bool)
        # dominance of every box point by every Monte-Carlo sample, once per
        # generation: (pool, |U|) counts of samples dominating the point
        cnt = np.zeros((len(pool), len(U)), dtype=np.int32)
        for t in range(n_mc):
            cnt += np.all(S[t][:, None, :] <= U[None, :, :], axis=2)
        alive = np.ones(len(U), dtype=bool)
        while len(chosen) < pop and avail.any() and alive.any():
            # EHVI ~ E[share of still-undominated box points a sample covers]
            ehvi = cnt[:, alive].sum(axis=1) / float(n_mc * len(U))
            ehvi[~avail] = -1.0
            i = int(np.argmax(ehvi))
            if ehvi[i] <= 0:
                break
            chosen.append(i)
            avail[i] = False
            # Kriging believer: the chosen point's mean joins the front
            alive &= ~np.all(U >= mean[i], axis=1)
        # fill the batch: best predicted rank among the rest, then random
        if len(chosen) < pop:
            rest = np.where(avail)[0]
            if len(rest):
                order = rest[np.argsort(mean[rest].sum(axis=1))]
                chosen += list(order[: pop - len(chosen)])
        kids = [pool[i] for i in chosen]
        before = eng.n_eval
        Q = eval_batch(eng, sp, kids, budget, rng, tracker, pop)
        if not Q or eng.n_eval == before:
            break
        from .search import environmental_selection
        P = environmental_selection(P + Q, pop)
    return {"gp_fits": n_fit}
