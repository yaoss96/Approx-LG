"""Multi-objective bookkeeping: dominance, crowding, hypervolume, IGD."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def dominates(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(np.all(x <= y) and np.any(x < y))


def nondominated_mask(F: np.ndarray) -> np.ndarray:
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        le = np.all(F <= F[i], axis=1)
        lt = np.any(F < F[i], axis=1)
        dom = le & lt
        if dom.any():
            keep[i] = False
    return keep


def fast_nondominated_sort(F: np.ndarray) -> List[np.ndarray]:
    n = F.shape[0]
    S = [[] for _ in range(n)]
    nd = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]
    for p in range(n):
        dom_p = np.all(F[p] <= F, axis=1) & np.any(F[p] < F, axis=1)
        dom_by_p = np.all(F <= F[p], axis=1) & np.any(F < F[p], axis=1)
        S[p] = list(np.where(dom_p)[0])
        nd[p] = int(dom_by_p.sum())
        if nd[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [np.array(f, dtype=int) for f in fronts[:-1]]


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        fk = F[order, k]
        d[order[0]] = d[order[-1]] = np.inf
        span = fk[-1] - fk[0]
        if span <= 0:
            continue
        d[order[1:-1]] += (fk[2:] - fk[:-2]) / span
    return d


# --------------------------------------------------------------------------
# Hypervolume (minimisation, reference point dominated by all points)
# --------------------------------------------------------------------------
def hypervolume(F: np.ndarray, ref: np.ndarray) -> float:
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    F = F[np.all(F <= ref, axis=1)]
    if F.size == 0:
        return 0.0
    F = F[nondominated_mask(F)]
    return _hv_rec(F, np.asarray(ref, dtype=float))


def _hv_rec(F: np.ndarray, ref: np.ndarray) -> float:
    m = F.shape[1]
    if m == 1:
        return float(ref[0] - F[:, 0].min())
    if m == 2:
        idx = np.argsort(F[:, 0])
        Fs = F[idx]
        hv, prev_y = 0.0, ref[1]
        for x, y in Fs:
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return float(hv)
    # slice along the last objective
    zs = np.unique(F[:, -1])
    zs = np.append(zs, ref[-1])
    hv = 0.0
    for i in range(len(zs) - 1):
        sl = F[F[:, -1] <= zs[i]][:, :-1]
        if sl.size:
            sl = sl[nondominated_mask(sl)]
            hv += _hv_rec(sl, ref[:-1]) * (zs[i + 1] - zs[i])
    return float(hv)


def igd(F: np.ndarray, ref_front: np.ndarray) -> float:
    if F.size == 0:
        return float("inf")
    d = np.sqrt(((ref_front[:, None, :] - F[None, :, :]) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


def normalise(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (F - lo) / rng
