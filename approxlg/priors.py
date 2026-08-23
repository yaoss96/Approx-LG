"""Two non-LLM proposal distributions over compressor styles.

The ablation of the guided search shows that essentially all of its advantage
travels through one channel: the ranked list of compressor styles per column
offset that biases every mutation.  That leaves the obvious question of whether
a *language model* is needed to produce that list, or whether any reasonable
prior over the same channel does as well.  This module supplies the two
controls that answer it, both driving the identical ``ModePrior`` interface
that the elicited prior uses, with the same epsilon and the same geometric
rank sampling, so the only thing that differs is where the ranking comes from.

``expert_prior``
    A fixed prior written down from the design rule the introduction states,
    with no search feedback and no model in the loop.  A slot ``d`` columns
    above the truncation point carries weight ``2^(T+d)``, so the error a cell
    may contribute has to shrink geometrically with ``d``; and a positively
    biased cell placed next to a negatively biased one cancels error for free.
    The prior is *derived* from the measured cell statistics by that rule
    rather than hand-tuned, so it is reproducible and was never fitted to a
    result.

``AdaptivePrior``
    A prior over the same channel estimated from the run's own data by
    credit assignment: how much more often a style appears at an offset among
    the non-dominated designs than among everything evaluated, with a UCB
    exploration term.  This is the standard non-LLM way to adapt a variation
    operator online, and it sees strictly more information about *this* run
    than the model does.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cells import CELL_BY_NAME, N_SLOT_MODES, SLOT_MODES, cell_statistics
from .llm import OFFSET_KEYS, ModePrior

N_RANK = 5          # length of a ranked list, matching the elicited prior


def style_error() -> Dict[int, tuple]:
    """(mean |ED|, signed mean ED) of every style, averaged over the cell roles
    it defines.  This is exactly the table the model is shown."""
    out = {}
    for i, m in enumerate(SLOT_MODES):
        a, s = [], []
        for nm in (m.fa, m.ha, m.c42):
            if nm:
                st = cell_statistics(CELL_BY_NAME[nm])
                a.append(st["mean_abs_ed"])
                s.append(st["mean_ed"])
        out[i] = (float(np.mean(a)), float(np.mean(s)))
    return out


def _alternate_bias(order: Sequence[int], signed: Dict[int, float]) -> List[int]:
    """Reorder a shortlist so consecutive entries alternate in bias sign.

    Mutation resamples several slots of a genome from the same ranked list, so
    making the head of the list mix a positive and a negative cell is what
    turns the error-cancellation rule into something the sampler can act on.
    Ranking is otherwise preserved: the best remaining cell of the wanted sign
    is taken, the best remaining cell of any sign when none is left, and an
    unbiased cell keeps its rank rather than being displaced -- otherwise the
    exact style, which is what the rule actually asks for high above the
    truncation point, gets pushed off the head of the list.
    """
    pool = list(order)
    out: List[int] = []
    want = 1.0
    while pool:
        if signed[pool[0]] == 0:
            out.append(pool.pop(0))
            continue
        pick = next((m for m in pool
                     if (signed[m] > 0) == (want > 0) and signed[m] != 0),
                    pool[0])
        out.append(pick)
        pool.remove(pick)
        want = -want
    return out


def expert_prior(epsilon: float = 0.25, n_rank: int = N_RANK) -> ModePrior:
    """The fixed designer prior: tolerable cell error halves per column."""
    err = style_error()
    tau_max = max(a for a, _ in err.values())
    by: Dict[str, List[int]] = {}
    for d, key in enumerate(OFFSET_KEYS):
        tau = tau_max * (2.0 ** -d)
        # closest to the tolerable magnitude, ties resolved towards the
        # cheaper (more approximate) cell
        order = sorted(range(N_SLOT_MODES),
                       key=lambda m: (abs(err[m][0] - tau), -err[m][0]))
        short = _alternate_bias(order[:n_rank + 2], {m: err[m][1] for m in
                                                     range(N_SLOT_MODES)})
        short = short[:n_rank]
        if 0 not in short:          # the exact style is never ruled out
            short.append(0)
        by[key] = short
    return ModePrior(by, epsilon=epsilon)


class AdaptivePrior:
    """Online credit assignment over (column offset, style), no model.

    ``update`` is called on the same schedule the elicited prior is refreshed
    on, and returns a ``ModePrior`` or ``None`` when there is not yet enough
    data, so the two enter the search at exactly the same point.
    """

    def __init__(self, n_head: int, nstage: int, ncol: int,
                 epsilon: float = 0.25, n_rank: int = N_RANK,
                 c_ucb: float = 0.5, min_obs: int = 40):
        self.n_head = n_head
        self.nstage = nstage
        self.ncol = ncol
        self.epsilon = epsilon
        self.n_rank = n_rank
        self.c_ucb = c_ucb
        self.min_obs = min_obs

    def _tally(self, genomes: Sequence[np.ndarray]):
        cnt = defaultdict(lambda: np.zeros(N_SLOT_MODES))
        for g in genomes:
            t, a = int(g[0]), int(g[1])
            slots = np.asarray(g[self.n_head:]).reshape(self.nstage, self.ncol)
            for j in range(t, min(t + a, self.ncol)):
                k = OFFSET_KEYS[min(j - t, len(OFFSET_KEYS) - 1)]
                for s in range(self.nstage):
                    cnt[k][int(slots[s, j])] += 1
        return cnt

    def update(self, all_genomes: Sequence[np.ndarray],
               elite_genomes: Sequence[np.ndarray]) -> Optional[ModePrior]:
        if len(elite_genomes) < 4:
            return None
        seen = self._tally(all_genomes)
        good = self._tally(elite_genomes)
        by: Dict[str, List[int]] = {}
        for k in OFFSET_KEYS:
            n_all = seen[k].sum()
            n_good = good[k].sum()
            if n_all < self.min_obs or n_good < 1:
                continue
            # smoothed lift of a style among the non-dominated designs
            p_good = (good[k] + 1.0) / (n_good + N_SLOT_MODES)
            p_all = (seen[k] + 1.0) / (n_all + N_SLOT_MODES)
            score = p_good / p_all
            bonus = self.c_ucb * np.sqrt(2.0 * math.log(max(n_all, 2.0)) /
                                         (seen[k] + 1.0))
            order = np.argsort(-(score + bonus))[: self.n_rank]
            by[k] = [int(m) for m in order]
        return ModePrior(by, epsilon=self.epsilon) if by else None
