"""One statistics table for the whole paper: effect sizes, CIs, and a
multiplicity correction applied inside declared families.

The first submission reported a p-value per comparison and nothing else, over
seven reference boxes, two indicators, six ablation variants and four prior
sources.  With that many tests "two of seven boxes reach p<0.05" is close to
what noise alone produces, so this script does three things the paper needed:

1. **Declares the families.**  The primary endpoint is the guided arm against
   the *sham* arm -- the placebo that walks the same code path with the model's
   answers discarded -- at each width.  Everything else is secondary and is
   corrected within its own family by Holm's step-down procedure, which needs
   no independence assumption.

2. **Reports effect sizes with intervals.**  Every comparison carries a paired
   bootstrap CI on the mean relative difference and the Vargha-Delaney
   \\hat{A}_{12}, so a reader can see the precision rather than a threshold.

3. **Calibrates the null.**  Two disjoint halves of the *same* arm's runs are
   compared, many times, to measure how large a difference this pipeline
   manufactures from nothing at these seed counts.  Any effect inside that
   band is not evidence, whatever its p-value.

Exact Wilcoxon p-values are used for n < 25 rather than the normal
approximation: at n = 8 the two-sided signed-rank p is confined to a grid whose
smallest value is 1/128, and quoting 0.03 from a normal approximation there is
not defensible.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")

BOOT = 20000
RNG = np.random.default_rng(20260822)


# --------------------------------------------------------------------------
def load(tags):
    by = defaultdict(dict)
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                by[d["algo"]][d["seed"]] = d["hv"]
    return by


def wilcoxon_exact(a, b):
    """Two-sided Wilcoxon signed-rank test.

    Exact for n <= 24: the 2^n sign assignments of the ranks are counted by
    their rank sum (a subset-sum recursion over doubled, hence integer, ranks,
    which gives the same counts as enumerating them one by one); normal
    approximation without continuity correction above that.  Ties in |d| get
    average ranks, zero differences are dropped.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan"), 0, 0
    order = np.argsort(np.abs(d))
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    absd = np.abs(d)
    for v in np.unique(absd):
        m = absd == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    wp = ranks[d > 0].sum()
    wins = int((d > 0).sum())
    if n <= 24:
        r2 = np.rint(2 * ranks).astype(np.int64)     # average ranks are k/2
        tot = int(r2.sum())
        counts = np.zeros(tot + 1)                    # exact below 2^53
        counts[0] = 1.0
        for r in r2:
            nxt = counts.copy()
            nxt[r:] += counts[:tot + 1 - r]
            counts = nxt
        stat = int(round(2 * min(wp, ranks.sum() - wp)))
        w = np.arange(tot + 1)
        hits = counts[np.minimum(w, tot - w) <= stat].sum()
        return float(hits / counts.sum()), wins, n
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    from math import erfc, sqrt
    z = (min(wp, ranks.sum() - wp) - mu) / sd if sd > 0 else 0.0
    return erfc(abs(z) / sqrt(2)), wins, n


def a12(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum((x > y) for x in a for y in b)
    eq = sum((x == y) for x in a for y in b)
    return (gt + 0.5 * eq) / (len(a) * len(b))


def _rng_for(*arrays):
    """A generator seeded by the data it will resample.

    The module used to draw every bootstrap and every null-band split from one
    global stream, which made each number depend on how many tests happened to
    run before it: adding a comparison silently moved the null band and every
    confidence interval after it.  Seeding from the input bytes instead makes
    each quantity reproducible on its own, whatever else the run computes.
    """
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "little"))


def boot_ci(a, b, level=0.95):
    """Percentile bootstrap CI on the paired mean relative difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = _rng_for(a, b).integers(0, n, size=(BOOT, n))
    stat = np.mean((a[idx] - b[idx]) / b[idx], axis=1)
    lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
    return 100 * lo, 100 * hi


def paired(by, x, y):
    seeds = sorted(set(by[x]) & set(by[y]))
    a = np.array([by[x][s] for s in seeds])
    b = np.array([by[y][s] for s in seeds])
    p, wins, n = wilcoxon_exact(a, b)
    lo, hi = boot_ci(a, b)
    return {"n": len(seeds), "delta": 100 * float(np.mean((a - b) / b)),
            "ci": (lo, hi), "p": p, "wins": wins, "npairs": n,
            "a12": a12(a, b)}


def holm(ps):
    """Holm step-down adjusted p-values, order preserved."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * ps[k])
        adj[k] = min(1.0, run)
    return adj


def null_band(by, arm, reps=200000, half=None):
    """How large a paired difference this pipeline invents from nothing.

    Splits one arm's runs into two disjoint halves and scores them against
    each other exactly as a real comparison is scored.  The 95th percentile of
    the absolute difference is the floor below which nothing is evidence.
    """
    v = np.array(list(by[arm].values()))
    rng = _rng_for(v)
    n = len(v)
    k = half or n // 2
    if n < 8:
        return float("nan"), float("nan")
    out = []
    for _ in range(reps):
        perm = rng.permutation(n)
        a, b = v[perm[:k]], v[perm[k:2 * k]]
        out.append(100 * float(np.mean((a - b) / b)))
    out = np.abs(np.array(out))
    p95 = float(np.quantile(out, 0.95))
    # Monte-Carlo error of the 95th percentile itself: the primary effect is
    # compared against it, so a reader has to know whether it is stable in the
    # digit the comparison turns on.
    boot = rng.integers(0, len(out), size=(400, len(out)))
    se = float(np.std(np.quantile(out[boot], 0.95, axis=1)))
    return float(np.median(out)), p95, se


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stats")
    args = ap.parse_args()
    os.makedirs(TABS, exist_ok=True)

    n8 = load(["n8", "n8L", "n8abl", "n8p"])
    u16 = load(["n16u"])
    s16 = load(["n16s"])

    report = {}

    # ---- primary endpoint: guided vs the placebo, at each width ----------
    primary = []
    for label, by in (("$8\\times8$", n8), ("$16\\times16$ unsigned", u16),
                      ("$16\\times16$ signed", s16)):
        if "approxlg" in by and "approxlg_sham" in by:
            r = paired(by, "approxlg", "approxlg_sham")
            r["what"] = label
            primary.append(r)
    if primary:
        adj = holm([r["p"] for r in primary])
        for r, q in zip(primary, adj):
            r["p_holm"] = float(q)
    report["primary"] = primary

    # ---- secondary families ---------------------------------------------
    fam = {}
    fam["reference"] = [("Approx-LG vs NSGA-II ($8\\times8$)", n8, "approxlg",
                         "nsga2"),
                        ("sham vs NSGA-II ($8\\times8$)", n8,
                         "approxlg_sham", "nsga2"),
                        ("GA vs random ($8\\times8$)", n8, "ga", "random"),
                        ("LLM-only vs NSGA-II ($8\\times8$)", n8, "llm_only",
                         "nsga2")]
    fam["prior"] = [("fixed prior vs sham", n8, "nsga2_expert",
                     "approxlg_sham"),
                    ("adaptive prior vs sham", n8, "nsga2_adaptive",
                     "approxlg_sham"),
                    ("elicited prior vs sham", n8, "nsga2_llmprior",
                     "approxlg_sham"),
                    ("elicited vs fixed prior", n8, "nsga2_llmprior",
                     "nsga2_expert")]
    fam["wide"] = []
    for lab, by in (("unsigned", u16), ("signed", s16)):
        fam["wide"] += [
            (f"Approx-LG vs RF surrogate ({lab})", by, "approxlg", "nsga2_rf"),
            (f"RF surrogate vs sham ({lab})", by, "nsga2_rf", "approxlg_sham"),
            (f"Approx-LG vs NSGA-II ({lab})", by, "approxlg", "nsga2"),
            (f"fixed prior vs sham ({lab})", by, "nsga2_expert",
             "approxlg_sham"),
            (f"elicited prior vs sham ({lab})", by, "nsga2_llmprior",
             "approxlg_sham")]
    fam["ablation"] = [(f"w/o {nm}", n8, key, "approxlg") for nm, key in
                       (("warm start", "approxlg_no_warm"),
                        ("injection", "approxlg_no_inject"),
                        ("mutation guidance", "approxlg_no_mutguide"),
                        ("the proposal distribution", "approxlg_no_prior"),
                        ("space refinement", "approxlg_no_refine"))]

    for name, entries in fam.items():
        rows = []
        for what, by, x, y in entries:
            if x in by and y in by and len(set(by[x]) & set(by[y])) >= 6:
                r = paired(by, x, y)
                r["what"] = what
                rows.append(r)
        if rows:
            adj = holm([r["p"] for r in rows])
            for r, q in zip(rows, adj):
                r["p_holm"] = float(q)
        report[name] = rows

    # ---- diagnostics that belong to no corrected family ------------------
    # The placebo is the arm the primary endpoint is measured against, so
    # whether the placebo is itself weaker than plain NSGA-II decides how much
    # of the primary effect is the model and how much is the control losing.
    # This is a sanity check on the control, not a test of the hypothesis, so
    # it is reported uncorrected and kept out of the families above.
    control = []
    for lab, by in (("unsigned", u16), ("signed", s16), ("$8\\times8$", n8)):
        if "approxlg_sham" in by and "nsga2" in by:
            r = paired(by, "approxlg_sham", "nsga2")
            r["what"] = f"sham vs NSGA-II ({lab})"
            control.append(r)
    report["control"] = control

    # ---- does the endpoint or the model size move the elicited prior? ----
    # Same arm, same seeds, same everything but the model behind the proposal
    # distribution: extended reasoning on, and a larger model.
    base = load(["n8p"])
    model = []
    for lab, tag in (("extended reasoning", "n8think"),
                     ("a larger model", "n8pro")):
        by = load([tag])
        if "nsga2_llmprior" in by and "nsga2_llmprior" in base:
            merged = {"variant": by["nsga2_llmprior"],
                      "base": base["nsga2_llmprior"]}
            if len(set(merged["variant"]) & set(merged["base"])) >= 6:
                r = paired(merged, "variant", "base")
                r["what"] = f"{lab} vs the default endpoint"
                model.append(r)
    # The two arms above only change the endpoint behind the proposal
    # distribution, which is the one channel shown to buy nothing.  This one
    # changes it behind the whole guided system, at the width the paper's
    # claim lives at, so it asks whether a stronger model writes better
    # candidate designs rather than better ranked lists.
    full = load(["n16upro"])
    if "approxlg" in full and "approxlg" in u16:
        merged = {"variant": full["approxlg"], "base": u16["approxlg"]}
        if len(set(merged["variant"]) & set(merged["base"])) >= 6:
            r = paired(merged, "variant", "base")
            r["what"] = "a larger model, whole system (unsigned)"
            model.append(r)
    if model:
        for r, q in zip(model, holm([r["p"] for r in model])):
            r["p_holm"] = float(q)
    report["model"] = model

    # ---- null calibration -------------------------------------------------
    cal = {}
    for label, by, arm in (("nsga2 8x8", n8, "nsga2"),
                           ("approxlg 8x8", n8, "approxlg"),
                           ("nsga2 16x16u", u16, "nsga2"),
                           ("approxlg 16x16u", u16, "approxlg"),
                           ("approxlg 16x16s", s16, "approxlg")):
        if arm in by and len(by[arm]) >= 8:
            med, p95, se = null_band(by, arm)
            cal[label] = {"median": med, "p95": p95, "p95_se": se,
                          "runs": len(by[arm])}
    report["null"] = cal

    # ---- print + emit ------------------------------------------------------
    def show(title, rows):
        if not rows:
            return
        print(f"\n{title}")
        print(f"  {'comparison':38s} {'n':>3s} {'delta':>7s} "
              f"{'95% CI':>17s} {'p':>8s} {'p_holm':>7s} {'A12':>5s}")
        for r in rows:
            print(f"  {r['what'][:38]:38s} {r['n']:3d} {r['delta']:+6.2f}% "
                  f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] "
                  f"{r['p']:8.4f} {r.get('p_holm', float('nan')):7.3f} "
                  f"{r['a12']:5.2f}")

    show("PRIMARY  guided vs sham guidance", report["primary"])
    for k in ("reference", "wide", "prior", "ablation", "model", "control"):
        show(f"secondary: {k}", report[k])
    print("\nnull calibration (same arm split in half, |paired delta|):")
    for k, v in cal.items():
        print(f"  {k:14s} {v['runs']:2d} runs   median {v['median']:.2f}%   "
              f"95th pct {v['p95']:.2f}% +- {v['p95_se']:.3f}")

    with open(os.path.join(TABS, f"{args.out}.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("\nwrote", os.path.join(TABS, f"{args.out}.json"))


if __name__ == "__main__":
    main()
