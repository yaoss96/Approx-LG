"""Aggregate the search-algorithm campaign into the paper's figure and tables."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
TABS = os.path.join(ROOT, "paper", "tabs")

LABEL = {
    "random": "Random",
    "ga": "GA",
    "nsga2": "NSGA-II",
    "llm_only": "LLM-only",
    "approxlg": "Approx-LG",
    "eamva": "EA-MVA~\\cite{devi2025}",
    "biased_ga": "BC-GA~\\cite{wen2025}",
    "approxlg_no_warm": "w/o warm start",
    "approxlg_no_inject": "w/o candidate injection",
    "approxlg_no_mutguide": "w/o mutation guidance",
    "approxlg_no_refine": "w/o space refinement",
    "approxlg_prior_only": "mutation prior only",
    "nsga2_expert": r"NSGA-II\,+\,fixed prior",
    "nsga2_adaptive": r"NSGA-II\,+\,adaptive prior",
    "nsga2_llmprior": r"NSGA-II\,+\,elicited prior",
    "approxlg_no_prior": "w/o the proposal distribution",
    "approxlg_sham": "sham guidance",
}
MAX_SEED = 10**9        # no cap; stale runs are purged at the source
# The fixed-prior control sits in the headline table on purpose: a reader who
# only looks at Table I should see the arm that decides what the guided row is
# evidence for.
ORDER = ["random", "ga", "nsga2", "eamva", "biased_ga", "llm_only",
         "nsga2_expert", "approxlg_sham", "approxlg"]
ABL = ["approxlg", "approxlg_no_warm", "approxlg_no_inject",
       "approxlg_no_mutguide", "approxlg_no_prior", "approxlg_no_refine",
       "approxlg_prior_only"]


def load(tags):
    rows = []
    for tag in (tags if isinstance(tags, (list, tuple)) else [tags]):
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("seed", 0) >= MAX_SEED:
                    continue          # runs that predate the flow fix
                rows.append(d)
    by = defaultdict(list)
    for r in rows:
        by[r["algo"]].append(r)
    return by


def curve(runs, grid):
    C = []
    for d in runs:
        tr = d["hv_trace"]
        c, h = [], 0.0
        for g in grid:
            for n, v in tr:
                if n <= g:
                    h = v
            c.append(h)
        C.append(c)
    return np.array(C)


def mannwhitney(x, y):
    """Two-sided Mann-Whitney U with a normal approximation (no SciPy dep)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3:
        return float("nan"), float("nan")
    allv = np.concatenate([x, y])
    order = allv.argsort()
    ranks = np.empty_like(allv)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u1 - mu) / sd if sd > 0 else 0.0
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2))
    # common-language effect size / A12
    a12 = u1 / (n1 * n2)
    return p, a12


def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank test on paired samples.

    The seed also seeds the initial population, which is drawn by the *same*
    sampler for every algorithm, so run ``s`` of two algorithms starts from the
    same designs and the runs are genuinely paired.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    n = len(d)
    if n < 6:
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
    wm = ranks[d < 0].sum()
    w = min(wp, wm)
    mu = n * (n + 1) / 4
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    from math import erfc, sqrt
    z = (w - mu) / sd if sd > 0 else 0.0
    return erfc(abs(z) / sqrt(2)), int((d > 0).sum()), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["n8"])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(TABS, exist_ok=True)
    by = load(args.tag)
    if not by:
        print("no data for", args.tag)
        return
    sfx = args.suffix or args.tag[0]

    grid = np.arange(16, args.budget + 1, 16)
    curves = {a: curve(rs, grid) for a, rs in by.items()}

    # ---------------- figure: HV vs budget ----------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif",
                         "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                         "legend.frameon": False, "figure.dpi": 300})
    style = {"random": ("#9e9e9e", ":"), "ga": ("#7b1fa2", "-."),
             "nsga2": ("#1565c0", "--"), "llm_only": ("#ef6c00", "-."),
             "approxlg": ("#c62828", "-"), "eamva": ("#2e7d32", (0, (3, 1, 1, 1))),
             "biased_ga": ("#00838f", (0, (1, 1)))}
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for a in ORDER:
        if a not in curves:
            continue
        C = curves[a]
        m, s = C.mean(0), C.std(0)
        col, ls = style.get(a, ("k", "-"))
        ax.plot(grid, m, color=col, ls=ls, label=LABEL[a].split("~")[0])
        ax.fill_between(grid, m - s, m + s, color=col, alpha=0.12, lw=0)
    ax.set_xlabel("EDA evaluations")
    ax.set_ylabel("hypervolume")
    ax.grid(alpha=0.25, lw=0.4)
    # the first two generations are common to every algorithm; clip the axis so
    # the region where they separate is legible
    tops = [c.mean(0)[-1] for c in curves.values() if len(c)]
    lo = min(c.mean(0)[3] for c in curves.values() if len(c) and len(c[0]) > 3)
    ax.set_ylim(max(0.0, lo * 0.8), max(tops) * 1.06)
    ax.set_xlim(grid[0], grid[-1])
    ax.legend(ncol=2, fontsize=6, loc="lower right")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIGS, f"hv_{sfx}.pdf"))
    plt.close(fig)

    # ---------------- table: final HV / speed-up ----------------------
    ref = np.mean([r["hv"] for r in by["nsga2"]]) if "nsga2" in by else None
    lines = []
    for a in ORDER:
        if a not in by:
            continue
        hv = np.array([r["hv"] for r in by[a]])
        # Evaluations until a run's own trace first reaches the *mean* final
        # NSGA-II hypervolume.  A run that never reaches it is censored at the
        # budget, so this is a censored mean: an arm whose own mean sits below
        # the reference can still show a value under B when a few of its runs
        # cross it, which is why the reach count is printed beside it.
        evs, reach = [], 0
        for d in by[a]:
            e = args.budget
            for n, h in d["hv_trace"]:
                if ref is not None and h >= ref:
                    e = n
                    reach += 1
                    break
            evs.append(e)
        p, a12 = mannwhitney(by["approxlg"] and
                             [r["hv"] for r in by["approxlg"]] or [], hv)
        pf = np.mean([len(r["pareto"]) for r in by[a]])
        lines.append((a, hv.mean(), hv.std(), np.mean(evs), pf, p, a12,
                      np.mean([r["seconds"] for r in by[a]]), reach))
    with open(os.path.join(TABS, f"search_{sfx}.tex"), "w") as fh:
        fh.write("% generated by scripts/analyze_search.py\n")
        for (a, m, s, e, pf, p, a12, sec, reach) in lines:
            star = "" if a == "approxlg" else (
                "$^{\\dagger}$" if p == p and p < 0.05 else "")
            fh.write(f"{LABEL[a]} & {len(by[a])} & {m:.4f} $\\pm$ "
                     f"{s:.4f}{star} & {pf:.0f} & "
                     f"{e:.0f}~({reach}/{len(by[a])}) & {sec:.0f} \\\\\n")
    print(f"{'algorithm':26s} {'HV mean':>9s} {'std':>7s} {'|PF|':>5s} "
          f"{'ev@ref':>7s} {'reach':>7s} {'p':>8s} {'A12':>5s}")
    for (a, m, s, e, pf, p, a12, sec, reach) in lines:
        print(f"{LABEL[a]:26s} {m:9.4f} {s:7.4f} {pf:5.0f} {e:7.0f} "
              f"{reach:3d}/{len(by[a]):<3d} {p:8.4f} {a12:5.2f}")

    # paired, per-seed comparison against Approx-LG
    if "approxlg" in by:
        ref_by_seed = {r["seed"]: r["hv"] for r in by["approxlg"]}
        print("\npaired per-seed comparison with Approx-LG "
              "(same seed = same initial population):")
        pairs = {}
        for a in ORDER:
            if a == "approxlg" or a not in by:
                continue
            xs, ys = [], []
            for r in by[a]:
                if r["seed"] in ref_by_seed:
                    xs.append(ref_by_seed[r["seed"]])
                    ys.append(r["hv"])
            if len(xs) < 6:
                continue
            p, wins, n = wilcoxon_paired(xs, ys)
            pairs[a] = (wins, n, p, float(np.mean(np.array(xs) - np.array(ys))))
            print(f"  vs {LABEL[a].split('~')[0]:12s} Approx-LG better in "
                  f"{wins}/{n} seeds, mean gain "
                  f"{100*np.mean((np.array(xs)-np.array(ys))/np.array(ys)):+.1f}%,"
                  f" p={p:.4f}")
        with open(os.path.join(TABS, f"paired_{sfx}.json"), "w") as fh:
            json.dump({k: {"wins": v[0], "n": v[1], "p": v[2], "gain": v[3]}
                       for k, v in pairs.items()}, fh, indent=1)

    # ---------------- table: ablation ---------------------------------
    have = [a for a in ABL if a in by]
    if len(have) > 1:
        base = np.mean([r["hv"] for r in by["approxlg"]])
        full = {r["seed"]: r["hv"] for r in by["approxlg"]}
        rows = list(have) + (["nsga2"] if "nsga2" in by else [])
        with open(os.path.join(TABS, f"ablation_{sfx}.tex"), "w") as fh:
            for a in rows:
                hv = np.array([r["hv"] for r in by[a]])
                # Variants have different seed counts, so a difference of raw
                # means compares different populations and can even flip sign
                # against the paired figure.  Score every row on the seeds it
                # shares with the full system.
                arm = {r["seed"]: r["hv"] for r in by[a]}
                common = sorted(set(arm) & set(full))
                if a != "approxlg" and len(common) >= 6:
                    A = np.array([arm[s] for s in common])
                    B = np.array([full[s] for s in common])
                    d = 100 * float(np.mean((A - B) / B))
                    p, _, _ = wilcoxon_paired(A, B)
                else:
                    d = 100 * (hv.mean() - base) / base
                    p, _ = mannwhitney([r["hv"] for r in by["approxlg"]], hv)
                lbl = "no guidance ($=$ NSGA-II)" if a == "nsga2" else LABEL[a]
                dd = "--" if a == "approxlg" else f"{d:+.1f}\\%"
                pp = "--" if a == "approxlg" else (
                    "--" if p != p else
                    ("$<$0.01" if p < 0.005 else f"{p:.2f}"))
                fh.write(f"{lbl} & {len(hv)} & {hv.mean():.4f} $\\pm$ "
                         f"{hv.std():.4f} & {dd} & {pp} \\\\\n")
        print("\nablation:")
        for a in have:
            hv = np.array([r["hv"] for r in by[a]])
            print(f"  {LABEL[a]:28s} {hv.mean():.4f} +- {hv.std():.4f} "
                  f"({100*(hv.mean()-base)/base:+.1f}%)")

    # ---------------- LLM usage statistics ----------------------------
    if "approxlg" in by:
        rs = by["approxlg"]
        st = {
            "calls": np.mean([r["llm"]["calls"] for r in rs if r.get("llm")]),
            "fail": np.mean([r["llm"]["failures"] for r in rs if r.get("llm")]),
            "ptok": np.mean([r["llm"]["prompt_tokens"] for r in rs if r.get("llm")]),
            "ctok": np.mean([r["llm"]["completion_tokens"] for r in rs
                             if r.get("llm")]),
            "sec": np.mean([r["llm"]["seconds"] for r in rs if r.get("llm")]),
            "accepted": np.mean([r["accepted"] for r in rs]),
            "llm_points": np.mean([r.get("llm_points", 0) for r in rs]),
            "on_front": np.mean([r.get("llm_on_front", 0) for r in rs]),
            "front": np.mean([len(r["pareto"]) for r in rs]),
        }
        with open(os.path.join(TABS, f"llmstats_{sfx}.json"), "w") as fh:
            json.dump(st, fh, indent=1)
        print("\nLLM usage per run:", {k: round(v, 1) for k, v in st.items()})


if __name__ == "__main__":
    main()
