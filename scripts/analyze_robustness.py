"""Is the comparison an artefact of the hypervolume box, and what does it cost?

Two questions the search tables cannot answer on their own.

*Sensitivity.*  A hypervolume is a number about a reference box as much as
about a front.  Ours caps the error axis at NMED = 0.02 and floors it at
1e-6, and normalises area and delay by the exact multiplier, all of which are
defensible and none of which are forced.  This script re-scores every stored
run under a grid of boxes -- tighter and looser error caps, deeper and
shallower floors, a linear rather than logarithmic error axis, a relaxed cost
reference -- and under two indicators that use no box at all, the additive
epsilon indicator and IGD+ against the union front of every run.  Recomputing
from the stored Pareto sets is exact: the objectives are monotone transforms
per axis, so which points are non-dominated does not depend on the box.

*Cost.*  Total run time stops being comparable across algorithms once the
persistent characterisation cache is warm, so the runs record where the time
actually went: simulation, synthesis, guidance, and how many syntheses were
really performed.  The EDA cost of a run is the per-characterisation cost
times the budget, which is the same for every algorithm by construction; what
differs is the guidance time stacked on top of it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_search import wilcoxon_paired                    # noqa: E402
from approxlg.metrics import hypervolume, nondominated_mask   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
TABS = os.path.join(ROOT, "paper", "tabs")

LABEL = {
    "random": "Random", "ga": "GA", "nsga2": "NSGA-II",
    "eamva": "EA-MVA", "biased_ga": "BC-GA", "llm_only": "LLM-only",
    "nsga2_expert": "NSGA-II + expert prior",
    "nsga2_adaptive": "NSGA-II + adaptive prior",
    "nsga2_llmprior": "NSGA-II + elicited prior",
    "approxlg": "Approx-LG",
}


# --------------------------------------------------------------------------
def load(tags):
    by = defaultdict(list)
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                by[d["algo"]].append(d)
    return by


def exact_box(tags):
    for tag in tags:
        p = os.path.join(RESULTS, f"exact_{tag}.json")
        if os.path.exists(p):
            d = json.load(open(p))
            return float(d["area"]), float(d["delay"])
    raise SystemExit("no exact_*.json for " + ",".join(tags))


def points(run):
    """(NMED, area, delay) of a run's stored Pareto set."""
    return np.array([[max(r["NMED"], 1e-30), r["area"], r["delay"]]
                     for r in run["pareto"]], dtype=float)


def to_box(P, floor, cap, area_hi, delay_hi, log=True):
    """Normalise raw points into the unit cube of one reference box."""
    if log:
        e = np.log10(np.maximum(P[:, 0], floor))
        lo0, hi0 = math.log10(floor), math.log10(cap)
    else:
        e = np.minimum(P[:, 0], cap)
        lo0, hi0 = 0.0, cap
    F = np.stack([(e - lo0) / (hi0 - lo0),
                  P[:, 1] / area_hi,
                  P[:, 2] / delay_hi], axis=1)
    return F


def eps_indicator(F, R):
    """Additive epsilon indicator I_eps+(F, R): the smallest amount by which
    every reference point has to be worsened for F to cover it.  Lower is
    better and no reference box enters it."""
    if len(F) == 0:
        return float("inf")
    d = (F[None, :, :] - R[:, None, :]).max(axis=2)      # |R| x |F|
    return float(d.min(axis=1).max())


def igd_plus(F, R):
    if len(F) == 0:
        return float("inf")
    d = np.sqrt((np.maximum(F[None, :, :] - R[:, None, :], 0.0) ** 2).sum(-1))
    return float(d.min(axis=1).mean())


# --------------------------------------------------------------------------
BOXES = [
    ("default ($\\eta\\in[10^{-6},0.02]$, log)",   1e-6, 0.02, 1.0, True),
    ("tight cap ($\\eta\\le 0.005$)",              1e-6, 0.005, 1.0, True),
    ("loose cap ($\\eta\\le 0.05$)",               1e-6, 0.05, 1.0, True),
    ("deep floor ($\\eta\\ge 10^{-7}$)",           1e-7, 0.02, 1.0, True),
    ("shallow floor ($\\eta\\ge 10^{-5}$)",        1e-5, 0.02, 1.0, True),
    ("linear error axis",                          1e-6, 0.02, 1.0, False),
    ("relaxed cost reference ($1.2\\times$)",      1e-6, 0.02, 1.2, True),
]

# The journal format has room for every box, and a reader checking robustness
# wants to see the rows that did *not* move as much as the ones that did.
TABLE_BOXES = {name for name, *_rest in BOXES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", default=["n8", "n8L", "n8p"])
    ap.add_argument("--suffix", default="n8")
    ap.add_argument("--base", default="nsga2")
    ap.add_argument("--arms", nargs="+",
                    default=["nsga2", "nsga2_expert", "nsga2_adaptive",
                             "nsga2_llmprior", "approxlg"])
    args = ap.parse_args()
    os.makedirs(TABS, exist_ok=True)
    by = load(args.tag)
    area_x, delay_x = exact_box(args.tag)
    arms = [a for a in args.arms if a in by]
    if args.base not in by:
        raise SystemExit(f"no {args.base} runs in {args.tag}")

    raw = {a: {r["seed"]: points(r) for r in by[a]} for a in arms}

    # ---------------- sensitivity of the ranking to the box ---------------
    # The default row must reproduce the recorded hypervolume exactly, not
    # merely to four decimals: recomputing it from the stored front shifts a
    # few runs by a hair and the paired p-value drifts away from the one the
    # other tables print for the same comparison.
    recorded = {a: {r["seed"]: r["hv"] for r in by[a]} for a in arms}
    rows = []
    for i_box, (name, floor, cap, cost_mul, log) in enumerate(BOXES):
        ref = np.ones(3)
        if i_box == 0:
            rows.append((name, {a: dict(recorded[a]) for a in arms}))
            continue
        hv = {}
        for a in arms:
            hv[a] = {s: hypervolume(
                to_box(P, floor, cap, area_x * cost_mul, delay_x * cost_mul,
                       log), ref) for s, P in raw[a].items()}
        rows.append((name, hv))

    # a common reference front for the box-free indicators: everything anyone
    # found, in the default normalisation
    allP = np.vstack([P for a in arms for P in raw[a].values()])
    Rn = to_box(allP, 1e-6, 0.02, area_x, delay_x, True)
    Rn = Rn[nondominated_mask(Rn)]
    # Where the indicators disagree, the fronts themselves say why: the box
    # stops paying below its error floor, so an arm that chases the box puts
    # fewer points in the very accurate corner and IGD+ charges it for them.
    out_cover = {a: float(np.mean([(P[:, 0] < 1e-5).sum()
                                   for P in raw[a].values()]))
                 for a in arms}

    eps = {a: {s: eps_indicator(to_box(P, 1e-6, 0.02, area_x, delay_x, True), Rn)
               for s, P in raw[a].items()} for a in arms}
    igdp = {a: {s: igd_plus(to_box(P, 1e-6, 0.02, area_x, delay_x, True), Rn)
                for s, P in raw[a].items()} for a in arms}

    def paired(x: dict, y: dict, lower_is_better=False):
        seeds = sorted(set(x) & set(y))
        if len(seeds) < 6:
            return float("nan"), 0, 0, float("nan")
        a = np.array([x[s] for s in seeds])
        b = np.array([y[s] for s in seeds])
        p, wins, n = wilcoxon_paired(a, b)
        # a positive gain always means "Approx-LG is better", whichever
        # direction the indicator runs in
        if lower_is_better:
            wins = int((a < b).sum())
            gain = 100.0 * float(np.mean(1.0 - a / b))
        else:
            gain = 100.0 * float(np.mean(a / b - 1.0))
        return p, wins, n, gain

    out = {"boxes": [], "eps": {}, "igd_plus": {}, "accurate_pts": out_cover}
    lines = []
    print(f"{'reference box':38s} " +
          " ".join(f"{LABEL[a][:14]:>14s}" for a in arms))
    for name, hv in rows:
        means = {a: float(np.mean(list(hv[a].values()))) for a in arms}
        p, wins, n, gain = paired(hv["approxlg"], hv[args.base]) \
            if "approxlg" in hv else (float("nan"), 0, 0, float("nan"))
        clean = (name.replace("$", "").replace("\\", "")
                     .replace("eta", "NMED").replace("times", "x"))
        print(f"{clean:38s} " + " ".join(f"{means[a]:14.4f}" for a in arms) +
              f"   d={gain:+5.1f}% p={p:.3f} {wins}/{n}")
        out["boxes"].append({"box": name, "mean": means, "gain": gain,
                             "p": p, "wins": wins, "n": n})
        lines.append((name, means, gain, p, wins, n))

    print()
    for tag, d, lo in (("eps+", eps, True), ("IGD+", igdp, True)):
        means = {a: float(np.mean(list(d[a].values()))) for a in arms}
        p, wins, n, gain = paired(d["approxlg"], d[args.base], lo) \
            if "approxlg" in d else (float("nan"), 0, 0, float("nan"))
        print(f"{tag:38s} " + " ".join(f"{means[a]:14.4f}" for a in arms) +
              f"   d={gain:+5.1f}% p={p:.3f} {wins}/{n}")
        out["eps" if tag == "eps+" else "igd_plus"] = {
            "mean": means, "gain": gain, "p": p, "wins": wins, "n": n}

    # The table keeps only the two arms the sensitivity question is about; the
    # rest stay in the JSON.  Seven numeric columns do not fit an IEEE column.
    def _p(p):
        """Never print a p-value as 0.00; a reader reads that as exactly zero."""
        if p != p:
            return "--"
        return "$<$0.01" if p < 0.005 else f"{p:.2f}"

    with open(os.path.join(TABS, f"hvsens_{args.suffix}.tex"), "w") as fh:
        fh.write("% generated by scripts/analyze_robustness.py\n")
        show = [a for a in (args.base, "approxlg") if a in arms]
        for (name, means, gain, p, wins, n) in lines:
            if name not in TABLE_BOXES:
                continue
            cells = " & ".join(f"{means[a]:.4f}" for a in show)
            fh.write(f"{name} & {cells} & {gain:+.1f}\\% & "
                     + (f"{wins}/{n}" if n else "--") + " & "
                     + _p(p) + " \\\\\n")
        for tag, d, key in (("$I_{\\varepsilon+}$", eps, "eps"),
                            ("IGD$+$", igdp, "igd_plus")):
            m = {a: float(np.mean(list(d[a].values()))) for a in show}
            r = out["eps" if key == "eps" else "igd_plus"]
            cells = " & ".join(f"{m[a]:.4f}" for a in show)
            fh.write(f"{tag} (lower better) & {cells} & "
                     f"{r['gain']:+.1f}\\% & "
                     + (f"{r['wins']}/{r['n']}" if r["n"] else "--") + " & "
                     + _p(r["p"]) + " \\\\\n")
    with open(os.path.join(TABS, f"hvsens_{args.suffix}.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    # ---------------- wall clock ------------------------------------------
    # The timing fields were added part-way through the campaign, so an arm
    # is summarised over the runs that carry them rather than skipped when its
    # first run predates them.  Guidance time comes from the LLM client and is
    # available for every guided run, timed or not.
    # A characterisation costs the same whichever algorithm asked for it, so
    # the per-circuit cost is pooled over every timed run rather than
    # estimated per arm: an arm whose runs happened to hit a warm cache
    # performs few syntheses and would otherwise look implausibly cheap.
    all_ns = sum(r.get("n_synth", 0) for rs in by.values() for r in rs
                 if "t_synth" in r)
    all_ts = sum(r.get("t_synth", 0.0) for rs in by.values() for r in rs
                 if "t_synth" in r)
    pooled = all_ts / all_ns if all_ns else float("nan")

    wc = {}
    for a in arms + [x for x in by if x not in arms]:
        rs = by[a]
        if not rs:
            continue
        timed = [r for r in rs if "t_synth" in r]
        wc[a] = {
            "runs": len(rs),
            "timed_runs": len(timed),
            "t_total": float(np.mean([r["seconds"] for r in rs])),
            # guidance time is recorded by the LLM client on every guided run,
            # whether or not that run predates the synthesis timers
            "t_llm": float(np.mean(
                [r.get("t_llm", (r.get("llm") or {}).get("seconds", 0.0))
                 for r in rs])),
            "t_sim": (float(np.mean([r.get("t_sim", 0.0) for r in timed]))
                      if timed else float("nan")),
            "per_synth": pooled,
            "n_eval": float(np.mean([r["n_eval"] for r in rs])),
        }
        wc[a]["t_eda_cold"] = pooled * wc[a]["n_eval"]
    if wc:
        print("\nwall clock (mean per run, s):")
        print(f"  {'algorithm':26s} {'total':>8s} {'EDA@cold':>9s} "
              f"{'guidance':>9s} {'s/circuit':>10s}")
        for a, d in wc.items():
            print(f"  {LABEL.get(a, a):26s} {d['t_total']:8.0f} "
                  f"{d['t_eda_cold']:9.0f} {d['t_llm']:9.0f} "
                  f"{d['per_synth']:10.2f}")
        with open(os.path.join(TABS, f"wallclock_{args.suffix}.json"), "w") as fh:
            json.dump(wc, fh, indent=1)


if __name__ == "__main__":
    main()
