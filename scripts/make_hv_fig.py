"""Two-panel hypervolume-vs-budget figure (8x8 and 16x16)."""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")

LABEL = {"random": "Random", "ga": "GA", "nsga2": "NSGA-II",
         "llm_only": "LLM-only", "approxlg": "Approx-LG",
         "eamva": "EA-MVA", "biased_ga": "BC-GA",
         "nsga2_expert": "+ fixed prior",
         "approxlg_sham": "sham guidance",
         "nsga2_rf": "+ RF surrogate"}
# The fixed-prior arm belongs on the figure: it is the control that decides
# what the guided curve is evidence for, and the two curves nearly coincide.
ORDER = ["random", "ga", "eamva", "biased_ga", "llm_only", "nsga2_rf",
         "nsga2", "nsga2_expert", "approxlg_sham", "approxlg"]
STYLE = {"random": ("#9e9e9e", ":"), "ga": ("#7b1fa2", "-."),
         "nsga2": ("#1565c0", "--"), "llm_only": ("#ef6c00", "-."),
         "approxlg": ("#c62828", "-"), "eamva": ("#2e7d32", (0, (3, 1, 1, 1))),
         "biased_ga": ("#00838f", (0, (1, 1))),
         "nsga2_expert": ("#455a64", (0, (5, 2))),
         "approxlg_sham": ("#f9a825", (0, (4, 1, 1, 1))),
         "nsga2_rf": ("#5d4037", (0, (2, 1)))}


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


def curves(by, grid):
    out = {}
    for a, runs in by.items():
        C = []
        for d in runs:
            c, h = [], 0.0
            for g in grid:
                for n, v in d["hv_trace"]:
                    if n <= g:
                        h = v
                c.append(h)
            C.append(c)
        out[a] = np.array(C)
    return out


def panel(ax, by, budget, title):
    grid = np.arange(16, budget + 1, 16)
    cs = curves(by, grid)
    for a in ORDER:
        if a not in cs or not len(cs[a]):
            continue
        m, s = cs[a].mean(0), cs[a].std(0)
        col, ls = STYLE[a]
        ax.plot(grid, m, color=col, ls=ls, label=LABEL[a])
        ax.fill_between(grid, m - s, m + s, color=col, alpha=0.10, lw=0)
    tops = [c.mean(0)[-1] for c in cs.values() if len(c)]
    lo = min(c.mean(0)[3] for c in cs.values() if len(c) and c.shape[1] > 3)
    ax.set_ylim(max(0.0, lo * 0.85), max(tops) * 1.05)
    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel("EDA evaluations")
    ax.set_ylabel("hypervolume")
    ax.set_title(title, fontsize=7, pad=3)
    ax.grid(alpha=0.25, lw=0.4)


def main():
    plt.rcParams.update(figstyle.rc())
    fig, axes = plt.subplots(1, 2, figsize=figstyle.size("hv_both"))
    panel(axes[0], load(["n8", "n8L", "n8p", "n8abl"]), 400,
          r"$8\times8$ unsigned, $B=400$")
    panel(axes[1], load(["n16u"]), 500, r"$16\times16$ unsigned, $B=500$")
    # Ten curves will not fit inside a single-column panel without covering
    # the data they describe, so the legend goes underneath both panels and is
    # shared.  Handles are pooled across the panels because the 16x16 campaign
    # runs a subset of the arms.
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.tight_layout(pad=0.25, rect=(0, 0.125, 1, 1))
    fig.legend(handles, labels, fontsize=5.9, loc="lower center", ncol=5,
               frameon=False, columnspacing=0.9, handlelength=1.5,
               handletextpad=0.35, labelspacing=0.25, borderaxespad=0.0,
               bbox_to_anchor=(0.5, -0.004))
    out = os.path.join(FIGS, "hv_both.pdf")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
