"""Which compressor styles does the search actually use, and where?

Produces (a) a table of the most used styles with their cells and measured
error bias and (b) a heat map of style usage against the distance of a column
above the truncation point -- the same coordinate the LLM prior is elicited in.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
TABS = os.path.join(ROOT, "paper", "tabs")

import figstyle                                                # noqa: E402
from approxlg.arch import Space                                # noqa: E402
from approxlg.cells import (CELL_BY_NAME, SLOT_MODES, bias_symbol,
                            cell_statistics)                   # noqa: E402


def load(tags, algos):
    pts = []
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d["algo"] in algos:
                    pts.extend(d["pareto"])
    return pts


def main():
    tags = sys.argv[1:] or ["n8L", "n8"]
    sp = Space(8)
    pts = load(tags, {"approxlg"})
    if not pts:
        print("no data")
        return
    use = Counter()
    grid = np.zeros((len(SLOT_MODES), 6))
    for d in pts:
        g = np.array(d["g"], dtype=np.int32)
        t, a = int(g[0]), int(g[1])
        sl = g[sp.n_head:].reshape(sp.nstage, sp.ncol)
        for j in range(t, min(t + a, sp.ncol)):
            for s in range(sp.nstage):
                m = int(sl[s, j])
                if m == 0:
                    continue
                use[m] += 1
                grid[m, min(j - t, 5)] += 1
    tot = sum(use.values())
    print(f"{len(pts)} Pareto designs, {tot} approximate slot instances")

    rows = []
    for m, c in use.most_common(6):
        mode = SLOT_MODES[m]
        parts = []
        for role, nm in (("3:2", mode.fa), ("2:2", mode.ha), ("4:2", mode.c42)):
            if nm:
                st = cell_statistics(CELL_BY_NAME[nm])
                parts.append((role, nm, st))
        fa = next((p for p in parts if p[0] == "3:2"), None)
        c42 = next((p for p in parts if p[0] == "4:2"), None)
        rows.append((mode.name, fa, c42, 100 * c / tot))
        print(f"  {mode.name:6s} {100*c/tot:5.1f}%  " +
              "  ".join(f"{r}={n}({s['mean_ed']:+.2f})" for r, n, s in parts))

    with open(os.path.join(TABS, "styles.tex"), "w") as fh:
        for nm, fa, c42, pct in rows:
            def cell(x):
                if not x:
                    return "--"
                _, n, s = x
                return (f"{n.replace('_', chr(92)+'_')} "
                        f"($\overline{{ED}}{s['mean_ed']:+.2f}$)")
            fh.write(f"{nm} & {cell(fa)} & {cell(c42)} & {pct:.1f}\\% "
                     + chr(92) * 2 + "\n")

    # A prose version of the same table, used in place of it in the paper.
    def group(pred):
        g = [(nm, fa[2]["mean_ed"], pct) for nm, fa, _, pct in rows
             if fa and pred(fa[2]["mean_ed"])]
        return g, sum(p for _, _, p in g)
    under, u_pct = group(lambda e: e > 0.05)
    over, o_pct = group(lambda e: e < -0.05)
    bal, b_pct = group(lambda e: abs(e) <= 0.05)

    def names(g, eds=True):
        nm = ", ".join(f"\\texttt{{{n}}}" for n, _, _ in g)
        if not eds or not g:
            return nm
        seen, vals = set(), []
        for _, e, _ in g:                 # styles that share an ED say it once
            v = f"{e:+.2f}"
            if v not in seen:
                seen.add(v)
                vals.append(v)
        return f"{nm} at $\\overline{{ED}}=" + ",\\,".join(vals) + "$"
    with open(os.path.join(TABS, "styles_text.tex"), "w") as fh:
        fh.write(
            # print the total as the three group shares actually add up to,
            # so a reader who sums the parts does not land one point away
            f"{len(rows)} styles cover {round(u_pct, 1) + round(o_pct, 1) + round(b_pct, 1):.1f}"
            f"\\,\\% of the approximate slots, and they mix under-estimating "
            f"cells ({names(under)}, {u_pct:.1f}\\,\\%) with over-estimating "
            f"ones ({names(over)}, {o_pct:.1f}\\,\\%) and the balanced "
            f"{names(bal, eds=False)} ({b_pct:.1f}\\,\\%)")

    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(figstyle.rc())
    keep = [m for m, _ in use.most_common(10)]
    G = grid[keep]
    G = G / np.maximum(G.sum(axis=0, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=figstyle.size("styles"))
    im = ax.imshow(G, aspect="auto", cmap="magma_r", vmin=0, vmax=G.max())
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([SLOT_MODES[m].name for m in keep])
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5+"])
    ax.set_xlabel("column distance $d$ above the truncation point")
    ax.set_ylabel("compressor style")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("share of slots at distance $d$", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout(pad=0.15)
    fig.savefig(os.path.join(FIGS, "styles.pdf"))
    print("wrote", os.path.join(FIGS, "styles.pdf"))


if __name__ == "__main__":
    main()
