"""Figures of the TCAD version (IEEE two-column geometry).

    hv_curves.pdf   hypervolume against evaluations, one panel per width
    metrics16.pdf   area, delay and power against NMED at 16x16 (unsigned),
                    the search front beside every reference design
    mre8.pdf        the 8x8 MRE/PDP plane: compressor-tree references, the
                    NMED-objective front and the MRE-retargeted front
    cites.json      the citation numbers the legends print

Run after building the paper, since the legends read main.aux.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "tcad", "figs")
sys.path.insert(0, ROOT)

COL = 3.5      # in, IEEE column
FULL = 7.16    # in, IEEE text width
plt.rcParams.update({"font.size": 7, "font.family": "serif",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "legend.frameon": False, "figure.dpi": 300,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})

ARMS = [("nsga2", "NSGA-II", "#1565c0", "--"),
        ("smsemoa", "SMS-EMOA", "#2e7d32", (0, (5, 1, 1, 1))),
        ("mobo", "BO", "#6a1b9a", (0, (1, 1))),
        ("nsga2_rf", "RF surrogate", "#5d4037", (0, (2, 1))),
        ("approxlg_sham", "sham", "#f9a825", (0, (4, 1, 1, 1))),
        ("approxlg", "Approx-LG", "#c62828", "-"),
        ("approxlg_sms", "Approx-LG on SMS-EMOA", "#000000", "-")]
TAGS = {"8": ["n8", "n8L", "n8abl", "n8moo", "n8hyb", "n8hyb2"],
        "16u": ["n16u", "n16umoo", "n16uhyb"],
        "32u": ["n32u", "n32ufill", "n32uhyb"]}
TITLE = {"8": r"$8\times8$", "16u": r"$16\times16$ (unsigned)",
         "32u": r"$32\times32$ (unsigned)"}
BUDGET = {"8": 400, "16u": 500, "32u": 500}

# a legend cites a baseline or a reference library as the tables do, with the
# numbers of the built paper (main.aux); package_tcad.py checks figs/cites.json
# against the paper before packaging
AUX = os.path.join(ROOT, "paper", "tcad", "main.aux")
CITE = {"nsga2": "deb2002", "smsemoa": "smsemoa", "mobo": "amg,qnehvi",
        "nsga2_rf": "jin2019", "OPACT": "opact", "Esposito": "esposito2018",
        "Wang": "wang2020", "AM-Lib": "amlib", "MulApprox": "mulapprox",
        "EvoApproxLib": "evoapproxlib"}
USED = {}


def cited(lab, name):
    """lab followed by what \\cite prints for the keys of name ("[3], [4]",
    "[5]-[7]"), or lab alone for an arm of this paper."""
    if name not in CITE:
        return lab
    num = dict(re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", open(AUX).read()))
    keys = CITE[name].split(",")
    USED.update((k, int(num[k])) for k in keys)
    runs = []
    for x in sorted(int(num[k]) for k in keys):
        if runs and x == runs[-1][-1] + 1:
            runs[-1].append(x)
        else:
            runs.append([x])
    return lab + " " + ", ".join(
        f"[{r[0]}]\u2013[{r[-1]}]" if len(r) > 2 else
        ", ".join(f"[{x}]" for x in r) for r in runs)


def load(tags):
    by = defaultdict(dict)
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                d = json.loads(line)
                by[d["algo"]].setdefault(d["seed"], d)
    return by


def curve(runs, grid):
    C = []
    for d in runs:
        tr = d["hv_trace"]
        c, h, i = [], 0.0, 0
        for g in grid:
            while i < len(tr) and tr[i][0] <= g:
                h = tr[i][1]
                i += 1
            c.append(h)
        C.append(c)
    return np.array(C)


def hv_curves():
    panels = [w for w in ("8", "16u", "32u") if load(TAGS[w])]
    fig, axes = plt.subplots(1, len(panels), figsize=(FULL, 1.75))
    axes = np.atleast_1d(axes)
    for ax, w in zip(axes, panels):
        by = load(TAGS[w])
        grid = np.arange(32, BUDGET[w] + 1, 4)
        for a, lab, col, ls in ARMS:
            if a not in by:
                continue
            C = curve(list(by[a].values()), grid)
            m, s = C.mean(0), C.std(0)
            ax.plot(grid, m, color=col, ls=ls,
                    label=f"{cited(lab, a)} ({len(C)})")
            ax.fill_between(grid, m - s / np.sqrt(len(C)), m + s / np.sqrt(len(C)),
                            color=col, alpha=0.15, lw=0)
        ax.set_title(TITLE[w], fontsize=7)
        ax.set_xlabel("distinct circuits evaluated")
        lo = min(np.percentile(curve(list(by["nsga2"].values()), grid)[:, len(grid) // 5], 10)
                 for _ in [0]) if "nsga2" in by else None
        if lo is not None:
            top = max(curve(list(by[a].values()), grid).mean(0)[-1]
                      for a, *_ in ARMS if a in by)
            ax.set_ylim(lo, top + 0.08 * (top - lo))
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("hypervolume (mean $\\pm$ s.e.)")
    shared_legend(fig, axes)
    fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(FIGS, "hv_curves.pdf"))
    plt.close(fig)


def shared_legend(fig, axes, **kw):
    """One legend above all panels, every entry once (run counts dropped)."""
    handles = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab.split(" (")[0], h)
    fig.legend(list(handles.values()), list(handles.keys()), fontsize=6,
               loc="upper center", ncol=len(handles), handlelength=2.4,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, **kw)


# ---------------------------------------------------------------------------
def pooled(by, algo, max_seeds=20):
    pts = []
    for s, d in by.get(algo, {}).items():
        if s < max_seeds:
            pts.extend(d["pareto"])
    return pts


def nondom(pts, keys):
    F = np.array([[p[k] for k in keys] for p in pts])
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        if np.any(np.all(F <= F[i], 1) & np.any(F < F[i], 1)):
            keep[i] = False
    return [p for p, k in zip(pts, keep) if k]


REFSTYLE = {"OPACT": ("#1565c0", "^"), "Esposito": ("#2e7d32", "s"),
            "Wang": ("#00838f", "D"), "AM-Lib": ("#6a1b9a", "v"),
            "MulApprox": ("#ef6c00", "P"), "EvoApproxLib": ("#5d4037", "x")}


def metrics16():
    by = load(["n16u"])
    front = pooled(by, "approxlg")
    refs = [json.loads(l) for l in open(os.path.join(RESULTS, "refgen_n16.jsonl"))]
    libs = []
    vp = os.path.join(RESULTS, "libs_verify.json")
    ver = {r["name"]: r["status"] for r in json.load(open(vp))} if os.path.exists(vp) else {}
    for l in open(os.path.join(RESULTS, "libs_n16.jsonl")):
        d = json.loads(l)
        if d.get("ok") and not d.get("signed") and ver.get(d["name"]) == "match":
            libs.append(d)
    allref = [r for r in refs + libs if 1e-7 <= r["NMED"] <= 0.02]
    ex = json.load(open(os.path.join(RESULTS, "exact_n16u.json")))
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.9))
    for ax, key, lab in zip(axes, ("area", "delay", "power"),
                            (r"area ($\mu$m$^2$)", "delay (ns)",
                             r"power ($\mu$W)")):
        fr = nondom([p for p in front if p["NMED"] >= 1e-7],
                    ["NMED", key])
        fr.sort(key=lambda p: p["NMED"])
        for lib, (col, mk) in REFSTYLE.items():
            sub = [r for r in allref if r["lib"] == lib]
            if sub:
                ax.scatter([r["NMED"] for r in sub], [r[key] for r in sub],
                           s=7 if mk != "x" else 9, marker=mk, color=col,
                           lw=0.5 if mk == "x" else 0, alpha=0.75,
                           label=cited(lib, lib))
        ax.step([p["NMED"] for p in fr], [p[key] for p in fr], where="post",
                color="#c62828", lw=1.1, label="Approx-LG front")
        ax.set_xscale("log")
        ax.set_xlabel("NMED")
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.25 * ex[key])
        ax.axhline(ex[key], color="0.4", lw=0.5, ls=":")
        ax.grid(alpha=0.25, lw=0.4)
    shared_legend(fig, axes, markerscale=1.4)
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(FIGS, "metrics16.pdf"))
    plt.close(fig)


def front2(xs, ys):
    """Non-dominated staircase of a 2-D point set (both minimized)."""
    o = np.lexsort((ys, xs))
    fx, fy, best = [], [], float("inf")
    for i in o:
        if ys[i] < best:
            best = ys[i]
            fx.append(xs[i])
            fy.append(ys[i])
    return fx, fy


def characterized(keep):
    """Every 8x8 design the campaigns characterized, one per genome."""
    import glob
    pts = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "designs_n8*.jsonl"))):
        if not keep(os.path.basename(f)):
            continue
        for line in open(f):
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("area") and d.get("MRE", 0) > 0:
                pts[tuple(d["g"])] = (d["MRE"], d["area"], d["delay"],
                                      d["power"])
    return np.array(list(pts.values()))


def mre8():
    """The 8x8 MRE plane in ADP and PDP: the compressor-tree references, the
    Approx-LG fronts of both objectives, and the envelope of every template
    design any campaign characterized (is the gap the search's or the
    template's?)."""
    by = load(["n8L"])
    bym = load(["n8mre"])
    bymx = load(["n8mrex"])
    refs = [json.loads(l) for l in open(os.path.join(RESULTS, "refgen_n8.jsonl"))]
    refs = [r for r in refs if r["MRE"] > 0]
    allc = characterized(lambda f: "mrex" not in f)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9))
    for ax, key, lab in zip(axes, ("ADP", "PDP"),
                            (r"area$\times$delay ($\mu$m$^2$ns)",
                             r"power$\times$delay (fJ)")):
        cost = ((lambda a, d, p: a * d) if key == "ADP"
                else (lambda a, d, p: p * d))
        ex, ey = front2(allc[:, 0], cost(allc[:, 1], allc[:, 2], allc[:, 3]))
        ax.step(ex, ey, where="post", color="0.6", lw=0.8,
                label="template, every design")
        for lib in ("OPACT", "Esposito", "Wang"):
            col, mk = REFSTYLE[lib]
            sub = [r for r in refs if r["lib"] == lib]
            ax.scatter([r["MRE"] for r in sub],
                       [cost(r["area"], r["delay"], r["power"]) for r in sub],
                       s=5, marker=mk, color=col, lw=0, alpha=0.8,
                       label=cited(lib, lib))
        for b, lb, col, ls in ((by, "NMED objective", "#c62828", "--"),
                               (bym, "MRE objective", "#c62828", "-"),
                               (bymx, "MRE objective, +E", "#ef6c00", "-")):
            pts = [p for p in pooled(b, "approxlg", 20 if b is by else 12)
                   if p.get("MRE", 0) > 0]
            if not pts:
                continue
            fx, fy = front2(np.array([p["MRE"] for p in pts]),
                            np.array([cost(p["area"], p["delay"], p["power"])
                                      for p in pts]))
            ax.step(fx, fy, where="post", color=col, ls=ls, lw=0.9, label=lb)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 0.1)
        win = [cost(r["area"], r["delay"], r["power"]) for r in refs
               if 1e-4 <= r["MRE"] <= 0.1]
        ax.set_ylim(0.85 * min(win), 1.05 * max(win))
        ax.set_xlabel("MRE")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.4)
    handles = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles.keys()), fontsize=5,
               loc="upper center", ncol=4, handlelength=1.8, markerscale=1.6,
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.8)
    fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0, 1, 0.80))
    fig.savefig(os.path.join(FIGS, "mre8.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    for f in (hv_curves, metrics16, mre8):
        try:
            f()
            print("ok", f.__name__)
        except Exception as e:                       # pragma: no cover
            print("!!", f.__name__, type(e).__name__, e)
    with open(os.path.join(FIGS, "cites.json"), "w") as fh:
        json.dump(USED, fh, indent=1, sort_keys=True)
