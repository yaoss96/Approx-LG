"""Compare the Approx-LG Pareto front against the reference designs.

For every reference design we look for the Approx-LG design that is *at least
as accurate* (NMED no larger) and report the relative change in area-delay
product and power-delay product.  The SOTA rate is the fraction of reference
designs that Approx-LG strictly improves on under this iso-accuracy rule.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import figstyle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
TABS = os.path.join(ROOT, "paper", "tabs")

MAX_SEED = 10**9      # no cap; stale runs are purged at the source
NMED_USABLE = 0.10      # a design with a larger mean error is not a multiplier


def load_designs(tags: List[str]) -> List[Dict]:
    out = []
    for tag in tags:
        p = os.path.join(RESULTS, f"designs_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    if d.get("area", 1e9) < 1e8:
                        out.append(d)
    return out


def load_front(tags: List[str], algos=("approxlg",),
               signed_tags=(), max_seeds: int = 0) -> List[Dict]:
    """Pooled non-dominated set of the runs of the given algorithms.

    ``max_seeds`` caps how many runs of each algorithm are pooled.  Comparing
    the front of one arm against another is meaningless without it: a union
    front only grows with the number of runs, so an arm with twenty seeds beats
    more library designs than one with eight for reasons that have nothing to
    do with the algorithm.
    """
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
                if d.get("seed", 0) >= MAX_SEED:
                    continue
                if algos and d["algo"] not in algos:
                    continue
                if max_seeds and d.get("seed", 0) >= max_seeds:
                    continue
                for q in d["pareto"]:
                    q["algo"] = d["algo"]
                    q["signed"] = tag in signed_tags
                pts.extend(d["pareto"])
    return pts


def iso_accuracy(front: List[Dict], ref: Dict, key, metrics=("NMED",)
                 ) -> Dict | None:
    """The cheapest design that is at least as accurate as ``ref`` on every
    requested metric.  Using both NMED and MRE keeps a heavily truncated design
    from being declared equivalent to a design with a bounded relative error."""
    cand = front
    for m in metrics:
        lim = ref.get(m)
        if lim is None or lim != lim:
            continue
        cand = [d for d in cand
                if d.get(m, float("inf")) <= lim * (1 + 1e-9)]
    return min(cand, key=key) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", nargs="+", default=["n8L"])
    ap.add_argument("--algos", nargs="+", default=["approxlg"])
    ap.add_argument("--refs", required=True,
                    help="libs jsonl, or a search tag for method baselines")
    ap.add_argument("--ref-algos", nargs="+", default=None)
    ap.add_argument("--max-seeds", type=int, default=0,
                    help="pool only seeds < this, so fronts from arms with "
                         "different run counts stay comparable")
    ap.add_argument("--signed", choices=["any", "yes", "no"], default="any")
    ap.add_argument("--name", default="sota")
    ap.add_argument("--exact", default=None)
    ap.add_argument("--select", type=int, default=8)
    ap.add_argument("--nmed-max", type=float, default=NMED_USABLE)
    ap.add_argument("--nmed-min", type=float, default=1e-6,
                    help="reference designs below this NMED are effectively "
                         "exact and are not part of the comparison")
    ap.add_argument("--signed-tags", nargs="*", default=[],
                    help="front tags that hold signed designs")
    ap.add_argument("--metrics", nargs="+", default=["NMED", "MRE"],
                    help="accuracy metrics that must not degrade")
    ap.add_argument("--match-signed", action="store_true",
                    help="only compare designs of equal signedness")
    ap.add_argument("--single", action="store_true",
                    help="single column-width panel (ADP only)")
    ap.add_argument("--short", action="store_true",
                    help="omit the library column in the selected table")
    args = ap.parse_args()
    os.makedirs(TABS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)

    front = load_front(args.front, tuple(args.algos), set(args.signed_tags),
                       max_seeds=args.max_seeds)
    if not front:
        print("empty front")
        return
    for d in front:
        d["ADP"] = d["area"] * d["delay"]
        d["PDP"] = d["power"] * d["delay"]

    # ---- reference designs ------------------------------------------
    refs: List[Dict] = []
    if args.refs.endswith(".jsonl") or os.path.exists(
            os.path.join(RESULTS, args.refs)):
        p = args.refs if os.path.isabs(args.refs) else os.path.join(RESULTS,
                                                                    args.refs)
        ver = {}
        vp = os.path.join(RESULTS, "libs_verify.json")
        if os.path.exists(vp):
            ver = {r["name"]: r["status"] for r in json.load(open(vp))}
        skipped = []
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("ok"):
                    continue
                if ver.get(r["name"]) != "match":
                    skipped.append((r["name"], "RTL cross-check"))
                    continue
                if r["NMED"] > args.nmed_max:
                    skipped.append((r["name"], f"NMED={r['NMED']:.2f}"))
                    continue
                if r["NMED"] < args.nmed_min:
                    skipped.append((r["name"], "effectively exact"))
                    continue
                if args.signed == "yes" and not r["signed"]:
                    continue
                if args.signed == "no" and r["signed"]:
                    continue
                refs.append(dict(name=r["name"], lib=r["lib"], NMED=r["NMED"],
                                 MRE=r.get("MRE", float("inf")),
                                 signed=bool(r.get("signed")),
                                 area=r["area"], delay=r["delay"],
                                 power=r["power"], ADP=r["ADP"], PDP=r["PDP"]))
        if skipped:
            print(f"excluded {len(skipped)} reference designs:")
            for nm, why in skipped:
                print(f"    {nm:20s} {why}")
    else:
        pts = load_front([args.refs], tuple(args.ref_algos or []))
        # keep only the non-dominated reference points
        F = np.array([[np.log10(max(d["NMED"], 1e-7)), d["area"], d["delay"]]
                      for d in pts])
        from approxlg.metrics import nondominated_mask
        keep = nondominated_mask(F)
        i = 0
        for d, k in zip(pts, keep):
            if k and args.nmed_min <= d["NMED"] <= args.nmed_max:
                d["ADP"] = d["area"] * d["delay"]
                d["PDP"] = d["power"] * d["delay"]
                pretty = {"eamva": "EA-MVA", "biased_ga": "BC-GA"}
                lbl = pretty.get(d.get("algo", ""), d.get("algo", "ref"))
                d.setdefault("name", f"{lbl}-{i:02d}")
                d.setdefault("lib", lbl)
                i += 1
                refs.append(d)

    if not refs:
        print("no reference designs")
        return

    rows = []
    for r in sorted(refs, key=lambda x: x["NMED"]):
        sub = front
        if args.match_signed:
            sub = [d for d in front
                   if bool(d.get("signed")) == bool(r.get("signed", False))]
            if not sub:
                continue
        a = iso_accuracy(sub, r, lambda d: d["ADP"], tuple(args.metrics))
        p = iso_accuracy(sub, r, lambda d: d["PDP"], tuple(args.metrics))
        if a is None or p is None:
            rows.append((r, None, None))
            continue
        rows.append((r, a, p))

    wins_adp = sum(1 for r, a, _ in rows if a and a["ADP"] < r["ADP"])
    wins_pdp = sum(1 for r, _, p in rows if p and p["PDP"] < r["PDP"])
    wins_both = sum(1 for r, a, p in rows
                    if a and p and a["ADP"] < r["ADP"] and p["PDP"] < r["PDP"])
    wins_any = sum(1 for r, a, p in rows
                   if (a and a["ADP"] < r["ADP"]) or (p and p["PDP"] < r["PDP"]))
    n = len(rows)
    gains_adp = [100 * (1 - a["ADP"] / r["ADP"]) for r, a, _ in rows if a]
    gains_pdp = [100 * (1 - p["PDP"] / r["PDP"]) for r, _, p in rows if p]

    print(f"\n{'reference':22s} {'NMED':>10s} {'ADP_ref':>9s} {'ADP_ours':>9s} "
          f"{'dADP%':>7s} {'PDP_ref':>8s} {'PDP_ours':>9s} {'dPDP%':>7s}")
    for r, a, p in rows:
        if a is None:
            print(f"{r['name']:22s} {r['NMED']:10.3e}   (no iso-accuracy design)")
            continue
        print(f"{r['name']:22s} {r['NMED']:10.3e} {r['ADP']:9.1f} {a['ADP']:9.1f} "
              f"{100*(1-a['ADP']/r['ADP']):7.1f} {r['PDP']:8.1f} {p['PDP']:9.1f} "
              f"{100*(1-p['PDP']/r['PDP']):7.1f}")
    print(f"\nSOTA rate  ADP {wins_adp}/{n} = {100*wins_adp/n:.1f}%   "
          f"PDP {wins_pdp}/{n} = {100*wins_pdp/n:.1f}%   "
          f"both {wins_both}/{n} = {100*wins_both/n:.1f}%   "
          f"either {wins_any}/{n} = {100*wins_any/n:.1f}%")
    # win rate by error decade, so the aggregate rate cannot hide a region
    dec = {}
    for r, a, p in rows:
        if a is None:
            continue
        d = int(np.floor(np.log10(max(r["NMED"], 1e-7))))
        w, t = dec.get(d, (0, 0))
        dec[d] = (w + (1 if a["ADP"] < r["ADP"] else 0), t + 1)
    print("win rate by NMED decade (ADP):")
    for d in sorted(dec):
        w, t = dec[d]
        print(f"    1e{d}: {w}/{t} = {100*w/t:.0f}%")
    print(f"mean ADP reduction {np.mean(gains_adp):.1f}% (median "
          f"{np.median(gains_adp):.1f}%), mean PDP reduction "
          f"{np.mean(gains_pdp):.1f}% (median {np.median(gains_pdp):.1f}%)")

    summary = dict(n=n, wins_adp=wins_adp, wins_pdp=wins_pdp,
                   wins_both=wins_both, wins_any=wins_any,
                   rate_adp=100 * wins_adp / n, rate_pdp=100 * wins_pdp / n,
                   rate_both=100 * wins_both / n, rate_any=100 * wins_any / n,
                   mean_adp=float(np.mean(gains_adp)),
                   median_adp=float(np.median(gains_adp)),
                   mean_pdp=float(np.mean(gains_pdp)),
                   median_pdp=float(np.median(gains_pdp)),
                   max_adp=float(np.max(gains_adp)),
                   max_pdp=float(np.max(gains_pdp)))
    with open(os.path.join(TABS, f"{args.name}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(TABS, f"{args.name}.tex"), "w") as fh:
        for r, a, p in rows:
            if a is None:
                continue
            fh.write(f"{r['name'].replace('_','\\_')} & {r.get('lib','')} & "
                     f"{r['NMED']:.2e} & {r['ADP']:.0f} & {r['PDP']:.0f} & "
                     f"{a['ADP']:.0f} & {100*(1-a['ADP']/r['ADP']):+.1f} & "
                     f"{p['PDP']:.0f} & {100*(1-p['PDP']/r['PDP']):+.1f} \\\\\n")

    # ---- selected rows for the paper ---------------------------------
    good = [(r, a, p) for r, a, p in rows if a is not None]
    if good:
        idx = np.linspace(0, len(good) - 1, min(args.select, len(good)))
        sel, seen, uniq = [good[int(round(i))] for i in idx], set(), []
        for r, a, p in sel:
            if r["name"] in seen:
                continue
            seen.add(r["name"])
            uniq.append((r, a, p))
        pct = chr(92) + "%"
        eol = " " + chr(92) * 2 + "\n"
        with open(os.path.join(TABS, f"{args.name}_sel.tex"), "w") as fh:
            for r, a, p in uniq:
                nm = r["name"].replace("_", chr(92) + "_")
                da = 100 * (1 - a["ADP"] / r["ADP"])
                # Every printed row describes ONE Approx-LG design, the
                # ADP-optimal iso-accuracy answer, so both deltas come from
                # that design.  The aggregate rates in the text may use a
                # different design per metric, which for 13 of the 75
                # references is not the same circuit.
                dp = 100 * (1 - a["PDP"] / r["PDP"])
                if args.short:
                    cols = [nm, f"{r['NMED']:.2e}", f"{r['ADP']:.0f}",
                            f"{a['ADP']:.0f}", f"{da:+.1f}{pct}",
                            f"{a['PDP']:.1f}", f"{dp:+.1f}{pct}"]
                else:
                    lib = r.get("lib", "")
                    # Signed and unsigned references are answered from
                    # different fronts, so the row is only readable if the
                    # table says which one it came from.
                    sgn = "s" if r.get("signed") else "u"
                    # Both accuracy metrics are printed for our design and for
                    # the reference it answers, so the reader can check the
                    # iso-accuracy rule row by row instead of taking it on
                    # trust: ours is never worse on either metric.
                    # Ours first, then the design it answers.  The name sits
                    # with its own numbers rather than four columns away.
                    cols = [f"{a['NMED']:.2e}", f"{a.get('MRE', 0):.4f}",
                            f"{a['ADP']:.0f}", f"{a['PDP']:.0f}",
                            nm, lib, sgn,
                            f"{r['NMED']:.2e}", f"{r.get('MRE', 0):.4f}",
                            f"{r['ADP']:.0f}", f"{r['PDP']:.0f}",
                            f"{da:+.1f}{pct}", f"{dp:+.1f}{pct}"]
                fh.write(" & ".join(cols) + eol)

    # ---- scatter figure ---------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "serif",
                         "legend.frameon": False, "figure.dpi": 300})
    from approxlg.metrics import nondominated_mask

    # The two cost/accuracy planes the paper argues about.  NMED and MRE are
    # not interchangeable, so a design that looks good in the first row can
    # sit well off the front in the second.
    PLANES = [("NMED", "ADP", "NMED",
               "area$\\times$delay ($\\mu$m$^2\\cdot$ns)"),
              ("MRE", "PDP", "MRE", "power$\\times$delay (fJ)")]

    # Marker shape identifies where a design comes from; marker colour says
    # whether it survives on the Pareto front of *everything* plotted in that
    # panel (red) or is dominated by some other design (black).
    MARK = {"Approx-LG": ("o", 11), "EvoApproxLib": ("D", 13),
            "AM-Lib": ("^", 15), "MulApprox": ("s", 13),
            "EA-MVA": ("v", 14), "BC-GA": ("P", 14)}
    ON, OFF = "#d32f2f", "#111111"

    def panel(ax, pts, title, xk, yk, xlbl, ylbl):
        """pts: list of (label, design dicts).  Colours by dominance in the
        (xk, yk) plane of this panel alone."""
        allp = [d for _, group in pts
                for d in group if d.get(xk, 0) > 0 and d.get(yk, 0) > 0]
        if not allp:
            return
        F = np.array([[np.log10(d[xk]), np.log10(d[yk])] for d in allp])
        on = {id(d) for d, k in zip(allp, nondominated_mask(F)) if k}
        for lb, group in pts:
            m, sz = MARK.get(lb, ("o", 12))
            g = [d for d in group if d.get(xk, 0) > 0 and d.get(yk, 0) > 0]
            for keep, col, z in ((True, ON, 5), (False, OFF, 3)):
                sel = [d for d in g if (id(d) in on) == keep]
                if not sel:
                    continue
                ax.scatter([d[xk] for d in sel], [d[yk] for d in sel],
                           s=sz, marker=m, facecolors="none", edgecolors=col,
                           linewidths=0.65, zorder=z)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_title(title, fontsize=7, pad=3)
        ax.grid(alpha=0.22, lw=0.4, which="both")

    def thin(seq, xk, yk, keep=0.8):
        """Reduce a pooled multi-run front to its own non-dominated subset in
        this panel's plane, then drop a fixed fraction of what is left, evenly
        along the front, so the markers do not run together.  The statistics
        use the full set."""
        g = [d for d in seq if d.get(xk, 0) > 0 and d.get(yk, 0) > 0]
        if not g:
            return []
        F = np.array([[np.log10(d[xk]), np.log10(d[yk])] for d in g])
        g = [d for d, k in zip(g, nondominated_mask(F)) if k]
        if keep >= 1.0 or len(g) < 4:
            return g
        g.sort(key=lambda d: d[xk])
        idx = np.unique(np.round(np.linspace(0, len(g) - 1,
                                             max(2, int(round(keep * len(g))))
                                             )).astype(int))
        return [g[i] for i in idx]

    from matplotlib.lines import Line2D
    libs = sorted({r.get("lib", "ref") for r in refs})
    if args.single:
        fig, ax0 = plt.subplots(figsize=figstyle.size("sota_single"))
        grid, splits = [[ax0]], [(None, "")]
        planes = PLANES[:1]
    else:
        # Columns are the two templates, rows the two cost/accuracy planes.
        fig, grid = plt.subplots(2, 2, figsize=figstyle.size("sota_grid"))
        splits = [(False, "unsigned"), (True, "signed")]
        planes = PLANES

    for row, (xk, yk, xlbl, ylbl) in enumerate(planes):
        for col, (sg, title) in enumerate(splits):
            ax = grid[row][col]

            def sel(seq, sg=sg):
                return [d for d in seq
                        if sg is None or bool(d.get("signed")) == sg]

            # The panels show the region the published designs occupy.  Our
            # search also returns points far outside it -- relative errors
            # above 100%, which no application uses -- and plotting them
            # stretches both axes by two decades and squashes the region the
            # comparison is actually about.  They are excluded from the plot
            # by the range of the references themselves, never from the rates:
            # a design outside that range has nothing to be compared against.
            rng_ref = sel(refs)
            if rng_ref:
                xmax = max(r[xk] for r in rng_ref)

                def in_range(seq, _x=xmax):
                    return [d for d in seq if d.get(xk, 0) <= _x]
            else:
                def in_range(seq):
                    return seq
            panel(ax, [("Approx-LG", thin(in_range(sel(front)), xk, yk))] +
                  [(lb, sel([r for r in refs if r.get("lib", "ref") == lb]))
                   for lb in libs],
                  (f"$16\\times16$ {title}" if title else ""),
                  xk, yk, xlbl, ylbl)
            # Both rows keep their x label: the two rows plot different
            # accuracy metrics, so a shared label would misread.
            if col > 0:
                ax.set_ylabel("")

    handles = [Line2D([], [], marker=MARK.get(lb, ("o", 12))[0], ls="none",
                      mfc="none", mec="#555555", mew=0.7, ms=3.6, label=lb)
               for lb in ["Approx-LG"] + libs]
    handles += [Line2D([], [], marker="o", ls="none", mfc="none", mec=ON,
                       mew=0.9, ms=3.6, label="on Pareto front")]
    grid[0][0].legend(handles=handles, fontsize=5.6, loc="lower left", ncol=2,
                      columnspacing=0.7, handletextpad=0.35, borderpad=0.2)
    fig.tight_layout(pad=0.25)
    fig.savefig(os.path.join(FIGS, f"{args.name}.pdf"))
    print("figure ->", os.path.join(FIGS, f"{args.name}.pdf"))


if __name__ == "__main__":
    main()
