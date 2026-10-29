"""Tables and macros of the TCAD version.

Reads every campaign the paper uses, recomputes the paired statistics, applies
Holm's correction inside each pre-declared family, and writes

    paper/tcad/tabs/*.tex       table bodies
    paper/tcad/numbers_tcad.tex macros quoted in the text
    paper/tcad/tabs/tcad_stats.json   everything, for audit

Families (each Holm-corrected on its own):
    primary   guided vs sham at every width/template
    placebo   sham vs NSGA-II (is the control handicapped?)
    space     guided vs every same-space optimizer
    priors    sources of the proposal distribution
    ablation  each operator removed, against the full system
    retarget  the MRE-objective campaign
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper", "tcad")
TABS = os.path.join(PAPER, "tabs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analyze_stats import holm, paired                       # noqa: E402

# width key -> campaign tags holding its runs
WIDTHS = {
    "8": ["n8", "n8L", "n8abl", "n8p", "n8moo"],
    "16u": ["n16u", "n16umoo", "n16uabl", "n16uhyb"],
    "16s": ["n16s", "n16smoo"],
    "32u": ["n32u"],
    "8mre": ["n8mre"],
    "8mrex": ["n8mrex"],
    # the full system re-run on the 16x16 ablation's seeds two months later
    "16udrift": ["n16udrift"],
}
WLABEL = {"8": r"$8\times8$", "16u": r"$16\times16$ u", "16s": r"$16\times16$ s",
          "32u": r"$32\times32$ u", "8mre": r"$8\times8$ (MRE)",
          "8mrex": r"$8\times8$ (MRE, +E)"}
LABEL = {
    "random": "Random", "ga": "GA", "nsga2": "NSGA-II", "smsemoa": "SMS-EMOA",
    "mobo": "BO (GP-EHVI)", "nsga2_rf": "NSGA-II + RF surrogate",
    "eamva": r"EA-MVA~\cite{devi2025}", "biased_ga": r"BC-GA~\cite{wen2025}",
    "llm_only": "LLM-only", "nsga2_expert": "NSGA-II + written prior",
    "nsga2_adaptive": "NSGA-II + UCB prior",
    "nsga2_llmprior": "NSGA-II + elicited prior",
    "approxlg_sham": "sham (placebo)", "approxlg": "Approx-LG",
    "approxlg_no_warm": "w/o O1 warm start",
    "approxlg_no_inject": "w/o O2 injection",
    "approxlg_no_mutguide": "w/o O3 (prior and local search)",
    "approxlg_no_prior": "w/o O3 prior (local search kept)",
    "approxlg_no_refine": "w/o O4 refinement",
    "approxlg_sms": "Approx-LG on SMS-EMOA",
    "approxlg_sms_sham": "sham on SMS-EMOA",
}
TABLE_ORDER = ["random", "ga", "eamva", "biased_ga", "llm_only", "nsga2",
               "smsemoa", "mobo", "nsga2_rf", "nsga2_expert", "nsga2_llmprior",
               "approxlg_sham", "approxlg", "approxlg_sms_sham", "approxlg_sms"]
ABLATIONS = ["approxlg_no_warm", "approxlg_no_inject", "approxlg_no_mutguide",
             "approxlg_no_prior", "approxlg_no_refine"]

M = {}


REWRITTEN = {}


def mac(name, value):
    # a name set twice to different values means two quantities share a
    # macro; the text would quote whichever was written last
    if name in M and M[name] != value:
        REWRITTEN[name] = (M[name], value)
    M[name] = value


def load():
    runs = {}
    for w, tags in WIDTHS.items():
        by = defaultdict(dict)
        for tag in tags:
            p = os.path.join(RESULTS, f"search_{tag}.jsonl")
            if not os.path.exists(p):
                continue
            for line in open(p):
                if not line.strip():
                    continue
                d = json.loads(line)
                # first record of a seed wins; duplicates are refused at
                # write time, this only guards against stale copies
                by[d["algo"]].setdefault(d["seed"], d)
        runs[w] = by
    return runs


def hv_by(runs, w):
    return {a: {s: r["hv"] for s, r in v.items()} for a, v in runs[w].items()}


def fmtp(p):
    if p != p:
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def mfmtp(p):
    """Relation and value, for $p\\Macro$ in the text."""
    if p < 0.001:
        return "{<}0.001"
    return "{=}" + f"{p:.3f}".rstrip("0").rstrip(".")


def evals_to(run, target, budget):
    for k, h in run["hv_trace"]:
        if h >= target:
            return k
    return budget


# ---------------------------------------------------------------------------
def search_table(runs):
    """One row per arm, one column group per width: HV mean (sd) and runs."""
    cols = ["8", "16u", "16s", "32u"]
    lines = []
    for a in TABLE_ORDER:
        cells = []
        present = False
        for w in cols:
            r = runs[w].get(a)
            if not r:
                cells.append("--")
                continue
            present = True
            hv = np.array([x["hv"] for x in r.values()])
            cells.append(f"{hv.mean():.4f} ({100 * hv.std() / hv.mean():.1f}) "
                         f"[{len(hv)}]")
        if present:
            lines.append(f"{LABEL[a]} & " + " & ".join(cells) + r" \\")
    with open(os.path.join(TABS, "search_all.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # macros: means per arm and width (the MRE campaigns as HV...Eightmre)
    for w in cols + ["8mre", "8mrex"]:
        for a, r in runs[w].items():
            hv = np.array([x["hv"] for x in r.values()])
            key = (a.replace("_", "").replace("approxlg", "alg")
                   .replace("nsga2", "nsga").replace("sham", "Sham"))
            mac(f"HV{key}{w}".replace("8", "Eight").replace("16", "Sixteen")
                .replace("32", "ThirtyTwo"), f"{hv.mean():.4f}")
    # unpaired gaps of the weak arms to NSGA-II at 8x8, in % of its mean, and
    # the spread of the full-space loops around it
    ref = np.mean([x["hv"] for x in runs["8"]["nsga2"].values()])
    for a, k in (("random", "Rand"), ("llm_only", "LlmOnly"),
                 ("eamva", "Eamva"), ("biased_ga", "Bcga")):
        if a in runs["8"]:
            m = np.mean([x["hv"] for x in runs["8"][a].values()])
            mac(f"Gap{k}Eight", f"{100 * (1 - m / ref):.0f}")
    loops = [a for a in ("smsemoa", "mobo", "nsga2_rf", "approxlg_sham",
                         "approxlg") if a in runs["8"]]
    if loops:
        dev = max(abs(np.mean([x["hv"] for x in runs["8"][a].values()])
                      / ref - 1) for a in loops)
        mac("SpreadLoopsEight", f"{100 * dev:.1f}")


def cost_table(runs):
    """Mean wall clock, generations and guidance seconds per run."""
    lines = []
    for w in ("8", "16u", "16s", "32u"):
        for a in ("nsga2", "smsemoa", "mobo", "nsga2_rf", "approxlg_sham",
                  "approxlg"):
            r = runs[w].get(a)
            if not r:
                continue
            sec = np.mean([x["seconds"] for x in r.values()])
            llm = np.mean([(x.get("t_llm") or (x.get("llm") or {}).get(
                "seconds", 0.0) or 0.0) for x in r.values()])
            calls = np.mean([((x.get("llm") or {}).get("calls", 0) or 0)
                             for x in r.values()])
            gens = np.mean([len(x["hv_trace"]) - 1 for x in r.values()])
            lines.append((w, a, sec, llm, calls, gens))
            # the optimizer's own computation: run time minus simulation,
            # synthesis and inference (runs that recorded the split only)
            split = [x for x in r.values() if "t_synth" in x]
            if split:
                ov = np.mean([x["seconds"] - x.get("t_sim", 0.0)
                              - x.get("t_synth", 0.0) - (x.get("t_llm") or 0.0)
                              for x in split])
                tag = {"8": "Eight", "16u": "SixteenU", "16s": "SixteenS",
                       "32u": "ThirtyTwo"}[w]
                short = {"nsga2": "Nsga", "smsemoa": "Sms", "mobo": "Bo",
                         "nsga2_rf": "Rf", "approxlg_sham": "Sham",
                         "approxlg": "Alg"}[a]
                mac(f"Overhead{short}{tag}", f"{ov:.0f}")
                if a == "nsga2":
                    # error simulation per newly characterized design (GPU)
                    sim = np.mean([x["t_sim"] / max(1, x.get("n_synth", 0))
                                   for x in split if x.get("n_synth")])
                    mac(f"SimPerDesign{tag}", f"{sim:.2f}")
    return lines


# ---------------------------------------------------------------------------
def families(runs):
    fam = defaultdict(list)          # family -> [(label, width, x, y, result)]

    def add(f, w, x, y, label=None):
        hv = hv_by(runs, w)
        if x not in hv or y not in hv:
            return
        common = set(hv[x]) & set(hv[y])
        if len(common) < 5:
            return
        r = paired(hv, x, y)
        fam[f].append((label or f"{LABEL[x]} vs {LABEL[y]}", w, x, y, r))

    for w in ("8", "16u", "16s", "32u"):
        add("primary", w, "approxlg", "approxlg_sham")
        add("placebo", w, "approxlg_sham", "nsga2")
        for y in ("nsga2", "smsemoa", "mobo", "nsga2_rf"):
            add("space", w, "approxlg", y)
        # the selection operator, and the Bayesian optimizer, without any model
        add("space", w, "smsemoa", "nsga2")
        add("space", w, "mobo", "nsga2")
    # knowledge sources
    for x in ("nsga2_expert", "nsga2_adaptive", "nsga2_llmprior"):
        add("priors", "8", x, "nsga2")
    add("priors", "8", "nsga2_llmprior", "nsga2_expert")
    for w in ("16u", "16s"):
        add("priors", w, "nsga2_expert", "nsga2")
        add("priors", w, "nsga2_llmprior", "nsga2")
        add("priors", w, "approxlg", "nsga2_expert")
    # ablations at both widths
    for v in ABLATIONS:
        add("ablation", "8", v, "approxlg")
    # At 16x16 the ablation ran two months after the full-system campaign, so
    # each variant is paired with the full system re-run at the same time on
    # the same seeds (n16udrift); the re-run against the earlier runs is the
    # drift check.  Without the re-run, fall back to the earlier runs.
    hvd = hv_by(runs, "16udrift").get("approxlg")
    hvo = hv_by(runs, "16u").get("approxlg")
    hv16 = hv_by(runs, "16u")
    for v in ABLATIONS:
        ref = hvd if hvd else hvo
        if v in hv16 and ref and len(set(hv16[v]) & set(ref)) >= 5:
            r = paired({"x": hv16[v], "y": ref}, "x", "y")
            fam["ablation"].append((f"{LABEL[v]} vs Approx-LG", "16u", v,
                                    "approxlg", r))
    if hvd and hvo and len(set(hvd) & set(hvo)) >= 5:
        r = paired({"x": hvd, "y": hvo}, "x", "y")
        fam["ablation"].append(("full system, re-run two months later",
                                "16u", "approxlg_drift", "approxlg", r))
        # the variants against the earlier runs, for the text only
        aug = []
        for v in ABLATIONS:
            if v in hv16 and len(set(hv16[v]) & set(hvo)) >= 5:
                ra = paired({"x": hv16[v], "y": hvo}, "x", "y")
                mac(f"AblAug{v.split('approxlg_')[1].title().replace('_', '')}",
                    f"{ra['delta']:+.2f}")
                aug.append(ra["delta"])
        if aug:
            mac("AblAugMin", f"{min(aug):+.2f}")
            mac("AblAugMax", f"{max(aug):+.2f}")
    # retargeting
    for y in ("approxlg_sham", "nsga2", "smsemoa"):
        add("retarget", "8mre", "approxlg", y)
    add("retarget", "8mre", "smsemoa", "nsga2")
    for y in ("nsga2", "smsemoa"):
        add("retarget", "8mrex", "approxlg", y)
    add("retarget", "8mrex", "smsemoa", "nsga2")
    # the new style itself: same arm and seeds, extended against base alphabet
    for a in ("approxlg", "nsga2", "smsemoa"):
        hvb = hv_by(runs, "8mre").get(a)
        hvx = hv_by(runs, "8mrex").get(a)
        if hvb and hvx and len(set(hvb) & set(hvx)) >= 5:
            r = paired({"x": hvx, "b": hvb}, "x", "b")
            fam["retarget"].append((f"{LABEL[a]}: with style E vs without",
                                    "8mre", a + "_ext", a, r))
    # complementarity: the guidance on top of SMS-EMOA's selection
    add("hybrid", "16u", "approxlg_sms", "approxlg_sms_sham")
    add("hybrid", "16u", "approxlg_sms", "smsemoa")
    add("hybrid", "16u", "approxlg_sms_sham", "smsemoa")
    add("hybrid", "16u", "approxlg_sms", "approxlg")
    for f, rows in fam.items():
        adj = holm([r["p"] for *_, r in rows])
        for (lab, w, x, y, r), pa in zip(rows, adj):
            r["p_holm"] = float(pa)
    return fam


def summary_table(fam):
    """Two table bodies: search comparisons (summary.tex) and attribution
    comparisons (summary2.tex).  The equal-wall-clock family is corrected
    with the others but printed in the cost table, beside its inputs."""
    parts = {"summary.tex": ["primary", "placebo", "space", "hybrid"],
             "summary2.tex": ["priors", "ablation", "retarget"]}
    name = {"primary": "primary", "placebo": "placebo check",
            "space": "same-space optimizers", "priors": "knowledge sources",
            "ablation": "operator ablation", "retarget": "MRE retargeting",
            "hybrid": "guidance on SMS-EMOA",
            "wallclock": "equal wall clock"}
    for fname, order in parts.items():
        lines = []
        for f in order:
            rows = fam.get(f, [])
            if not rows:
                continue
            lines.append(r"\multicolumn{7}{@{}l}{\emph{" + name[f] +
                         f" ({len(rows)} test{'s' if len(rows) > 1 else ''})"
                         + r"}}\\")
            for lab, w, x, y, r in rows:
                lines.append(
                    f"{lab} & {WLABEL[w]} & ${r['delta']:+.2f}$ & "
                    f"$[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}]$ & "
                    f"{r['wins']}/{r['npairs']} & {fmtp(r['p'])} & "
                    f"{fmtp(r['p_holm'])} \\\\")
        with open(os.path.join(TABS, fname), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def fam_macros(fam):
    tag = {"8": "Eight", "16u": "SixteenU", "16s": "SixteenS",
           "32u": "ThirtyTwo", "8mre": "Mre", "8mrex": "MreX"}
    short = {"approxlg": "Alg", "approxlg_sham": "Sham", "nsga2": "Nsga",
             "smsemoa": "Sms", "mobo": "Bo", "nsga2_rf": "Rf",
             "nsga2_expert": "Expert", "nsga2_adaptive": "Ucb",
             "nsga2_llmprior": "Elicited", "approxlg_no_warm": "NoWarm",
             "approxlg_no_inject": "NoInject",
             "approxlg_no_mutguide": "NoMut", "approxlg_no_prior": "NoPrior",
             "approxlg_no_refine": "NoRefine", "approxlg_sms": "AlgSms",
             "approxlg_sms_sham": "ShamSms", "approxlg_ext": "AlgExt",
             "approxlg_drift": "AlgDrift", "approxlg_eqt": "AlgEqT",
             "nsga2_ext": "NsgaExt", "smsemoa_ext": "SmsExt"}
    seen = {}
    for f, rows in fam.items():
        for lab, w, x, y, r in rows:
            k = f"{short[x]}V{short[y]}{tag[w]}"
            # two tests must never share a macro: the later would silently
            # overwrite the earlier in numbers_tcad.tex
            if k in seen:
                raise SystemExit(f"macro {k} used by both {seen[k]!r} and "
                                 f"{(f, lab)!r}")
            seen[k] = (f, lab)
            mac(k, f"{r['delta']:+.2f}")
            mac(k + "CI", f"[{r['ci'][0]:+.2f},\\,{r['ci'][1]:+.2f}]")
            mac(k + "P", mfmtp(r["p"]))
            mac(k + "Holm", mfmtp(r["p_holm"]))
            mac(k + "Wins", f"{r['wins']}/{r['npairs']}")
            mac(k + "A", f"{r['a12']:.2f}")
            # unsigned, one decimal: for running prose ("improves by 2.3%")
            mac(k + "Abs", f"{abs(r['delta']):.1f}")


def ablation_table(fam):
    rows = {(w, x): r for lab, w, x, y, r in fam.get("ablation", [])}
    lines = []
    for v in ABLATIONS:
        cells = []
        for w in ("8", "16u"):
            r = rows.get((w, v))
            cells.append("--" if r is None else
                         f"${r['delta']:+.2f}$ & {r['wins']}/{r['npairs']} & "
                         f"{fmtp(r['p_holm'])}")
        lines.append(f"{LABEL[v]} & " + " & ".join(cells) + r" \\")
    with open(os.path.join(TABS, "ablation_all.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Reference methods (OPACT, Esposito, Wang): iso-accuracy and hypervolume
# ---------------------------------------------------------------------------
def pooled_front(runs, w, algo, max_seeds):
    pts = []
    for s, r in sorted(runs[w].get(algo, {}).items()):
        if s >= max_seeds:
            continue
        for q in r["pareto"]:
            q = dict(q)
            q["ADP"] = q["area"] * q["delay"]
            q["PDP"] = q["power"] * q["delay"]
            pts.append(q)
    return pts


def iso(front, ref, key, metrics):
    cand = front
    for m in metrics:
        lim = ref[m]
        cand = [d for d in cand if d.get(m, float("inf")) <= lim * (1 + 1e-9)]
    return min(cand, key=key) if cand else None


def load_als(n, window=True):
    """Circuits of the ALS baseline (scripts/run_als.py), each netlist once."""
    p = os.path.join(RESULTS, f"als_n{n}.jsonl")
    if not os.path.exists(p):
        return []
    recs = [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in recs if r.get("ok") and not r.get("dup")
            and r.get("area") and (not window or 1e-6 <= r["NMED"] <= 0.02)]


def ref_compare(runs):
    out = {}
    lines = []
    last_group = None
    for n, w in ((8, "8"), (16, "16u"), (32, "32u")):
        rp = os.path.join(RESULTS, f"refgen_n{n}.jsonl")
        if not os.path.exists(rp):
            continue
        refs = [json.loads(l) for l in open(rp) if l.strip()]
        refs = [r for r in refs if r.get("ok", True) and r.get("area")
                and 1e-6 <= r["NMED"] <= 0.02]
        if not refs:
            continue
        # ALS is scored by the same rules but kept out of "all", which is the
        # reproduced compressor-tree families
        als = load_als(n)
        libs = ("OPACT", "Esposito", "Wang", "all") + (("ALSRAC",) if als else ())
        arms = [("approxlg", w), ("approxlg_sham", w), ("nsga2", w),
                ("smsemoa", w)]
        if n == 8:
            arms += [("approxlg", "8mre"), ("nsga2", "8mre"),
                     ("smsemoa", "8mre"), ("approxlg", "8mrex"),
                     ("nsga2", "8mrex"), ("smsemoa", "8mrex")]
        seeds = 20
        for algo, ww in arms:
            fr = pooled_front(runs, ww, algo,
                              seeds if not ww.startswith("8mre") else 12)
            if not fr:
                continue
            res = {}
            for metrics, tagm in ((("NMED",), "n"), (("NMED", "MRE"), "nm")):
                for lib in libs:
                    sub = als if lib == "ALSRAC" else \
                        [r for r in refs if lib == "all" or r["lib"] == lib]
                    wa = wp = 0
                    ga = []
                    for r in sub:
                        a = iso(fr, r, lambda d: d["ADP"], metrics)
                        p = iso(fr, r, lambda d: d["PDP"], metrics)
                        if a is not None and a["ADP"] < r["ADP"]:
                            wa += 1
                        if p is not None and p["PDP"] < r["PDP"]:
                            wp += 1
                        if a is not None:
                            ga.append(100 * (1 - a["ADP"] / r["ADP"]))
                    res[(tagm, lib)] = (wa, wp, len(sub),
                                        float(np.median(ga)) if ga else float("nan"))
            out[(n, algo, ww)] = res
            # first column names the group (width, or the MRE objective and
            # alphabet) once; arm names stay short to fit a column
            group = {"8mre": "MRE obj.", "8mrex": "MRE, +E"}.get(
                ww, WLABEL[w].replace(" u", ""))
            lab = {"approxlg_sham": "sham"}.get(algo, LABEL[algo])
            cells = []
            for tagm in ("n", "nm"):
                wa, wp, tot, med = res[(tagm, "all")]
                cells.append(f"{wa}/{tot} & {wp}/{tot} & ${med:+.0f}$")
            if lines and group != last_group:
                lines.append(r"\addlinespace[2pt]")
            lines.append(f"{group if group != last_group else ''} & {lab} & " +
                         " & ".join(cells) + r" \\")
            last_group = group
    with open(os.path.join(TABS, "refcmp.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


def als_macros(runs):
    """The ALS baseline: how many designs, how far ALSRAC got within its time
    cap, and the hypervolume of its designs in the box of ref_hv -- of all of
    them and of its widest-bound run alone, against the best single run of
    the search arms at the width."""
    from approxlg.metrics import hypervolume
    for n, w, wn, exactf in ((8, "8", "Eight", "exact_n8.json"),
                             (16, "16u", "Sixteen", "exact_n16u.json")):
        allr = load_als(n, window=False)
        if not allr:
            continue
        ex = json.load(open(os.path.join(RESULTS, exactf)))
        lo = np.array([math.log10(1e-6), 0, 0])
        hi = np.array([math.log10(0.02), ex["area"], ex["delay"]])

        def hv(pts):
            F = np.array([[math.log10(max(r["NMED"], 1e-7)), r["area"],
                           r["delay"]] for r in pts])
            return hypervolume((F - lo) / (hi - lo), np.ones(3))
        mac(f"AlsHV{wn}", f"{hv(allr):.4f}")
        # the run to the widest bound, counting the circuits it shares with
        # the other run (recorded there as duplicates)
        raw = [json.loads(l) for l in open(os.path.join(RESULTS, f"als_n{n}.jsonl"))
               if l.strip()]
        byh = {r["hash"]: r for r in allr}
        wide = max(r["bound"] for r in raw)
        one = [byh[r["hash"]] for r in raw
               if r.get("ok") and r["bound"] == wide and r.get("hash") in byh]
        if one:
            mac(f"AlsHVRun{wn}", f"{hv(one):.4f}")
        best = max((v for a in hv_by(runs, w).values() for v in a.values()),
                   default=None)
        if best is not None:
            mac(f"BestRunHV{wn}", f"{best:.4f}")
        # ALSRAC's sampled error estimate against our measurement
        est = [abs(r["NMED"] / r["als_err"] - 1) for r in allr
               if r.get("als_err") and r["NMED"] > 0]
        if est:
            mac(f"AlsEstDiff{wn}", f"{100 * np.median(est):.0f}")
        sp = os.path.join(RESULTS, f"als_runs_n{n}.json")
        if os.path.exists(sp):
            st = json.load(open(sp))
            mac(f"AlsWritten{wn}", f"{sum(s['written'] for s in st.values())}")
            for tag, b in zip(("Lo", "Hi"), sorted(st, key=float)):
                mac(f"AlsRounds{wn}{tag}", f"{st[b]['rounds']}")
                mac(f"AlsCpuMin{wn}{tag}", f"{st[b]['cpu_s'] / 60:.0f}")
                mac(f"AlsCpuH{wn}{tag}", f"{st[b]['cpu_s'] / 3600:.1f}")
        mac(f"AlsN{wn}", f"{len(allr)}")
        mac(f"AlsNWin{wn}", f"{len(load_als(n))}")
        top = max(r["NMED"] for r in allr)
        e = math.floor(math.log10(top)) if top > 0 else 0
        mac(f"AlsMaxNmed{wn}",
            f"{top / 10 ** e:.1f}\\times10^{{{e}}}" if top > 0 else "0")
        mac(f"AlsExactArea{wn}", f"{min(r['area'] for r in allr if r['NMED'] == 0):.0f}"
            if any(r["NMED"] == 0 for r in allr) else "--")
        print(f"als n={n}: {len(allr)} designs, {len(load_als(n))} in the window, "
              f"NMED up to {top:.2e}")


def ref_hv(runs):
    from approxlg.metrics import hypervolume
    res = {}
    for n, w, exactf in ((8, "8", "exact_n8.json"),
                         (16, "16u", "exact_n16u.json"),
                         (32, "32u", "exact_n32u.json")):
        rp = os.path.join(RESULTS, f"refgen_n{n}.jsonl")
        if not os.path.exists(rp):
            continue
        ex = json.load(open(os.path.join(RESULTS, exactf)))
        lo = np.array([math.log10(1e-6), 0, 0])
        hi = np.array([math.log10(0.02), ex["area"], ex["delay"]])
        refs = [json.loads(l) for l in open(rp) if l.strip()]
        refs = [r for r in refs if r.get("ok", True) and r.get("area")]

        def hv(pts):
            if not pts:
                return 0.0
            F = np.array([[math.log10(max(p["NMED"], 1e-7)), p["area"],
                           p["delay"]] for p in pts])
            return hypervolume((F - lo) / (hi - lo), np.ones(3))
        for lib in ("OPACT", "Esposito", "Wang"):
            res[(n, lib)] = hv([r for r in refs if r["lib"] == lib])
        res[(n, "all")] = hv(refs)
        res[(n, "count")] = len(refs)
        res[(n, "count_opact")] = sum(r["lib"] == "OPACT" for r in refs)
    return res


def wallclock(runs, fam):
    """Guidance time against the circuits it saves, per width.

    k* is the number of circuits a guided run needs to reach the mean final
    hypervolume of an unguided arm (censored at the budget).  Reaching it
    costs k* characterisations and the guidance spent by then, G k*/B; the
    unguided arm pays B characterisations.  Guidance is therefore faster to
    that target when one characterisation costs more than
        c* = G k* / (B (B - k*))
    seconds of wall clock, amortised over the concurrent synthesis jobs.
    With the serial cost c_s of a characterisation measured on an idle
    machine (results/time_serial.json), guidance pays for any number of
    concurrent jobs up to W* = c_s / c*.
    The equal-time comparison scores each guided run at the moment the
    unguided arm finishes its budget, t = B c, where it has bought only
    k = B c / (c + G/B) circuits."""
    tp = os.path.join(RESULTS, "time_serial.json")
    timing = json.load(open(tp)) if os.path.exists(tp) else {}
    budget = {"8": 400, "16u": 500, "16s": 500, "32u": 500}
    wkey = {"8": "8", "16u": "16", "16s": "16", "32u": "32"}
    tag = {"8": "Eight", "16u": "SixteenU", "16s": "SixteenS",
           "32u": "ThirtyTwo"}
    lines = []
    for w, B in budget.items():
        t = timing.get(wkey[w], {})
        # the mean, not the median: throughput is what a run pays, and at
        # 32x32 a few near-exact designs take ten times the typical one
        cs = t.get("mean")
        cp = t.get("parallel_s_per_circuit")
        if cs:
            mac(f"Serial{tag[w]}", f"{cs:.2f}")
            mac(f"SerialMedian{tag[w]}", f"{t['median']:.2f}")
            # (not "Speedup...": numbers.tex has those names for another
            # quantity, and \newcommand there wins over \providecommand here;
            # scripts/check_macro_clash.py lists such collisions)
            mac(f"ParSpeedup{tag[w]}", f"{cs / cp:.1f}" if cp else "--")
        if cp:
            mac(f"Par{tag[w]}", f"{cp:.2f}")
        g_runs = runs[w].get("approxlg", {})
        if not g_runs:
            continue
        G = float(np.mean([(r.get("t_llm") or (r.get("llm") or {}).get(
            "seconds", 0.0) or 0.0) for r in g_runs.values()]))
        mac(f"Guide{tag[w]}", f"{G:.0f}")
        for x, short in (("nsga2", "Nsga"), ("smsemoa", "Sms"),
                         ("mobo", "Bo"), ("approxlg_sham", "Sham")):
            u = runs[w].get(x)
            if not u:
                continue
            target = float(np.mean([r["hv"] for r in u.values()]))
            ks = [evals_to(r, target, B) for r in g_runs.values()]
            reach = sum(k < B for k in ks)
            k = float(np.median(ks))
            cstar = G * k / (B * (B - k)) if k < B else float("inf")
            mac(f"Kstar{tag[w]}{short}", f"{k:.0f}")
            mac(f"Reach{tag[w]}{short}", f"{reach}/{len(ks)}")
            mac(f"BreakEven{tag[w]}{short}",
                "\\infty" if cstar == float("inf") else f"{cstar:.2f}")
            wmax = (cs / cstar) if (cs and cstar < float("inf")) else None
            if wmax is not None:
                mac(f"Wmax{tag[w]}{short}", f"{wmax:.1f}")
            # the same crossover with the parallel efficiency we measured at
            # twelve jobs instead of ideal 1/W scaling from the serial cost
            if cp and cstar < float("inf"):
                mac(f"WmaxMeas{tag[w]}{short}", f"{12 * cp / cstar:.0f}")
            # equal wall clock at the unguided arm's finishing time
            eq = None
            if cp:
                kg = B * cp / (cp + G / B)
                gh = {s: next((h for e, h in reversed(r["hv_trace"])
                               if e <= kg), 0.0)
                      for s, r in g_runs.items()}
                uh = {s: r["hv"] for s, r in u.items()}
                eq = paired({"g": gh, "u": uh}, "g", "u")
                mac(f"EqT{tag[w]}{short}", f"{eq['delta']:+.2f}")
                mac(f"EqT{tag[w]}{short}P", mfmtp(eq["p"]))
                mac(f"EqT{tag[w]}{short}K", f"{kg:.0f}")
                # its own arm key: (approxlg, x) would reuse the macro names
                # of the primary and same-space tests and overwrite them
                fam["wallclock"].append((f"Approx-LG at equal time vs {LABEL[x]}",
                                         w, "approxlg_eqt", x, eq))
            lines.append((w, x, G, k, reach, len(ks), cstar, cs, cp, wmax, eq))
            print(f"wallclock {w:4s} vs {x:14s} G={G:5.0f}s k*={k:5.0f} "
                  f"reach {reach}/{len(ks)} c*={cstar:.3f}s serial={cs} "
                  f"par={cp} W*={wmax} eq={None if eq is None else round(eq['delta'], 2)}")
    if fam.get("wallclock"):
        adj = holm([r["p"] for *_, r in fam["wallclock"]])
        for (lab, w, x, y, r), pa in zip(fam["wallclock"], adj):
            r["p_holm"] = float(pa)
    # table: one row per width and unguided arm
    rows, last = [], None
    for w, x, G, k, reach, nk, cstar, cs, cp, wmax, eq in lines:
        cell_w = WLABEL[w] if w != last else ""
        cell_g = f"{G:.0f}" if w != last else ""
        last = w
        rows.append(
            f"{cell_w} & {cell_g} & "
            f"{ {'approxlg_sham': 'sham'}.get(x, LABEL[x])} & {k:.0f} & "
            f"{reach}/{nk} & "
            + ("$\\infty$" if cstar == float("inf") else f"{cstar:.2f}")
            + " & " + ("0" if cstar == float("inf") else
                       "--" if wmax is None else f"{wmax:.1f}") + " & "
            + ("--" if eq is None else
               f"${eq['delta']:+.1f}$ ({fmtp(eq['p_holm'])})") + r" \\")
    with open(os.path.join(TABS, "cost.tex"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return lines


def template_envelope():
    """Search limit or template limit?  The 8x8 references at matched MRE
    against every template design any campaign characterized, base alphabet
    and with style E, and what separates the designs that win."""
    import glob
    refs = [json.loads(l) for l in open(os.path.join(
        RESULTS, "refgen_n8.jsonl")) if l.strip()]
    refs = [r for r in refs if 1e-6 <= r["NMED"] <= 0.02]

    from approxlg.arch import Space
    sp = Space(8)

    def designs(keep, canon):
        # one record per circuit, as in scripts/analyze_mechanisms.py (the
        # base-alphabet count is the same set); extended-alphabet genomes are
        # keyed raw, since the default decoder does not know style E
        pts = {}
        for f in sorted(glob.glob(os.path.join(RESULTS, "designs_n8*.jsonl"))):
            if not keep(os.path.basename(f)):
                continue
            for line in open(f):
                if not line.strip():
                    continue
                d = json.loads(line)
                if not d.get("area") or d["area"] > 1e8:
                    continue
                k = (sp.canonical(np.array(d["g"], dtype=np.int32)).tobytes()
                     if canon else tuple(d["g"]))
                pts[k] = (d.get("MRE", 0.0), d["NMED"], d["area"],
                          d["delay"], d["power"])
        return np.array(list(pts.values()))

    for tag, keep, canon in (("", lambda f: "mrex" not in f, True),
                             ("X", lambda f: True, False)):
        P = designs(keep, canon)
        adp, pdp = P[:, 2] * P[:, 3], P[:, 4] * P[:, 3]
        wa = wp = 0
        ratio, dly, stages_lost = [], [], []
        for r in refs:
            m = P[:, 0] <= r["MRE"] * (1 + 1e-9)
            if not m.any():
                stages_lost.append(r.get("stages"))
                continue
            i = np.flatnonzero(m)[np.argmin(adp[m])]
            a_win = adp[i] < r["area"] * r["delay"]
            wa += a_win
            wp += pdp[m].min() < r["power"] * r["delay"]
            if P[i, 1] > 0:
                ratio.append(r["NMED"] / P[i, 1])
            dly.append(r["delay"] / P[i, 3])
            if not a_win:
                stages_lost.append(r.get("stages"))
        mac(f"Env{tag}N", f"{len(P):,}".replace(",", "\\,"))
        mac(f"Env{tag}ADP", f"{wa}")
        mac(f"Env{tag}PDP", f"{wp}")
        mac(f"Env{tag}Tot", f"{len(refs)}")
        mac(f"Env{tag}NmedRatio", f"{np.median(ratio):.1f}")
        mac(f"Env{tag}DelayRatio", f"{100 * (1 - np.median(dly)):.0f}")
        lost = [s for s in stages_lost if s is not None]
        mac(f"Env{tag}LostShallow",
            f"{sum(s <= 3 for s in lost)}")
        mac(f"Env{tag}Lost", f"{len(lost)}")
        print(f"envelope{tag or ' base'}: {len(P)} designs, matched MRE "
              f"ADP {wa}/{len(refs)} PDP {wp}/{len(refs)}; ref/template NMED "
              f"median x{np.median(ratio):.2f}; ref delay "
              f"{100 * (1 - np.median(dly)):.0f}% shorter; refs not beaten "
              f"in ADP with <=3 stages: {sum(s <= 3 for s in lost)}/{len(lost)}")
    st = [r.get("stages") for r in refs if r["lib"] == "OPACT"]
    mac("OpactStagesTwoThree", f"{sum(s in (2, 3) for s in st)}")
    mac("OpactCount", f"{len(st)}")


def ablation_macros(runs, fam):
    """Range of the single-operator effects per width, and the share of the
    evaluated circuits that the model proposed (run records: llm_points)."""
    for w, tag in (("8", "Eight"), ("16u", "SixteenU")):
        d = [r["delta"] for lab, ww, x, y, r in fam.get("ablation", [])
             if ww == w and x in ABLATIONS]
        if d:
            mac(f"AblMin{tag}", f"{min(d):+.1f}")
            mac(f"AblMax{tag}", f"{max(d):+.1f}")
        for a, k in (("approxlg", "Alg"), ("approxlg_no_inject", "NoInject"),
                     ("approxlg_no_mutguide", "NoMut")):
            rs = [r for r in runs[w].get(a, {}).values()
                  if r.get("llm_points") is not None and r.get("n_eval")]
            if rs:
                share = np.mean([r["llm_points"] / r["n_eval"] for r in rs])
                front = np.mean([r["llm_on_front"] / max(1, len(r["pareto"]))
                                 for r in rs])
                mac(f"LlmShare{k}{tag}", f"{100 * share:.0f}")
                mac(f"LlmFront{k}{tag}", f"{100 * front:.0f}")


def charcount_macros():
    """Distinct characterized designs (scripts/count_characterized.py)."""
    p = os.path.join(TABS, "charcount.json")
    if os.path.exists(p):
        c = json.load(open(p))
        mac("NcharDistinct", f"{c['total']:,}".replace(",", "\\,"))
        mac("NcharRecords", f"{c['records']:,}".replace(",", "\\,"))


def mech_macros():
    """Macros of the mechanism analysis (scripts/analyze_mechanisms.py)."""
    p = os.path.join(TABS, "mechanisms.json")
    if not os.path.exists(p):
        return
    mech = json.load(open(p))
    key = {"without truncation ($T=0$)": "Trunc",
           "without compensation ($c=0$)": "Comp",
           "without OR-based adder ($k=0$)": "Or",
           "without approximate cells": "Cells",
           "compressors only ($T=k=0$)": "CompOnly"}
    for w, tag in (("8", "Eight"), ("16u", "Sixteen")):
        if w not in mech:
            continue
        m = mech[w]
        mac(f"MechN{tag}", f"{m['n_designs']:,}".replace(",", "\\,"))
        full = m["restrict"]["full template"]["hv"]
        mac(f"MechHVFull{tag}", f"{full:.4f}")
        for name, k in key.items():
            hv = m["restrict"][name]["hv"]
            mac(f"MechHV{k}{tag}", f"{hv:.4f}")
            mac(f"MechLoss{k}{tag}", f"{100 * (1 - hv / full):.0f}")
        for u, k in (("T>0", "Trunc"), ("c>0", "Comp"), ("k>0", "Or"),
                     ("approx cells", "Cells")):
            mac(f"MechUse{k}{tag}", f"{100 * m['usage'][u]:.0f}")


def apps_mre_macros():
    """Macros of the matched-MRE application check (scripts/run_apps_mre.py):
    per MRE target, the cheapest design of each source in power x delay."""
    p = os.path.join(RESULTS, "apps_mre.json")
    if not os.path.exists(p):
        return
    rows = json.load(open(p))
    # accuracy of the exact network of the same run: apps_mre_exact.json, or
    # the run's log for the run made before that file existed
    ex = None
    pe = os.path.join(RESULTS, "apps_mre_exact.json")
    pl = os.path.join(RESULTS, "log_apps_mre.txt")
    if os.path.exists(pe):
        ex = json.load(open(pe)).get("cnn_acc")
    elif os.path.exists(pl):
        for line in open(pl):
            if line.startswith("exact top-1"):
                ex = float(line.split()[-1])
    if ex is not None:
        mac("AppsMreExactTop", f"{100 * ex:.2f}")
    acc = [100 * r["cnn_acc"] for r in rows if r.get("cnn_acc") is not None]
    if acc:
        mac("AppsMreTopLo", f"{min(acc):.1f}")
        mac("AppsMreTopHi", f"{max(acc):.1f}")
    targets = sorted({r["target"] for r in rows})
    save = []
    for t in targets:
        at = [r for r in rows if r["target"] == t]
        op = [r for r in at if r["source"] == "OPACT"]
        alg = [r for r in at if r["source"].startswith("Approx-LG")]
        if op and alg:
            best = min(alg, key=lambda r: r["PDP"])
            save.append(100 * (1 - op[0]["PDP"] / best["PDP"]))
    if save:
        mac("AppsMreOpactSaveLo", f"{min(save):.0f}")
        mac("AppsMreOpactSaveHi", f"{max(save):.0f}")
        mac("AppsMreTargets", f"{len(save)}")
    # the image workloads at MRE <= 3e-3: smoothing quality against NMED
    at = {r["source"]: r for r in rows if abs(r["target"] - 3e-3) < 1e-12}
    if "OPACT" in at and "Approx-LG (MRE objective)" in at:
        o, a = at["OPACT"], at["Approx-LG (MRE objective)"]
        mac("AppsMreSmoothGap", f"{a['smooth_psnr'] - o['smooth_psnr']:.1f}")
        mac("AppsMreNmedRatio", f"{o['NMED'] / a['NMED']:.1f}")
        mac("AppsMreOpactPdpThree", f"{o['PDP']:.1f}")
        mac("AppsMreAlgPdpThree", f"{a['PDP']:.1f}")
    short = {"Approx-LG (MRE objective, +E)": "Approx-LG, MRE obj., +E",
             "Approx-LG (MRE objective)": "Approx-LG, MRE obj.",
             "Approx-LG (NMED objective)": "Approx-LG, NMED obj.",
             "OPACT": "OPACT"}
    lines, last = [], None
    for r in rows:
        t = r["target"]
        exp = int(math.floor(math.log10(t)))
        man = t / 10 ** exp
        tt = (f"$10^{{{exp}}}$" if abs(man - 1) < 1e-9
              else f"${man:g}\\cdot10^{{{exp}}}$")
        def psnr(v):
            return "lossless" if v >= 99 else f"{v:.1f}"
        lines.append(f"{tt if t != last else ''} & {short[r['source']]} & "
                     f"{r['MRE']:.4f} & {r['NMED']:.2e} & {r['PDP']:.1f} & "
                     f"{psnr(r['blend_psnr'])} & {psnr(r['smooth_psnr'])} & "
                     + (f"{100 * r['cnn_acc']:.2f}" if r.get("cnn_acc")
                        is not None else "--") + r" \\")
        last = t
        print(f"apps_mre {r['target']:.0e} {r['source']:30s} PDP {r['PDP']:6.2f} "
              f"NMED {r['NMED']:.2e} smooth {r['smooth_psnr']:6.2f} "
              f"top-1 {r.get('cnn_acc')}")
    with open(os.path.join(TABS, "apps_mre.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def glitch_macros():
    """Macros of the unit-delay check (scripts/analyze_glitch.py)."""
    gp = os.path.join(TABS, "glitch.json")
    ip = os.path.join(TABS, "glitch_iso.json")
    if not (os.path.exists(gp) and os.path.exists(ip)):
        return
    rows = json.load(open(gp))
    tag = {"Approx-LG": "Alg", "AM-Lib": "Am", "EvoApproxLib": "Evo",
           "OPACT": "Opact", "MulApprox": "Mul"}
    for lib, t in tag.items():
        v = [r["glitch"] for r in rows if r["lib"] == lib]
        if v:
            mac(f"Glitch{t}", f"{np.median(v):.1f}")
            mac(f"Glitch{t}N", f"{len(v)}")
            mac(f"Glitch{t}Lo", f"{min(v):.1f}")
            mac(f"Glitch{t}Hi", f"{max(v):.1f}")
    iso = json.load(open(ip))
    libs_n = sum(1 for r in iso if r["lib"] in ("AM-Lib", "EvoApproxLib",
                                                 "MulApprox"))
    mac("GlitchLibN", f"{libs_n}")
    for match in ("N", "NM"):
        for grp, members in (("Lib", ("AM-Lib", "EvoApproxLib", "MulApprox")),
                             ("Opact", ("OPACT",))):
            sub = [r[match] for r in iso if r["lib"] in members and r[match]]
            if not sub:
                continue
            for model in ("zero", "unit"):
                wins = sum(s[f"win_{model}"] for s in sub)
                med = np.median([s[f"gain_{model}"] for s in sub])
                m = "Zero" if model == "zero" else "Unit"
                mac(f"GlitchIso{grp}{match}{m}Wins", f"{wins}")
                mac(f"GlitchIso{grp}{match}{m}Med", f"{med:.0f}")
            mac(f"GlitchIso{grp}{match}Tot", f"{len(sub)}")


# ---------------------------------------------------------------------------
def main():
    os.makedirs(TABS, exist_ok=True)
    glitch_macros()
    template_envelope()
    apps_mre_macros()
    mech_macros()
    charcount_macros()
    runs = load()
    search_table(runs)
    fam = families(runs)
    mac("NSpaceTests", f"{len(fam.get('space', []))}")
    wallclock(runs, fam)
    summary_table(fam)
    fam_macros(fam)
    ablation_table(fam)
    ablation_macros(runs, fam)
    cost = cost_table(runs)
    for w, a, sec, llm, calls, gens in cost:
        print(f"cost {w:4s} {a:14s} {sec:7.0f}s  llm {llm:6.0f}s  "
              f"calls {calls:5.1f}  gens {gens:5.1f}")

    rc = ref_compare(runs)
    rh = ref_hv(runs)
    als_macros(runs)
    wname = {8: "Eight", 16: "Sixteen", 32: "ThirtyTwo"}
    for (n, algo, ww), res in rc.items():
        for tagm in ("n", "nm"):
            for lib in ("OPACT", "Esposito", "Wang", "all", "ALSRAC"):
                if (tagm, lib) not in res:
                    continue
                wa, wp, tot, med = res[(tagm, lib)]
                key = (f"Ref{wname[n]}"
                       f"{ {'8mre': 'Mre', '8mrex': 'MreX'}.get(ww, '')}"
                       f"{ {'approxlg': 'Alg', 'approxlg_sham': 'Sham', 'nsga2': 'Nsga', 'smsemoa': 'Sms'}[algo]}"
                       f"{'NM' if tagm == 'nm' else 'N'}{lib}")
                mac(key + "ADP", f"{wa}")
                mac(key + "PDP", f"{wp}")
                mac(key + "Tot", f"{tot}")
                mac(key + "Med", f"{med:.0f}")
                print(f"ref n={n} {algo:14s} {ww:5s} {tagm:2s} {lib:9s} "
                      f"ADP {wa}/{tot} PDP {wp}/{tot} median dADP {med:.1f}%")
    for (n, lib), v in rh.items():
        if lib.startswith("count"):
            mac(f"RefCount{wname[n]}"
                f"{'Opact' if lib == 'count_opact' else ''}", f"{v}")
        else:
            mac(f"RefHV{wname[n]}{lib.capitalize()}", f"{v:.4f}")
    for f, rows in fam.items():
        for lab, w, x, y, r in rows:
            print(f"{f:9s} {w:5s} {lab:48s} n={r['n']:2d} {r['delta']:+6.2f}% "
                  f"CI[{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] wins {r['wins']}/"
                  f"{r['npairs']} p={r['p']:.4f} holm={r['p_holm']:.4f} "
                  f"A12={r['a12']:.2f}")

    with open(os.path.join(PAPER, "numbers_tcad.tex"), "w") as fh:
        fh.write("% auto-generated by scripts/analyze_tcad.py -- do not edit\n")
        for k in sorted(M):
            fh.write(f"\\providecommand{{\\{k}}}{{{M[k]}}}\n")
    dump = {f: [(lab, w, x, y, r) for lab, w, x, y, r in rows]
            for f, rows in fam.items()}
    with open(os.path.join(TABS, "tcad_stats.json"), "w") as fh:
        json.dump(dump, fh, indent=1, default=float)
    print(f"wrote {len(M)} macros")
    for k, (a, b) in sorted(REWRITTEN.items()):
        print(f"!! macro {k} written twice: {a!r} then {b!r}")


if __name__ == "__main__":
    main()
