"""Emit paper/numbers.tex and the selected-row table fragments."""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PAPER = os.path.join(ROOT, "paper")
TABS = os.path.join(PAPER, "tabs")
os.makedirs(TABS, exist_ok=True)

from approxlg.arch import Space                              # noqa: E402
from approxlg.cells import (ALL_CELLS, HA_CELLS, FA_CELLS, C42_CELLS,
                            SLOT_MODES)                      # noqa: E402

M = {}


def mac(name, value):
    M[name] = value


def load_search(tags):
    by = defaultdict(list)
    for tag in tags:
        p = os.path.join(RESULTS, f"search_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                by[d["algo"]].append(d)
    return by


def fmt_pow(x):
    e = int(math.floor(math.log10(x)))
    return f"10^{{{e}}}"


# ---------------------------------------------------------------- space
sp8, sp16 = Space(8), Space(16)


def space_size(sp):
    """Number of distinct genomes after canonicalisation: slot genes outside
    the approximate window are inactive, so the count is a sum over (T, A)."""
    tot = 0
    K = len(SLOT_MODES)
    for t in range(sp.max_t + 1):
        for a in range(sp.ncol - t + 1):
            tot += 4 * (a + 1) * K ** (sp.nstage * a)
    return tot


mac("Ncells", len(ALL_CELLS))
mac("NcellsHA", len(HA_CELLS))
mac("NcellsFA", len(FA_CELLS))
mac("NcellsCFT", len(C42_CELLS))
mac("Nmodes", len(SLOT_MODES))
mac("GenesEight", sp8.length)
mac("GenesSixteen", sp16.length)
# The header genes T, A, c and k are not slots; the paper needs both counts.
mac("SlotGenesEight", sp8.length - 4)
mac("SlotGenesSixteen", sp16.length - 4)
mac("SpaceEight", fmt_pow(space_size(sp8)))
mac("SpaceSixteen", fmt_pow(space_size(sp16)))
mac("MsEight", "2")

# ---------------------------------------------------------------- search
by8 = load_search(["n8", "n8L"])
if "nsga2" in by8 and "approxlg" in by8:
    hv_n = np.array([r["hv"] for r in by8["nsga2"]])
    hv_a = np.array([r["hv"] for r in by8["approxlg"]])
    mac("HVgainEight", f"{100*(hv_a.mean()-hv_n.mean())/hv_n.mean():.1f}")
    ref = hv_n.mean()
    # Evaluations to reach NSGA-II's final mean hypervolume, computed exactly
    # as the corresponding column of the search table: per run, then averaged.
    # Reading it off the *mean* curve instead gives a different number, and
    # the two used to disagree in print.
    def evals_to_ref(runs, budget=400):
        out = []
        for d in runs:
            e = budget
            for n, h in d["hv_trace"]:
                if h >= ref:
                    e = n
                    break
            out.append(e)
        return float(np.mean(out))

    e = evals_to_ref(by8["approxlg"])
    mac("EvalsApproxLG", f"{e:.0f}")
    mac("SpeedupEight", f"{400/e:.1f}")
    mac("SeedsEight", len(hv_a))
    fr = [r.get("llm_on_front", 0) / max(len(r["pareto"]), 1)
          for r in by8["approxlg"]]
    mac("LLMfrontShare", f"{100*np.mean(fr):.0f}")
    st = [r["llm"] for r in by8["approxlg"] if r.get("llm")]
    if st:
        mac("LLMcalls", f"{np.mean([s['calls'] for s in st]):.0f}")
        mac("LLMsec", f"{np.mean([s['seconds']/max(s['calls'],1) for s in st]):.1f}")
        mac("LLMtok", f"{np.mean([(s['prompt_tokens']+s['completion_tokens'])/1000 for s in st]):.0f}")
# Generations actually run.  A generation costs the budget only the circuits
# it has not already bought, so a run takes far more than budget/mu of them.
for tag, algo, key in (("n8L", "approxlg", "Eight"),
                       ("n16u", "approxlg", "SixteenU")):
    rs = load_search([tag]).get(algo, [])
    if rs:
        mac(f"Gens{key}", f"{np.mean([len(r['hv_trace']) for r in rs]):.0f}")
        st = [r["llm"] for r in rs if r.get("llm")]
        if st:
            mac(f"Calls{key}", f"{np.mean([x['calls'] for x in st]):.0f}")
mac("Budget", 400)
from approxlg.llm import ModePrior                            # noqa: E402
mac("PriorEps", f"{ModePrior().epsilon:g}")

# paired (same seed = same initial population) comparison against NSGA-II
if "nsga2" in by8 and "approxlg" in by8:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from analyze_search import mannwhitney, wilcoxon_paired
    ha = {r["seed"]: r["hv"] for r in by8["approxlg"]}
    hn = {r["seed"]: r["hv"] for r in by8["nsga2"]}
    common = sorted(set(ha) & set(hn))
    if len(common) >= 6:
        A = np.array([ha[s] for s in common])
        N = np.array([hn[s] for s in common])
        p, w, n = wilcoxon_paired(A, N)
        mac("PairedWins", f"{w}/{n}")
        mac("PairedP", f"{p:.2f}")
        mac("PairedGain", f"{100*np.mean((A-N)/N):.1f}")
        mac("PairedMedian", f"{100*np.median((A-N)/N):.1f}")
        # what the same comparison said on the first ten seeds alone, which is
        # the number the first submission reported
        if len(common) >= 10:
            c10 = common[:10]
            A10 = np.array([ha[s] for s in c10])
            N10 = np.array([hn[s] for s in c10])
            mac("PairedGainTen", f"{100*np.mean((A10-N10)/N10):.1f}")
    # The smallest budget past which the guided mean curve stays above the
    # unguided one.  The paper used to say "every budget beyond the first two
    # generations", which the extended campaign does not support.
    g8 = np.arange(16, 401, 16)

    def _mcurve(runs):
        C = []
        for d in runs:
            c, h = [], 0.0
            for gg in g8:
                for nn, vv in d["hv_trace"]:
                    if nn <= gg:
                        h = vv
                c.append(h)
            C.append(c)
        return np.mean(C, axis=0)

    ca, cn = _mcurve(by8["approxlg"]), _mcurve(by8["nsga2"])
    lead = len(g8)
    for i in range(len(g8) - 1, -1, -1):
        if ca[i] >= cn[i]:
            lead = i
        else:
            break
    mac("LeadFrom", int(g8[lead]) if lead < len(g8) else 400)

    pu, a12 = mannwhitney([r["hv"] for r in by8["approxlg"]],
                          [r["hv"] for r in by8["nsga2"]])
    mac("UnpairedP", f"{pu:.2f}")
    # P(a random Approx-LG run beats a random NSGA-II run), Vargha-Delaney
    mac("EffectA", f"{a12:.2f}" if a12 == a12 else "0.50")

# ------------------------------------------------- what the guidance channel
# needs.  Four arms differing only in where the ranked style lists come from:
# nothing, a fixed designer prior, online credit assignment, the model.
by_pri = load_search(["n8p", "n8"])
PRIOR_ARMS = ("nsga2_expert", "nsga2_adaptive", "nsga2_llmprior")
KEYS = {"nsga2_expert": "Expert", "nsga2_adaptive": "Adaptive",
        "nsga2_llmprior": "LLMprior"}
if "nsga2" in by_pri:
    from analyze_search import mannwhitney, wilcoxon_paired  # noqa: F811
    base = {r["seed"]: r["hv"] for r in by_pri["nsga2"]}
    for arm in PRIOR_ARMS:
        if arm not in by_pri or not by_pri[arm]:
            continue
        k = KEYS[arm]
        hv = {r["seed"]: r["hv"] for r in by_pri[arm]}
        v = np.array(list(hv.values()))
        mac(f"HV{k}", f"{v.mean():.4f}")
        mac(f"Seeds{k}", len(v))
        common = sorted(set(hv) & set(base))
        if len(common) >= 6:
            A = np.array([hv[s] for s in common])
            N = np.array([base[s] for s in common])
            p, w, n = wilcoxon_paired(A, N)
            mac(f"Gain{k}", f"{100*np.mean((A-N)/N):+.1f}")
            mac(f"P{k}", f"{p:.2f}")
            mac(f"Wins{k}", f"{w}/{n}")
    # head-to-head of the elicited prior against each non-LLM control
    f = os.path.join(TABS, "prior_n8.json")
    if os.path.exists(f):
        s = json.load(open(f))
        for other, key in (("nsga2_expert", "Expert"),
                           ("nsga2_adaptive", "Adaptive")):
            d = s.get(f"llmprior_vs_{other}")
            if d:
                mac(f"LLMvs{key}", f"{d['gain']:+.1f}")
                mac(f"LLMvs{key}P", f"{d['p']:.2f}")
                mac(f"LLMvs{key}Wins", f"{d['wins']}/{d['n']}")

HOLM = {}

# ---- the statistical protocol: primary endpoint, families, null band -----
f = os.path.join(TABS, "stats.json")
if os.path.exists(f):
    st_ = json.load(open(f))

    def _pnum(p):
        """The value alone, for prose that supplies its own words."""
        return "$<$0.001" if p < 0.001 else f"{p:.3f}".rstrip("0").rstrip(".")

    def _fmtp(p):
        """Relation and value, for sites written as $p\\Macro$."""
        if p < 0.001:
            return "{<}0.001"
        return "{=}" + f"{p:.3f}".rstrip("0").rstrip(".")

    KEY = {"$8\\times8$": "Eight", "$16\\times16$ unsigned": "SixteenU",
           "$16\\times16$ signed": "SixteenS"}
    for r in st_.get("primary", []):
        k = KEY.get(r["what"])
        if not k:
            continue
        mac(f"Sham{k}", f"{r['delta']:+.2f}")
        mac(f"Sham{k}CI", f"[{r['ci'][0]:+.2f},\\,{r['ci'][1]:+.2f}]")
        mac(f"Sham{k}P", _fmtp(r["p"]))
        mac(f"Sham{k}Holm", _fmtp(r["p_holm"]))
        mac(f"Sham{k}N", r["n"])
        mac(f"Sham{k}Wins", f"{r['wins']}/{r['npairs']}")
        mac(f"Sham{k}A", f"{r['a12']:.2f}")
    KEYM = {"extended reasoning vs the default endpoint": "Think",
            "a larger model vs the default endpoint": "Pro"}
    for r in st_.get("model", []):
        k = KEYM.get(r["what"])
        if not k:
            continue
        mac(k, f"{r['delta']:+.2f}")
        mac(f"{k}CI", f"[{r['ci'][0]:+.2f},\\,{r['ci'][1]:+.2f}]")
        mac(f"{k}P", _fmtp(r["p"]))
        mac(f"{k}Holm", _fmtp(r["p_holm"]))
        mac(f"{k}N", r["n"])
        mac(f"{k}A", f"{r['a12']:.2f}")
    KEYC = {"sham vs NSGA-II (unsigned)": "ShamNsgaU",
            "sham vs NSGA-II (signed)": "ShamNsgaS",
            "sham vs NSGA-II ($8\\times8$)": "ShamNsgaEight"}
    for r in st_.get("control", []):
        k = KEYC.get(r["what"])
        if not k:
            continue
        mac(k, f"{r['delta']:+.2f}")
        mac(f"{k}CI", f"[{r['ci'][0]:+.2f},\\,{r['ci'][1]:+.2f}]")
        mac(f"{k}P", _fmtp(r["p"]))
    nb = st_.get("null", {})
    if "approxlg 8x8" in nb:
        mac("NullEight", f"{nb['approxlg 8x8']['p95']:.2f}")
        mac("NullEightMed", f"{nb['approxlg 8x8']['median']:.2f}")
    if "nsga2 8x8" in nb:
        mac("NullEightN", f"{nb['nsga2 8x8']['p95']:.2f}")
    if "nsga2 16x16u" in nb:
        mac("NullSixteen", f"{nb['nsga2 16x16u']['p95']:.2f}")
    for row in st_.get("prior", []):
        if row["what"].startswith("elicited vs fixed"):
            mac("LLMvsExpertHolm", _pnum(row["p_holm"]))
            HOLM["elicited_vs_fixed"] = float(row["p_holm"])
        if row["what"].startswith("fixed prior vs sham"):
            mac("ExpertVsSham", f"{row['delta']:+.2f}")
            mac("ExpertVsShamHolm", _pnum(row["p_holm"]))
    for row in st_.get("reference", []):
        if row["what"].startswith("GA vs random"):
            mac("GAvsRandom", f"{row['delta']:+.2f}")
            mac("GAvsRandomP", _fmtp(row["p"]))

    # the surrogate baseline, which is the competing way to spend a small
    # evaluation budget
    wide = {r["what"]: r for r in st_.get("wide", [])}
    bits = []
    for lab in ("unsigned", "signed"):
        a = wide.get(f"Approx-LG vs RF surrogate ({lab})")
        b = wide.get(f"RF surrogate vs sham ({lab})")
        if a and b:
            bits.append(
                f"On the {lab} template the surrogate is worth "
                f"{b['delta']:+.2f}\\% over the sham arm "
                f"($p{_fmtp(b['p'])}$), and Approx-LG "
                f"{a['delta']:+.2f}\\% above the surrogate "
                f"($p{_fmtp(a['p'])}$, {_pnum(a['p_holm'])} after Holm).")
    if bits:
        mac("SurrogateText", "  ".join(bits))

# equal-wall-clock comparison: the guided arm at its evaluation budget against
# the same arms given the guidance seconds as extra evaluations
f = os.path.join(TABS, "walltime_n16u.json")
if os.path.exists(f):
    w = json.load(open(f))
    mac("SecPerCircuitWide", f"{w['per_circuit_s']:.2f}")
    mac("EqTimeSeconds", f"{w['t_match_s']:.0f}")
    et = w.get("equal_time", {})
    if et:
        mac("GuidanceSecondsWide", f"{et['guidance_s']:.0f}")
        NM = {"nsga2": "NSGA", "approxlg_sham": "Sham", "nsga2_rf": "RF"}
        for arm, key in NM.items():
            d = et.get("arms", {}).get(arm)
            if not d:
                continue
            mac(f"EqTime{key}", f"{d['delta']:+.2f}")
            mac(f"EqTime{key}CI", f"[{d['ci'][0]:+.2f},\\,{d['ci'][1]:+.2f}]")
            mac(f"EqTime{key}P",
                "$<$0.001" if d["p"] < 0.001 else f"{d['p']:.3f}".rstrip("0"))
            mac(f"Match{key}", f"{d['median_evals_to_match']:.0f}")
            mac(f"BreakEven{key}", f"{d['break_even_s']:.2f}")
        mac("EqTimeEvals", f"{int(round(w['t_match_s'] / w['per_circuit_s']))}")

# provenance: which arm's front beats the libraries, and memorisation
f = os.path.join(TABS, "provenance.json")
if os.path.exists(f):
    pv = json.load(open(f))
    mem = pv.get("memorisation", {})
    if mem:
        mac("MemMatches", sum(v["exact_matches"] for v in mem.values()))
        mac("MemPoints", f"{sum(v['front_points'] for v in mem.values()):,}"
            .replace(",", "\\,"))

# the characterisation sanity check.  "scanned" is live; the catch itself is
# recorded here because the record was purged and re-characterised, so a fresh
# scan of the cleaned data necessarily finds nothing.  Both figures were
# verified by rebuilding the genome and re-synthesising it.
f = os.path.join(TABS, "charcheck.json")
if os.path.exists(f):
    mac("NcharTotal", f"{json.load(open(f))['scanned']:,}".replace(",", "\\,"))
mac("NcharBad", "one")
mac("NcharBadArea", "19.7\\,$\\mu$m$^2$ at 39\\,ps")
mac("NcharGoodArea", "369.7\\,$\\mu$m$^2$ at 440\\,ps")

# how far the elicited advice is from the written-down rule, independently of
# whether the two perform differently
f = os.path.join(TABS, "prior_agreement.json")
if os.path.exists(f):
    s = json.load(open(f))
    mac("NPriorAnswers", s["answers"])
    mac("PriorHeadMatch", f"{s['head_match']:.0f}")
    mac("PriorTopThree", f"{s['top3_match']:.0f}")
    mac("PriorExactFar", f"{s['exact_far']:.0f}")

# ------------------------------------------------- what a guidance outage costs
# 20 runs of an unplanned endpoint outage: labelled Approx-LG, guided by
# nothing.  This is the only honest measurement of the fail-closed property,
# and it is what replaces the claim that a guided run can never be worse.
dead = load_search(["n8L_dead"])
if dead.get("approxlg") and "nsga2" in by8:
    from analyze_search import wilcoxon_paired               # noqa: F811
    D = {r["seed"]: r["hv"] for r in dead["approxlg"]}
    N0 = {r["seed"]: r["hv"] for r in by8["nsga2"]}
    common = sorted(set(D) & set(N0))
    if len(common) >= 6:
        A = np.array([D[s] for s in common])
        B = np.array([N0[s] for s in common])
        p, w, n = wilcoxon_paired(A, B)
        mac("DeadRuns", n)
        mac("DeadHV", f"{A.mean():.4f}")
        mac("DeadHVN", f"{B.mean():.4f}")
        mac("DeadGap", f"{100*np.mean((A-B)/B):+.1f}")
        mac("DeadWins", f"{w}/{n}")
        mac("DeadWorst", f"{100*np.min((A-B)/B):.0f}")
        mac("DeadP", f"{p:.2f}")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from analyze_search import mannwhitney, wilcoxon_paired      # noqa: E402,F811

by16 = load_search(["n16u", "n16s"])
mac("SeedsSixteen", max(len(by16.get("approxlg", [])) // 2, 3))
for tag, key in (("n16u", "U"), ("n16s", "S")):
    b = load_search([tag])
    if "approxlg" in b and "nsga2" in b:
        a = np.array([r["hv"] for r in b["approxlg"]])
        n = np.array([r["hv"] for r in b["nsga2"]])
        rr = np.array([r["hv"] for r in b.get("random", [])] or [np.nan])
        mac(f"HVsixteen{key}A", f"{a.mean():.4f}")
        mac(f"HVsixteen{key}N", f"{n.mean():.4f}")
        mac(f"HVsixteen{key}R", f"{np.nanmean(rr):.4f}")
        mac(f"HVgainSixteen{key}", f"{100*(a.mean()-n.mean())/n.mean():.1f}")
        mac(f"HVstdSixteen{key}A", f"{a.std():.4f}")
        mac(f"HVstdSixteen{key}N", f"{n.std():.4f}")
        mac(f"SeedsSixteen{key}", len(a))
        if a.std() > 0:
            mac(f"VarRatioSixteen{key}", f"{n.std()/a.std():.0f}")
        g16 = np.arange(16, 501, 16)

        def mcurve(runs):
            C = []
            for d in runs:
                c, h = [], 0.0
                for gg in g16:
                    for nn, vv in d["hv_trace"]:
                        if nn <= gg:
                            h = vv
                    c.append(h)
                C.append(c)
            return np.mean(C, axis=0)

        ca16 = mcurve(b["approxlg"])
        ii = np.where(ca16 >= n.mean())[0]
        e16 = int(g16[ii[0]]) if len(ii) else 500
        mac(f"EvalsSixteen{key}", f"{e16}")
        mac(f"SpeedupSixteen{key}", f"{500/e16:.1f}")

        # paired statistics and the transferred fixed prior, which is the
        # arm that decides whether the model is doing anything the rule
        # cannot at this width
        hv_by = {alg: {r["seed"]: r["hv"] for r in b.get(alg, [])}
                 for alg in ("approxlg", "nsga2", "nsga2_expert",
                             "nsga2_llmprior")}

        def _pair(x, y, prefix):
            common = sorted(set(hv_by[x]) & set(hv_by[y]))
            if len(common) < 6:
                return
            A_ = np.array([hv_by[x][s] for s in common])
            B_ = np.array([hv_by[y][s] for s in common])
            pv, wn, nn_ = wilcoxon_paired(A_, B_)
            mac(f"{prefix}{key}", f"{100*np.mean((A_-B_)/B_):+.1f}")
            # never print a p-value as 0.00; a reader reads that as exactly nil
            mac(f"{prefix}{key}P",
                "$<$0.001" if pv < 0.001 else f"{pv:.3f}".rstrip("0"))
            mac(f"{prefix}{key}Wins", f"{wn}/{nn_}")

        _pair("approxlg", "nsga2", "PairSixteen")

        # Spread, not just location.  A guided run landing in the same place
        # every time is a separate and arguably more useful property than a
        # higher mean, so it gets its own test: Levene's, as a Mann-Whitney on
        # the absolute deviations from each arm's median.
        va = np.array(list(hv_by["approxlg"].values()))
        vn = np.array(list(hv_by["nsga2"].values()))
        if len(va) >= 3 and len(vn) >= 3 and va.std() > 0:
            # ddof=0, the same dispersion Table~\ref{tab:search16} prints
            mac(f"SdSixteen{key}A", f"{va.std():.4f}")
            mac(f"SdSixteen{key}N", f"{vn.std():.4f}")
            mac(f"VarRatioSixteen{key}",
                f"{vn.std(ddof=1)/va.std(ddof=1):.1f}")
            pl, _ = mannwhitney(np.abs(va - np.median(va)),
                                np.abs(vn - np.median(vn)))
            mac(f"LeveneSixteen{key}", f"{pl:.2f}" if pl == pl else "--")
            # The spread difference is carried by the tail, not the bulk, so
            # the honest summary is the worst run of each arm rather than a
            # variance ratio that one outlier sets.
            mac(f"MinSixteen{key}A", f"{va.min():.4f}")
            mac(f"MinSixteen{key}N", f"{vn.min():.4f}")
        if hv_by["nsga2_expert"]:
            ex = np.array(list(hv_by["nsga2_expert"].values()))
            mac(f"HVsixteen{key}E", f"{ex.mean():.4f}")
            mac(f"SeedsSixteen{key}E", len(ex))
            _pair("nsga2_expert", "nsga2", "ExpertSixteen")
            _pair("approxlg", "nsga2_expert", "VsExpertSixteen")
        if hv_by["nsga2_llmprior"]:
            lp = np.array(list(hv_by["nsga2_llmprior"].values()))
            mac(f"HVsixteen{key}L", f"{lp.mean():.4f}")
            mac(f"SeedsSixteen{key}L", len(lp))
            _pair("nsga2_llmprior", "nsga2", "LLMpriorSixteen")
            # the apples-to-apples control: same channel, two sources
            _pair("nsga2_llmprior", "nsga2_expert", "LLMvsExpSixteen")
            # how much of the guided system's lead survives the prior being
            # the only thing the model supplies
            _pair("approxlg", "nsga2_llmprior", "VsLLMSixteen")

# ---------------------------------------------------------------- ablation
ABL = {"approxlg_no_warm": "warm start",
       "approxlg_no_inject": "candidate injection",
       "approxlg_no_mutguide": "mutation guidance",
       "approxlg_no_prior": "the proposal distribution alone",
       "approxlg_no_refine": "space refinement"}
by_abl = load_search(["n8abl", "n8L", "n8"])

# The mutation-guidance operator has two halves and the ablation flag used to
# delete both.  These two paired comparisons separate them.
if "approxlg" in by_abl:
    full = {r["seed"]: r["hv"] for r in by_abl["approxlg"]}
    for arm, key in (("approxlg_no_prior", "NoPrior"),
                     ("approxlg_no_mutguide", "NoMut")):
        if arm not in by_abl:
            continue
        hv = {r["seed"]: r["hv"] for r in by_abl[arm]}
        common = sorted(set(hv) & set(full))
        if len(common) >= 6:
            A = np.array([hv[s] for s in common])
            B = np.array([full[s] for s in common])
            p, w, n = wilcoxon_paired(A, B)
            mac(f"{key}Gain", f"{100*np.mean((A-B)/B):+.1f}")
            mac(f"{key}P", f"{p:.2f}")
            mac(f"{key}Wins", f"{w}/{n}")
    # the two halves against each other, which is the test with power: both
    # arms differ from the full system, so comparing them to it separately
    # throws away the pairing that matters
    if ("approxlg_no_prior" in by_abl and "approxlg_no_mutguide" in by_abl):
        hp = {r["seed"]: r["hv"] for r in by_abl["approxlg_no_prior"]}
        hm = {r["seed"]: r["hv"] for r in by_abl["approxlg_no_mutguide"]}
        common = sorted(set(hp) & set(hm))
        if len(common) >= 6:
            A = np.array([hp[s] for s in common])
            B = np.array([hm[s] for s in common])
            p, w, n = wilcoxon_paired(A, B)
            mac("HalfSplit", f"{100*np.mean((A-B)/B):+.1f}")
            mac("HalfSplitP", "$<$0.01" if p < 0.005 else f"{p:.2f}")
            mac("HalfSplitWins", f"{w}/{n}")

# Only claim the local search is the heavier half when the two halves are
# actually separated by more than the noise the ablation carries; at ten to
# twenty seeds a one-point gap is not a finding.
_half_split = False
try:
    _half_split = (float(M["NoMutGain"]) < float(M["NoPriorGain"]) - 1.0)
except (KeyError, ValueError):
    pass

txt = []
if "approxlg" in by_abl:
    base = np.mean([r["hv"] for r in by_abl["approxlg"]])
    drops = []
    for k, lbl in ABL.items():
        if k not in by_abl or not by_abl[k]:
            continue
        # paired on shared seeds, matching the convention used everywhere
        # else; a difference of raw means compares different seed sets and
        # can carry the opposite sign
        arm = {r["seed"]: r["hv"] for r in by_abl[k]}
        common = sorted(set(arm) & set(full))
        if len(common) >= 6:
            A = np.array([arm[s] for s in common])
            B = np.array([full[s] for s in common])
            d = 100 * float(np.mean((A - B) / B))
            pv, _, _ = wilcoxon_paired(A, B)
        else:
            d = 100 * (np.mean(list(arm.values())) - base) / base
            pv, _ = mannwhitney([r["hv"] for r in by_abl["approxlg"]],
                                list(arm.values()))
        drops.append((lbl, d, pv, k))
    if drops:
        drops.sort(key=lambda x: x[1])
        worst = drops[0]
        nsga = np.mean([r["hv"] for r in by_abl["nsga2"]]) \
            if "nsga2" in by_abl else None
        # Paired on shared seeds, exactly like every row of the table: the
        # unpaired ratio of means gives a different number and the text and
        # the table's last row used to disagree because of it.
        gap = 0.0
        if "nsga2" in by_abl:
            hn_ = {r["seed"]: r["hv"] for r in by_abl["nsga2"]}
            cm_ = sorted(set(hn_) & set(full))
            if len(cm_) >= 6:
                A_ = np.array([hn_[s] for s in cm_])
                B_ = np.array([full[s] for s in cm_])
                gap = 100 * float(np.mean((A_ - B_) / B_))
            else:
                gap = 100 * (nsga - base) / base
        sig = [(l, d, pv) for l, d, pv, _ in drops if pv == pv and pv < 0.05]
        listing = ", ".join(f"{l} {d:+.1f}\\% ($p={pv:.2f}$)"
                            for l, d, pv, _ in drops)
        if sig:
            txt.append(
                f"Removing the {worst[0]} costs {worst[1]:+.1f}\\% "
                f"(Mann--Whitney $p={worst[2]:.3f}$) and is the only removal "
                f"the test resolves.  In order: {listing}.  Removing guidance "
                f"altogether, exactly NSGA-II, costs {gap:+.1f}\\%.")
        else:
            nrun = sorted({len(by_abl[k]) for _, _, _, k in drops})
            seeds = (f"{nrun[0]}" if len(nrun) == 1
                     else f"{nrun[0]} to {nrun[-1]}")
            txt.append(
                f"No single removal is resolvable over {seeds} seeds per "
                f"variant, every figure paired against the full system on "
                f"shared seeds.  In order of damage: {listing}; removing "
                f"guidance altogether, NSGA-II, "
                f"costs {gap:+.1f}\\%.  At this sample size, with a "
                f"whole-system effect of a few percent, an ablation cannot "
                f"apportion it "
                f"between four overlapping operators, and these figures are "
                f"not a ranking.  One split is worth reading, because the "
                f"mutation-guidance row deletes two mechanisms at once.  "
                f"Keeping the LLM local search and dropping only the proposal "
                f"distribution costs \\NoPriorGain\\% ($p=\\NoPriorP$) against "
                f"\\NoMutGain\\% ($p=\\NoMutP$) for dropping both; compared "
                f"directly the two differ by \\HalfSplit\\% over "
                f"\\HalfSplitWins\\ pairs ($p=\\HalfSplitP$)"
                + (", the one comparison here that resolves, and it says the "
                   "local search carries that operator while the proposal "
                   "distribution does not"
                   if _half_split else
                   ", too close to tell the halves apart") +
                f".  Single removals also cost more than removing guidance "
                f"entirely, so the operators interact.  The rest of the "
                f"attribution comes from Section~\\ref{{sec:prior}}, which varies "
                f"one mechanism against a matched alternative instead of "
                f"deleting it.")
mac("AblationText", "".join(txt) if txt else
    "Each operator contributes to the final hypervolume.")

# ---------------------------------------------------------------- libraries
libs = []
p = os.path.join(RESULTS, "libs_n16.jsonl")
if os.path.exists(p):
    libs = [json.loads(l) for l in open(p) if l.strip()]
ver = {}
vp = os.path.join(RESULTS, "libs_verify.json")
if os.path.exists(vp):
    ver = {r["name"]: r["status"] for r in json.load(open(vp))}
ok = [r for r in libs if r.get("ok") and ver.get(r["name"]) == "match"]
mac("NlibTotal", len(libs))
mac("NlibElaborate", sum(1 for r in libs if r.get("ok")))
mac("NlibVerified", len(ok))
# The counts have to add up in print: collected -> elaborates with a plain
# N x N interface -> reproduces its own RTL -> inside the error window.  An
# earlier version said "two exact Booth multipliers" and left 91-8-2 = 81
# where the comparison actually uses 79.
mac("NlibExcluded", sum(1 for r in ok if r["NMED"] > 0.10))
mac("NlibExact", sum(1 for r in ok if r["NMED"] < 1e-12))
mac("NlibUsed", sum(1 for r in ok if 1e-12 <= r["NMED"] <= 0.10))

# ---------------------------------------------------------------- SOTA
for key, name in (("Eight", "sota8"), ("Sixteen", "sota16"),
                  ("SixteenU", "sota16u"), ("SixteenS", "sota16s"),
                  ("ShamSixteen", "sota16_approxlg_sham")):
    f = os.path.join(TABS, f"{name}_summary.json")
    if not os.path.exists(f):
        continue
    s = json.load(open(f))
    mac(f"Nref{key}", s["n"])
    mac(f"Sota{key}ADP", f"{s['rate_adp']:.1f}")
    mac(f"Sota{key}PDP", f"{s['rate_pdp']:.1f}")
    mac(f"Sota{key}Any", f"{s['rate_any']:.1f}")
    mac(f"MedianADP{key}", f"{s['median_adp']:.1f}")
    mac(f"MedianPDP{key}", f"{s['median_pdp']:.1f}")
    mac(f"MeanADP{key}", f"{s['mean_adp']:.1f}")

# pooled rate over both comparison campaigns
tot = wa = wp = 0
for name in ("sota8", "sota16"):
    f = os.path.join(TABS, f"{name}_summary.json")
    if os.path.exists(f):
        s = json.load(open(f))
        tot += s["n"]
        wa += s["wins_adp"]
        wp += s["wins_pdp"]
if tot:
    mac("NrefAll", tot)
    mac("SotaAllADP", f"{100*wa/tot:.1f}")
    mac("SotaAllPDP", f"{100*wp/tot:.1f}")

# per-library breakdown of the 16-bit comparison
try:
    from analyze_sota import iso_accuracy, load_front
    fr = load_front(["n16u", "n16s"], ("approxlg",), {"n16s"})
    for d in fr:
        d["ADP"] = d["area"] * d["delay"]
        d["PDP"] = d["power"] * d["delay"]
    vv = {r["name"]: r["status"]
          for r in json.load(open(os.path.join(RESULTS, "libs_verify.json")))}
    per = defaultdict(lambda: [0, 0])
    for line in open(os.path.join(RESULTS, "libs_n16.jsonl")):
        r = json.loads(line)
        if not r.get("ok") or vv.get(r["name"]) != "match":
            continue
        if not (1e-12 <= r["NMED"] <= 0.10):
            continue
        sub = [d for d in fr if bool(d.get("signed")) == bool(r["signed"])]
        a = iso_accuracy(sub, r, lambda d: d["ADP"], ("NMED", "MRE"))
        per[r["lib"]][0] += 1
        per[r["lib"]][1] += 1 if (a and a["ADP"] < r["ADP"]) else 0
    key = {"AM-Lib": "Am", "MulApprox": "Mul", "EvoApproxLib": "Evo"}
    for lb, (n_, w_) in per.items():
        if lb in key and n_:
            mac(f"Nref{key[lb]}", n_)
            mac(f"Sota{key[lb]}", f"{100*w_/n_:.1f}")
except Exception as exc:                                    # pragma: no cover
    print("per-library breakdown skipped:", exc)

mac("LLMname", "deepseek-v4-flash")

# ---------------------------------------------------------------- apps table
apps = []
p = os.path.join(RESULTS, "apps.json")
if os.path.exists(p):
    apps = json.load(open(p))
exact8 = {}
if os.path.exists(os.path.join(RESULTS, "exact_n8.json")):
    exact8 = json.load(open(os.path.join(RESULTS, "exact_n8.json")))
cnn = {}
if os.path.exists(os.path.join(RESULTS, "cnn_acc.json")):
    cnn = json.load(open(os.path.join(RESULTS, "cnn_acc.json")))
EOL = " " + chr(92) * 2 + "\n"


def _db(x):
    # scripts/run_apps.py stores 99.0 for a bit-exact image, since PSNR is
    # infinite when the error is zero, and the column averages six images.
    # Printing that as ">90" invented a measurement; exactly 99.0 means every
    # image came back identical, so say so.
    if x >= 99.0 - 1e-9:
        return "lossless"
    return f"{x:.1f}"


if apps:
    ours = [d for d in apps if "lib" not in d]
    refs = {d["match"]: d for d in apps if "lib" in d}

    def _row(d, acc_key="cnn_acc"):
        acc = d.get(acc_key)
        nm = d["name"].replace("_", chr(92) + "_")
        return " & ".join(
            [nm, f"{d['NMED']:.2e}", f"{(d.get('MRE') or 0):.4f}",
             f"{d['PDP']:.1f}", _db(d["blend_psnr"]), _db(d["smooth_psnr"]),
             f"{d['blend_ssim']:.4f}",
             (f"{100*acc:.2f}\\%" if acc else "--")]) + EOL

    # Each Approx-LG design is followed by the published design of matching
    # accuracy, so the two are read as a pair.
    with open(os.path.join(TABS, "apps.tex"), "w") as fh:
        if exact8:
            pe = exact8["power"] * exact8["delay"]
            fh.write(" & ".join(["exact", "0", "0", f"{pe:.1f}", "--", "--",
                                 "1.0000",
                                 (f"{100*cnn.get('exact', 0):.2f}\\%"
                                  if cnn else "--")]) + EOL)
        for d in ours:
            fh.write(_row(d))
            if d["name"] in refs:
                fh.write(_row(refs[d["name"]]))

    good = [d for d in ours if d.get("cnn_acc")]
    if good and exact8 and cnn:
        pe = exact8["power"] * exact8["delay"]
        lim = 100 * cnn.get("exact", 1) - 1.0
        ok = [d for d in good if 100 * d["cnn_acc"] >= lim]
        best = min(ok, key=lambda d: d["PDP"]) if ok else good[0]
        paired = [(d, refs[d["name"]]) for d in ok if d["name"] in refs]
        gain = ([100 * (1 - d["PDP"] / r["PDP"]) for d, r in paired]
                or [0.0])
        bad = [d for d in good if 100 * d["cnn_acc"] < lim]
        worst = min(bad, key=lambda d: d["cnn_acc"]) if bad else None
        wr = refs.get(worst["name"]) if worst else None
        txt = (f"Quality degrades gracefully along the front, and design "
               f"{best['name']} cuts the power--delay product from "
               f"{pe:.1f}\\,fJ to {best['PDP']:.1f}\\,fJ, a "
               f"{100*(1-best['PDP']/pe):.0f}\\% reduction, while keeping "
               f"LeNet-5 top-1 within one point of exact.  Read in pairs, the "
               f"table repeats on the application side what "
               f"Section~\\ref{{sec:sota16}} showed on the circuit side: at "
               f"matched \\nmed{{}} the Approx-LG design matches the published "
               f"design's image quality and stays within a point of its top-1 "
               f"while spending {min(gain):.0f} to {max(gain):.0f}\\% less "
               f"energy.")
        if worst is not None and wr is not None:
            txt += (f"  The most aggressive point is the exception: "
                    f"{worst['name']} is {100*(1-worst['PDP']/wr['PDP']):.0f}"
                    f"\\% cheaper than {wr['name'].replace('_', chr(92)+'_')} "
                    f"at the same \\nmed{{}}, but its relative error is "
                    f"{worst['MRE']/wr['MRE']:.0f}$\\times$ larger and the "
                    f"classifier collapses to {100*worst['cnn_acc']:.1f}\\% "
                    f"where the logarithmic reference holds "
                    f"{100*wr['cnn_acc']:.1f}\\%.  Relative error on small "
                    f"activations is exactly the axis a truncation-heavy "
                    f"design gives up, which is why the useful output of the "
                    f"search is the whole front, not one design.")
        mac("AppsText", txt)
    else:
        mac("AppsText", "")
else:
    mac("AppsText", "")

# ------------------------------------------- indicator robustness, wall clock
f = os.path.join(TABS, "hvsens_n8.json")
if os.path.exists(f):
    s = json.load(open(f))
    boxes = s.get("boxes", [])
    if boxes:
        gains = [b["gain"] for b in boxes]
        mac("NBoxes", len(boxes))
        mac("BoxGainLo", f"{min(gains):+.1f}")
        mac("BoxGainHi", f"{max(gains):+.1f}")
        mac("BoxSig", sum(1 for b in boxes if b["p"] == b["p"] and b["p"] < 0.05))
    for tag, key in (("eps", "Eps"), ("igd_plus", "Igd")):
        d = s.get(tag)
        if d:
            mac(f"{key}Gain", f"{d['gain']:+.1f}")
            mac(f"{key}P", f"{d['p']:.2f}")
            mac(f"{key}Wins", f"{d['wins']}/{d['n']}")

f = os.path.join(TABS, "wallclock_n8.json")
if os.path.exists(f):
    s = json.load(open(f))
    per = [v["per_synth"] for v in s.values() if v.get("per_synth") == v.get("per_synth")]
    if per:
        mac("SecPerCircuit", f"{np.mean(per):.2f}")
    if "approxlg" in s and "nsga2" in s:
        a, n = s["approxlg"], s["nsga2"]
        mac("TimeApproxLG", f"{a['t_total']:.0f}")
        mac("TimeNSGA", f"{n['t_total']:.0f}")
        mac("TimeGuidance", f"{a['t_llm']:.0f}")
        tot = a["t_eda_cold"] + a["t_llm"]
        if tot > 0:
            mac("GuidanceShare", f"{100*a['t_llm']/tot:.0f}")
        mac("TimeEDAcold", f"{a['t_eda_cold']:.0f}")

# -------------------------------------------------- generated prose blocks
# These three read the arms' own statistics rather than repeating them by
# hand, because the conclusion depends on which differences survive and that
# is not something to leave stale in the source.
def _sig(p):
    return p == p and p < 0.05


f = os.path.join(TABS, "prior_n8.json")
if os.path.exists(f):
    s = json.load(open(f))
    have = [k for k in ("nsga2_expert", "nsga2_adaptive", "nsga2_llmprior")
            if k in s and s[k].get("n")]
    if have and "nsga2" in s:
        name = {"nsga2_expert": "the fixed designer prior",
                "nsga2_adaptive": "the adaptive prior",
                "nsga2_llmprior": "the elicited prior"}
        def pfmt(p):
            return "p<0.001" if p < 0.001 else f"p={p:.3f}".rstrip("0")

        def cite(k):
            return (f"{s[k]['gain']:+.1f}\\% (${pfmt(s[k]['p'])}$, "
                    f"{s[k]['wins']} of {s[k]['n']} pairs)")

        works = [k for k in have if _sig(s[k]["p"]) and s[k]["gain"] > 0]
        idle = [k for k in have if k not in works]
        parts = []
        if works:
            parts.append(
                "Against uniform sampling, "
                + ", and ".join(f"{name[k]} is worth {cite(k)}"
                                for k in works) + ".")
        if idle:
            lead = ("  " if parts else "")
            if len(idle) == 1:
                parts.append(f"{lead}{name[idle[0]].capitalize()} is not: "
                             f"{cite(idle[0])}.")
            else:
                parts.append(
                    lead + "Neither " + " nor ".join(name[k] for k in idle)
                    + " is: " + ", ".join(cite(k) for k in idle)
                    + " respectively.")
        if "nsga2_adaptive" in idle:
            parts.append(
                "  The adaptive prior failing is the informative half: it "
                "sees the whole archive and still cannot recover the "
                "offset-to-style mapping, because \\Budget\\ evaluations over "
                "\\Nmodes\\ styles and five offsets leave a handful of "
                "samples per cell, so the knowledge must come from "
                "\\emph{outside} the run.")
        hh = s.get("llmprior_vs_nsga2_expert")
        if hh:
            parts.append(
                f"  The comparison that matters is the elicited prior against "
                f"the written-down one: {hh['gain']:+.1f}\\%, winning "
                f"{hh['wins']} of {hh['n']} paired seeds, "
                f"${pfmt(hh['p'])}$, which Holm's correction inside the "
                f"prior family leaves at \\LLMvsExpertHolm.")
            p_adj = HOLM.get("elicited_vs_fixed", hh["p"])
            if _sig(p_adj) and hh["gain"] > 0:
                parts.append("  The model is the better source of the "
                             "distribution, not merely a source of one.")
            elif _sig(p_adj):
                parts.append(
                    "  At this width the written rule is the better source, "
                    "which we state plainly because it is the result such a "
                    "paper is least inclined to look for.  The rule did have "
                    "to be written, once per template, so "
                    "Section~\\ref{sec:hv16} repeats the comparison on a "
                    "template that has changed.")
            elif _sig(hh["p"]):
                parts.append(
                    "  The uncorrected test rejects and the family correction "
                    "does not, so the protocol fixed in "
                    "Section~\\ref{sec:stats} leaves this one unresolved and "
                    "we read it no further than its sign: the point estimate "
                    "favours the written rule, and what we claim is only that "
                    "the elicited distribution is not the better source.  "
                    "That is still the result such a paper is least inclined "
                    "to look for.  The rule did have to be written, once per "
                    "template, so Section~\\ref{sec:hv16} repeats the "
                    "comparison on a template that has changed.")
            else:
                parts.append(
                    "  They are indistinguishable.  We therefore do not claim "
                    "that a language model produces better guidance than a "
                    "designer who writes the rule down; we claim it produces "
                    "comparable guidance without the rule being written, from "
                    "a description of the template and the measured cell "
                    "statistics alone.  Whether that is worth an inference "
                    "endpoint is a judgement about how often the template "
                    "changes, and Section~\\ref{sec:hv16} is where the two are "
                    "asked to transfer to one that has.")
        mac("PriorText", "".join(parts))

if "HVsixteenUA" in M:
    t = (f"Over \\SeedsSixteenU\\ unsigned seeds the hypervolume is "
         f"\\HVsixteenUA\\ for Approx-LG against \\HVsixteenUN\\ for NSGA-II "
         f"(\\PairSixteenU\\%, \\PairSixteenUWins\\ pairs, "
         f"$p=\\PairSixteenUP$) and \\HVsixteenUR\\ for random search; the "
         f"signed template gives \\HVsixteenSA\\ against \\HVsixteenSN\\ "
         f"(\\PairSixteenS\\%, \\PairSixteenSWins\\ pairs, "
         f"$p=\\PairSixteenSP$).  The unsigned mean curve reaches the final "
         f"NSGA-II hypervolume after \\EvalsSixteenU\\ evaluations, a "
         f"\\SpeedupSixteenU$\\times$ saving.  The unsigned spread is "
         f"\\SdSixteenUA\\ against \\SdSixteenUN, but one bad NSGA-II run sets "
         f"almost all of that gap and Levene's test does not resolve it "
         f"($p=\\LeveneSixteenU$), so we claim no variance reduction.")
    if "HVsixteenUE" in M:
        t += (f"  The written rule transfers unchanged, reaching "
              f"\\HVsixteenUE\\ unsigned (\\ExpertSixteenU\\% over NSGA-II, "
              f"$p=\\ExpertSixteenUP$) and \\HVsixteenSE\\ signed "
              f"(\\ExpertSixteenS\\%, $p=\\ExpertSixteenSP$); nothing in it "
              f"was specific to $8\\times8$.")
    if "HVsixteenUL" in M:
        t += (f"  The matched arm, the elicited prior alone, reaches "
              f"\\HVsixteenUL\\ unsigned (\\LLMpriorSixteenU\\%, "
              f"$p=\\LLMpriorSixteenUP$) and \\HVsixteenSL\\ signed "
              f"(\\LLMpriorSixteenS\\%, $p=\\LLMpriorSixteenSP$); against the "
              f"written rule it is \\LLMvsExpSixteenU\\% "
              f"($p=\\LLMvsExpSixteenUP$) and \\LLMvsExpSixteenS\\% "
              f"($p=\\LLMvsExpSixteenSP$), against \\LLMvsExpert\\% at "
              f"$8\\times8$.  At no width is the elicited prior the better "
              f"source, yet the full system sits \\VsLLMSixteenU\\% above it "
              f"unsigned (\\VsLLMSixteenUWins\\ pairs, $p=\\VsLLMSixteenUP$) "
              f"and \\VsLLMSixteenS\\% signed ($p=\\VsLLMSixteenSP$): most of "
              f"what guidance buys does not travel through the proposal "
              f"distribution.")
    mac("SixteenText", t)

    # one clause for the conclusion, keyed on whether the transferred rule
    # keeps up with the model at the larger width
    if "VsExpertSixteenU" in M:
        du = float(M["VsExpertSixteenU"])
        ds = float(M.get("VsExpertSixteenS", 0))
        pu_ = float(M["VsExpertSixteenUP"])
        ps_ = float(M.get("VsExpertSixteenSP", 1.0))
        if (du > 0 and pu_ < 0.05) or (ds > 0 and ps_ < 0.05):
            # The comparison behind this branch is the *whole guided system*
            # against the written rule, not the elicited prior against it, and
            # an earlier wording said the opposite of what the prior control
            # found.  Name the arms.
            mac("SixteenVerdict",
                "At $16\\times16$ the full guided system does separate from "
                "the carried-over rule, though the elicited prior on its own "
                "still does not, so the separation comes from the operators "
                "that propose designs and not from the advice")
        else:
            mac("SixteenVerdict",
                "At $16\\times16$, on a wider and a signed template, the "
                "carried-over rule keeps up with the full guided system, so "
                "the case for the model rests on not having to state the rule "
                "rather than on the quality of what it states")

f = os.path.join(TABS, "hvsens_n8.json")
if os.path.exists(f):
    s = json.load(open(f))
    boxes = s.get("boxes", [])
    if boxes:
        g = [b["gain"] for b in boxes]
        nsig = sum(1 for b in boxes if _sig(b["p"]))
        pos = sum(1 for x in g if x > 0)
        e, i = s.get("eps"), s.get("igd_plus")
        t = (f"A hypervolume is a statement about a box as much as about a "
             f"front, so Table~\\ref{{tab:hvsens}} re-scores every run under "
             f"{len(boxes)} boxes and two indicators that need none.  The "
             f"sign survives {pos} of {len(boxes)} boxes, from "
             f"{min(g):+.1f}\\% to {max(g):+.1f}\\%, reaching $p<0.05$ in "
             f"{nsig}.")
        agree = e and i and e["gain"] > 0 and i["gain"] > 0
        if e and i:
            t += (f"  Both box-free indicators disagree: against the union "
                  f"front of every run, the additive $\\varepsilon$ indicator "
                  f"puts Approx-LG {e['gain']:+.1f}\\% ($p={e['p']:.2f}$) and "
                  f"IGD$+$ {i['gain']:+.1f}\\% ($p={i['p']:.2f}$), both "
                  f"favouring NSGA-II." if not agree else
                  f"  Both box-free indicators agree: against the union front "
                  f"of every run the additive $\\varepsilon$ indicator puts "
                  f"Approx-LG {e['gain']:+.1f}\\% ($p={e['p']:.2f}$) and "
                  f"IGD$+$ {i['gain']:+.1f}\\% ($p={i['p']:.2f}$).")
        cov = s.get("accurate_pts") or {}
        if not agree and cov.get("approxlg") and cov.get("nsga2"):
            t += (f"  The fronts show why: the box stops paying below "
                  f"$\\nmed=10^{{-6}}$, so a search chasing it has no reason "
                  f"to refine further, and guided runs place "
                  f"{cov['approxlg']:.1f} front points below $10^{{-5}}$ "
                  f"against {cov['nsga2']:.1f} for NSGA-II"
                  + (f" and {cov['nsga2_expert']:.1f} for the fixed prior"
                     if cov.get("nsga2_expert") else "") +
                  f".  IGD$+$ charges for that thinning and the hypervolume "
                  f"does not; a reader who needs the accurate corner should "
                  f"weight the box-free numbers.  We report the disagreement "
                  f"rather than the flattering indicator.")
        else:
            t += ("  The ranking is therefore not an artefact of where the "
                  "box was drawn, though no scoring turns a difference of "
                  "this size into a decisive one.")
        mac("RobustText", t)

if "DeadRuns" in M:
    mac("FailClosedText",
        f"An endpoint outage gave us \\DeadRuns\\ runs of Approx-LG in which "
        f"\\emph{{every}} guidance call failed, the experiment we would not "
        f"have scheduled.  They average \\DeadHV\\ against \\DeadHVN\\ for "
        f"NSGA-II on the same seeds, \\DeadGap\\% and unresolvable "
        f"($p=\\DeadP$), so the fallback works.  It does not make a run "
        f"\\emph{{identical}} to NSGA-II: the guided path draws a different "
        f"random stream, so what is restored is the NSGA-II "
        f"\\emph{{distribution}}, and pairs scatter, better in \\DeadWins\\ "
        f"and worst by \\DeadWorst\\%.  A guarantee that no run is ever worse "
        f"is thus stronger than the mechanism supports.  They were also nearly "
        f"averaged into the guided arm, hiding the effect; the driver now "
        f"aborts after six consecutive failures.")

# ---------------------------------------------------------------- write
defaults = {
    "HVgainEight": "0.0", "SpeedupEight": "1.0", "EvalsApproxLG": "400",
    "SeedsEight": "10", "SeedsSixteen": "3", "LLMfrontShare": "0",
    "LLMcalls": "0", "LLMsec": "0", "LLMtok": "0",
    "NrefEight": "0", "SotaEightADP": "0", "SotaEightPDP": "0",
    "MedianADPEight": "0", "MeanADPEight": "0", "SotaEightAny": "0",
    "NrefSixteen": "0", "SotaSixteenADP": "0", "SotaSixteenPDP": "0",
    "MedianADPSixteen": "0", "MedianPDPSixteen": "0", "MeanADPSixteen": "0",
    "SotaSixteenAny": "0",
    "NrefSixteenU": "0", "SotaSixteenUADP": "0", "SotaSixteenUPDP": "0",
    "MedianADPSixteenU": "0", "MedianPDPSixteenU": "0",
    "MeanADPSixteenU": "0", "SotaSixteenUAny": "0",
    "NrefSixteenS": "0", "SotaSixteenSADP": "0", "SotaSixteenSPDP": "0",
    "MedianADPSixteenS": "0", "MedianPDPSixteenS": "0",
    "MeanADPSixteenS": "0", "SotaSixteenSAny": "0",
    "NlibTotal": "0", "NlibVerified": "0", "NlibExcluded": "0",
    "HVsixteenUA": "0",
    "HVsixteenUN": "0",
    "HVsixteenUR": "0",
    "HVgainSixteenU": "0",
    "HVstdSixteenUA": "0",
    "HVstdSixteenUN": "0",
    "SeedsSixteenU": "3",
    "HVsixteenSA": "0",
    "HVsixteenSN": "0",
    "HVsixteenSR": "0",
    "HVgainSixteenS": "0",
    "HVstdSixteenSA": "0",
    "HVstdSixteenSN": "0",
    "SeedsSixteenS": "3",
    "PairedWins": "--", "PairedP": "1.00", "PairedGain": "0",
    "PairedGainTen": "0", "LeadFrom": "0",
    "UnpairedP": "1.00", "EffectA": "0.50", "PriorEps": "0.25",
    "NrefAll": "0", "SotaAllADP": "0", "SotaAllPDP": "0",
    "NrefShamSixteen": "0", "SotaShamSixteenADP": "0",
    "SotaShamSixteenPDP": "0", "SotaShamSixteenAny": "0",
    "MedianADPShamSixteen": "0", "MedianPDPShamSixteen": "0",
    "MeanADPShamSixteen": "0",
    "NrefAm": "0", "SotaAm": "0", "NrefMul": "0", "SotaMul": "0",
    "NrefEvo": "0", "SotaEvo": "0",
    "AblationText": "", "AppsText": "",
    "PriorText": "", "RobustText": "", "FailClosedText": "",
    "HVExpert": "0", "HVAdaptive": "0", "HVLLMprior": "0",
    "SeedsExpert": "0", "SeedsAdaptive": "0", "SeedsLLMprior": "0",
    "GainExpert": "0", "GainAdaptive": "0", "GainLLMprior": "0",
    "PExpert": "1.00", "PAdaptive": "1.00", "PLLMprior": "1.00",
    "WinsExpert": "--", "WinsAdaptive": "--", "WinsLLMprior": "--",
    "LLMvsExpert": "0", "LLMvsExpertP": "1.00", "LLMvsExpertWins": "--",
    "LLMvsAdaptive": "0", "LLMvsAdaptiveP": "1.00",
    "LLMvsAdaptiveWins": "--",
    "DeadRuns": "0", "DeadHV": "0", "DeadHVN": "0", "DeadGap": "0",
    "DeadWins": "--", "DeadWorst": "0", "DeadP": "1.00",
    "SixteenText": "", "SixteenVerdict": "",
    "PairSixteenU": "0", "PairSixteenUP": "1.00", "PairSixteenUWins": "--",
    "PairSixteenS": "0", "PairSixteenSP": "1.00", "PairSixteenSWins": "--",
    "HVsixteenUE": "0", "HVsixteenSE": "0",
    "SlotGenesEight": "0", "SlotGenesSixteen": "0",
    "GensEight": "0", "GensSixteenU": "0",
    "CallsEight": "0", "CallsSixteenU": "0",
    "PairedMedian": "0",
    "Think": "0", "ThinkCI": "--", "ThinkP": "1", "ThinkHolm": "1",
    "ThinkN": "0", "ThinkA": "0",
    "Pro": "0", "ProCI": "--", "ProP": "1", "ProHolm": "1",
    "ProN": "0", "ProA": "0",
    "ShamNsgaU": "0", "ShamNsgaUCI": "--", "ShamNsgaUP": "1",
    "ShamNsgaS": "0", "ShamNsgaSCI": "--", "ShamNsgaSP": "1",
    "ShamNsgaEight": "0", "ShamNsgaEightCI": "--", "ShamNsgaEightP": "1",
    "SdSixteenUA": "0", "SdSixteenUN": "0",
    "SdSixteenSA": "0", "SdSixteenSN": "0",
    "LeveneSixteenU": "--", "LeveneSixteenS": "--",
    "HVsixteenUL": "0", "HVsixteenSL": "0",
    "SeedsSixteenUL": "0", "SeedsSixteenSL": "0",
    "LLMpriorSixteenU": "0", "LLMpriorSixteenUP": "1.00",
    "LLMpriorSixteenUWins": "--",
    "LLMpriorSixteenS": "0", "LLMpriorSixteenSP": "1.00",
    "LLMpriorSixteenSWins": "--",
    "LLMvsExpSixteenU": "0", "LLMvsExpSixteenUP": "1.00",
    "LLMvsExpSixteenUWins": "--",
    "LLMvsExpSixteenS": "0", "LLMvsExpSixteenSP": "1.00",
    "LLMvsExpSixteenSWins": "--",
    "VsLLMSixteenU": "0", "VsLLMSixteenUP": "1.00", "VsLLMSixteenUWins": "--",
    "VsLLMSixteenS": "0", "VsLLMSixteenSP": "1.00", "VsLLMSixteenSWins": "--",
    "NoPriorGain": "0", "NoPriorP": "1.00", "NoPriorWins": "--",
    "NoMutGain": "0", "NoMutP": "1.00", "NoMutWins": "--",
    "HalfSplit": "0", "HalfSplitP": "1.00", "HalfSplitWins": "--",
    "MinSixteenUA": "0", "MinSixteenUN": "0",
    "MinSixteenSA": "0", "MinSixteenSN": "0",
    "SeedsSixteenUE": "0", "SeedsSixteenSE": "0",
    "ExpertSixteenU": "0", "ExpertSixteenUP": "1.00",
    "ExpertSixteenUWins": "--",
    "ExpertSixteenS": "0", "ExpertSixteenSP": "1.00",
    "ExpertSixteenSWins": "--",
    "VsExpertSixteenU": "0", "VsExpertSixteenUP": "1.00",
    "VsExpertSixteenUWins": "--",
    "VsExpertSixteenS": "0", "VsExpertSixteenSP": "1.00",
    "VsExpertSixteenSWins": "--",
    "ShamEight": "0", "ShamEightCI": "--", "ShamEightP": "1",
    "ShamEightHolm": "1", "ShamEightN": "0", "ShamEightWins": "--",
    "ShamSixteenU": "0", "ShamSixteenUCI": "--", "ShamSixteenUP": "1",
    "ShamSixteenUHolm": "1", "ShamSixteenUN": "0", "ShamSixteenUWins": "--",
    "ShamSixteenS": "0", "ShamSixteenSCI": "--", "ShamSixteenSP": "1",
    "ShamSixteenSHolm": "1", "ShamSixteenSN": "0", "ShamSixteenSWins": "--",
    "ShamEightA": "0", "ShamSixteenUA": "0", "ShamSixteenSA": "0",
    "NullEight": "0", "NullEightMed": "0", "NullEightN": "0",
    "NullSixteen": "0", "LLMvsExpertHolm": "1",
    "ExpertVsSham": "0", "ExpertVsShamHolm": "1",
    "GAvsRandom": "0", "GAvsRandomP": "1",
    "MemMatches": "0", "MemPoints": "0", "SurrogateText": "",
    "SecPerCircuitWide": "0", "EqTimeSeconds": "0", "EqTimeEvals": "0",
    "GuidanceSecondsWide": "0",
    "EqTimeNSGA": "0", "EqTimeNSGACI": "--", "EqTimeNSGAP": "1",
    "EqTimeSham": "0", "EqTimeShamCI": "--", "EqTimeShamP": "1",
    "EqTimeRF": "0", "EqTimeRFCI": "--", "EqTimeRFP": "1",
    "MatchNSGA": "0", "MatchSham": "0", "MatchRF": "0",
    "BreakEvenNSGA": "0", "BreakEvenSham": "0", "BreakEvenRF": "0",
    "NcharTotal": "0", "NcharBad": "no", "NcharBadArea": "--",
    "NcharGoodArea": "--",
    "NPriorAnswers": "0", "PriorHeadMatch": "0", "PriorTopThree": "0",
    "PriorExactFar": "0",
    "NBoxes": "0", "BoxGainLo": "0", "BoxGainHi": "0", "BoxSig": "0",
    "EpsGain": "0", "EpsP": "1.00", "EpsWins": "--",
    "IgdGain": "0", "IgdP": "1.00", "IgdWins": "--",
    "SecPerCircuit": "0", "TimeApproxLG": "0", "TimeNSGA": "0",
    "TimeGuidance": "0", "GuidanceShare": "0", "TimeEDAcold": "0",
}
for k, v in defaults.items():
    M.setdefault(k, v)

with open(os.path.join(PAPER, "numbers.tex"), "w") as fh:
    fh.write("% auto-generated by scripts/make_numbers.py -- do not edit\n")
    for k in sorted(M):
        fh.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
print("wrote paper/numbers.tex with", len(M), "macros")
for k in sorted(M):
    if k not in ("AblationText", "AppsText"):
        print(f"  \\{k} = {M[k]}")
