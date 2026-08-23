"""Main search-algorithm comparison for Approx-LG.

Usage:
    python scripts/run_search.py --n 8 --budget 400 --seeds 10 \
        --algos random ga nsga2 llm_only approxlg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approxlg import baselines, search                     # noqa: E402
from approxlg.arch import Space                            # noqa: E402
from approxlg.engine import Engine, pareto_of              # noqa: E402
from approxlg.llm import LLM, LLMOutage                    # noqa: E402
from approxlg.metrics import hypervolume                   # noqa: E402
from approxlg.simulate import Evaluator                    # noqa: E402
from approxlg.synth import Synthesizer                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

ALL = dict(search.ALGORITHMS)
from approxlg.surrogate import run_nsga2_rf                 # noqa: E402

ALL["nsga2_rf"] = run_nsga2_rf
ALL["eamva"] = baselines.run_eamva
ALL["biased_ga"] = baselines.run_biased_ga

# ablations
ALL["approxlg_no_inject"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_inject": False})
ALL["approxlg_no_mutguide"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_mutguide": False})
ALL["approxlg_no_refine"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_refine": False})
ALL["approxlg_no_warm"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_warm": False})
# Splits the mutation-guidance operator in two: this keeps the LLM's local
# search over the decade elites and drops only the proposal distribution, so
# the ablation can say which half its effect belongs to.
ALL["approxlg_no_prior"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_prior": False})
# Placebo control: the full guided control flow with the model's content
# removed.  Plain NSGA-II is not the right comparison for guidance, because the
# guided loop consumes different random draws and tops the generation up
# differently even when every answer is discarded.
ALL["approxlg_sham"] = search.run_approxlg
ALL["approxlg_prior_only"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_warm": False, "use_inject": False, "use_refine": False})
ALL["approxlg_inject_only"] = lambda *a, **k: search.run_approxlg(
    *a, **{**k, "use_mutguide": False, "use_refine": False})

NEEDS_LLM = {"llm_only", "approxlg", "approxlg_no_inject",
             "approxlg_no_mutguide", "approxlg_no_refine",
             "approxlg_no_warm", "approxlg_prior_only",
             "approxlg_inject_only", "approxlg_no_prior",
             "nsga2_llmprior", "approxlg_sham"}

# Arms that walk the guidance code path but must never receive an answer.
SHAM_ALGOS = {"approxlg_sham"}

# A run whose guidance calls nearly all failed is a run of the *unguided*
# algorithm wearing the guided algorithm's label, and averaging it into the
# guided arm dilutes the effect towards zero.  Such a run is reported and
# dropped instead of written.  See scripts/quarantine_dead_llm.py for the
# campaign this rule was written after.
MAX_FAIL_RATE = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--algos", nargs="+", default=["random", "ga", "nsga2",
                                                   "llm_only", "approxlg"])
    ap.add_argument("--nmed-ref", type=float, default=0.02)
    ap.add_argument("--nmed-floor", type=float, default=1e-6)
    ap.add_argument("--nvec", type=int, default=0,
                    help="0 = exhaustive (8x8); otherwise Monte-Carlo size")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--signed", action="store_true")
    ap.add_argument("--inject", type=int, default=6)
    ap.add_argument("--mutguide", type=int, default=6)
    ap.add_argument("--warm-frac", type=float, default=0.5)
    ap.add_argument("--inject-every", type=int, default=2)
    ap.add_argument("--prior-every", type=int, default=3)
    ap.add_argument("--refine-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", default="")
    ap.add_argument("--llm-temp", type=float, default=None,
                    help="override the LLM sampling temperature, so the "
                         "sensitivity of the result to it can be measured")
    ap.add_argument("--llm-model", default=None,
                    help="override the LLM endpoint")
    ap.add_argument("--llm-thinking", action="store_true",
                    help="enable extended thinking, to test whether the "
                         "paper's negative results survive a stronger "
                         "configuration of the same endpoint")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    tag = args.tag or f"n{args.n}"
    sp = Space(args.n, signed=args.signed)
    syn = Synthesizer(cache=os.path.join(RESULTS, f"ppa_cache_{tag}.json"),
                      n=args.n, power_vectors=1 << 14)
    ev = Evaluator(sp, nvec=(args.nvec or None), gpu=args.gpu)
    eng = Engine(sp, syn, ev, nworkers=args.workers,
                 cache_file=os.path.join(RESULTS, f"designs_{tag}.jsonl"))

    print(f"space: n={sp.n} genes={sp.length} stages={sp.nstage} "
          f"columns={sp.ncol} modes={len(__import__('approxlg.cells', fromlist=['x']).SLOT_MODES)}")
    print(f"yosys: {syn.yosys}\nsta:   {syn.sta}")

    # reference design: the exact multiplier (not charged to any budget)
    exact = eng.evaluate([np.zeros(sp.length, dtype=np.int32)])[0]
    print(f"exact {args.n}x{args.n}: area={exact.area:.2f} um^2  "
          f"delay={exact.delay:.3f} ns  power={exact.power:.2f} uW  "
          f"gates={exact.gates}")
    with open(os.path.join(RESULTS, f"exact_{tag}.json"), "w") as fh:
        json.dump(exact.to_json(), fh, indent=1)

    import math
    lo = np.array([math.log10(args.nmed_floor), 0.0, 0.0])
    hi = np.array([math.log10(args.nmed_ref), exact.area, exact.delay])
    ref = np.array([1.0, 1.0, 1.0])
    norm = hi - lo

    out_path = os.path.join(RESULTS, f"search_{tag}.jsonl")

    def already_done():
        got = set()
        if os.path.exists(out_path):
            with open(out_path) as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        got.add((d["algo"], d["seed"]))
        return got

    # Two campaigns on the same tag write the same (algo, seed) twice, and a
    # duplicated run silently doubles a seed's weight in every average.  The
    # start-up snapshot is not enough, because the other campaign writes while
    # this one runs, so the check is repeated just before each write.
    done = already_done()

    for algo in args.algos:
        for s in range(args.seed0, args.seed0 + args.seeds):
            if (algo, s) in done:
                print(f"skip {algo} seed {s} (already done)")
                continue
            llm = None
            if algo in NEEDS_LLM:
                kw = {}
                if args.llm_temp is not None:
                    kw["temperature"] = args.llm_temp
                    kw["fixed_temperature"] = True
                if args.llm_model:
                    kw["model"] = args.llm_model
                if args.llm_thinking:
                    kw["thinking"] = True
                if algo in SHAM_ALGOS:
                    kw["sham"] = True
                    kw.pop("log_path", None)
                llm = LLM(log_path=(None if algo in SHAM_ALGOS else
                                    os.path.join(RESULTS, f"llm_{tag}.jsonl")),
                          **kw)
                if not llm.available():
                    print(f"!! no API key, skipping {algo}")
                    break
            t0 = time.time()
            eng.start_run()
            tracker = search.Tracker(lo, hi, ref)
            kw = dict(pop=args.pop, llm=llm, norm=norm)
            if algo.startswith("approxlg"):
                kw.update(n_inject=args.inject, n_mutguide=args.mutguide,
                          warm_frac=args.warm_frac,
                          inject_every=args.inject_every,
                          prior_every=args.prior_every,
                          refine_every=args.refine_every)
            try:
                extra = ALL[algo](eng, sp, args.budget, s, tracker, **kw) or {}
            except LLMOutage as e:
                print(f"!! {algo} seed {s}: {e}")
                print("   endpoint looks down; stopping this algorithm")
                break
            dt = time.time() - t0
            if llm is not None:
                st = llm.stats
                rate = st.failures / max(1, st.calls + st.failures)
                if rate > MAX_FAIL_RATE:
                    print(f"!! {algo} seed {s}: {rate:.0%} of guidance calls "
                          f"failed ({st.calls} ok, {st.failures} bad); "
                          f"discarding the run")
                    continue
            pf = pareto_of(tracker.records)
            hv = tracker.hv()
            rec = {
                "algo": algo, "seed": s, "budget": args.budget,
                "n_eval": eng.n_eval, "n_points": len(tracker.records),
                "hv": hv, "seconds": dt,
                "hv_trace": tracker.trace,
                "pareto": [r.to_json() for r in pf],
                "llm": (dict(llm.stats.__dict__) if llm else None),
                "accepted": extra.get("accepted", 0),
                "proposed": extra.get("proposed", 0),
                "llm_points": extra.get("llm_points", 0),
                "llm_on_front": extra.get("llm_on_front", 0),
                "restrictions": extra.get("restrictions", 0),
                "prior_updates": extra.get("prior_updates", 0),
                # wall-clock breakdown; n_synth < n_eval when the persistent
                # cache already held the circuit, so the EDA cost of a run is
                # t_synth/n_synth * n_eval rather than the measured t_synth
                "n_synth": eng.n_synth,
                "t_sim": eng.t_sim,
                "t_synth": eng.t_synth,
                "t_llm": (llm.stats.seconds if llm else 0.0),
            }
            if (algo, s) in already_done():
                print(f"!! {algo} seed {s} was written by another campaign "
                      f"while this one ran; discarding the duplicate")
                continue
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{algo:22s} seed {s:2d}: HV={hv:.4f} "
                  f"|PF|={len(pf):3d} evals={eng.n_eval} {dt:.1f}s"
                  + (f" llm={llm.stats.calls}c/{llm.stats.failures}f"
                     f" acc={extra.get('accepted',0)}/{extra.get('proposed',0)}"
                     if llm else ""))
    syn.save_cache()


if __name__ == "__main__":
    main()
