"""Review campaign, lane B: everything that needs the LLM endpoint.

The 2026-08-20 seed-extension campaign ran against a dead endpoint and every
guidance call in it failed, so the runs it recorded as Approx-LG were plain
NSGA-II.  They have been moved to ``search_*_dead.jsonl`` by
``scripts/quarantine_dead_llm.py``; this replaces them with real ones and adds
the LLM arm of the three-way prior control.

Jobs are ordered by how much the paper depends on them, because the endpoint
may go down again and a truncated campaign should still leave the important
comparison complete.  ``run_search.py`` now aborts an algorithm after six
consecutive failed calls instead of quietly recording an unguided run, so a
second outage costs time rather than data.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

JOBS = [
    # the headline comparison: 10 seeds cannot resolve the effect
    (["--n", "8", "--budget", "400", "--seeds", "20", "--seed0", "10",
      "--algos", "approxlg", "--tag", "n8L"], "Approx-LG seeds 10..29"),
    # the LLM arm of the prior control, paired with lane A's two arms
    (["--n", "8", "--budget", "400", "--seeds", "20",
      "--algos", "nsga2_llmprior", "--tag", "n8p"], "elicited prior only"),
    # the ablation: 5 valid seeds for two of the four variants
    (["--n", "8", "--budget", "400", "--seeds", "5",
      "--algos", "approxlg_no_mutguide", "approxlg_no_refine",
      "--tag", "n8abl"], "ablation gaps, seeds 0..4"),
    (["--n", "8", "--budget", "400", "--seeds", "10", "--seed0", "10",
      "--algos", "approxlg_no_warm", "approxlg_no_inject",
      "approxlg_no_mutguide", "approxlg_no_refine", "--tag", "n8abl"],
     "ablation to 20 seeds"),
    (["--n", "8", "--budget", "400", "--seeds", "10", "--seed0", "10",
      "--algos", "llm_only", "--tag", "n8L"], "LLM-only to 20 seeds"),
]


def main():
    for argv, what in JOBS:
        print(f"\n=== {what}\n    {' '.join(argv)}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, os.path.join("scripts", "run_search.py")]
                           + argv, cwd=ROOT, capture_output=True, text=True)
        print(f"    exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        for line in (r.stdout or "").strip().splitlines()[-8:]:
            print("   ", line, flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-1500:], flush=True)


if __name__ == "__main__":
    main()
