"""Review campaign, lane D: splitting the mutation-guidance operator.

The prior control settled one question and opened another.  The elicited
proposal distribution on its own is worth +0.3% over uniform sampling and
nothing that a test can see, yet removing the whole mutation-guidance operator
from Approx-LG costs several percent.  Those two facts are only compatible if
the operator's value is in its *other* half, the LLM local search that is
handed the cheapest design in each error decade and asked for a child that
dominates it.

``approxlg_no_prior`` keeps that local search and drops only the proposal
distribution, which is the arm that tells the two apart.  It runs first because
the paper's mechanism claim depends on it; the ablation extension comes after.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

JOBS = [
    (["--n", "8", "--budget", "400", "--seeds", "10",
      "--algos", "approxlg_no_prior", "--tag", "n8abl"],
     "guided mutation without the proposal distribution"),
    (["--n", "8", "--budget", "400", "--seeds", "10", "--seed0", "10",
      "--algos", "approxlg_no_mutguide", "approxlg_no_refine",
      "approxlg_no_inject", "approxlg_no_warm", "--tag", "n8abl"],
     "ablation to 20 seeds"),
]


def main():
    for argv, what in JOBS:
        print(f"\n=== {what}\n    {' '.join(argv)}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, os.path.join("scripts", "run_search.py")]
                           + argv, cwd=ROOT, capture_output=True, text=True)
        print(f"    exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        for line in (r.stdout or "").strip().splitlines()[-10:]:
            print("   ", line, flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-1500:], flush=True)


if __name__ == "__main__":
    main()
