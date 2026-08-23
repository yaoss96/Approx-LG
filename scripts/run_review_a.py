"""Review campaign, lane A: everything that needs no LLM endpoint.

Two jobs the reviewers asked for and one the data asked for:

  * the two non-LLM proposal distributions (fixed designer prior, online
    credit-assignment prior).  The ablation says the elicited prior is the
    operator that carries the result, so the paper cannot claim the language
    model is what matters until the same channel driven by something else has
    been measured;
  * the unguided baselines extended from 10 to 30 seeds, so every arm of the
    main table rests on the same number of runs;
  * nothing here touches the network, so it can run beside the LLM lane
    without either waiting on the other.  Two concurrent campaigns is the
    limit before Yosys starts failing to spawn on this machine.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

JOBS = [
    (["--n", "8", "--budget", "400", "--seeds", "20",
      "--algos", "nsga2_expert", "nsga2_adaptive", "--tag", "n8p"],
     "non-LLM priors, 20 seeds"),
    (["--n", "8", "--budget", "400", "--seeds", "20", "--seed0", "10",
      "--algos", "random", "ga", "eamva", "biased_ga", "--tag", "n8"],
     "unguided baselines to 30 seeds"),
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
