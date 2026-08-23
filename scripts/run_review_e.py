"""Review campaign, lane E: the elicited prior at 16x16.

At 8x8 the written-down rule beats the elicited one on the same channel.  At
16x16 the full guided system beats the written-down rule on the unsigned
template.  Those two facts do not compose into a statement about the model,
because the second compares four operators against one.  The missing arm is the
elicited prior alone at the larger width, which is the only apples-to-apples
version of the 8x8 control and the one that says whether the model's advice
gets better, or merely more of it gets used, when the genome grows.

Random search is deliberately last: it already has three seeds, it is far
below everything else, and nothing in the paper turns on its error bars.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

COMMON = ["--n", "16", "--seeds", "8", "--budget", "500", "--pop", "32",
          "--nvec", "4194304", "--gpu", "--workers", "12"]

JOBS = [
    (COMMON + ["--algos", "nsga2_llmprior", "--tag", "n16u"],
     "elicited prior alone, unsigned"),
    (COMMON + ["--signed", "--algos", "nsga2_llmprior", "--tag", "n16s"],
     "elicited prior alone, signed"),
    (COMMON + ["--algos", "random", "--tag", "n16u"],
     "random search to 8 seeds, unsigned"),
    (COMMON + ["--signed", "--algos", "random", "--tag", "n16s"],
     "random search to 8 seeds, signed"),
]


def main():
    for argv, what in JOBS:
        print(f"\n=== {what}\n    {' '.join(argv)}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, os.path.join("scripts", "run_search.py")]
                           + argv, cwd=ROOT, capture_output=True, text=True)
        print(f"    exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        for line in (r.stdout or "").strip().splitlines()[-12:]:
            print("   ", line, flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-1500:], flush=True)


if __name__ == "__main__":
    main()
