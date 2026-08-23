"""The surrogate-assisted baseline at 16x16, where the claim actually lives.

At 8x8 nothing separates surrogate assistance, LLM guidance and the sham
control, so the comparison the reviewer asked for has to be made at the width
where the guided arm is significant at all.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

COMMON = ["--n", "16", "--budget", "500", "--pop", "32",
          "--nvec", "4194304", "--gpu", "--workers", "10",
          "--seeds", "20", "--algos", "nsga2_rf"]

JOBS = [
    (COMMON + ["--tag", "n16u"], "surrogate NSGA-II, unsigned"),
    (COMMON + ["--signed", "--tag", "n16s"], "surrogate NSGA-II, signed"),
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
