"""Re-run the campaigns that scripts/finalize.py destroyed.

finalize.py used to delete every search record with seed >= 10.  That rule was
written to purge runs predating the synthesis-retry fix and should have been
removed when the seed budget was extended.  It was not, and one finalize run
wiped ~80 completed runs.  The deleter is gone; this replaces the data.

run_search.py skips (algo, seed) pairs it already has, so this is safe to
re-run and will only fill the gaps.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

JOBS = [
    (["--n", "8", "--budget", "400", "--seeds", "20", "--seed0", "10",
      "--algos", "nsga2", "--tag", "n8"], "NSGA-II seeds 10..29"),
    (["--n", "8", "--budget", "400", "--seeds", "20", "--seed0", "10",
      "--algos", "approxlg", "--tag", "n8L"], "Approx-LG seeds 10..29"),
    (["--n", "8", "--budget", "400", "--seeds", "10", "--seed0", "10",
      "--algos", "approxlg_no_warm", "approxlg_no_inject",
      "approxlg_no_mutguide", "approxlg_no_refine", "--tag", "n8abl"],
     "ablation seeds 10..19"),
]


def main():
    for argv, what in JOBS:
        print(f"\n=== {what}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, os.path.join("scripts", "run_search.py")]
                           + argv, cwd=ROOT, capture_output=True, text=True)
        print(f"    exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        for line in (r.stdout or "").strip().splitlines()[-4:]:
            print("   ", line, flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-800:], flush=True)


if __name__ == "__main__":
    main()
