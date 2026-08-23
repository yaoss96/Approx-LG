"""Review campaign, lane C: the 16x16 templates from three seeds to eight.

Three seeds cannot separate anything, and 16x16 is where the paper's claim now
lives, because the 8x8 prior control showed a written-down rule matching the
guided run.  The question here is transfer: the rule was stated once, in terms
of a column offset and a measured cell error, and is carried to a wider and to
a signed template unchanged, while the model is re-asked about a circuit it has
not seen.

Jobs are ordered by priority *and* completed for both templates before the next
priority starts, because a 16x16 run costs minutes and the campaign may have to
be cut short.  Stopping after any job leaves a complete, reportable comparison
rather than half of one.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

COMMON = ["--n", "16", "--seeds", "8", "--budget", "500", "--pop", "32",
          "--nvec", "4194304", "--gpu", "--workers", "12"]

JOBS = []
for algos, what in ((["approxlg", "nsga2"], "guided and unguided"),
                    (["nsga2_expert"], "the transferred fixed prior"),
                    (["random"], "random search")):
    for tag, extra, name in (("n16u", [], "unsigned"),
                             ("n16s", ["--signed"], "signed")):
        JOBS.append((COMMON + extra + ["--algos"] + algos + ["--tag", tag],
                     f"{what}, {name}"))


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
