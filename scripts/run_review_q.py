"""Second review round, 16x16: the sham control and the seeds the claim needs.

The reviewer's two structural points about this width are that (a) the only
significant claim in the paper rests on eight seeds, and (b) plain NSGA-II is
the wrong control, because the guided loop consumes different random draws and
tops generations up differently even when every answer is discarded -- an
unplanned outage measured that path effect at about a percent on its own.

So the sham arm is run first and to the full seed count on both templates: it
becomes the primary comparison, and NSGA-II drops to a secondary reference.
Then the guided arm is extended to match.  Jobs finish a template at a time so
that stopping early still leaves a complete comparison.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

COMMON = ["--n", "16", "--budget", "500", "--pop", "32",
          "--nvec", "4194304", "--gpu", "--workers", "12"]

JOBS = [
    (COMMON + ["--seeds", "20", "--algos", "approxlg_sham", "--tag", "n16u"],
     "sham guidance, unsigned"),
    (COMMON + ["--seeds", "20", "--signed", "--algos", "approxlg_sham",
               "--tag", "n16s"], "sham guidance, signed"),
    (COMMON + ["--seeds", "20", "--algos", "approxlg", "--tag", "n16u"],
     "guided to 20 seeds, unsigned"),
    (COMMON + ["--seeds", "20", "--signed", "--algos", "approxlg",
               "--tag", "n16s"], "guided to 20 seeds, signed"),
    (COMMON + ["--seeds", "20", "--algos", "nsga2", "--tag", "n16u"],
     "NSGA-II to 20 seeds, unsigned"),
    (COMMON + ["--seeds", "20", "--signed", "--algos", "nsga2",
               "--tag", "n16s"], "NSGA-II to 20 seeds, signed"),
    (COMMON + ["--seeds", "20", "--algos", "nsga2_expert", "nsga2_llmprior",
               "--tag", "n16u"], "prior arms to 20 seeds, unsigned"),
    (COMMON + ["--seeds", "20", "--signed", "--algos", "nsga2_expert",
               "nsga2_llmprior", "--tag", "n16s"],
     "prior arms to 20 seeds, signed"),
]


def main():
    for argv, what in JOBS:
        print(f"\n=== {what}\n    {' '.join(argv)}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, os.path.join("scripts", "run_search.py")]
                           + argv, cwd=ROOT, capture_output=True, text=True)
        print(f"    exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        for line in (r.stdout or "").strip().splitlines()[-14:]:
            print("   ", line, flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-1500:], flush=True)


if __name__ == "__main__":
    main()
