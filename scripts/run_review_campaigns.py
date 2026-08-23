"""Run the campaigns needed to answer the statistical review.

Three things are missing from the first submission's evidence:

  1. the 8x8 comparison rests on 10 seeds, where a ~4% hypervolume difference
     is not resolvable;
  2. the ablation rests on 5 seeds per variant, where only the largest effect
     (mutation guidance) clears significance, so nothing can be concluded
     about the other three operators either way;
  3. every number comes from one LLM endpoint at one temperature.

The runs are serialised: two concurrent Yosys campaigns is the most this
machine takes before spawn failures start.
"""
import json
import os
import subprocess
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PY = sys.executable


def counts(tag):
    p = os.path.join(RESULTS, f"search_{tag}.jsonl")
    by = defaultdict(int)
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                by[json.loads(line)["algo"]] += 1
    return by


def wait_until_quiet(tag, algo, target, poll=60):
    """Block until an already-running campaign has produced `target` runs."""
    while counts(tag).get(algo, 0) < target:
        time.sleep(poll)


JOBS = [
    # ablation to 20 seeds per variant: 5 is far too few to separate a ~3%
    # effect from noise, so the current "not significant" says nothing
    (["--n", "8", "--budget", "400", "--seeds", "15", "--seed0", "5",
      "--algos", "approxlg_no_warm", "approxlg_no_inject",
      "approxlg_no_mutguide", "approxlg_no_refine", "--tag", "n8abl"],
     "ablation to 20 seeds"),
    # LLM sensitivity: same algorithm, colder and hotter sampling
    (["--n", "8", "--budget", "400", "--seeds", "8", "--algos", "approxlg",
      "--llm-temp", "0.2", "--tag", "n8t02"], "temperature 0.2"),
    (["--n", "8", "--budget", "400", "--seeds", "8", "--algos", "approxlg",
      "--llm-temp", "1.2", "--tag", "n8t12"], "temperature 1.2"),
]


def main():
    print("waiting for the seed-extension campaigns to finish")
    wait_until_quiet("n8", "nsga2", 30)
    wait_until_quiet("n8L", "approxlg", 30)
    print("both finished")
    for argv, what in JOBS:
        print(f"\n=== {what}\n    {' '.join(argv)}", flush=True)
        t0 = time.time()
        r = subprocess.run([PY, os.path.join("scripts", "run_search.py")]
                           + argv, cwd=ROOT, capture_output=True, text=True)
        print(f"    exit {r.returncode} in {time.time()-t0:.0f}s", flush=True)
        for line in (r.stdout or "").strip().splitlines()[-6:]:
            print("   ", line, flush=True)
        if r.returncode != 0:
            print((r.stderr or "")[-800:], flush=True)


if __name__ == "__main__":
    main()
