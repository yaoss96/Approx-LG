"""Regenerate every figure, table and macro of the paper, then build it."""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

def cleanup():
    """Drop any design whose characterisation failed.

    This used to also delete every search record with seed >= 10, to purge
    runs that predated the synthesis-retry fix.  Those were purged long ago and
    the seed budget has since been extended well past 10, so the rule now only
    destroys good data.  Never filter results by seed number here.
    """
    for p in glob.glob(os.path.join(ROOT, "results", "designs_*.jsonl")):
        if time.time() - os.path.getmtime(p) < 300:
            print(f"  {os.path.basename(p)} still being written, left alone")
            continue
        keep, drop = [], 0
        for line in open(p):
            if not line.strip():
                continue
            if json.loads(line).get("area", 0) > 1e8:
                drop += 1
                continue
            keep.append(line.rstrip("\n"))
        if drop:
            open(p, "w").write("\n".join(keep) + "\n")
            print(f"  dropped {drop} failed designs from {os.path.basename(p)}")


STEPS = [
    # Reports, does not purge: a corrupt characterisation invalidates whatever
    # run put it on a front, so removing it has to be followed by re-running
    # that seed and neither should happen silently inside a paper build.
    ["scripts/purge_bad_ppa.py"],
    ["scripts/make_template_fig.py"],
    ["scripts/analyze_search.py", "--tag", "n8", "n8L", "n8abl", "n8p"],
    ["scripts/analyze_priors.py", "--tag", "n8", "n8L", "n8p"],
    ["scripts/analyze_prior_agreement.py"],
    ["scripts/analyze_robustness.py", "--tag", "n8", "n8L", "n8p",
     "--suffix", "n8"],
    ["scripts/analyze_search.py", "--tag", "n16u", "--budget", "500",
     "--suffix", "n16u"],
    ["scripts/analyze_search.py", "--tag", "n16s", "--budget", "500",
     "--suffix", "n16s"],
    ["scripts/make_hv_fig.py"],
    ["scripts/analyze_styles.py", "n8L"],
    ["scripts/make_app_figs.py"],
    # The statistics the paper quotes.  These are slow -- the exact Wilcoxon
    # enumerates 2^n sign assignments and the accuracy check re-simulates on
    # the GPU -- but a build that reuses a stale stats.json would print
    # yesterday's p-values beside today's tables, which is the failure this
    # campaign has already had once.
    ["scripts/analyze_stats.py"],
    ["scripts/analyze_provenance.py"],
    ["scripts/analyze_walltime.py", "--tags", "n16u", "--budget", "500"],
    ["scripts/analyze_sota.py", "--front", "n8L", "--algos", "approxlg",
     "--refs", "n8", "--ref-algos", "eamva", "biased_ga", "--name", "sota8",
     "--select", "8", "--short", "--nmed-max", "0.02"],
    ["scripts/analyze_sota.py", "--front", "n16u", "--algos", "approxlg",
     "--refs", "libs_n16.jsonl", "--signed", "no", "--name", "sota16u",
     "--select", "8", "--nmed-min", "1e-12"],
    ["scripts/analyze_sota.py", "--front", "n16s", "--algos", "approxlg",
     "--refs", "libs_n16.jsonl", "--signed", "yes", "--name", "sota16s",
     "--select", "8", "--nmed-min", "1e-12"],
    ["scripts/analyze_sota.py", "--front", "n16u", "n16s",
     "--signed-tags", "n16s", "--match-signed", "--algos", "approxlg",
     "--refs", "libs_n16.jsonl", "--name", "sota16", "--select", "8",
     "--nmed-min", "1e-12"],
    ["scripts/make_numbers.py"],
    ["scripts/build_paper.py"],
]


def main():
    print("=== cleanup")
    cleanup()
    for step in STEPS:
        print(f"\n=== {' '.join(step)}")
        r = subprocess.run([PY] + step, cwd=ROOT, capture_output=True,
                           text=True)
        out = (r.stdout or "").strip().splitlines()
        for line in out[-14:]:
            print("   ", line)
        if r.returncode != 0:
            print("    !! failed")
            print((r.stderr or "")[-1200:])
    # Keep the deliverable in step with the build.  Copying this by hand is how
    # paper/Approx-LG.pdf silently went several turns out of date.
    src = os.path.join(ROOT, "paper", "main_build.pdf")
    dst = os.path.join(ROOT, "paper", "Approx-LG.pdf")
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print("\npaper/main_build.pdf -> paper/Approx-LG.pdf")
    else:
        print("\n!! no main_build.pdf, paper/Approx-LG.pdf left untouched")


if __name__ == "__main__":
    main()
