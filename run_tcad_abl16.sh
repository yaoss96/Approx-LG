#!/bin/bash
# TCAD revision: operator ablation at 16x16 unsigned (reviewer 2), paired by
# seed with the approxlg / approxlg_sham runs of the n16u campaign.  The design
# and PPA caches were seeded from n16u, which changes wall clock only: the
# budget charges every circuit a run asks for, cached or not.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
ALGOS="approxlg_no_prior approxlg_no_inject approxlg_no_mutguide approxlg_no_warm approxlg_no_refine"
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 --seed0 0 \
  --algos $ALGOS --nvec 4194304 --gpu --workers 12 --tag n16uabl \
  >> results/log_n16uabl.txt 2>&1
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 --seed0 10 \
  --algos $ALGOS --nvec 4194304 --gpu --workers 12 --tag n16uabl \
  >> results/log_n16uabl.txt 2>&1
