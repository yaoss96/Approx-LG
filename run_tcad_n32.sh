#!/bin/bash
# TCAD revision: 32x32 unsigned (reviewer 1, larger widths).  2^20 Monte-Carlo
# vectors per design; the guided arm, its placebo, NSGA-II and SMS-EMOA,
# paired by seed.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
python scripts/run_search.py --n 32 --budget 500 --pop 32 --seeds 10 \
  --algos nsga2 approxlg_sham approxlg smsemoa --nvec 1048576 --gpu \
  --workers 12 --tag n32u >> results/log_n32u.txt 2>&1
