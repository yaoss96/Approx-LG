#!/bin/bash
# TCAD revision: retargeting (reviewer 1).  The same framework re-run with
# MRE instead of NMED as the error objective, at 8x8, where the MRE-matched
# comparison with OPACT-style trees is the weak point of the NMED front.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 12 \
  --algos nsga2 approxlg_sham approxlg smsemoa --error-metric MRE \
  --nmed-floor 1e-5 --nmed-ref 0.1 --workers 6 --tag n8mre \
  >> results/log_n8mre.txt 2>&1
