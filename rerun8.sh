#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 10 \
  --algos nsga2 random ga eamva biased_ga --workers 8 --tag n8 \
  >> results/log_n8_clean.txt 2>&1
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 5 \
  --algos approxlg_no_warm approxlg_no_inject approxlg_no_mutguide approxlg_no_refine \
  --workers 8 --tag n8abl >> results/log_n8abl_clean.txt 2>&1
