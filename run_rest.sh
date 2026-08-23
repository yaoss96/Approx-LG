#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
# 16x16 signed
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 3 \
    --algos approxlg nsga2 random --nvec 4194304 --gpu --signed --workers 12 \
    --tag n16s >> results/log_n16s.txt 2>&1
# ablations at 8x8
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 5 \
    --algos approxlg_no_warm approxlg_no_inject approxlg_no_mutguide \
            approxlg_no_refine --workers 12 --tag n8abl >> results/log_n8abl.txt 2>&1
