#!/bin/bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 10 \
  --algos approxlg llm_only --workers 8 --tag n8L >> results/log_n8L_clean.txt 2>&1
