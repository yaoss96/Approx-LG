#!/bin/bash
# TCAD revision: second 16x16 ablation batch (seeds 10-19), together with the
# full system re-run on the same seeds in the same period (tag n16udrift),
# which every ablation variant is paired with -- the first batch ran two
# months after the main campaign, so pairing with the August runs would mix
# a possible drift of the endpoint's model into the operator effects.
# Two chains of 12 workers; the driver skips seeds that are already recorded.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
echo "$(date) second ablation batch and drift seeds 10-19 started" >> results/log_resume_llm.txt
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 \
  --seed0 10 --algos approxlg --nvec 4194304 --gpu --workers 12 \
  --tag n16udrift >> results/log_n16udrift.txt 2>&1 &
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 \
  --seed0 10 --algos approxlg_no_prior approxlg_no_inject \
  approxlg_no_mutguide approxlg_no_warm approxlg_no_refine \
  --nvec 4194304 --gpu --workers 12 --tag n16uabl \
  >> results/log_n16uabl.txt 2>&1
wait
echo "$(date) second ablation batch and drift seeds 10-19 done" >> results/log_resume_llm.txt
