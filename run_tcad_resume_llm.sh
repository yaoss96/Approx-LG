#!/bin/bash
# TCAD revision: resume the LLM-guided arms after the endpoint started
# refusing calls (HTTP 402, account balance exhausted, 2026-10-29 ~02:00).
# Polls the endpoint every 10 minutes with a two-token request; once it
# answers, re-runs every guided arm with missing seeds -- the driver skips
# seeds that are already recorded -- in two chains of at most 12 workers.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
until python scripts/api_ok.py 2>>results/log_resume_llm.txt; do sleep 600; done
# never during the serial-timing measurement (run_tcad_timing.sh)
while [ -f results/time_serial.lock ]; do sleep 60; done
echo "$(date) endpoint back; resuming guided arms" >> results/log_resume_llm.txt

wide() {
  # 32x32 primary arm -- once the campaign's own chain has moved past it, so
  # the two never compute the same seeds -- then the extended-alphabet arm
  until grep -q "^smsemoa" results/log_n32u.txt 2>/dev/null; do sleep 300; done
  python scripts/run_search.py --n 32 --budget 500 --pop 32 --seeds 10 \
    --algos approxlg --nvec 1048576 --gpu --workers 12 --tag n32u \
    >> results/log_n32u.txt 2>&1
  APPROXLG_STYLES=ext python scripts/run_search.py --n 8 --budget 400 \
    --pop 32 --seeds 12 --algos approxlg --error-metric MRE \
    --nmed-floor 1e-5 --nmed-ref 0.1 --workers 6 --tag n8mrex \
    >> results/log_n8mrex.txt 2>&1
}
sixteen() {
  # first ablation batch; the full system re-run on the ablation's seeds now,
  # against its August runs, which tests whether the model behind the
  # endpoint name drifted between the campaigns (the ablation pairs with the
  # August runs); the guided hybrid arm; then the second ablation batch
  python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 \
    --seed0 0 --algos approxlg_no_refine --nvec 4194304 --gpu --workers 12 \
    --tag n16uabl >> results/log_n16uabl.txt 2>&1
  [ -f results/designs_n16udrift.jsonl ] || cp results/designs_n16u.jsonl results/designs_n16udrift.jsonl
  [ -f results/ppa_cache_n16udrift.json ] || cp results/ppa_cache_n16u.json results/ppa_cache_n16udrift.json
  python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 \
    --seed0 0 --algos approxlg --nvec 4194304 --gpu --workers 12 \
    --tag n16udrift >> results/log_n16udrift.txt 2>&1
  python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 20 \
    --algos approxlg_sms --nvec 4194304 --gpu --workers 12 --tag n16uhyb \
    >> results/log_n16uhyb.txt 2>&1
  python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 10 \
    --seed0 10 --algos approxlg_no_prior approxlg_no_inject \
    approxlg_no_mutguide approxlg_no_warm approxlg_no_refine \
    --nvec 4194304 --gpu --workers 12 --tag n16uabl \
    >> results/log_n16uabl.txt 2>&1
}
wide &
sixteen
wait
echo "$(date) resume done" >> results/log_resume_llm.txt
