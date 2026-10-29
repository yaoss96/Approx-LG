#!/bin/bash
# TCAD revision: is the guidance complementary to a stronger optimizer?
# The four operators on top of SMS-EMOA's selection, against the same hybrid
# with every answer discarded, 16x16 unsigned, paired by seed.  The placebo
# arm needs no endpoint and runs here; the guided arm runs from
# run_tcad_resume_llm.sh once the endpoint answers again (it stopped
# answering on 2026-10-29 when the account balance ran out).
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 20 \
  --algos approxlg_sms_sham --nvec 4194304 --gpu --workers 12 \
  --tag n16uhyb >> results/log_n16uhyb.txt 2>&1
