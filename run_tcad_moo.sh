#!/bin/bash
# TCAD revision: stronger same-space optimisers (reviewer 1) -- SMS-EMOA and
# batch MOBO over the template, at every width of the paper, then the 32x32
# scaling campaign.  Caches were seeded from the campaign of the same width,
# which changes wall clock only.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 30 \
  --algos smsemoa mobo --workers 12 --tag n8moo >> results/log_n8moo.txt 2>&1
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 20 \
  --algos smsemoa mobo --nvec 4194304 --gpu --workers 12 --tag n16umoo \
  >> results/log_n16umoo.txt 2>&1
python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 20 --signed \
  --algos smsemoa mobo --nvec 4194304 --gpu --workers 12 --tag n16smoo \
  >> results/log_n16smoo.txt 2>&1
