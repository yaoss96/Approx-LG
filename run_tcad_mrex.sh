#!/bin/bash
# TCAD revision: retargeting the cell library as well as the objective.  The
# same MRE-objective campaign as run_tcad_mre.sh, with the carry-free
# compressors of Esposito et al. added to the template as a 22nd style
# (APPROXLG_STYLES=ext).  Separate tag and caches: the alphabet differs.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
export APPROXLG_STYLES=ext
[ -f results/designs_n8mrex.jsonl ] || cp results/designs_n8mre.jsonl results/designs_n8mrex.jsonl
[ -f results/ppa_cache_n8mrex.json ] || cp results/ppa_cache_n8mre.json results/ppa_cache_n8mrex.json
python scripts/run_search.py --n 8 --budget 400 --pop 32 --seeds 12 \
  --algos nsga2 smsemoa approxlg --error-metric MRE \
  --nmed-floor 1e-5 --nmed-ref 0.1 --workers 6 --tag n8mrex \
  >> results/log_n8mrex.txt 2>&1
