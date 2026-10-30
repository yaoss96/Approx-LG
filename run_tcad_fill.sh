#!/bin/bash
# TCAD revision: complete the search table, so that every arm of it is run at
# every width.  At 32x32 the arms that had run at 8 and 16 bits only and the
# arms that had run at 8 bits only (10 seeds); at 16x16 unsigned and signed
# the 8x8-only arms (20 seeds) and the missing random-search seeds 8-19.
# Each chain has its own tag, so no two processes append to the same cache.
# The caches start from the main campaign of the width, which saves synthesis
# time only: a design still costs budget the first time a run asks for it.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
seed_cache() {  # new tag, campaign whose cache it starts from
  [ -f results/designs_$1.jsonl ] || cp results/designs_$2.jsonl results/designs_$1.jsonl
  [ -f results/ppa_cache_$1.json ] || cp results/ppa_cache_$2.json results/ppa_cache_$1.json
}
seed_cache n32ufill n32u
seed_cache n16ufill n16u
seed_cache n16sfill n16s
echo "$(date) table completion started" >> results/log_fill.txt

wide() {
  python scripts/run_search.py --n 32 --budget 500 --pop 32 --seeds 10 \
    --algos random mobo nsga2_rf nsga2_expert nsga2_llmprior ga eamva \
    biased_ga llm_only --nvec 1048576 --gpu --workers 12 --tag n32ufill \
    >> results/log_n32ufill.txt 2>&1
}
sixteen() {
  for s in u s; do
    flag=""
    [ "$s" = s ] && flag="--signed"
    python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 20 $flag \
      --algos ga eamva biased_ga llm_only --nvec 4194304 --gpu --workers 12 \
      --tag n16${s}fill >> results/log_n16${s}fill.txt 2>&1
    python scripts/run_search.py --n 16 --budget 500 --pop 32 --seeds 12 \
      --seed0 8 $flag --algos random --nvec 4194304 --gpu --workers 12 \
      --tag n16${s}fill >> results/log_n16${s}fill.txt 2>&1
  done
}
wide &
sixteen
wait
echo "$(date) table completion done" >> results/log_fill.txt
