#!/bin/bash
# TCAD revision: the four guidance operators on top of SMS-EMOA's selection
# (approxlg_sms) and the same hybrid with every answer discarded
# (approxlg_sms_sham) at the widths run_tcad_hyb.sh did not cover: 8x8 and
# 16x16 signed on 20 seeds and 32x32 on 10, paired by seed, with the budgets,
# populations and stimuli of the main campaigns.  Every chain has its own tag
# and cache (seeded from a cache of the same width, which saves synthesis
# time only); analyze_tcad pools the tags of a width.  The guided 8x8 and
# 16x16 signed arms ran as two chains each (seeds 0-16 and 17-19, 0-9 and
# 10-19) to finish sooner.  The placebo sends no request, so only the guided
# chains need the endpoint.
#   bash run_tcad_hyb2.sh                 # every chain
#   bash run_tcad_hyb2.sh g32 s32         # a subset
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
export PYTHONUNBUFFERED=1
# chain: tag  seed-cache  algo  width-flags
declare -A CHAIN=(
  [g8]="n8hyb n8moo approxlg_sms --n 8 --budget 400 --seeds 17"
  [g8b]="n8hyb2 n8moo approxlg_sms --n 8 --budget 400 --seeds 3 --seed0 17"
  [s8]="n8hybs n8moo approxlg_sms_sham --n 8 --budget 400 --seeds 20"
  [g16s]="n16shyb n16smoo approxlg_sms --n 16 --signed --budget 500 --seeds 10 --nvec 4194304 --gpu"
  [g16s2]="n16shyb2 n16smoo approxlg_sms --n 16 --signed --budget 500 --seeds 10 --seed0 10 --nvec 4194304 --gpu"
  [s16s]="n16shybs n16smoo approxlg_sms_sham --n 16 --signed --budget 500 --seeds 20 --nvec 4194304 --gpu"
  [g32]="n32uhyb n32ufill_e2 approxlg_sms --n 32 --budget 500 --seeds 10 --nvec 1048576 --gpu"
  [s32]="n32uhybs n32ufill_e2 approxlg_sms_sham --n 32 --budget 500 --seeds 10 --nvec 1048576 --gpu"
)
run_chain() {
  set -- ${CHAIN[$1]}
  local tag=$1 from=$2 algo=$3
  shift 3
  [ -f results/designs_$tag.jsonl ] || cp results/designs_$from.jsonl results/designs_$tag.jsonl
  [ -f results/ppa_cache_$tag.json ] || cp results/ppa_cache_$from.json results/ppa_cache_$tag.json
  python scripts/run_search.py "$@" --pop 32 --algos $algo --workers 6 \
    --tag $tag >> results/log_$tag.txt 2>&1
}
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(s8 s16s s32 g16s g16s2 g32 g8 g8b)
echo "$(date) hybrid campaigns started: ${names[*]}" >> results/log_fill.txt
for c in "${names[@]}"; do run_chain $c & done
wait
echo "$(date) hybrid campaigns done: ${names[*]}" >> results/log_fill.txt
