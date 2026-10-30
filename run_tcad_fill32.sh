#!/bin/bash
# TCAD revision: the 32x32 tail of run_tcad_fill.sh (EA-MVA seeds 2-9, BC-GA
# and LLM-only), split into parallel chains.  EA-MVA's near-exact 32x32
# circuits keep ABC busy for minutes each, so a single chain spends most of a
# run waiting for one circuit while the rest of the machine idles; here every
# EA-MVA seed has its own chain.  Each chain has its own tag, and
# analyze_tcad pools the tags of a width; the chains were seeded with the
# union of the purged 32x32 caches (a cache only saves synthesis time).
# The n32ufill chain itself stopped after EA-MVA seed 1.
# GA seed 8 of n32ufill held the one corrupted characterisation found by
# scripts/purge_bad_ppa.py (a torn ABC netlist, see approxlg/synth.py), so it
# is re-run here as chain ga8 on a purged cache and the old record is moved
# to search_n32ufill_dead.jsonl.  BC-GA runs with its upper-bound probes
# simulated rather than synthesised (approxlg/baselines.py): ABC does not map
# those near-exact 32x32 circuits within the flow's time limit, and at 8 and
# 16 bits the bound and its budget charge are unchanged.
#   bash run_tcad_fill32.sh            # every chain
#   bash run_tcad_fill32.sh e2 lo      # a subset
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
export PYTHONUNBUFFERED=1
declare -A CHAIN=(
  [e2]="2 1 eamva 4"
  [e3]="3 1 eamva 4"
  [e4]="4 1 eamva 4"
  [e5]="5 1 eamva 4"
  [e67]="6 2 eamva 4"
  [e8]="8 1 eamva 4"
  [e9]="9 1 eamva 4"
  [bc04]="0 5 biased_ga 6"
  [bc59]="5 5 biased_ga 6"
  [lo]="0 10 llm_only 4"
  [ga8]="8 1 ga 6"
)
run_chain() {
  local tag=n32ufill_$1
  set -- ${CHAIN[$1]}
  [ -f results/designs_$tag.jsonl ] || cp results/designs_n32ufill.jsonl results/designs_$tag.jsonl
  [ -f results/ppa_cache_$tag.json ] || cp results/ppa_cache_n32ufill.json results/ppa_cache_$tag.json
  python scripts/run_search.py --n 32 --budget 500 --pop 32 --seeds $2 \
    --seed0 $1 --algos $3 --nvec 1048576 --gpu --workers $4 --tag $tag \
    >> results/log_$tag.txt 2>&1
}
names=("$@")
[ ${#names[@]} -eq 0 ] && names=(e2 e3 e4 e5 e67 e8 e9 bc04 bc59 lo ga8)
echo "$(date) 32x32 tail started: ${names[*]}" >> results/log_fill.txt
for c in "${names[@]}"; do run_chain $c & done
wait
echo "$(date) 32x32 tail done: ${names[*]}" >> results/log_fill.txt
