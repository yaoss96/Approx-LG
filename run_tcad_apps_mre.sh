#!/bin/bash
# TCAD revision: applications at matched MRE, once the extended-alphabet MRE
# campaign is complete (36 runs).  Needs the extended alphabet to decode the
# style-E genomes; base-alphabet genomes decode to the same circuits under it.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
count() { [ -f results/search_n8mrex.jsonl ] && grep -c '"algo"' results/search_n8mrex.jsonl || echo 0; }
while [ "$(count)" -lt 36 ]; do sleep 120; done
APPROXLG_STYLES=ext python scripts/run_apps_mre.py --cnn --epochs 4 \
  > results/log_apps_mre.txt 2>&1
