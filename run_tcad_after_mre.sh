#!/bin/bash
# Start the 32x32 campaign once the MRE retargeting campaign is complete
# (48 runs), keeping three campaigns on the machine at most.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
count() { [ -f results/search_n8mre.jsonl ] && grep -c '"algo"' results/search_n8mre.jsonl || echo 0; }
while [ "$(count)" -lt 48 ]; do sleep 120; done
bash run_tcad_n32.sh
