#!/bin/bash
# Start the 32x32 campaign once the same-space chain has written all of its
# signed 16x16 runs (two campaigns at a time is what this machine tolerates).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
count() { [ -f results/search_n16smoo.jsonl ] && grep -c '"algo"' results/search_n16smoo.jsonl || echo 0; }
while [ "$(count)" -lt 40 ]; do sleep 300; done
bash run_tcad_n32.sh
