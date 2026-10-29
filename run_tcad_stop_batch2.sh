#!/bin/bash
# TCAD revision: stop the resume chain before the second 16x16 ablation batch
# (seeds 10-19).  Ten seeds per variant answer the reviewer; the extra batch
# would cost about six hours and most of the remaining endpoint credit.
# Waits for the hybrid's guided arm (20 runs), then terminates the batch's
# process as soon as it appears (and nothing else).
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
cnt() { local c=0; [ -f results/search_n16uhyb.jsonl ] && c=$(grep -c '"algo": "approxlg_sms"' results/search_n16uhyb.jsonl); echo "${c:-0}"; }
until [ "$(cnt)" -ge 20 ]; do sleep 60; done
for i in $(seq 1 60); do
  n=$(powershell.exe -NoProfile -Command '$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and $_.CommandLine -like "*run_search.py*" -and $_.CommandLine -like "*--seed0 10*" -and $_.CommandLine -like "*n16uabl*" }); $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; $p.Count' | tr -d '\r')
  if [ "$n" != "0" ]; then
    echo "$(date) stopped the second ablation batch ($n process)" >> results/log_resume_llm.txt
    exit 0
  fi
  sleep 10
done
echo "$(date) second ablation batch never started" >> results/log_resume_llm.txt
