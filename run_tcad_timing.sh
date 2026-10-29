#!/bin/bash
# TCAD revision: serial cost of a characterization, measured once no campaign
# is running.  Holds results/time_serial.lock while it measures, which
# run_tcad_resume_llm.sh waits for, so guided runs cannot load the machine
# during the measurement.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
cd "$HERE"
busy() {
  powershell.exe -NoProfile -Command '@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like "python*" -and ($_.CommandLine -like "*run_search.py*" -or $_.CommandLine -like "*run_refgen.py*" -or $_.CommandLine -like "*run_apps*" -or $_.CommandLine -like "*analyze_glitch*") }).Count' | tr -d '\r'
}
until [ "$(busy)" = "0" ]; do sleep 120; done
touch results/time_serial.lock
echo "$(date) machine idle; measuring serial characterization cost" >> results/log_timing.txt
python scripts/time_serial.py --widths 8 16 32 --samples 40 >> results/log_timing.txt 2>&1
rm -f results/time_serial.lock
echo "$(date) done" >> results/log_timing.txt
