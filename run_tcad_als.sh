#!/bin/bash
# TCAD revision: ALS baseline.  ALSRAC (Meng et al., ICCAD'20; public code of
# SJTU-ECTL/ALSRAC, built in WSL at ~/ALSRAC, see README) simplifies the exact
# multiplier of our template under an NMED bound and writes the current
# circuit as it goes (every 10 rounds, every round above 1% of the bound).
# One greedy run per bound: 2e-4 for the accurate end of the window, 2e-2 for
# all of it.  Default options (-f 64, mcnc.genlib); the run seeds itself from
# std::random_device.  Wall-clock caps: 8 h at 8x8, 10 h at 16x16.
# scripts/run_als.py then characterizes every written circuit in our flow.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
echo "$(date) ALS runs started" >> results/log_als.txt
for n in 8 16; do
  for b in 2e-4 2e-2; do
    cap=$([ $n = 8 ] && echo 28800 || echo 36000)
    wsl.exe -e bash -lc "cd ~/ALSRAC && mkdir -p runs/n$n/nmed_$b && \
      timeout $cap ./main -i runs/exact$n.blif -m NMED -b $b -f 64 \
        -o runs/n$n/nmed_$b/ > runs/n$n/nmed_$b.log 2>&1; \
      echo \"\$(date) n=$n bound=$b exit \$?\" >> runs/done.txt" &
  done
done
wait
echo "$(date) ALS runs done" >> results/log_als.txt
