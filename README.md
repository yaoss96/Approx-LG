# Approx-LG — LLM-Guided Design Space Exploration for Approximate Multipliers

Approx-LG couples a large language model with NSGA-II to explore the
accuracy/area/delay trade-off of approximate multipliers.  The LLM reads the
current Pareto front together with the measured error and PPA feedback, and
then proposes candidate designs, steers mutation and refines the search space;
every point it suggests is still evaluated by the same RTL simulation and logic
synthesis flow as every other point, so the comparison against Random Search,
GA, NSGA-II and LLM-only runs is made under an identical EDA budget.

This repository holds everything needed to reproduce the paper: the multiplier
generator, the evaluation flow, the search algorithms, the baselines, the
analysis scripts and the LaTeX sources.

## Layout

```
approxlg/           the framework
  cells.py          approximate compressor library (Boolean expressions that
                    are valid NumPy *and* Verilog at the same time)
  arch.py           genome -> structural netlist -> Verilog
                    (AND array / Baugh-Wooley, Dadda tree, LOA final adder)
  simulate.py       bit-packed functional simulator (CPU or GPU)
  synth.py          Yosys + ABC + Nangate45 PPA characterisation
  power.py          activity-based power on the mapped netlist
  liberty.py        minimal Liberty reader (area, leakage, caps, functions)
  netjson.py        gate-level evaluator for third-party Verilog
  metrics.py        dominance, crowding distance, hypervolume
  engine.py         evaluation pipeline + EDA-budget accounting
  llm.py            DeepSeek client and the four guidance operators
  search.py         random / GA / NSGA-II / LLM-only / Approx-LG
  baselines.py      re-implementations of EA-MVA and the biased-compressor GA
scripts/            experiment drivers and analysis
paper/              LaTeX sources, generated figures and tables
results/            raw measurements (jsonl) and caches
```

## Getting started

```bash
git clone <this repository>
cd Approx-LG
python -m pip install numpy cupy-cuda12x requests matplotlib scikit-image \
                      torch torchvision
```

Three external pieces are not redistributed here and have to be put in place
first.

**1. Native EDA tools.**  `yosys`, `abc` and `iverilog` are taken from a local
[OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build/releases)
checkout.  Extract it into `oss-cad-suite/` inside the clone; `env.sh` puts it
on `PATH`.  A system-wide installation works as well, in which case `env.sh` is
not needed.

**2. Standard cells.**  Copy the Nangate45 Liberty file to
`lib/NangateOpenCellLibrary_typical.lib` — see [`lib/README.md`](lib/README.md).

**3. Reference libraries**, for the published-design comparison.  They are
expected *next to* the clone:

```
<parent>/
  Approx-LG/          this repository
  AM-Lib-main/        AM-Lib   (Wu et al., ACM TODAES 2023, doi 10.1145/3610291)
  MulApprox-main/     https://github.com/RatkoFri/MulApprox
  evoapproxlib/       https://github.com/ehw-fit/evoapproxlib
```

`AM-Lib-main/` and `MulApprox-main/` keep the directory names that GitHub's
"download ZIP" produces, i.e. the sources sit one level further down
(`AM-Lib-main/AM-Lib-main/`).

`DEEPSEEK_API_KEY` must be set for the LLM-driven algorithms; without it the
guided operators degrade to plain NSGA-II.  A GPU is used for 16-bit
simulation (`--gpu`) and for the CNN application study.

## Reproducing

```bash
source env.sh

# correctness: functional model == RTL == mapped netlist
python scripts/verify_rtl.py 8 8      # 8x8, unsigned and signed
python scripts/verify_rtl.py 6 16     # 16x16, unsigned and signed

# search campaigns
python scripts/run_search.py --n 8  --budget 400 --seeds 10 \
       --algos nsga2 random ga eamva biased_ga --tag n8
python scripts/run_search.py --n 8  --budget 400 --seeds 10 \
       --algos approxlg llm_only --tag n8L
python scripts/run_search.py --n 8  --budget 400 --seeds 5 \
       --algos approxlg_no_warm approxlg_no_inject approxlg_no_mutguide \
               approxlg_no_refine --tag n8abl
python scripts/run_search.py --n 16 --budget 500 --seeds 4 --gpu \
       --nvec 4194304 --algos approxlg nsga2 random --tag n16u
python scripts/run_search.py --n 16 --budget 500 --seeds 4 --gpu --signed \
       --nvec 4194304 --algos approxlg nsga2 random --tag n16s

# published libraries (AM-Lib, MulApprox, EvoApproxLib), identical flow
python scripts/run_libs.py --n 16 --nvec 4194304 --gpu
python scripts/run_libs.py --n 8                    # exhaustive at 8 bits
python scripts/verify_libs.py --n 16   # RTL cross-check, gates the comparison
python scripts/verify_libs.py --n 8

# applications, ours next to a published design of matching accuracy
python scripts/pick_app_designs.py --tags n8L --algos approxlg
python scripts/pick_app_refs.py
python scripts/run_apps.py --refs results/app_refs.json --cnn --epochs 4

# figures, tables, macros and the PDF
python scripts/finalize.py             # -> paper/main_build.pdf
```

`rerun8.sh`, `rerunL.sh` and `run_rest.sh` are the campaign drivers as they
were actually run.

Run at most two campaigns concurrently: on Windows a heavily oversubscribed
machine makes Yosys fail to spawn, and although a failed characterisation is
now retried and never cached, the retries cost time.

## Data in this repository

`results/` ships the measurements the paper is built from, so every figure and
table can be regenerated without re-running the EDA flow:

* `search_*.jsonl` — one record per search step (algorithm, seed, EDA budget
  spent, Pareto front), including the quarantined `*_dead.jsonl` runs kept for
  the provenance audit
* `libs_*.jsonl`, `libs_verify*.json` — the published designs characterised in
  our flow, and their bit-exactness cross-check
* `exact_*.json`, `ppa_sweep_*.json`, `apps.json`, `app_refs.json`,
  `cnn_acc.json`, `verify_rtl_*.txt` — reference points, synthesis-recipe
  sweeps, application results and the correctness logs

Four kinds of file are deliberately left out because the scripts rebuild them
and they are large: the per-design evaluation caches (`designs_*.jsonl`), the
raw LLM transcripts (`llm_*.jsonl`), the synthesis caches (`ppa_cache_*.json`)
and the gate-level JSON of third-party RTL (`results/libjson/`).  Re-running a
campaign regenerates whichever of them it needs.

## Notes

* The EDA budget counts *distinct circuits*, identified by the SHA-1 of the
  emitted netlist, so an algorithm cannot buy extra evaluations by proposing
  genomes that decode to the same design.
* A design point is dropped, never cached, if its synthesis fails.
* Third-party designs enter the comparison only if their gate-level evaluation
  reproduces their own RTL bit-exactly (`scripts/verify_libs.py`) and their
  NMED stays in (0, 10 %].  Exact multipliers are excluded because they are not
  approximate designs.
* `results/libjson/` holds one netlist per design name and is shared by the
  8- and 16-bit runs, so re-run `scripts/run_libs.py` for a width before
  verifying that width.

## TCAD version (`paper/tcad/`)

The journal version adds reproduced compressor-tree baselines, stronger
same-space optimisers, a 32x32 width, an error-metric retargeting experiment
and several analyses.  New modules:

```
approxlg/ctree.py   compressor-tree multipliers of the reference methods
                    (Esposito et al. TCAS-I'18 cells and allocation rule,
                    Wang et al. ISCAS'20 strategy, generic stage builder)
approxlg/opact.py   OPACT (DATE'22): allocation ILP in its three modes and the
                    connection-order objective (SCIP; Gurobi optional)
approxlg/moo.py     SMS-EMOA and batch MOBO (GP + Monte-Carlo EHVI)
```

`approxlg.engine.set_error_metric("MRE")` (or `run_search.py --error-metric
MRE`) retargets the objective vector, the decade elites and the prompts from
NMED to MRE; NMED behaviour is byte-identical to the earlier campaigns.

```bash
source env.sh
# reference designs, characterised in the same flow, and their check
# against the published error figures
python scripts/run_refgen.py --n 8
python scripts/run_refgen.py --n 16 --gpu --nvec 4194304 --extra-stages 3
python scripts/analyze_repro.py
# campaigns as run for the revision
bash run_tcad_abl16.sh      # operator ablation at 16x16
bash run_tcad_moo.sh        # SMS-EMOA and MOBO at 8x8 and 16x16 u/s
bash run_tcad_mre.sh        # MRE-objective retargeting at 8x8
bash run_tcad_mrex.sh       # the same with the carry-free cells as style E
bash run_tcad_n32.sh        # 32x32 unsigned
bash run_tcad_hyb.sh        # placebo of the guidance-on-SMS-EMOA hybrid
bash run_tcad_resume_llm.sh # guided arms after an endpoint outage: waits
                            # for scripts/api_ok.py, re-runs missing seeds
# serial timing (on an otherwise idle machine), analyses, figures, prompts;
# analyze_tcad.py reads the outputs of the four scripts before it
python scripts/time_serial.py --widths 8 16 32
python scripts/analyze_mechanisms.py
python scripts/analyze_glitch.py
APPROXLG_STYLES=ext python scripts/run_apps_mre.py --cnn --epochs 4
python scripts/analyze_tcad.py
python scripts/make_tcad_figs.py
python scripts/make_prompts_tex.py
# the scripts above write their macros, tables and figures into paper/tcad/;
# the manuscript sources themselves are not part of this repository.  With
# them in place (Fig. 1 is drawn by hand in paper/tcad/template.drawio):
draw.io --export --format pdf --crop --output paper/tcad/figs/template.pdf paper/tcad/template.drawio
cd paper/tcad && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The OPACT ILP and connection-order model are solved with SCIP through
`pyscipopt`; `opact.allocate(..., solver="gurobi")` is the hook for a licensed
Gurobi.  The 32x32 template simulates 64-bit products in unsigned 64-bit
arithmetic (`simulate.Stimulus.wide`), checked against Icarus Verilog.

**ALS baseline.**  ALSRAC (Meng et al., ICCAD'20) is run from its public code,
built in WSL (Ubuntu 24.04, GCC 13).  The sources predate GCC 13 and the
current Boost, hence the two extra flags; `scripts/alsrac_compat.h` only
includes standard headers.

```bash
sudo apt-get install -y build-essential libreadline-dev libncurses-dev libboost-dev universal-ctags
git clone https://github.com/SJTU-ECTL/ALSRAC.git ~/ALSRAC && cd ~/ALSRAC
git clone https://github.com/changmg/abc.git && (cd abc && make -j16 libabc.a)
git clone https://github.com/changmg/espresso.git && (cd espresso && make lib)
make -j16 FLAG="-DBOOST_TIMER_ENABLE_DEPRECATED -include <repo>/scripts/alsrac_compat.h"
```

Then, from the repository (`run_als.py` expects the WSL home at `D:/eda/home`;
`--alsrac` points it elsewhere):

```bash
python scripts/run_als.py --prep --n 8      # the template's exact multiplier as BLIF
python scripts/run_als.py --prep --n 16
bash run_tcad_als.sh                        # ALSRAC under NMED bounds 2e-4 and 2e-2
python scripts/run_als.py --n 8             # every circuit it wrote, in our flow
python scripts/run_als.py --n 16 --gpu
```

On Windows, keep the repository on a path without non-ASCII characters (or
reach it through a directory junction): Yosys cannot locate its share
directory otherwise.

## Licence

MIT, see [LICENSE](LICENSE).  The reference libraries, the Nangate45 cell
library and the OSS CAD Suite are covered by their own licences and are not
redistributed here.
