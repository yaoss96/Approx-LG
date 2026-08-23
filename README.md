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

## Licence

MIT, see [LICENSE](LICENSE).  The reference libraries, the Nangate45 cell
library and the OSS CAD Suite are covered by their own licences and are not
redistributed here.
