# DBC-SDDiP

Codebase for stochastic dual dynamic programming with disjunctive strengthening (DBC/iDBC) and cutting-plane-tree (CPT) variants.

This public repository is **code-only**:
- Included: source code, experiment configs, notebooks, docs.
- Excluded: generated results and binary datasets (`*.jld2`).

## 1. Requirements

- Julia `1.11.x`
- Gurobi + valid license
- macOS/Linux shell

Install dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## 2. Repository layout

- `src/multistage_SCUC`: multistage SCUC models and SDDP/SDDiP implementations
- `src/multistage_generation_expansion`: GEP models and algorithms
- `src/sslp`: SSLP models and algorithms
- `src/common`: shared runtime/branching utilities
- `experiments`: experiment configs + run entrypoints
- `experiments/plotting/plot_results.jl`: unified plotting script for saved runs
- `notebooks`: interactive quick-run and paper-suite notebooks
- `docs`: curated notes and algorithm documentation

## 3. Data expectations (code-only release)

Runtime scripts load prebuilt `.jld2` files from these paths:

- SCUC:
  - `src/multistage_SCUC/experiment_<case>/initialStateInfo.jld2`
  - `src/multistage_SCUC/experiment_<case>/stage(<T>)real(<R>)/{indexSets,paramOPF,paramDemand,scenarioTree}.jld2`
- GEP:
  - `src/multistage_generation_expansion/testData/stage(<T>)real(<R>)/{stageDataList,Ω,binaryInfo,probList}.jld2`
- SSLP:
  - `src/sslp/testData/J<J>-I<I>-Ω<Omega>/{stageData,randomVariables}.jld2`

If these files are missing, use your private dataset bundle or regenerate data with your internal pipeline.

## 4. Run experiments

SCUC:

```bash
julia --project=. experiments/scuc/run_experiment.jl experiments/scuc/configs/default.jl
```

GEP:

```bash
julia --project=. experiments/generation_expansion/run_experiment.jl experiments/generation_expansion/configs/default.jl
```

SSLP:

```bash
julia --project=. experiments/sslp/run_experiment.jl experiments/sslp/configs/default.jl
```

Dry-run (config + orchestration path, no solve):

```bash
julia --project=. experiments/scuc/run_experiment.jl experiments/scuc/configs/default.jl --dry-run
```

## 5. Plot saved results

Example:

```bash
julia --project=. experiments/plotting/plot_results.jl \
  --project scuc \
  --dataset-filter 'case=case_RTS_GMLC__T=6__R=5' \
  --run-filter 'cut=DBC|cut=iDBC' \
  --format png
```

Outputs are written to `results/figures/<project>/<dataset>/`.

## 6. License

MIT License (see `LICENSE`).
