# DBC-SDDiP

Codebase for stochastic dual dynamic programming with disjunctive strengthening (DBC/iDBC) and cutting-plane-tree (CPT) variants.

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

## 3. Included test datasets

The repository already includes test datasets in these paths:

- SCUC:
  - `src/multistage_SCUC/data/`
  - `src/multistage_SCUC/experiment_case_RTS_GMLC/`
- GEP:
  - `src/multistage_generation_expansion/testData/`
- SSLP:
  - `src/sslp/testData/`

With these datasets present, default experiment configs can run directly.

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
