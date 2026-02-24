# DBC-SDDiP

A research codebase for stochastic dual dynamic programming with disjunctive cuts and cutting-plane-tree variants.

Note: this repository intentionally excludes experimental results and datasets.

## Repository structure

- `src/multistage_SCUC`: multistage SCUC model and SDDP/SDDiP/SDDPL implementations.
- `src/sslp`: two-stage SSLP model and cut-generation variants.
- `src/multistage_generation_expansion`: multistage generation expansion model.
- `src/common`: shared utilities (for example, branching policy selection).
- `experiments`: standardized experiment configs and entrypoints.
- `docs`: design notes and architecture documents.

## Experiment entrypoints

Configuration reference: `docs/config_reference.md`  
Run guide: `docs/run_new_experiments.md`

Run SCUC with default config:

```bash
julia --project experiments/scuc/run_experiment.jl experiments/scuc/configs/default.jl
```

Run SSLP with default config:

```bash
julia --project experiments/sslp/run_experiment.jl experiments/sslp/configs/default.jl
```

Run generation expansion with default config:

```bash
julia --project experiments/generation_expansion/run_experiment.jl experiments/generation_expansion/configs/default.jl
```

Dry-run (configuration and data loading only):

```bash
julia --project experiments/scuc/run_experiment.jl experiments/scuc/configs/default.jl --dry-run
```

Notebook workflow: `notebooks/experiment_workflow.ipynb`

## Entry points

Use the experiment runners directly:

- `experiments/scuc/run_experiment.jl`
- `experiments/sslp/run_experiment.jl`
- `experiments/generation_expansion/run_experiment.jl`
