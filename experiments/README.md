# Experiment Layout

This directory centralizes reproducible experiment configuration for all three models:

- `scuc/`: multistage SCUC with SDDP/SDDiP/SDDPL.
- `sslp/`: two-stage SSLP experiments.
- `generation_expansion/`: multistage generation expansion experiments.

Each subdirectory contains:

- `exp_config.jl`: typed configuration definitions and helper builders.
- `configs/*.jl`: concrete experiment presets.
- `run_experiment.jl`: entrypoint that loads one config and runs the full batch.

Paper study presets:

- `scuc/configs/paper_suite/`
- `sslp/configs/paper_suite/`
- `generation_expansion/configs/paper_suite/`
- `paper_suite_runbook.md`

Global config reference:

- `docs/config_reference.md`
- `docs/run_new_experiments.md`
- `docs/result_storage_design.md`
- `docs/naming_migration.md`
- `docs/uc_imdc_stability.md`
- `docs/fenchel_cut_audit.md`
- `docs/split_cut_validation.md`
- `docs/benchmark_extension_assessment.md`
- `docs/bug_fix_report.md`

## Usage

Run with a preset:

```bash
julia --project experiments/scuc/run_experiment.jl experiments/scuc/configs/default.jl
```

Dry-run plan without solving:

```bash
julia --project experiments/scuc/run_experiment.jl experiments/scuc/configs/default.jl --dry-run
```

The same pattern works for `sslp` and `generation_expansion`.

Default presets in `configs/default.jl` are editable templates. Copy one and customize fields directly.

By default, result files are saved under `results/<project>/...`.
Use `results_root`, `experiment_tag`, and `legacy_logger_paths` in `static` config to control output paths.

For iDBC runs, `copy_branch_policy` controls whether CPT branches on copy variables
(`:adaptive`, `:always`, `:never`, `:fallback_only`, `:dominant`, `:blend`).

Fenchel cut is included in the default sweeps (`:FC`).
Split-cut benchmark is included as `:SPC` (aliases: `:Split`, `:SplitCut`).
No separate Fenchel-only pipeline is required: use the same `run_experiment.jl` + config workflow for all benchmark cuts.
