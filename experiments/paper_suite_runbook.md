# Paper Suite Runbook

This runbook executes the new paper-study experiment presets under:

- `experiments/sslp/configs/paper_suite/`
- `experiments/generation_expansion/configs/paper_suite/`
- `experiments/scuc/configs/paper_suite/`

## 1) SSLP

Run one preset:

```bash
julia --project=. experiments/sslp/run_experiment.jl experiments/sslp/configs/paper_suite/exp1_idbc_dynamic_inherit_off.jl
```

Run all SSLP paper-suite presets:

```bash
for f in experiments/sslp/configs/paper_suite/*.jl; do
  julia --project=. experiments/sslp/run_experiment.jl "$f"
done
```

## 2) Generation Expansion

Run one preset:

```bash
julia --project=. experiments/generation_expansion/run_experiment.jl experiments/generation_expansion/configs/paper_suite/exp1_idbc_dynamic_inherit_on_snc.jl
```

Run all generation-expansion paper-suite presets:

```bash
for f in experiments/generation_expansion/configs/paper_suite/*.jl; do
  julia --project=. experiments/generation_expansion/run_experiment.jl "$f"
done
```

## 3) SCUC

Run one preset:

```bash
julia --project=. experiments/scuc/run_experiment.jl experiments/scuc/configs/paper_suite/exp1_idbc_dynamic_inherit_on_snc.jl
```

Run all SCUC paper-suite presets:

```bash
for f in experiments/scuc/configs/paper_suite/*.jl; do
  julia --project=. experiments/scuc/run_experiment.jl "$f"
done
```

## Notes

- Experiment 1 compares iDBC with and without cut inheritance across normalizations.
- Experiment 2 uses iDBC with fixed `D` via `disjunction_iteration_limit = 10`.
- Experiment 3 runs benchmark cuts:
  - SSLP: `:BC`, `:LC`, `:SMC`
  - Generation Expansion: `:BC`, `:LC`, `:SMC` (BC is routed as DBC with `D=0`)
  - SCUC: `:BC`, `:LC`, `:SMC` (BC is routed as DBC with `D=0`)
- Experiment 4 runs DBC with inheritance and a `BEST_NORMALIZATION` placeholder in each config.
