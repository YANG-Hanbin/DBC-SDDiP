# Algorithm 3 Validation Matrix (DBC vs iDBC)

This matrix uses 5-iteration debug presets to validate Algorithm 3 behavior
across all three examples.

## Key toggles

- `enable_copy_branching = false` -> DBC path (no copy-variable split).
- `enable_copy_branching = true` + `enable_surrogate_copy_split = true` +
  `copy_split_strategy = :surrogate_delta` -> iDBC path (Algorithm 3 surrogate
  copy split + Delta trigger).

`copy_split_enabled` is kept as a compatibility alias and is mirrored to
`enable_copy_branching`.

## Run commands

| Example | Cut | Config | Command |
|---|---|---|---|
| SCUC | DBC | `experiments/scuc/configs/debug_dbc_sddip_5iter.jl` | `julia --project=. experiments/scuc/run_experiment.jl experiments/scuc/configs/debug_dbc_sddip_5iter.jl` |
| SCUC | iDBC | `experiments/scuc/configs/debug_idbc_sddip_5iter.jl` | `julia --project=. experiments/scuc/run_experiment.jl experiments/scuc/configs/debug_idbc_sddip_5iter.jl` |
| SSLP | DBC | `experiments/sslp/configs/debug_dbc_5iter.jl` | `julia --project=. experiments/sslp/run_experiment.jl experiments/sslp/configs/debug_dbc_5iter.jl` |
| SSLP | iDBC | `experiments/sslp/configs/debug_idbc_5iter.jl` | `julia --project=. experiments/sslp/run_experiment.jl experiments/sslp/configs/debug_idbc_5iter.jl` |
| GEP | DBC | `experiments/generation_expansion/configs/debug_dbc_5iter.jl` | `julia --project=. experiments/generation_expansion/run_experiment.jl experiments/generation_expansion/configs/debug_dbc_5iter.jl` |
| GEP | iDBC | `experiments/generation_expansion/configs/debug_idbc_5iter.jl` | `julia --project=. experiments/generation_expansion/run_experiment.jl experiments/generation_expansion/configs/debug_idbc_5iter.jl` |

## Reference logs (latest run)

- `/tmp/codex_alg3_runs2/scuc_dbc_sddip_5iter_debug.log`
- `/tmp/codex_alg3_runs2/scuc_idbc_sddip_5iter_debug.log`
- `/tmp/codex_alg3_runs2/sslp_dbc_5iter_repo.log`
- `/tmp/codex_alg3_runs2/sslp_idbc_5iter_repo.log`
- `/tmp/codex_alg3_runs2/ge_dbc_5iter_repo.log`
- `/tmp/codex_alg3_runs2/ge_idbc_5iter_repo.log`

## Algorithm 3 code anchors

- SSLP: `src/sslp/utilities/cutting_plane_tree_alg.jl`
  - surrogate copy-split phase before integrality gate: around `# Algorithm 3 order`
- GEP: `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl`
  - surrogate copy-split phase before integrality gate: around `# Algorithm 3 order`
- SCUC: `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl`
  - surrogate copy-split phase before integrality gate: around `# Algorithm 3 order`
