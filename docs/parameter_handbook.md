# Parameter Handbook

This document is the canonical parameter guide for all three examples:

1. SCUC (`experiments/scuc`)
2. Generation Expansion (`experiments/generation_expansion`)
3. SSLP (`experiments/sslp`)

It covers:

- meaning of each parameter,
- where it is consumed,
- legacy aliases,
- practical combinations.

---

## 1. Parameter flow (all projects)

All projects follow the same pipeline:

1. `experiments/*/configs/*.jl`: define `EXPERIMENT_CONFIG`.
2. `experiments/*/exp_config.jl`: transform config into `param`, `param_cut`, `param_levelset`.
3. `src/*/utilities/utils.jl`: `param_setup(...)` + `resolve_*_runtime_params(...)`.
4. `src/*/algorithm.jl` or `src/multistage_SCUC/sddp.jl`: SDDP loop consumes runtime params.
5. `src/*/backward_pass.jl` + `src/*/utilities/cutting_plane_tree_alg.jl`: cut-specific logic (DBC/iDBC/SPC/FC/etc.).

The `resolve_*_runtime_params(...)` layer is the effective source of truth for what algorithm code actually reads.

---

## 2. Cut-name semantics and aliases

Canonical names:

- `:BC`
- `:LC`
- `:SMC`
- `:SBC`
- `:FC`
- `:DBC`
- `:iDBC`
- `:SPC`

Accepted aliases and normalization:

- `:MDC -> :DBC`
- `:iMDC -> :iDBC`
- `:Split -> :SPC`
- `:SplitCut -> :SPC`
- SCUC also accepts `:CPT -> :DBC`

Meaning:

- `DBC`: disjunctive Benders cut (disjunctive strengthening on original mixed-integer subproblem variables).
- `iDBC`: DBC with copy-variable branching enabled.
- `SPC`: one split-disjunction round benchmark (`MDCiter = 1` behavior).

---

## 3. Cross-project core controls

These fields exist in all three projects (possibly with legacy names):

| Canonical field | Purpose | Typical values |
|---|---|---|
| `cut_selection` | benchmark/cut family | `:BC/:LC/:SMC/:SBC/:FC/:DBC/:iDBC/:SPC` |
| `algorithm` | main algorithm mode | project-specific symbols |
| `disjunction_iteration_limit` | max DBC rounds per backward subproblem (`<0` dynamic in some paths) | `-1`, `1`, `5`, `10` |
| `inherit_disjunctive_cuts` | reuse previously generated disjunctive cuts | `true/false` |
| `enforce_binary_copies` | enforce binary copy-state variables | `true/false` |
| `branch_selection_strategy` | variable selector for CPT split | `:MFV`, `:LFV`, `:Random`, `:First`, `:ML` |
| `time_limit` | per-model solver time limit | positive `Float64` |
| `max_iter` | outer SDDP iterations | positive `Int` |
| `terminate_time` | wall-clock stop seconds | positive `Float64` |
| `terminate_threshold` | gap threshold | small positive `Float64` |

Copy-split controls (for Algorithm 3 / iDBC path):

| Field | Purpose |
|---|---|
| `enable_copy_branching` | master switch for copy-variable branching in iDBC |
| `enable_surrogate_copy_split` | enable surrogate-Delta pre-split stage (Algorithm 3 style) |
| `copy_split_strategy` | current strategy key (`:surrogate_delta` is canonical) |
| `copy_split_delta_tol` | minimum Delta to accept surrogate-copy split |
| `copy_split_enabled` | legacy alias of `enable_copy_branching` |
| `idbc_warm_pass` | optional warm backward pass before normal backward pass in iDBC |

Copy-branch pool controls:

| Field | Purpose |
|---|---|
| `copy_branch_policy` | policy for mixing copy vs primary candidates |
| `copy_branch_min_deviation` | min copy deviation for candidacy |
| `copy_branch_dominance_ratio` | copy-vs-primary max deviation trigger |
| `copy_branch_mean_ratio` | copy-vs-primary mean deviation trigger |
| `copy_branch_boost` | copy score multiplier in blend mode |
| `branching_ml_weights` | linear weights used when `:ML` strategy is selected |

---

## 4. SCUC parameter map

Source: `experiments/scuc/exp_config.jl`

### 4.1 `ScucStaticConfig`

| Field | Meaning |
|---|---|
| `case_name` | dataset folder key under `src/multistage_SCUC/experiment_*` |
| `tightness` | legacy alias of copy integrality enforcement |
| `enforce_binary_copies` | enforce binary copy variables in model/state |
| `inherit_mdc` | legacy alias of disjunctive-cut inheritance |
| `inherit_disjunctive_cuts` | persist/reuse disjunctive cuts |
| `imdc_state_projection` | sanitize projected iDBC warm states before backward |
| `algorithm_cpt_method` | CPT implementation mode (e.g., `:modifiedCPT`) |
| `var_selection` | legacy branch strategy selector |
| `branch_selection_strategy` | canonical branch strategy selector |
| `copy_branch_*` | copy-vs-primary candidate pool behavior |
| `enable_copy_branching` | master copy branching switch |
| `enable_surrogate_copy_split` | enable Algorithm-3 surrogate Delta stage |
| `copy_split_enabled` | legacy alias of `enable_copy_branching` |
| `copy_split_strategy` | copy split policy key |
| `copy_split_delta_tol` | surrogate Delta acceptance threshold |
| `copy_split_max_candidates` | cap number of surrogate copy candidates (UC only) |
| `copy_split_min_violation` | min copy fractional violation before evaluation (UC only) |
| `idbc_warm_pass` | optional iDBC warm backward sweep |
| `branching_ml_weights` | weights for ML branch scorer |
| `branch_variable` | SDDPL partition-tree branching selector |
| `branch_threshold` | SDDPL branch trigger threshold |
| `normalization` | CGLP normalization type |
| `mdc_iter` | legacy disjunction iteration limit |
| `disjunction_iteration_limit` | canonical disjunction iteration limit |
| `logger_save` | save run history |
| `results_root` | output root override |
| `legacy_logger_paths` | mirror to old logger paths |
| `experiment_tag` | run id tag |
| `num_scenarios` | forward sampled scenarios |
| `num_backward_scenarios` | scenarios used in backward cut generation |
| `num_partition_scenarios` | scenarios used for SDDPL partition updates |
| `epsilon` | SDDiP binarization precision |
| `terminate_time` | wall-clock stop |
| `per_mip_time_limit` | legacy solver per-MIP limit |
| `time_limit` | canonical solver per-MIP limit |
| `terminate_threshold` | stopping gap threshold |
| `max_iter` | max SDDP iterations |
| `lift_iter_threshold` | SDDPL partition-start iteration |
| `mip_focus`/`numeric_focus`/`feasibility_tol`/`mip_gap` | solver controls |

### 4.2 `ScucSweepConfig`

| Field | Meaning |
|---|---|
| `algorithms` | outer algorithm sweep (`:SDDPL/:SDDP/:SDDiP`) |
| `cuts` | cut sweep symbols |
| `periods` | number of stages |
| `realizations` | number of realizations |

### 4.3 `ScucCutConfig` and `ScucLevelSetConfig`

| Field | Meaning |
|---|---|
| `core_point_strategy` | core-point mode used by cut generation |
| `delta`/`ell` | cut/level parameters |
| `mu`/`lambda`/`threshold`/`next_bound`/`max_iter`/`verbose` | level-set controls |

---

## 5. Generation Expansion parameter map

Source: `experiments/generation_expansion/exp_config.jl`

### 5.1 `GeStaticConfig`

| Field | Meaning |
|---|---|
| `enhancement` | additional enhancement toggle |
| `tightness` / `enforce_binary_copies` | copy integrality enforcement |
| `sample_count` | forward Monte Carlo path count |
| `num_backward_samples` | sampled paths used for backward sweep (`-1` means all sampled paths) |
| `count_benders_cut` | whether to track Benders-cut counting mode |
| `opt` | reference OPT value for logs |
| solver controls | `feasibility_tol`, `mip_focus`, `numeric_focus`, `mip_gap` |
| `default_mdc_iter` / `disjunction_iteration_limit` | DBC rounds |
| branching controls | `var_selection`, `branch_selection_strategy`, copy branch parameters |
| copy split controls | `enable_copy_branching`, `enable_surrogate_copy_split`, `copy_split_strategy`, `copy_split_delta_tol` |
| `idbc_warm_pass` | optional iDBC warm backward pass |
| `branching_ml_weights` | ML branch weights |
| `algorithm` | e.g., `:modifiedCPT` |
| `normalization` | CGLP normalization |
| `inherit_mdc` / `inherit_disjunctive_cuts` | disjunctive-cut inheritance |
| `improvement` | improvement mode toggle |
| stop controls | `terminate_time`, `terminate_threshold`, `max_iter` |
| time limit | `time_limit` (legacy), `time_limit` (canonical) |
| `epsilon` | algorithm precision parameter |
| output controls | `results_root`, `legacy_logger_paths`, `experiment_tag`, `logger_save` |

### 5.2 `GeSweepConfig`, `GeCutConfig`, `GeLevelSetConfig`

| Struct | Key fields |
|---|---|
| `GeSweepConfig` | `cuts`, `realizations`, `periods` |
| `GeCutConfig` | `delta`, `ell1`, `ell2`, `epsilon` |
| `GeLevelSetConfig` | `mu`, `lambda`, `threshold`, `next_bound`, `max_iter`, `verbose` |

---

## 6. SSLP parameter map

Source: `experiments/sslp/exp_config.jl`

### 6.1 `SslpStaticConfig`

| Field | Meaning |
|---|---|
| `opt` | reference objective for logs |
| solver controls | `feasibility_tol`, `mip_focus`, `numeric_focus`, `mip_gap` |
| `default_mdc_iter` / `disjunction_iteration_limit` | DBC rounds |
| branching controls | `var_selection`, `branch_selection_strategy`, copy branch parameters |
| copy split controls | `enable_copy_branching`, `enable_surrogate_copy_split`, `copy_split_strategy`, `copy_split_delta_tol` |
| `copy_split_enabled` | legacy alias of `enable_copy_branching` |
| `branching_ml_weights` | ML branch weights |
| `normalization` | CGLP normalization |
| `algorithm` | e.g., `:CPT` |
| `cut_warm_start` | legacy warm-pass flag |
| `idbc_warm_pass` | canonical warm-pass flag |
| `inherit_mdc` / `inherit_disjunctive_cuts` | disjunctive-cut inheritance |
| `tightness` / `enforce_binary_copies` | copy integrality enforcement |
| stop controls | `terminate_time`, `terminate_threshold`, `max_iter` |
| time limit | `time_limit` (legacy), `time_limit` (canonical) |
| `theta_lower` | lower bound for theta |
| `epsilon` | precision parameter |
| output controls | `results_root`, `legacy_logger_paths`, `experiment_tag`, `logger_save` |

### 6.2 `SslpSweepConfig`, `SslpCutConfig`, `SslpLevelSetConfig`

| Struct | Key fields |
|---|---|
| `SslpSweepConfig` | `omegas`, `problem_sizes`, `base_cuts`, `mdc_iters`, `mdc_cuts`, normalization sweep |
| `SslpCutConfig` | `delta`, `ell1`, `ell2`, `epsilon` |
| `SslpLevelSetConfig` | `mu`, `lambda`, `threshold`, `next_bound`, `max_iter`, `verbose` |

---

## 7. Effective behavior combinations

### 7.1 DBC baseline

- `cut_selection = :DBC`
- `enable_copy_branching = false` (or true; ignored because `cut_selection` is not `:iDBC`)
- `disjunction_iteration_limit >= 1` (or dynamic if negative)

Effect:

- CPT branches only on primary mixed-integer variables.
- Disjunctive cuts strengthen LP, then duals produce Benders cut.

### 7.2 iDBC (Algorithm 3 style)

Required:

- `cut_selection = :iDBC`
- `enable_copy_branching = true`
- `enable_surrogate_copy_split = true`
- `copy_split_strategy = :surrogate_delta`

Recommended:

- `copy_split_delta_tol = 1e-6` (tune upward for fewer surrogate splits)
- `idbc_warm_pass = false` as strict baseline; enable if you want a two-pass variant

Effect:

1. surrogate Delta evaluation over copy variables,
2. optional copy-variable disjunctive cut,
3. regular incumbent-separation DBC loop,
4. dual extraction for final Benders cut.

### 7.3 SPC benchmark

- `cut_selection = :SPC`
- backward pass forces single disjunction round (`MDCiter = 1`)

Effect:

- split-cut style benchmark via one-round disjunctive strengthening.

### 7.4 FC benchmark

- `cut_selection = :FC`

Effect:

- Fenchel separation path on convex-hull approximation,
- then dual-based Benders extraction.

---

## 8. Important compatibility rules

1. If `cut_selection = :iDBC` and `enable_copy_branching = false`, runtime normalization degrades effective behavior to `DBC` semantics.
2. Use canonical names in new configs; legacy names remain accepted but should be treated as compatibility only.
3. Prefer setting both high-level and canonical fields consistently to avoid ambiguity in custom scripts.

---

## 9. Minimal config examples

### 9.1 iDBC with surrogate copy split (SSLP)

```julia
EXPERIMENT_CONFIG = SslpExperimentConfig(
    static = SslpStaticConfig(
        algorithm = :CPT,
        cut_selection = :iDBC,   # via run grid or custom run case
        enable_copy_branching = true,
        enable_surrogate_copy_split = true,
        copy_split_strategy = :surrogate_delta,
        copy_split_delta_tol = 1e-6,
        disjunction_iteration_limit = 5,
    ),
)
```

### 9.2 Force DBC behavior from an iDBC preset

```julia
static = ScucStaticConfig(
    enable_copy_branching = false,  # runtime degenerates iDBC branch behavior to DBC
)
```

### 9.3 SPC benchmark

```julia
static = GeStaticConfig(
    disjunction_iteration_limit = 1,
)
# run case cut_selection = :SPC
```
