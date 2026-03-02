# Paper-Code Alignment Report (Algorithm 1/2/3)

Date: 2026-02-14

## Scope

This report compares:

1. Paper Algorithm 1 (`alg:scdp`) in `SCDiP/sec2_MSMIP.tex`
2. Paper Algorithm 2 (`alg:cptalgorithm`) in `SCDiP/sec4_cpt.tex`
3. Paper Algorithm 3 (`alg:satds`) in `SCDiP/sec4_cpt.tex`

against current implementations in:

- SCUC: `src/multistage_SCUC/sddp.jl`, `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl`
- SSLP: `src/sslp/algorithm.jl`, `src/sslp/utilities/cutting_plane_tree_alg.jl`
- GEP: `src/multistage_generation_expansion/algorithm.jl`, `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl`

## A. Confirmed Matches

### A1. Algorithm 1 outer structure

Paper: sample forward paths, compute statistical UB, run backward pass by stage and child nodes, inject Benders cuts.

Code matches:

- SCUC loop structure: `src/multistage_SCUC/sddp.jl:141`
- SSLP loop structure: `src/sslp/algorithm.jl:69`
- GEP loop structure: `src/multistage_generation_expansion/algorithm.jl:66`

All three do:

1. Forward simulation under current cut approximation.
2. UB/LB update.
3. Backward cut generation and aggregation.
4. Add stage cuts and continue iterations.

### A2. Algorithm 2 (CPT on primary mixed-integer variables)

Paper: split leaf boxes on fractional integer components of primary variables, solve CGLP, append MDC.

Code matches:

- Tree update (`find_node`, `update_cutting_plane_tree`) and CGLP setup in all three CPT files.
- Standard primary split + MDC loop in `CPT_optimization!`:
  - SSLP: `src/sslp/utilities/cutting_plane_tree_alg.jl:1033`
  - SCUC: `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:1362`
  - GEP: `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:1087`

### A3. Algorithm 3 (structure-aware copy-variable split)

Paper key steps (Section 4.2):

1. Compute surrogate Delta per copy coordinate.
2. Choose `j* = argmax Delta`.
3. If `Delta > eps`, split on copy variable and add an MDC cutting the surrogate point.
4. Re-solve anchored LP.
5. Continue standard incumbent-point split/cut.

Code matches this sequence:

- SSLP surrogate stage in `CPT_optimization!`: `src/sslp/utilities/cutting_plane_tree_alg.jl:908`
- SCUC surrogate stage in `CPT_optimization!`: `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:1213`
- GEP surrogate stage in `CPT_optimization!`: `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:969`

All three use helper functions to:

1. Temporarily replace nonanticipativity with surrogate edge interpolation constraints.
2. Solve surrogate LP and compute Delta.
3. Restore anchored nonanticipativity.

## B. Intentional Engineering Extensions (Not in paper pseudocode)

1. `idbc_warm_pass` optional extra backward sweep before standard backward sweep.
- SCUC: `src/multistage_SCUC/sddp.jl:251`
- SSLP: `src/sslp/algorithm.jl:106`
- GEP: `src/multistage_generation_expansion/algorithm.jl:121`

2. Runtime downgrade `:iDBC -> :DBC` when copy branching is disabled.
- SCUC: `src/multistage_SCUC/utils.jl:430`
- SSLP: `src/sslp/utilities/utils.jl:330`
- GEP: `src/multistage_generation_expansion/utilities/utils.jl:417`

3. Copy-branch heuristics (`copy_branch_policy`, dominance ratios, boost, optional ML scorer) beyond pure `argmax Delta`.

These are performance controls and do not change DBC validity logic.

## C. Material Differences to Review

### C1. Backward sampling policy differs across examples

Paper Algorithm 1 describes backward over nodes visited by sampled scenarios.

Current behavior:

- SCUC uses `num_backward_scenarios` subset of sampled paths (`src/multistage_SCUC/sddp.jl:148`), default often `1`.
- GEP uses `num_backward_samples` subset (`src/multistage_generation_expansion/algorithm.jl:112`).
- SSLP runs all `omega` states each iteration (`src/sslp/algorithm.jl:81`, `src/sslp/algorithm.jl:102`).

Impact:

- This changes cut sample coverage per iteration and can change convergence speed/variance.
- To match paper style strictly, set SCUC/GEP backward sample count to all forward samples.

### C2. SCUC has neutral-cut fallback branch

SCUC CPT includes robust fallback when no primal-feasible/dual-valid state is found (neutral cut return).

- `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:964`

Impact:

- Stabilizes runtime in pathological solver statuses.
- Not part of paper pseudocode; should be viewed as a safeguard.

## D. Naming Consistency (MDC vs DBC)

The code now follows:

- `MDC`: disjunctive cut generated inside CPT to tighten LP region.
- `DBC`: final Benders cut extracted from strengthened LP dual.
- `iDBC`: DBC produced with copy-variable split enabled in the CPT strengthening phase.

Legacy names (`:MDC`, `:iMDC`, `:Split`) are normalized to canonical names in config resolvers.

## E. Practical “paper-faithful” settings

To run closest to paper Algorithm 1 + Algorithm 3 behavior:

1. Use `cut_selection = :iDBC`.
2. Set `enable_copy_branching = true` and `enable_surrogate_copy_split = true`.
3. Set `copy_split_strategy = :surrogate_delta`.
4. Use full backward coverage each iteration:
- SCUC: `num_backward_scenarios = num_scenarios`
- GEP: `num_backward_samples = sample_count`
- SSLP: already full omega sweep.
5. Keep `idbc_warm_pass = false` unless you intentionally evaluate warm-start behavior.

## F. Verdict

- Core Algorithm 2/3 mechanics (surrogate copy split + incumbent split + CGLP-based MDC generation + dual-derived DBC extraction) are implemented in all three examples.
- Main divergence from paper pseudocode is sampling policy and optional warm-pass engineering extensions, not the mathematical cut construction itself.
