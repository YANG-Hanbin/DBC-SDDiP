# Algorithm 3 + SDDP Code Walkthrough (All Three Examples)

This document maps your paper algorithms to the current implementation:

- Algorithm 1: SDDP outer loop (forward sampling + backward cut generation).
- Algorithm 2: CPT disjunctive strengthening loop (primary variables).
- Algorithm 3: copy-variable (`z_m`) surrogate-Delta split integrated into CPT.

The focus is exact implementation components and control flow.

---

## 1. What Algorithm 3 means in code

Paper intent:

1. Standard CPT originally splits only on original mixed-integer variables.
2. For copy variables `z_m`, compute surrogate Delta to detect when copy split is useful.
3. In each backward call, do two logical strengthening steps:
   - one surrogate-point DBC on copy variable to tighten convex-hull approximation,
   - one incumbent-point DBC to cut off current fractional incumbent.

Current implementation structure:

1. Build/maintain a CPT tree (`Tree`, `Leaf`).
2. In each CPT iteration:
   - solve anchored LP (`z_m = x_n^i` nonanticipativity),
   - evaluate surrogate Delta for copy-variable candidates,
   - if `best_delta > tol`, add one disjunctive cut from that surrogate split,
   - re-solve model,
   - continue standard CPT branch/cut on primary fractional variables,
   - extract dual and return strengthened Benders cut.

---

## 2. Trigger and effective-runtime switches

### 2.1 Canonical resolution layer

All three projects normalize parameters in:

- SSLP: `src/sslp/utilities/utils.jl:376`
- GEP: `src/multistage_generation_expansion/utilities/utils.jl:433`
- SCUC: `src/multistage_SCUC/utils.jl:502`

Key booleans produced by resolver:

- `cut_selection`
- `use_copy_branching = (cut_selection == :iDBC) && copy_split_enabled`
- `enable_surrogate_copy_split`
- `copy_split_strategy`
- `copy_split_delta_tol`

So Algorithm 3 surrogate stage is active only when:

1. effective cut is `:iDBC`,
2. copy branching is enabled,
3. surrogate split is enabled,
4. strategy is `:surrogate_delta`.

---

## 3. Function-level map for Algorithm 3

### 3.1 SSLP (reference implementation)

File: `src/sslp/utilities/cutting_plane_tree_alg.jl`

Core helper functions:

1. `_sslp_remove_surrogate_constraints!` at `src/sslp/utilities/cutting_plane_tree_alg.jl:529`
2. `_sslp_restore_anchor_nonanticipativity!` at `src/sslp/utilities/cutting_plane_tree_alg.jl:543`
3. `_sslp_add_surrogate_edge_constraints!` at `src/sslp/utilities/cutting_plane_tree_alg.jl:553`
4. `_sslp_endpoint_objective!` at `src/sslp/utilities/cutting_plane_tree_alg.jl:586`
5. `_sslp_evaluate_surrogate_candidate!` at `src/sslp/utilities/cutting_plane_tree_alg.jl:599`
6. `CPT_optimization!` at `src/sslp/utilities/cutting_plane_tree_alg.jl:676`

### 3.2 Generation expansion

File: `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl`

Analogous helper set:

1. `_gep_remove_surrogate_constraints!` at `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:594`
2. `_gep_restore_anchor_nonanticipative!` at `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:608`
3. `_gep_add_surrogate_edge_constraints!` at `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:618`
4. `_gep_endpoint_objective!` at `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:649`
5. `_gep_evaluate_surrogate_candidate!` at `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:662`
6. `CPT_optimization!` at `src/multistage_generation_expansion/utilities/cutting_plane_tree_alg.jl:736`

### 3.3 SCUC

File: `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl`

Analogous helper set:

1. `_uc_remove_surrogate_constraints!` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:546`
2. `_uc_restore_anchor_nonanticipative!` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:560`
3. `_uc_collect_copy_components` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:577`
4. `_uc_add_surrogate_edge_constraints!` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:611`
5. `_uc_endpoint_objective!` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:644`
6. `_uc_evaluate_surrogate_candidate!` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:664`
7. `CPT_optimization!` at `src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:765`

SCUC detail: current copy-component collection explicitly prioritizes `y_copy` family as Algorithm-3 copy split candidate pool (`src/multistage_SCUC/utilities/cutting_plane_tree_alg.jl:584`).

---

## 4. Statement-by-statement walkthrough (SSLP key functions)

This section is the most granular mapping.

### 4.1 `_sslp_evaluate_surrogate_candidate!`

File: `src/sslp/utilities/cutting_plane_tree_alg.jl:599`

1. `endpoint_state = copy(state)` (`:607`)
   - clone anchor state.
2. `endpoint_state[branch_position] = 1.0 - endpoint_state[branch_position]` (`:608`)
   - flip selected copy-state component to opposite binary endpoint.
3. `endpoint_value = _sslp_endpoint_objective!(...)` (`:609`)
   - solve anchored LP at endpoint state.
4. infeasible endpoint branch (`:610-:618`)
   - restore anchor constraints and return `feasible=false`, `delta=-Inf`.
5. `_sslp_add_surrogate_edge_constraints!(...)` (`:620`)
   - replace exact nonanticipativity with surrogate segment constraints.
6. `optimize!(model)` (`:621`) and status check (`:622-:631`)
   - solve surrogate interpolation LP.
7. `alpha = value(model[:SurrogateAlpha])` (`:633`)
   - interpolation scalar.
8. `objective_value = JuMP.objective_value(model)` (`:634`)
   - value at surrogate point.
9. `delta = anchor_value * alpha + endpoint_value * (1 - alpha) - objective_value` (`:635`)
   - Jensen-gap style surrogate Delta.
10. `x_fractional = Dict(index => value(var) for (var, index) in columns)` (`:636`)
    - cache current LP point for tree update and CGLP cut.
11. `branch_value = value(copy_vars[branch_position])` (`:637`)
    - actual split value used in tree update.
12. restore anchor nonanticipativity (`:639`) and return candidate tuple (`:640-:645`).

### 4.2 `CPT_optimization!` (Algorithm-2 + Algorithm-3 fusion)

File: `src/sslp/utilities/cutting_plane_tree_alg.jl:676`

1. Runtime switches are resolved (`:685-:695`).
2. CPT tree initialized (`:697-:710`).
3. Remove nonanticipativity, extract LP matrix, optionally inherit old cuts (`:712-:727`), then re-add anchor nonanticipativity (`:728-:731`).
4. Loop `while d < MDCiter` (`:734`):
   - solve tightened LP (`:735`),
   - cache incumbent fractional point (`:738-:744`),
   - if integral primary vars, extract dual Benders cut and exit (`:765-:809`).
5. Algorithm-3 surrogate stage (`:812-:882`):
   - check trigger (`:813`),
   - evaluate each copy candidate by `_sslp_evaluate_surrogate_candidate!` (`:820-:835`),
   - if `best_delta > copy_split_delta_tol` (`:837`):
     - update CPT tree on copy variable (`:844-:851`),
     - solve CGLP and add one disjunctive cut (`:852-:875`),
     - refresh matrix data after structural change (`:876-:879`).
6. If surrogate cut added, re-solve and re-check integrality/duals (`:884-:935`).
7. Standard primary branching stage:
   - compute primary and optional copy deviations (`:937-:947`),
   - construct branch candidate pool (`:949-:969`),
   - choose branch variable (`:972-:979`),
   - update tree (`:1014-:1021`),
   - solve CGLP and add standard DBC cut (`:1024-:1057`).
8. After loop, solve final tightened LP and extract dual Benders cut (`:1122-:1187`).

This realizes:

- Algorithm 3 pre-cut from copy split (conditional),
- Algorithm 2 incumbent-separation cut loop,
- final Benders cut from strengthened LP dual.

---

## 5. SDDP outer components (Algorithm 1 mapping)

### 5.1 SSLP main loop

File: `src/sslp/algorithm.jl`

Components:

1. Worker model initialization (`:46-:68`)
2. Forward pass and UB/LB update (`:69-:101`)
3. Optional iDBC warm backward sweep (`:105-:146`)
4. Standard backward sweep and cut aggregation (`:147-:187`)
5. Logging + stop checks (`:189-:215`)

Backward scenarios:

- SSLP backward pass runs over all `omega_ids = 1:omega_count` each iteration (`:81`, `:102`, `:148`).

### 5.2 Generation expansion main loop

File: `src/multistage_generation_expansion/algorithm.jl`

Components:

1. Forward path sampling (`:67-:115`)
2. Backward stage loop `for t in reverse(2:param.T)` (`:118`)
3. Optional iDBC warm backward sweep (`:121-:163`)
4. Standard backward sweep (`:165-:205`)
5. Logging + termination (`:213-:239`)

Backward scenarios:

- controlled by `num_backward_samples`; if `<0`, defaults to all sampled forward paths (`:112-:114`).

### 5.3 SCUC main loop

File: `src/multistage_SCUC/sddp.jl`

Components:

1. Forward scenario sampling and forward pass (`:129-:172`)
2. Optional SDDPL partition updates (`:175-:231`)
3. Backward stage loop (`:236-:358`)
4. Optional iDBC warm backward sweep (`:239-:296`)
5. Standard backward sweep and stage-(t-1) cut injection (`:298-:354`)
6. Logging + termination (`:366-:394`)

Backward scenarios:

- SCUC backward uses `backward_scenarios` subset selected by `num_backward_scenarios` (`:136`, `:237`).

---

## 6. Where DBC vs iDBC actually diverge

In all three `CPT_optimization!` implementations:

1. `use_copy_branching` is true only for effective `:iDBC`.
2. If `use_copy_branching` is false:
   - surrogate-copy stage is skipped,
   - algorithm behaves as DBC (primary-variable CPT only).
3. If true and surrogate stage enabled:
   - one additional surrogate-based disjunctive cut may be generated before incumbent split.

So iDBC is not a new dual extraction formula; it is a strengthened pre-dual LP via copy-variable disjunction.

---

## 7. Consistency checklist to compare with paper text

Use this checklist when auditing consistency:

1. Nonanticipativity removal before CGLP matrix extraction:
   - SSLP `remove_nonanticipativity_constraint(...)`
   - GEP `RemoveNonAnticipative!(...)`
   - SCUC `RemoveContVarNonAnticipative!(...)`
2. Nonanticipativity restored before final dual extraction.
3. Surrogate Delta computed from endpoint-anchor interpolation formula.
4. Surrogate split accepted only when `Delta > copy_split_delta_tol`.
5. Standard CPT split still runs on incumbent fractional point afterward.
6. Benders cut always extracted from duals of strengthened LP with anchored nonanticipativity.
