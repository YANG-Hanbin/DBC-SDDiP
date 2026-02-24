if !isdefined(Main, :select_branch_variable)
    include(joinpath(@__DIR__, "..", "common", "branching_policy.jl"))
end
if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    _resolve_scuc_namedtuple_arg(
        explicit_value::Union{Nothing, NamedTuple},
        key::Symbol,
        legacy_symbol::Symbol,
    )::NamedTuple

Resolve one SCUC parameter bundle with compatibility fallback order:
`explicit` -> `RuntimeContext(:scuc)[key]` -> `Main.legacy_symbol`.
"""
function _resolve_scuc_namedtuple_arg(
    explicit_value::Union{Nothing, NamedTuple},
    key::Symbol,
    legacy_symbol::Symbol,
)::NamedTuple
    if !isnothing(explicit_value)
        return explicit_value
    end
    if RuntimeContext.has_context(:scuc)
        context = RuntimeContext.get_context(:scuc)
        if haskey(context, key) && !isnothing(context[key])
            return context[key]
        end
    end
    if isdefined(Main, legacy_symbol)
        legacy_value = getfield(Main, legacy_symbol)
        if legacy_value isa NamedTuple
            return legacy_value
        end
    end
    error("Missing SCUC parameter `$(key)`: pass it explicitly or initialize runtime context.")
end

"""
    _resolve_scuc_initial_state(
        initial_state_info::Union{Nothing, StateInfo},
        initialStateInfo::Union{Nothing, StateInfo},
    )::StateInfo

Resolve initial state with compatibility fallback order:
`initial_state_info` -> `initialStateInfo` -> `RuntimeContext(:scuc)` -> legacy `Main`.
"""
function _resolve_scuc_initial_state(
    initial_state_info::Union{Nothing, StateInfo},
    initialStateInfo::Union{Nothing, StateInfo},
)::StateInfo
    if !isnothing(initial_state_info)
        return initial_state_info
    end
    if !isnothing(initialStateInfo)
        return initialStateInfo
    end
    if RuntimeContext.has_context(:scuc)
        context = RuntimeContext.get_context(:scuc)
        if haskey(context, :initial_state_info) && context[:initial_state_info] isa StateInfo
            return context[:initial_state_info]
        end
    end
    if isdefined(Main, :initialStateInfo)
        legacy_value = getfield(Main, :initialStateInfo)
        if legacy_value isa StateInfo
            return legacy_value
        end
    end
    if isdefined(Main, :initial_state_info)
        legacy_value = getfield(Main, :initial_state_info)
        if legacy_value isa StateInfo
            return legacy_value
        end
    end
    error("Missing SCUC initial state: pass `initial_state_info` or `initialStateInfo` explicitly.")
end

"""
    function stochastic_dual_dynamic_programming_algorithm(
            scenario_tree::ScenarioTree,
            index_sets::IndexSets,
            param_demand::ParamDemand,
            param_opf::ParamOPF;
            initial_state_info::StateInfo = initial_state_info,
            param_cut::Union{Nothing, NamedTuple},
            param_levelsetmethod::Union{Nothing, NamedTuple},
            param::Union{Nothing, NamedTuple}
    )

# Arguments
  1. `scenario_tree::ScenarioTree`: A scenario tree
  2. `index_sets::IndexSets`: A set of indices
  3. `param_demand::ParamDemand`: Demand parameters
  4. `param_opf::ParamOPF`: OPF parameters
  5. `initial_state_info::StateInfo`: Initial state information
  6. `param_cut::NamedTuple`: Named tuple of parameters for the PLC algorithm
  7. `param_levelsetmethod::NamedTuple`: Named tuple of parameters for the level set method
  8. `param::NamedTuple`: Named tuple of parameters for the SDDiP algorithm
"""
function select_scenario_subset(scenario_ids::Vector{Int64}, requested_count::Int64)::Vector{Int64}
    if isempty(scenario_ids)
        return Int64[]
    end
    if requested_count <= 0
        @warn "Requested scenario count must be positive. Falling back to 1."
        return [first(scenario_ids)]
    end
    return scenario_ids[1:min(requested_count, length(scenario_ids))]
end

"""
    build_iteration_state_map(
        forward_pass_result::Vector,
        scenario_ids::Vector{Int64},
        stage_count::Int64,
    )::Tuple{Dict{Tuple{Int64, Int64}, StateInfo}, Dict{Int64, Float64}}

# Purpose
    Convert forward-pass outputs into a compact per-iteration state map keyed by `(stage, scenario_id)`.
    This avoids keeping and broadcasting full history across iterations.

# Arguments
    1. `forward_pass_result`: vector of scenario state trajectories from forward pass.
    2. `scenario_ids`: sampled scenario ids (same ordering used in `forward_pass_result`).
    3. `stage_count`: total number of stages.

# Returns
    1. `iteration_state_map`: map `(t, scenario_id) => StateInfo` for current iteration only.
    2. `scenario_values`: realized total forward value per sampled scenario.
"""
function build_iteration_state_map(
    forward_pass_result::Vector,
    scenario_ids::Vector{Int64},
    stage_count::Int64,
)::Tuple{Dict{Tuple{Int64, Int64}, StateInfo}, Dict{Int64, Float64}}
    iteration_state_map = Dict{Tuple{Int64, Int64}, StateInfo}()
    scenario_values = Dict{Int64, Float64}()

    for (result_idx, scenario_id) in enumerate(scenario_ids)
        scenario_stage_values = 0.0
        for t in 1:stage_count
            stage_state = forward_pass_result[result_idx][t]
            iteration_state_map[(t, scenario_id)] = stage_state
            scenario_stage_values += stage_state.StageValue
        end
        scenario_values[scenario_id] = scenario_stage_values
    end

    return iteration_state_map, scenario_values
end

"""
    function stochastic_dual_dynamic_programming_algorithm(...)

# Purpose
    Run the full SDDP workflow for one problem family: forward simulation, backward cut generation, cut injection, and convergence logging.

# Arguments
    1. Scenario/problem data plus `param`, `param_cut`, and level-set parameter bundles.

# Returns
    1. A result dictionary containing at least iteration history and gap trajectory.
"""
function stochastic_dual_dynamic_programming_algorithm(
    scenario_tree::ScenarioTree,
    index_sets::IndexSets,
    param_demand::ParamDemand,
    param_opf::ParamOPF;
    initialStateInfo::Union{Nothing, StateInfo} = nothing,
    initial_state_info::Union{Nothing, StateInfo} = nothing,
    param_cut::Union{Nothing, NamedTuple} = nothing,
    param_levelsetmethod::Union{Nothing, NamedTuple} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::Dict
    initial_state_info_value = _resolve_scuc_initial_state(initial_state_info, initialStateInfo)
    local_param = _resolve_scuc_namedtuple_arg(param, :param, :param)
    local_param_cut = _resolve_scuc_namedtuple_arg(param_cut, :param_cut, :param_cut)
    local_param_levelsetmethod = _resolve_scuc_namedtuple_arg(param_levelsetmethod, :param_level_method, :param_levelsetmethod)

    # Local aliases keep existing body readable and preserve behavior.
    param = local_param
    param_cut = local_param_cut
    param_levelsetmethod = local_param_levelsetmethod

    runtime_param = resolve_scuc_runtime_params(param)
    num_scenarios = runtime_param.num_scenarios
    num_partition_scenarios = runtime_param.num_partition_scenarios
    num_backward_scenarios = runtime_param.num_backward_scenarios
    lift_iter_threshold = runtime_param.lift_iter_threshold
    cut_selection = runtime_param.cut_selection
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    branch_threshold = runtime_param.branch_threshold
    branch_variable = runtime_param.branch_variable
    max_iter = runtime_param.max_iter
    imdc_state_projection = runtime_param.imdc_state_projection
    idbc_warm_pass = runtime_param.idbc_warm_pass
    ml_weights = runtime_param.branching_ml_weights

    start_time = now()
    iter = 1
    lower_bound::Float64 = -Inf
    upper_bound::Float64 = Inf
    iter_time::Float64 = 0.0
    total_time::Float64 = 0.0
    iter_start_time = 0.0
    lm_iter::Int64 = 0
    gap::Float64 = 100.0
    gap_string = "100%"
    branch_decision = false

    column_names = [:Iter, :LB, :OPT, :UB, :gap, :time, :LM_iter, :Time, :Branch]
    column_types = [Int64, Float64, Union{Float64, Nothing}, Float64, String, Float64, Int64, Float64, Bool]
    named_tuple = (; zip(column_names, type[] for type in column_types)...)
    sol_history = DataFrame(named_tuple)
    gap_history = []

    @everywhere initialize_scuc_runtime_context!(
        $scenario_tree,
        $index_sets,
        $param_demand,
        $param_opf,
        $initial_state_info_value,
        $param;
        param_cut_input = $param_cut,
        param_level_method_input = $param_levelsetmethod,
    )
    runtime_context = RuntimeContext.get_context(:scuc)
    model_list = runtime_context[:model_list]

    while true
        # Phase 1: sample scenario paths, run forward pass, and update statistical bounds.
        iter_start_time = now()
        branch_decision = false
        sampled_scenarios = sample_scenarios(; scenario_tree = scenario_tree, num_scenarios = num_scenarios)
        scenario_ids = sort(collect(keys(sampled_scenarios)))
        partition_scenarios = select_scenario_subset(scenario_ids, num_partition_scenarios)
        backward_scenarios = select_scenario_subset(scenario_ids, num_backward_scenarios)

        forward_pass_result = pmap(scenario_ids) do scenario_id
            forwardPass(
                sampled_scenarios[scenario_id];
                param_demand = param_demand,
                param_opf = param_opf,
                index_sets = index_sets,
                initial_state_info = initial_state_info_value,
                param = param,
            )
        end

        iteration_state_map, scenario_values = build_iteration_state_map(
            forward_pass_result,
            scenario_ids,
            index_sets.T,
        )

        first_scenario_id = first(scenario_ids)
        lower_bound = maximum([iteration_state_map[(1, first_scenario_id)].StateValue, lower_bound])
        scenario_value_array = collect(values(scenario_values))
        mean_value = mean(scenario_value_array)
        variance_value = length(scenario_value_array) > 1 ? Statistics.var(scenario_value_array; corrected = false) : 0.0
        upper_bound = mean_value + 1.96 * sqrt(variance_value / length(scenario_value_array))
        gap = if isfinite(upper_bound) && abs(upper_bound) > 1e-9
            round((upper_bound - lower_bound) / upper_bound * 100, digits = 2)
        else
            Inf
        end
        gap_string = string(gap, "%")

        lm_iter = 0

        # Phase 2: (SDDPL only) optionally update partition trees before backward cutting.
        if iter >= lift_iter_threshold && param.algorithm == :SDDPL
            for t in reverse(1:index_sets.T-1)
                for scenario_id in partition_scenarios
                    fractional_deviation = Dict()
                    for g in index_sets.G
                        if iteration_state_map[(t, scenario_id)].BinVar[:y][g] > 0.5
                            k = maximum(
                                [leaf_idx for (leaf_idx, leaf_info) in iteration_state_map[(t, scenario_id)].ContVarLeaf[:s][g] if leaf_info[:var] > 0.5]
                            )
                            leaf_info = model_list[t].ContVarLeaf[:s][g][k]
                            fractional_deviation[g] = round(
                                minimum(
                                    [
                                        (leaf_info[:ub] - iteration_state_map[(t, scenario_id)].ContVar[:s][g]) / (leaf_info[:ub] - leaf_info[:lb] + 1e-6),
                                        (iteration_state_map[(t, scenario_id)].ContVar[:s][g] - leaf_info[:lb]) / (leaf_info[:ub] - leaf_info[:lb] + 1e-6),
                                    ]
                                ),
                                digits = 5,
                            )
                        end
                    end

                    branch_candidates = Int64[]
                    if branch_variable == :ALL
                        branch_candidates = [g for (g, g_dev) in fractional_deviation if g_dev >= branch_threshold]
                    else
                        selected_branch = select_branch_variable(
                            fractional_deviation;
                            strategy = branch_variable,
                            fallback = :MFV,
                            ml_weights = ml_weights,
                        )
                        if !isnothing(selected_branch)
                            branch_candidates = [selected_branch]
                        end
                    end

                    for g in branch_candidates
                        branch_decision = true
                        stage_state = iteration_state_map[(t, scenario_id)]
                        @everywhere begin
                            t = $t
                            g = $g
                            stage_state = $stage_state
                            update_partition_tree!(
                                ModelList,
                                stage_state,
                                t,
                                g;
                                param = param,
                            )
                        end
                    end
                end
            end
        end

        # Phase 3 + 4: backward pass.
        #   Phase 3 (iDBC only): warm-start backward to inherit stronger cut pools.
        #   Phase 4: standard backward and cut injection into stage-(t-1) value function.
        for t in reverse(2:index_sets.T)
            for scenario_id in backward_scenarios
                node_ids = sort(collect(keys(scenario_tree.tree[t].nodes)))
                parent_state = iteration_state_map[(t - 1, scenario_id)]
                current_stage_state_value = iteration_state_map[(t, scenario_id)].StateValue
                if cut_selection == :iDBC && idbc_warm_pass
                    warm_state = setup_core_point(
                        parent_state;
                        index_sets = index_sets,
                        param_opf = param_opf,
                        param = param,
                        param_cut = param_cut,
                    )
                    # Keep carry-over commitment states fixed and sanitize copied binarization states.
                    warm_state.BinVar = deepcopy(parent_state.BinVar)
                    if param.algorithm == :SDDiP && imdc_state_projection
                        warm_state.ContStateBin = deepcopy(parent_state.ContStateBin)
                        warm_state = sanitize_imdc_copy_state(
                            warm_state;
                            enforce_binary = true,
                            zero_copy_when_off = true,
                            index_sets = index_sets,
                            param = param,
                        )
                    end
                    warm_backward_jobs = [
                        (0.0, t, node_id, scenario_id, cut_selection, param_cut.core_point_strategy) for node_id in node_ids
                    ]
                    warm_backward_vector = pmap(warm_backward_jobs) do backward_node_info
                        backwardPass(
                            backward_node_info;
                            index_sets = index_sets,
                            param_demand = param_demand,
                            param_opf = param_opf,
                            scenario_tree = scenario_tree,
                            param = param,
                            param_cut = param_cut,
                            param_levelsetmethod = param_levelsetmethod,
                            parent_state = parent_state,
                            stage_state_value = current_stage_state_value,
                            state_override = warm_state,
                        )
                    end
                    warm_backward_result = Dict(
                        node_id => warm_backward_vector[idx] for (idx, node_id) in enumerate(node_ids)
                    )

                    if inherit_disjunctive_cuts
                        cut_collection = Dict()
                        for node_id in node_ids
                            for (key, cut) in warm_backward_result[node_id][3]
                                cut_collection[key] = cut
                            end
                        end
                        @everywhere begin
                            t = $t
                            cut_collection = $cut_collection
                            ModelList[t].model[:cut_expression] = cut_collection
                        end
                    end

                    lm_iter += sum(warm_backward_result[node_id][2] for node_id in node_ids)
                end

                backward_node_jobs = [
                    (iter, t, node_id, scenario_id, cut_selection, param_cut.core_point_strategy) for node_id in node_ids
                ]
                backward_pass_vector = pmap(backward_node_jobs) do backward_node_info
                    backwardPass(
                        backward_node_info;
                        index_sets = index_sets,
                        param_demand = param_demand,
                        param_opf = param_opf,
                        scenario_tree = scenario_tree,
                        param = param,
                        param_cut = param_cut,
                        param_levelsetmethod = param_levelsetmethod,
                        parent_state = parent_state,
                        stage_state_value = current_stage_state_value,
                    )
                end
                backward_pass_result = Dict(
                    node_id => backward_pass_vector[idx] for (idx, node_id) in enumerate(node_ids)
                )

                cut_collection = Dict()
                if inherit_disjunctive_cuts
                    for node_id in node_ids
                        for (key, cut) in backward_pass_result[node_id][3]
                            cut_collection[key] = cut
                        end
                    end
                end

                for node_id in node_ids
                    @everywhere begin
                        node_id = $node_id
                        t = $t
                        (λ₀, λ₁) = $backward_pass_result[node_id][1]
                        if $inherit_disjunctive_cuts
                            cut_collection = $cut_collection
                            ModelList[t].model[:cut_expression] = cut_collection
                        end
                        @constraint(
                            ModelList[t-1].model,
                            ModelList[t-1].model[:θ][node_id] / scenario_tree.tree[t-1].prob[node_id] >= λ₀ +
                            sum(
                                (
                                    param.algorithm == :SDDiP ?
                                    sum(λ₁.ContStateBin[:s][g][bit_idx] * ModelList[t-1].model[:λ][g, bit_idx] for bit_idx in 1:param.kappa[g]; init = 0.0)
                                    : λ₁.ContVar[:s][g] * ModelList[t-1].model[:s][g]
                                ) +
                                λ₁.BinVar[:y][g] * ModelList[t-1].model[:y][g] +
                                (
                                    param.algorithm == :SDDPL ?
                                    sum(λ₁.ContAugState[:s][g][k] * ModelList[t-1].model[:augmentVar][g, k] for k in keys(λ₁.ContAugState[:s][g]); init = 0.0)
                                    : 0.0
                                ) for g in index_sets.G
                            )
                        )
                    end
                end

                lm_iter += sum(backward_pass_result[node_id][2] for node_id in node_ids)
            end
        end

        backward_normalizer = max(
            1,
            length(backward_scenarios) * sum(length(scenario_tree.tree[t].nodes) for t in 2:index_sets.T),
        )
        lm_iter = floor(Int64, lm_iter / backward_normalizer)

        iter_end_time = now()
        iter_time = (iter_end_time - iter_start_time).value / 1000
        total_time = (iter_end_time - start_time).value / 1000
        push!(sol_history, [iter, lower_bound, param.opt, upper_bound, gap_string, iter_time, lm_iter, total_time, branch_decision])
        push!(gap_history, gap)

        if iter == 1
            print_iteration_info_bar()
        end
        print_iteration_info(iter, lower_bound, upper_bound, gap, iter_time, lm_iter, total_time)
        save_info(
            param,
            param_cut,
            Dict(
                :sol_history => sol_history,
                :gap_history => gap_history,
            );
            logger_save = param.logger_save,
        )

        if total_time > param.terminate_time || iter >= max_iter || gap <= param.terminate_threshold * 100
            return Dict(
                :sol_history => sol_history,
                :gap_history => gap_history,
            )
        end

        iter += 1
    end
end
