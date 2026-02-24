if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    _resolve_ge_namedtuple_arg(
        explicit_value::Union{Nothing, NamedTuple},
        key::Symbol,
        legacy_symbol::Symbol,
    )::NamedTuple

Resolve one GEP parameter bundle with compatibility fallback order:
`explicit` -> `RuntimeContext(:generation_expansion)[key]` -> `Main.legacy_symbol`.
"""
function _resolve_ge_namedtuple_arg(
    explicit_value::Union{Nothing, NamedTuple},
    key::Symbol,
    legacy_symbol::Symbol,
)::NamedTuple
    if !isnothing(explicit_value)
        return explicit_value
    end
    if RuntimeContext.has_context(:generation_expansion)
        context = RuntimeContext.get_context(:generation_expansion)
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
    error("Missing GEP parameter `$(key)`: pass it explicitly or initialize runtime context.")
end

"""
    function stochastic_dual_dynamic_programming_algorithm(
        Ω::Dict{Int64,Dict{Int64,RandomVariables}},
        prob_list::Dict{Int64,Vector{Float64}},
        stage_data_list::Dict{Int64, StageData};
        param::Union{Nothing, NamedTuple},
        param_cut::Union{Nothing, NamedTuple},
        param_LevelMethod::Union{Nothing, NamedTuple}
    )::Dict

# Arguments
  1. `Ω`: Each stage has a dictionary of realizations of demand
  2. `prob_list`: Probability of each realization
  3. `stage_data_list`: Stage parameters information
  4. `param`: Named tuple of parameters for the PLC algorithm
  5. `param_cut`: Named tuple of parameters for the level set method
  6. `param_LevelMethod`: Named tuple of parameters for the SDDiP algorithm
"""
function stochastic_dual_dynamic_programming_algorithm(
    Ω::Dict{Int64, Dict{Int64, RandomVariables}},
    prob_list::Dict{Int64, Vector{Float64}},
    stage_data_list::Dict{Int64, StageData};
    param::Union{Nothing, NamedTuple} = nothing,
    param_cut::Union{Nothing, NamedTuple} = nothing,
    param_LevelMethod::Union{Nothing, NamedTuple} = nothing,
)::Dict
    local_param = _resolve_ge_namedtuple_arg(param, :param, :param)
    local_param_cut = _resolve_ge_namedtuple_arg(param_cut, :param_cut, :param_cut)
    local_param_levelmethod = _resolve_ge_namedtuple_arg(param_LevelMethod, :param_level_method, :param_LevelMethod)

    runtime_param = resolve_ge_runtime_params(local_param)
    sample_count = runtime_param.sample_count
    num_backward_samples = runtime_param.num_backward_samples
    count_benders_cut = runtime_param.count_benders_cut
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    cut_selection = runtime_param.cut_selection
    max_iter = runtime_param.max_iter
    idbc_warm_pass = runtime_param.idbc_warm_pass

    start_time = now()
    iter_time = 0.0
    total_time = 0.0
    iter_start_time = 0.0
    iter = 1
    lower_bound = -Inf
    upper_bound = Inf
    solution_collection = Dict()
    disjunction_rounds = 0

    column_names = [:Iter, :LB, :OPT, :UB, :gap, :time, :MDCiter, :Time, :count_benders_cut]
    column_types = [Int64, Float64, Union{Float64, Nothing}, Float64, String, Float64, Int64, Float64, Bool]
    named_tuple = (; zip(column_names, type[] for type in column_types)...)
    sol_history = DataFrame(named_tuple)
    gap_history = []

    @everywhere initialize_ge_runtime_context!(
        $Ω,
        $prob_list,
        $stage_data_list,
        $local_param;
        param_cut_input = $local_param_cut,
        param_level_method_input = $local_param_levelmethod,
    )

    while true
        # Phase 1: forward simulation over sampled paths and statistical bound update.
        iter_start_time = now()
        Random.seed!(iter * 3)
        solution_collection = Dict()
        scenario_values = zeros(Float64, sample_count)
        sampled_paths = SampleScenarios(
            Ω,
            prob_list;
            M = sample_count,
        )

        forward_pass_result = pmap(1:sample_count) do path_id
            forwardPass(
                path_id,
                sampled_paths;
                param = local_param,
            )
        end

        for path_id in 1:sample_count
            for t in 1:local_param.T
                solution_collection[t, path_id] = (
                    stage_solution = forward_pass_result[path_id][t, path_id].stage_solution,
                    stage_value = forward_pass_result[path_id][t, path_id].stage_value,
                    OPT = forward_pass_result[path_id][t, path_id].OPT,
                )
            end
            scenario_values[path_id] = sum(solution_collection[t, path_id].stage_value for t in 1:local_param.T)
        end

        lower_bound = solution_collection[1, 1].OPT
        mean_value = mean(scenario_values)
        variance_value = sample_count > 1 ? var(scenario_values; corrected = false) : 0.0
        upper_bound = mean_value + 1.96 * sqrt(variance_value / sample_count)
        gap = if isfinite(upper_bound) && abs(upper_bound) > 1e-9
            round((upper_bound - lower_bound) / upper_bound * 100, digits = 2)
        else
            Inf
        end
        gap_string = string(gap, "%")
        requested_backward_samples = num_backward_samples <= 0 ? sample_count : min(num_backward_samples, sample_count)
        backward_sample_ids = collect(1:requested_backward_samples)
        disjunction_rounds = 0

        # Phase 2: backward pass and cut construction.
        # iDBC adds a warm-start backward sweep before the standard sweep.
        for t in reverse(2:local_param.T)
            for path_id in backward_sample_ids
                node_ids = sort(collect(keys(Ω[t])))
                if cut_selection == :iDBC && idbc_warm_pass
                    warm_state = solution_collection[t-1, path_id].stage_solution .* local_param_cut.ℓ2 .+ (1 - local_param_cut.ℓ2) / 2
                    warm_backward_jobs = [(0.0, t, node_id, path_id) for node_id in node_ids]
                    warm_result_vector = pmap(warm_backward_jobs) do backward_node_info
                        backwardPass(
                            backward_node_info;
                            param = local_param,
                            param_cut = local_param_cut,
                            param_LevelMethod = local_param_levelmethod,
                            prob_list = prob_list,
                            sol_collection = solution_collection,
                            state_override = warm_state,
                        )
                    end
                    warm_backward_results = Dict(
                        node_id => warm_result_vector[idx] for (idx, node_id) in enumerate(node_ids)
                    )

                    aggregated_cut = [0, zeros(Float64, local_param.binary_info.n)]
                    cut_collection = Dict()
                    for node_id in node_ids
                        aggregated_cut = aggregated_cut .+ prob_list[t][node_id] .* warm_backward_results[node_id].cut_info
                        disjunction_rounds += warm_backward_results[node_id].iter
                        if inherit_disjunctive_cuts
                            for (key, cut) in warm_backward_results[node_id].cut_collection
                                cut_collection[key] = cut
                            end
                        end
                    end

                    @everywhere begin
                        runtime_context = RuntimeContext.get_context(:generation_expansion)
                        forward_info_list = runtime_context[:forward_info_list]
                        t = $t
                        aggregated_cut = $aggregated_cut
                        @constraint(
                            forward_info_list[t-1],
                            forward_info_list[t-1][:θ] >= aggregated_cut[1] + aggregated_cut[2]' * forward_info_list[t-1][:Lt],
                        )
                        if $inherit_disjunctive_cuts
                            cut_collection = $cut_collection
                            forward_info_list[t][:cut_expression] = cut_collection
                        end
                    end
                end

                backward_node_jobs = [(iter, t, node_id, path_id) for node_id in node_ids]
                backward_result_vector = pmap(backward_node_jobs) do backward_node_info
                        backwardPass(
                            backward_node_info;
                            param = local_param,
                            param_cut = local_param_cut,
                            param_LevelMethod = local_param_levelmethod,
                            prob_list = prob_list,
                            sol_collection = solution_collection,
                        )
                end
                backward_results = Dict(
                    node_id => backward_result_vector[idx] for (idx, node_id) in enumerate(node_ids)
                )

                aggregated_cut = [0, zeros(Float64, local_param.binary_info.n)]
                cut_collection = Dict()
                for node_id in node_ids
                    aggregated_cut = aggregated_cut .+ prob_list[t][node_id] .* backward_results[node_id].cut_info
                    disjunction_rounds += backward_results[node_id].iter
                    if inherit_disjunctive_cuts
                        for (key, cut) in backward_results[node_id].cut_collection
                            cut_collection[key] = cut
                        end
                    end
                end

                @everywhere begin
                    runtime_context = RuntimeContext.get_context(:generation_expansion)
                    forward_info_list = runtime_context[:forward_info_list]
                    t = $t
                    aggregated_cut = $aggregated_cut
                    @constraint(
                        forward_info_list[t-1],
                        forward_info_list[t-1][:θ] >= aggregated_cut[1] + aggregated_cut[2]' * forward_info_list[t-1][:Lt],
                    )
                    if $inherit_disjunctive_cuts
                        cut_collection = $cut_collection
                        forward_info_list[t][:cut_expression] = cut_collection
                    end
                end
            end
        end

        backward_normalizer = max(
            1,
            length(backward_sample_ids) * sum(length(Ω[t]) for t in 2:local_param.T),
        )
        disjunction_rounds = floor(Int64, disjunction_rounds / backward_normalizer)

        iter_end_time = now()
        iter_time = (iter_end_time - iter_start_time).value / 1000
        total_time = (iter_end_time - start_time).value / 1000
        push!(sol_history, [iter, lower_bound, local_param.opt, upper_bound, gap_string, iter_time, disjunction_rounds, total_time, count_benders_cut])
        push!(gap_history, gap)

        if iter == 1
            print_iteration_info_bar()
        end
        print_iteration_info(iter, lower_bound, upper_bound, gap, iter_time, disjunction_rounds, total_time)
        save_info(
            local_param,
            Dict(
                :sol_history => sol_history,
                :gap_history => gap_history,
            );
            logger_save = local_param.logger_save,
        )

        if total_time > local_param.terminate_time || iter >= max_iter || gap <= local_param.terminate_threshold * 100
            return Dict(
                :sol_history => sol_history,
                :gap_history => gap_history,
            )
        end
        iter += 1
    end
end
