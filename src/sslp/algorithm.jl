if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    _resolve_sslp_namedtuple_arg(
        explicit_value::Union{Nothing, NamedTuple},
        key::Symbol,
        legacy_symbol::Symbol,
    )::NamedTuple

Resolve one SSLP parameter bundle with compatibility fallback order:
`explicit` -> `RuntimeContext(:sslp)[key]` -> `Main.legacy_symbol`.
"""
function _resolve_sslp_namedtuple_arg(
    explicit_value::Union{Nothing, NamedTuple},
    key::Symbol,
    legacy_symbol::Symbol,
)::NamedTuple
    if !isnothing(explicit_value)
        return explicit_value
    end
    if RuntimeContext.has_context(:sslp)
        context = RuntimeContext.get_context(:sslp)
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
    error("Missing SSLP parameter `$(key)`: pass it explicitly or initialize runtime context.")
end

"""
    function stochastic_dual_dynamic_programming_algorithm(
        stage_data::StageData,
        random_variables::RandomVariables;
        param::Union{Nothing, NamedTuple},
        param_cut::Union{Nothing, NamedTuple},
        param_LevelMethod::Union{Nothing, NamedTuple}
    )::Dict
# Arguments
  1. `stage_data`: stage data
  2. `random_variables`: random variables for each scenario
  3. `param`: Named tuple of parameters for the PLC algorithm
  4. `param_cut`: Named tuple of parameters for the level set method
  5. `param_LevelMethod`: Named tuple of parameters for the SDDiP algorithm
"""
function stochastic_dual_dynamic_programming_algorithm(
    stage_data::StageData,
    random_variables::RandomVariables;
    param::Union{Nothing, NamedTuple} = nothing,
    param_cut::Union{Nothing, NamedTuple} = nothing,
    param_LevelMethod::Union{Nothing, NamedTuple} = nothing,
)::Dict
    local_param = _resolve_sslp_namedtuple_arg(param, :param, :param)
    local_param_cut = _resolve_sslp_namedtuple_arg(param_cut, :param_cut, :param_cut)
    local_param_levelmethod = _resolve_sslp_namedtuple_arg(param_LevelMethod, :param_level_method, :param_LevelMethod)

    runtime_param = resolve_sslp_runtime_params(local_param)
    omega_count = runtime_param.omega_count
    cut_selection = runtime_param.cut_selection
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    idbc_warm_pass = runtime_param.idbc_warm_pass
    max_iter = runtime_param.max_iter

    start_time = now()
    iter_time = 0.0
    total_time = 0.0
    iter_start_time = 0.0
    iter = 1
    lower_bound = -Inf
    upper_bound = Inf
    solution_collection = Dict()
    disjunction_rounds = 0

    column_names = [:Iter, :LB, :OPT, :UB, :gap, :time, :MDCiter, :Time]
    column_types = [Int64, Float64, Union{Float64, Nothing}, Float64, String, Float64, Int64, Float64]
    named_tuple = (; zip(column_names, type[] for type in column_types)...)
    sol_history = DataFrame(named_tuple)
    gap_history = []

    @everywhere initialize_sslp_runtime_context!(
        $stage_data,
        $random_variables,
        $local_param,
        $omega_count;
        param_cut_input = $local_param_cut,
        param_level_method_input = $local_param_levelmethod,
    )
    runtime_context = RuntimeContext.get_context(:sslp)
    forward_model = runtime_context[:forward_model]

    while true
        # Phase 1: forward solve and statistical upper/lower bound update.
        iter_start_time = now()
        solution_collection = Dict()

        optimize!(forward_model)
        solution_collection[1, iter, 1] = (
            state = round.(JuMP.value.(forward_model[:x]), digits = 2),
            stage_value = JuMP.objective_value(forward_model) - JuMP.value(forward_model[:θ]),
            OPT = JuMP.objective_value(forward_model),
        )

        omega_ids = collect(1:omega_count)
        subproblem_results = pmap(omega_ids) do omega_id
            forwardPass(
                omega_id;
                state = solution_collection[1, iter, 1].state,
            )
        end

        for result in subproblem_results
            solution_collection[2, iter, result.ω] = result.state_info
        end

        lower_bound = solution_collection[1, iter, 1].OPT
        upper_bound = minimum([
            solution_collection[1, iter, 1].stage_value +
            sum(stage_data.p[omega_id] * solution_collection[2, iter, omega_id].OPT for omega_id in 1:omega_count),
            upper_bound,
        ])

        gap = round(abs(upper_bound - lower_bound) / abs(lower_bound) * 100, digits = 2)
        gap_string = string(gap, "%")
        backward_jobs = [(iter, omega_id) for omega_id in omega_ids]
        disjunction_rounds = 0

        # Phase 2: iDBC warm-start backward sweep (optional).
        if cut_selection == :iDBC && idbc_warm_pass
            warm_start_results_vec = pmap(backward_jobs) do (iter_id, omega_id)
                backwardPass(
                    (0.0, omega_id),
                    solution_collection[1, iter_id, 1].state .* local_param_cut.ℓ2 .+ (1 - local_param_cut.ℓ2) / 2,
                    solution_collection[2, iter_id, omega_id].OPT;
                    param = local_param,
                    param_cut = local_param_cut,
                    param_LevelMethod = local_param_levelmethod,
                )
            end
            warm_start_results = Dict(
                omega_id => warm_start_results_vec[idx] for (idx, omega_id) in enumerate(omega_ids)
            )

            aggregated_cut = [0, zeros(Float64, local_param.J)]
            cut_collection = Dict()
            for omega_id in omega_ids
                aggregated_cut = aggregated_cut + stage_data.p[omega_id] .* warm_start_results[omega_id].backward_info.cut_info
                disjunction_rounds += warm_start_results[omega_id].backward_info.iter
                if inherit_disjunctive_cuts
                    for (key, cut) in warm_start_results[omega_id].cut_collection
                        cut_collection[key] = cut
                    end
                end
            end

            if inherit_disjunctive_cuts
                @everywhere begin
                    runtime_context = RuntimeContext.get_context(:sslp)
                    backward_info_list = runtime_context[:backward_info_list]
                    cut_collection = $cut_collection
                    for (key, cut) in cut_collection
                        backward_info_list[key[1]][:cut_expression][key] = cut
                    end
                end
            end
            @constraint(
                forward_model,
                forward_model[:θ] >= aggregated_cut[1] + aggregated_cut[2]' * forward_model[:x],
            )
        end

        # Phase 3: standard backward sweep and cut injection.
        backward_results_vec = pmap(backward_jobs) do (iter_id, omega_id)
            backwardPass(
                (iter_id, omega_id),
                solution_collection[1, iter_id, 1].state,
                solution_collection[2, iter_id, omega_id].OPT;
                param = local_param,
                param_cut = local_param_cut,
                param_LevelMethod = local_param_levelmethod,
            )
        end
        backward_results = Dict(
            omega_id => backward_results_vec[idx] for (idx, omega_id) in enumerate(omega_ids)
        )

        aggregated_cut = [0, zeros(Float64, local_param.J)]
        cut_collection = Dict()
        for omega_id in omega_ids
            aggregated_cut = aggregated_cut + stage_data.p[omega_id] .* backward_results[omega_id].backward_info.cut_info
            disjunction_rounds += backward_results[omega_id].backward_info.iter
            if inherit_disjunctive_cuts
                for (key, cut) in backward_results[omega_id].cut_collection
                    cut_collection[key] = cut
                end
            end
        end

        if inherit_disjunctive_cuts
            @everywhere begin
                runtime_context = RuntimeContext.get_context(:sslp)
                backward_info_list = runtime_context[:backward_info_list]
                cut_collection = $cut_collection
                for (key, cut) in cut_collection
                    backward_info_list[key[1]][:cut_expression][key] = cut
                end
            end
        end

        disjunction_rounds = floor(Int64, disjunction_rounds / omega_count)
        @constraint(
            forward_model,
            forward_model[:θ] >= aggregated_cut[1] + aggregated_cut[2]' * forward_model[:x],
        )

        iter_end_time = now()
        iter_time = (iter_end_time - iter_start_time).value / 1000
        total_time = (iter_end_time - start_time).value / 1000
        push!(sol_history, [iter, lower_bound, local_param.opt, upper_bound, gap_string, iter_time, disjunction_rounds, total_time])
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
