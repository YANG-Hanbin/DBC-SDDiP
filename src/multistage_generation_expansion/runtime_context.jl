if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    initialize_ge_runtime_context!(
        scenario_realizations::Dict{Int64, Dict{Int64, RandomVariables}},
        prob_list::Dict{Int64, Vector{Float64}},
        stage_data_list::Dict{Int64, StageData},
        param::NamedTuple;
        param_cut::Union{Nothing, NamedTuple} = nothing,
        param_level_method::Union{Nothing, NamedTuple} = nothing,
    )::Nothing

# Purpose
    Build and register worker-local generation-expansion runtime context objects.
"""
function initialize_ge_runtime_context!(
    scenario_realizations_input::Dict{Int64, Dict{Int64, RandomVariables}},
    prob_list_input::Dict{Int64, Vector{Float64}},
    stage_data_list_input::Dict{Int64, StageData},
    param_input::NamedTuple;
    param_cut_input::Union{Nothing, NamedTuple} = nothing,
    param_level_method_input::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    runtime_scenario_realizations = scenario_realizations_input
    runtime_prob_list = prob_list_input
    runtime_stage_data_list = stage_data_list_input
    runtime_param = param_input
    runtime_param_cut = param_cut_input
    runtime_param_level_method = param_level_method_input

    local_forward_info_list = Dict{Int, Model}()
    for t in 1:runtime_param.T
        local_forward_info_list[t] = forwardModel!(
            runtime_stage_data_list[t],
            runtime_param,
        )
    end

    RuntimeContext.set_context!(
        :generation_expansion;
        scenario_realizations = runtime_scenario_realizations,
        prob_list = runtime_prob_list,
        stage_data_list = runtime_stage_data_list,
        param = runtime_param,
        param_cut = runtime_param_cut,
        param_level_method = runtime_param_level_method,
        forward_info_list = local_forward_info_list,
    )

    # Compatibility aliases for legacy code paths that may still access globals.
    global Ω = runtime_scenario_realizations
    global forwardInfoList = local_forward_info_list
    global forward_info_list = local_forward_info_list
    global prob_list = runtime_prob_list
    global param = runtime_param
    global param_cut = runtime_param_cut
    global param_LevelMethod = runtime_param_level_method
    return nothing
end
