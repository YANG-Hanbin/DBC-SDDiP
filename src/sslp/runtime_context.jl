if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    initialize_sslp_runtime_context!(
        stage_data::StageData,
        random_variables::RandomVariables,
        param::NamedTuple,
        omega_count::Int64;
        param_cut::Union{Nothing, NamedTuple} = nothing,
        param_level_method::Union{Nothing, NamedTuple} = nothing,
    )::Nothing

# Purpose
    Build and register worker-local SSLP runtime context objects.
"""
function initialize_sslp_runtime_context!(
    stage_data_input::StageData,
    random_variables_input::RandomVariables,
    param_input::NamedTuple,
    omega_count_input::Int64;
    param_cut_input::Union{Nothing, NamedTuple} = nothing,
    param_level_method_input::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    runtime_stage_data = stage_data_input
    runtime_random_variables = random_variables_input
    runtime_param = param_input
    runtime_omega_count = omega_count_input
    runtime_param_cut = param_cut_input
    runtime_param_level_method = param_level_method_input

    local_forward_model = forwardModel!(
        runtime_stage_data,
        runtime_param,
    )
    local_backward_info_list = Dict{Int, Model}()
    for omega_id in 1:runtime_omega_count
        local_backward_info_list[omega_id] = backward_model!(
            runtime_stage_data,
            runtime_random_variables,
            omega_id,
            runtime_param,
        )
    end

    RuntimeContext.set_context!(
        :sslp;
        stage_data = runtime_stage_data,
        random_variables = runtime_random_variables,
        param = runtime_param,
        param_cut = runtime_param_cut,
        param_level_method = runtime_param_level_method,
        omega_count = runtime_omega_count,
        forward_model = local_forward_model,
        backward_info_list = local_backward_info_list,
    )

    # Compatibility aliases for legacy code paths that may still access globals.
    global forwardModel = local_forward_model
    global backwardInfoList = local_backward_info_list
    global forward_model = local_forward_model
    global backward_info_list = local_backward_info_list
    global param = runtime_param
    global param_cut = runtime_param_cut
    global param_LevelMethod = runtime_param_level_method
    return nothing
end
