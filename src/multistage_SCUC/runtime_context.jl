if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    initialize_scuc_runtime_context!(
        scenario_tree::ScenarioTree,
        index_sets::IndexSets,
        param_demand::ParamDemand,
        param_opf::ParamOPF,
        initial_state_info::StateInfo,
        param::NamedTuple;
        param_cut::Union{Nothing, NamedTuple} = nothing,
        param_level_method::Union{Nothing, NamedTuple} = nothing,
    )::Nothing

# Purpose
    Build and register worker-local SCUC runtime context objects.
"""
function initialize_scuc_runtime_context!(
    scenario_tree_input::ScenarioTree,
    index_sets_input::IndexSets,
    param_demand_input::ParamDemand,
    param_opf_input::ParamOPF,
    initial_state_info_input::StateInfo,
    param_input::NamedTuple;
    param_cut_input::Union{Nothing, NamedTuple} = nothing,
    param_level_method_input::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    runtime_scenario_tree = scenario_tree_input
    runtime_index_sets = index_sets_input
    runtime_param_demand = param_demand_input
    runtime_param_opf = param_opf_input
    runtime_initial_state_info = initial_state_info_input
    runtime_param = param_input
    runtime_param_cut = param_cut_input
    runtime_param_level_method = param_level_method_input

    if runtime_param.algorithm == :SDDiP
        for g in runtime_index_sets.G
            if runtime_param_opf.smax[g] >= runtime_param.epsilon
                runtime_param.kappa[g] = ceil(Int, log2(runtime_param_opf.smax[g] / runtime_param.epsilon))
            else
                runtime_param.kappa[g] = 1
            end
        end

        cont_state_bin = Dict(
            g => binarize_continuous_variable(runtime_initial_state_info.ContVar[:s][g], runtime_param_opf.smax[g], runtime_param)
            for g in runtime_index_sets.G
        )
        runtime_initial_state_info.ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(bit_idx => cont_state_bin[g][bit_idx] for bit_idx in 1:runtime_param.kappa[g])
                for g in runtime_index_sets.G
            )
        )
    end

    model_list = Dict{Int, SDDPModel}()
    for t in 1:runtime_index_sets.T
        model_list[t] = forwardModel!(
            runtime_param_demand,
            runtime_param_opf,
            runtime_scenario_tree.tree[t];
            index_sets = runtime_index_sets,
            param = runtime_param,
        )
    end

    RuntimeContext.set_context!(
        :scuc;
        scenario_tree = runtime_scenario_tree,
        index_sets = runtime_index_sets,
        param_demand = runtime_param_demand,
        param_opf = runtime_param_opf,
        initial_state_info = runtime_initial_state_info,
        param = runtime_param,
        param_cut = runtime_param_cut,
        param_level_method = runtime_param_level_method,
        model_list = model_list,
        state_info_collection = Dict(),
    )

    # Compatibility aliases for legacy code paths that may still access globals.
    global scenarioTree = runtime_scenario_tree
    global indexSets = runtime_index_sets
    global paramDemand = runtime_param_demand
    global paramOPF = runtime_param_opf
    global initialStateInfo = runtime_initial_state_info
    global scenario_tree = runtime_scenario_tree
    global index_sets = runtime_index_sets
    global param_demand = runtime_param_demand
    global param_opf = runtime_param_opf
    global initial_state_info = runtime_initial_state_info
    global ModelList = model_list
    global stateInfoCollection = Dict()
    global state_info_collection = Dict()
    global param = runtime_param
    global param_cut = runtime_param_cut
    global param_levelsetmethod = runtime_param_level_method
    return nothing
end
