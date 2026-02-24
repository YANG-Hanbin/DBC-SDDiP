export backwardPass
if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

function _resolve_scuc_backward_param(param::Union{Nothing, NamedTuple})::NamedTuple
    if !isnothing(param)
        return param
    end
    if RuntimeContext.has_context(:scuc)
        context = RuntimeContext.get_context(:scuc)
        if haskey(context, :param) && context[:param] isa NamedTuple
            return context[:param]
        end
    end
    if isdefined(Main, :param) && getfield(Main, :param) isa NamedTuple
        return getfield(Main, :param)
    end
    error("Missing SCUC `param` in backward_pass; pass explicitly or initialize runtime context.")
end

function _resolve_scuc_backward_index_sets(index_sets::Union{Nothing, IndexSets})::IndexSets
    if !isnothing(index_sets)
        return index_sets
    end
    if RuntimeContext.has_context(:scuc)
        context = RuntimeContext.get_context(:scuc)
        if haskey(context, :index_sets) && context[:index_sets] isa IndexSets
            return context[:index_sets]
        end
    end
    if isdefined(Main, :index_sets) && getfield(Main, :index_sets) isa IndexSets
        return getfield(Main, :index_sets)
    end
    if isdefined(Main, :indexSets) && getfield(Main, :indexSets) isa IndexSets
        return getfield(Main, :indexSets)
    end
    error("Missing SCUC `index_sets` in backward_pass; pass explicitly or initialize runtime context.")
end

function _resolve_scuc_backward_model(model::Union{Nothing, Model})::Model
    if !isnothing(model)
        return model
    end
    if isdefined(Main, :model) && getfield(Main, :model) isa Model
        return getfield(Main, :model)
    end
    error("Missing SCUC `model` in backward_pass; pass explicitly.")
end
"""
AddContVarNonAnticipative!(; model::Model = model)

# Arguments

    1. `model::Model` : a forward pass model of stage t
    2. `state_info::StateInfo` : the parent's node decisions
    3. `index_sets::IndexSets` : the index sets of the model
    4. `param::NamedTuple` : the parameters of the model
  
# Modification
    1. Add the Non-anticipativity constraints
"""
function AddContVarNonAnticipative!( 
    model::Model, 
    state_info::StateInfo;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    local_index_sets = _resolve_scuc_backward_index_sets(index_sets)
    local_param = _resolve_scuc_backward_param(param)
    if local_param.algorithm !== :SDDiP
        if :ContVarNonAnticipative ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                ContVarNonAnticipative[g in local_index_sets.G], 
                model[:s_copy][g] == state_info.ContVar[:s][g]
            );
        end

        if :BinVarNonAnticipative ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                BinVarNonAnticipative[g in local_index_sets.G], 
                model[:y_copy][g] == state_info.BinVar[:y][g]
            );
        end
    elseif local_param.algorithm == :SDDiP
        if :BinarizationNonAnticipative ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                BinarizationNonAnticipative[g in local_index_sets.G, i in 1:local_param.kappa[g]], 
                model[:λ_copy][g, i] == state_info.ContStateBin[:s][g][i]
            );
        end
        if :BinVarNonAnticipative ∉ keys(model.obj_dict) 
            @constraint(
                model, 
                BinVarNonAnticipative[g in local_index_sets.G], 
                model[:y_copy][g] == state_info.BinVar[:y][g]
            );
        end
    end

    return
end

"""
RemoveContVarNonAnticipative!(model::Model)

# Arguments

    1. `model::Model` : a nodal problem
  
# Modification
    1. Remove the Non-anticipativity constraints
"""
function RemoveContVarNonAnticipative!(
    model::Union{Nothing, Model} = nothing;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    local_model = _resolve_scuc_backward_model(model)
    local_index_sets = _resolve_scuc_backward_index_sets(index_sets)
    local_param = _resolve_scuc_backward_param(param)

    if :λ_copy ∉ keys(local_model.obj_dict)
        for g in local_index_sets.G
            delete(local_model, local_model[:ContVarNonAnticipative][g]);
            delete(local_model, local_model[:BinVarNonAnticipative][g]);
        end
        unregister(local_model, :ContVarNonAnticipative);
        unregister(local_model, :BinVarNonAnticipative);
    else
        for g in local_index_sets.G
            delete(local_model, local_model[:BinVarNonAnticipative][g]);
            for i in 1:local_param.kappa[g]
                delete(local_model, local_model[:BinarizationNonAnticipative][g, i]);
            end
        end
        unregister(local_model, :BinVarNonAnticipative);
        unregister(local_model, :BinarizationNonAnticipative);
    end
    
    return
end

"""
    setup_initial_point(state_info::StateInfo)
# Arguments
    state_info::StateInfo : the parent's node decisions
    Utility routine for setting up the initial dual variables.
"""
function setup_initial_point(
    state_info::StateInfo;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::StateInfo
    local_index_sets = _resolve_scuc_backward_index_sets(index_sets)
    local_param = _resolve_scuc_backward_param(param)
    BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
        g => 0.0 for g in local_index_sets.G)
    );
    ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
        g => 0.0 for g in local_index_sets.G)
    );
    if state_info.ContAugState == nothing 
        ContAugState = nothing
    else
        ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    k => 0.0 for k in keys(state_info.ContAugState[:s][g])
                ) for g in local_index_sets.G
            )
        );
    end

    if state_info.ContStateBin == nothing 
        ContStateBin = nothing
    else
        ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => 0.0 for i in 1:local_param.kappa[g]
                ) for g in local_index_sets.G
            )
        );
    end

    return StateInfo(
        BinVar, 
        nothing, 
        ContVar, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        ContAugState,
        nothing,
        ContStateBin
    );
end

"""
    backwardPass(backward_node_info)

    Backward pass routine for parallel execution.
"""
function backwardPass(
    backward_node_info::Tuple; 
    ModelList::Union{Nothing, Dict{Int64, SDDPModel}} = nothing,
    index_sets::Union{Nothing, IndexSets} = nothing,
    param_demand::Union{Nothing, ParamDemand} = nothing,
    param_opf::Union{Nothing, ParamOPF} = nothing,
    scenario_tree::Union{Nothing, ScenarioTree} = nothing,
    state_info_collection::Union{Nothing, Dict{Any, Any}} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
    param_cut::Union{Nothing, NamedTuple} = nothing,
    param_levelsetmethod::Union{Nothing, NamedTuple} = nothing,
    parent_state::Union{Nothing, StateInfo} = nothing,
    stage_state_value::Union{Nothing, Float64} = nothing,
    state_override::Union{Nothing, StateInfo} = nothing,
)
    runtime_context = RuntimeContext.get_context(:scuc)
    local_model_list = isnothing(ModelList) ? runtime_context[:model_list] : ModelList
    local_index_sets = isnothing(index_sets) ? runtime_context[:index_sets] : index_sets
    local_param_demand = isnothing(param_demand) ? runtime_context[:param_demand] : param_demand
    local_param_opf = isnothing(param_opf) ? runtime_context[:param_opf] : param_opf
    local_scenario_tree = isnothing(scenario_tree) ? runtime_context[:scenario_tree] : scenario_tree
    local_state_info_collection = isnothing(state_info_collection) ? runtime_context[:state_info_collection] : state_info_collection
    local_param = isnothing(param) ? runtime_context[:param] : param
    local_param_cut = isnothing(param_cut) ? runtime_context[:param_cut] : param_cut
    local_param_levelsetmethod = isnothing(param_levelsetmethod) ? runtime_context[:param_level_method] : param_levelsetmethod

    (
        iteration_index,
        stage_index,
        node_index,
        scenario_index,
        cut_selection,
        core_point_strategy,
    ) = backward_node_info
    runtime_param = resolve_scuc_runtime_params(local_param)
    cut_selection = _normalize_cut_selection(cut_selection)
    parent_state = if !isnothing(state_override)
        state_override
    elseif !isnothing(parent_state)
        parent_state
    else
        local_state_info_collection[iteration_index, stage_index - 1, scenario_index]
    end
    current_stage_value = isnothing(stage_state_value) ?
        local_state_info_collection[iteration_index, stage_index, scenario_index].StateValue :
        stage_state_value
    imdc_state_projection = runtime_param.imdc_state_projection
    disjunction_iteration_limit = runtime_param.disjunction_iteration_limit
    if local_param.algorithm == :SDDiP && cut_selection == :iDBC && imdc_state_projection
        parent_state = sanitize_imdc_copy_state(
            parent_state;
            enforce_binary = true,
            zero_copy_when_off = true,
            index_sets = local_index_sets,
            param = local_param,
        )
    end
    ModelModification!( 
        local_model_list[stage_index].model,
        local_scenario_tree.tree[stage_index].nodes[node_index],
        local_param_demand,
        parent_state;
        index_sets = local_index_sets,
        param = local_param,
    );
    if cut_selection ∈ [:DBC, :iDBC, :SPC, :BC]
        if cut_selection == :SPC
            MDCiter = 1
        elseif cut_selection == :BC
            # BC is the no-disjunction baseline in the disjunctive-CPT family.
            MDCiter = 0
        elseif disjunction_iteration_limit < 0 
            ## Paper-style DBC schedule: increase max disjunction count by one each iteration.
            MDCiter = iteration_index - 1.;
        else
            ## we use the fixed number of disjunctions
            MDCiter = disjunction_iteration_limit;
        end
        ((λ₀, λ₁), LMiter) = CPT_optimization!(
            local_model_list[stage_index].model,
            parent_state,
            node_index;
            index_sets = local_index_sets,
            param_demand = local_param_demand,
            param_opf = local_param_opf,
            param = local_param,
            MDCiter = MDCiter
        );
        RemoveContVarNonAnticipative!(
            local_model_list[stage_index].model;
            index_sets = local_index_sets,
            param = local_param
        );
        return (
            (λ₀, λ₁), 
            LMiter, 
            local_model_list[stage_index].model[:cut_expression]
        )  
    elseif cut_selection == :FC 
        if disjunction_iteration_limit < 0 
            ## Paper-style DBC schedule: increase max disjunction count by one each iteration.
            MDCiter = iteration_index;
        else
            ## we use the fixed number of disjunctions
            MDCiter = disjunction_iteration_limit;
        end
        ((λ₀, λ₁), LMiter) = Fenchel_cut_optimization!(
            local_model_list[stage_index].model,
            parent_state,
            node_index;
            index_sets = local_index_sets,
            param_demand = local_param_demand,
            param_opf = local_param_opf,
            param = local_param,
            param_levelsetmethod = local_param_levelsetmethod,
            MDCiter = MDCiter
        );
        RemoveContVarNonAnticipative!(
            local_model_list[stage_index].model;
            index_sets = local_index_sets,
            param = local_param
        );

        return (
            (λ₀, λ₁), 
            LMiter, 
            local_model_list[stage_index].model[:cut_expression]
        )  
    elseif cut_selection == :SBC
        if disjunction_iteration_limit < 0 
            ## Paper-style DBC schedule: increase max disjunction count by one each iteration.
            MDCiter = iteration_index;
        else
            ## we use the fixed number of disjunctions
            MDCiter = disjunction_iteration_limit;
        end
        ((λ₀, λ₁), LMiter) = CPT_optimization!(
            local_model_list[stage_index].model,
            parent_state,
            node_index;
            index_sets = local_index_sets,
            param_demand = local_param_demand,
            param_opf = local_param_opf,
            param = local_param,
            MDCiter = MDCiter
        );
        RemoveContVarNonAnticipative!(
            local_model_list[stage_index].model;
            index_sets = local_index_sets,
            param = local_param
        );

        CutGenerationInfo = StrengthenedBendersCutGeneration{StateInfo}(
            λ₁
        );

        (λ₀, λ₁) = StrengthedBendersCut_optimization!(
            local_model_list[stage_index].model,
            parent_state,
            CutGenerationInfo;
            index_sets = local_index_sets,
            param_demand = local_param_demand,
            param_opf = local_param_opf,
            param = local_param,
            param_levelsetmethod = local_param_levelsetmethod
        );

        return (
            (λ₀, λ₁), 
            LMiter, 
            local_model_list[stage_index].model[:cut_expression]
        )  

    end


    RemoveContVarNonAnticipative!(
        local_model_list[stage_index].model;
        index_sets = local_index_sets,
        param = local_param
    );

    if cut_selection == :PLC
        CutGenerationInfo = ParetoLagrangianCutGeneration{Float64}(
            core_point_strategy, 
            setup_core_point(
                parent_state;
                index_sets = local_index_sets,
                param_opf = local_param_opf,
                param_cut = local_param_cut,
                param = local_param
            ), 
            local_param_cut.δ,
            current_stage_value
        );
    elseif cut_selection == :LC
        CutGenerationInfo = LagrangianCutGeneration{Float64}(
            current_stage_value
        );
    elseif cut_selection == :SMC
        CutGenerationInfo = SquareMinimizationCutGeneration{Float64}(
            local_param_cut.δ,
            current_stage_value
        );
    else
        @warn "Invalid cut_selection value: $cut_selection. Defaulting to :SMC."
        CutGenerationInfo = SquareMinimizationCutGeneration{Float64}(
            local_param_cut.δ,
            current_stage_value
        )
    end

    levelsetmethod_oracle_param = SetupLevelSetMethodOracleParam(
        parent_state;
        index_sets = local_index_sets,
        param = local_param,
        param_levelsetmethod = local_param_levelsetmethod
    );
    
    ((λ₀, λ₁), LMiter) = LevelSetMethod_optimization!(
        local_model_list[stage_index].model,
        levelsetmethod_oracle_param, 
        parent_state,
        CutGenerationInfo;
        index_sets = local_index_sets,
        param_demand = local_param_demand,
        param_opf = local_param_opf,
        param = local_param,
        param_levelsetmethod = local_param_levelsetmethod
    );

    return (
        (λ₀, λ₁), 
        LMiter, 
        local_model_list[stage_index].model[:cut_expression]
    )
end
