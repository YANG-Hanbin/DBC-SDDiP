"""
    function update_partition_tree!(
        ModelList, 
        state_info, 
        i, 
        t, 
        ω, 
        g, 
        param
    )
# Arguments

    1. `ModelList`: A dictionary of `SDDPModel` objects
    2. `state_info`: StateInfo
    3. `t`: the stage index
    4. `g`: the generator index

    # Returns
    1. `Nothing`
"""
function _resolve_scuc_partition_param(param::Union{Nothing, NamedTuple})::NamedTuple
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
    error("Missing SCUC `param` in partition_tree; pass explicitly or initialize runtime context.")
end

function update_partition_tree!(
    ModelList::Dict{Int64, SDDPModel}, 
    state_info::StateInfo,
    t, 
    g;
    param::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    local_param = _resolve_scuc_partition_param(param)
    # find the active leaf node 
    keys_with_value_1 = maximum([k for (k, v) in state_info.ContVarLeaf[:s][g] if v[:var] > 0.5]); ## find the active leaf node: maximum(values(state_info.stage_solution[:sur][g]))
    # find the lb and ub of this leaf node 
    info = ModelList[t].ContVarLeaf[:s][g][keys_with_value_1]; (lb, ub) = info[:lb], info[:ub]; 
    if local_param.med_method == :IntervalMed
        med = (lb + ub)/2; 
    elseif local_param.med_method == :ExactPoint
        med = state_info.ContVar[:s][g] == nothing ? (lb + ub)/2 : state_info.ContVar[:s][g]; # round(state_info.ContVar[:s][g], digits = 3); 
    end
    # create two new leaf nodes, and update their info (lb, ub)
    left = maximum(keys(ModelList[t].ContVarLeaf[:s][g])) + 1; 
    right = left + 1; 
    ModelList[t].model[:augmentVar][g, left] = @variable(
        ModelList[t].model, 
        base_name = "augmentVar[$g, $left]", 
        binary = true
    ); 
    ModelList[t].model[:augmentVar][g, right] = @variable(
        ModelList[t].model, 
        base_name = "augmentVar[$g, $right]", 
        binary = true
    );
    # delete the parent node and create new leaf nodes
    ModelList[t].ContVarLeaf[:s][g][left] = Dict{Symbol, Any}(
        :lb => lb, 
        :ub => med, 
        :var => ModelList[t].model[:augmentVar][g, left], 
        :sibling => right, 
        :parent => keys_with_value_1
    );
    ModelList[t].ContVarLeaf[:s][g][right] = Dict{Symbol, Any}(
        :lb => med, 
        :ub => ub, 
        :var => ModelList[t].model[:augmentVar][g, right], 
        :sibling => left, 
        :parent => keys_with_value_1
    );
    delete!(ModelList[t].ContVarLeaf[:s][g], keys_with_value_1);

    # add logic constraints
    ## for forward models
    ### Parent-Child relationship
    @constraint(
        ModelList[t].model, 
        ModelList[t].model[:augmentVar][g, left] + 
        ModelList[t].model[:augmentVar][g, right] == 
        ModelList[t].model[:augmentVar][g, keys_with_value_1]
    );
    ### bounding constraints
    if :BoundingConstraintUpper ∈ keys(ModelList[t].model.obj_dict) 
        delete(ModelList[t].model, ModelList[t].model[:BoundingConstraintUpper]);
        delete(ModelList[t].model, ModelList[t].model[:BoundingConstraintLower]);
        unregister(ModelList[t].model, :BoundingConstraintUpper);
        unregister(ModelList[t].model, :BoundingConstraintLower);
    end
    @constraint(
        ModelList[t].model, 
        BoundingConstraintUpper,
        ModelList[t].ContVar[:s][g] ≤ 
        sum(ModelList[t].ContVarLeaf[:s][g][k][:ub] * ModelList[t].ContVarLeaf[:s][g][k][:var] 
            for k in keys(ModelList[t].ContVarLeaf[:s][g]))
    );
    @constraint(
        ModelList[t].model, 
        BoundingConstraintLower,
        ModelList[t].ContVar[:s][g] ≥ 
        sum(
            ModelList[t].ContVarLeaf[:s][g][k][:lb] * ModelList[t].ContVarLeaf[:s][g][k][:var] 
            for k in keys(ModelList[t].ContVarLeaf[:s][g])
        )
    );
    

    ## for backward pass, we need to add the logic constraints and bounding constraints for the copy variables in the next stage
    ### Parent-Child relationship
    enforce_binary_copies = local_param.enforce_binary_copies
    if enforce_binary_copies
        ModelList[t+1].model[:augmentVar_copy][g, left] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $left]", 
            binary = true
        ); 
        ModelList[t+1].model[:augmentVar_copy][g, right] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $right]", 
            binary = true
        );
    else
        ModelList[t+1].model[:augmentVar_copy][g, left] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $left]", 
            lower_bound = 0, 
            upper_bound = 1
        ); 
        ModelList[t+1].model[:augmentVar_copy][g, right] = @variable(
            ModelList[t+1].model, 
            base_name = "augmentVar_copy[$g, $right]", 
            lower_bound = 0, 
            upper_bound = 1
        );
    end

    @constraint(
        ModelList[t+1].model, 
        ModelList[t+1].model[:augmentVar_copy][g, left] + 
        ModelList[t+1].model[:augmentVar_copy][g, right] == 
        ModelList[t+1].model[:augmentVar_copy][g, keys_with_value_1]
    );
    if :AugBoundingConstraintUpper ∈ keys(ModelList[t+1].model.obj_dict) 
        delete(ModelList[t+1].model, ModelList[t+1].model[:AugBoundingConstraintUpper]);
        unregister(ModelList[t+1].model, :AugBoundingConstraintUpper);

        delete(ModelList[t+1].model, ModelList[t+1].model[:AugBoundingConstraintLower]);
        unregister(ModelList[t+1].model, :AugBoundingConstraintLower);
    end
    @constraint(
        ModelList[t+1].model, 
        AugBoundingConstraintUpper,
        ModelList[t+1].model[:s_copy][g] ≤ 
        sum(ModelList[t].ContVarLeaf[:s][g][k][:ub] * ModelList[t+1].model[:augmentVar_copy][g, k] for k in keys(ModelList[t].ContVarLeaf[:s][g]))
    );
    @constraint(
        ModelList[t+1].model, 
        AugBoundingConstraintLower,
        ModelList[t+1].model[:s_copy][g] ≥ 
        sum(ModelList[t].ContVarLeaf[:s][g][k][:lb] * ModelList[t+1].model[:augmentVar_copy][g, k] for k in keys(ModelList[t].ContVarLeaf[:s][g]))
    );

    ## update the stage_decision
    if state_info.ContVar[:s][g] ≤ med
        state_info.ContAugState[:s][g] = Dict{Any, Any}(
            left => 1.0 for k in keys(ModelList[t].ContVarLeaf[:s][g])
        );
    else
        state_info.ContAugState[:s][g] = Dict{Any, Any}(
            right => 1.0 for k in keys(ModelList[t].ContVarLeaf[:s][g])
        );
    end
    return
end
