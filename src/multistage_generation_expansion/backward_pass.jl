if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    function AddNonAnticipative!(
        model::Model,
        state::Vector{Float64}
    )::Nothing

# Arguments
  1. `model`: stage model

# usage: 
        Add the non-anticipative constraints

"""
function AddNonAnticipative!(
    model::Model,
    state::Vector{Float64}
)::Nothing

    if :NonAnticipative ∉ keys(model.obj_dict) 
        @constraint(
            model, 
            NonAnticipative[i in 1:length(model[:Lt])], 
            model[:Lc][i] == state[i]
        );
    else 
        for i in 1:length(model[:Lt])
            delete(
                model, 
                model[:NonAnticipative][i]
            );
        end
        unregister(
            model, 
            :NonAnticipative
        );
        @constraint(
            model, 
            NonAnticipative[i in 1:length(model[:Lt])], 
            model[:Lc][i] == state[i]
        );
    end
    return
end

"""
    function RemoveNonAnticipative!(
        model::Model
    )::Nothing

# Arguments
  1. `model`: stage model

# usage: 
        Remove the non-anticipative constraints

"""
function RemoveNonAnticipative!(
    model::Model
)::Nothing

    if :NonAnticipative ∉ keys(model.obj_dict) 
        return 
    else 
        for i in 1:length(model[:Lt])
            delete(
                model, 
                model[:NonAnticipative][i]
            );
        end
        unregister(
            model, 
            :NonAnticipative
        );
        return 
    end
end

"""
    function setup_levelset_param(
        model::Model, 
        state::Vector{Float64},
        param::NamedTuple,
        param_cut::NamedTuple,
        param_LevelMethod::NamedTuple
    )::NamedTuple

# Arguments
  1. `model`: stage model
  2. `state`: current state
# usage: 
        Setup the level set method parameters

"""
function setup_levelset_param(
    model::Model, 
    state::Vector{Float64},
    param::NamedTuple,
    param_cut::NamedTuple,
    param_LevelMethod::NamedTuple
)::NamedTuple
    if param.cut_selection == :PLC
        optimize!(model); 
        f_star_value = JuMP.objective_value(model);
        core_point = state .* param_cut.ℓ2 .+ (1 - param_cut.ℓ2)/2;

        threshold = 1.0; 
        level_set_method_param = LevelSetMethodParam(
            0.95, 
            param_LevelMethod.λ, 
            threshold, 
            1e14, 
            2e2,  
            state, 
            param.cut_selection, 
            core_point, 
            f_star_value
        );

    elseif param.cut_selection == :SMC 
        optimize!(model); 
        f_star_value = JuMP.objective_value(model);
        threshold = 1.0; 
        level_set_method_param = LevelSetMethodParam(
            0.95, 
            param_LevelMethod.λ, 
            threshold, 
            1e14, 
            1e2, 
            state, 
            param.cut_selection, 
            nothing, 
            f_star_value
        );

    elseif param.cut_selection ∈ [:LC, :SBC]
        f_star_value = 0.0;
        threshold = 1.0; 
        level_set_method_param = LevelSetMethodParam(
            0.95, 
            param_LevelMethod.λ, 
            threshold, 
            1e14, 
            1e2,  
            state, 
            param.cut_selection, 
            nothing, 
            f_star_value
        );
    end

    x₀ =  state .* 0.0;

    return (
        level_set_method_param = level_set_method_param, 
        x₀ = x₀
    );
end

"""
    function lagrangian_cut_generation(
        model::Model, 
        state::Vector{Float64},
        param::NamedTuple,
        param_cut::NamedTuple,
        param_LevelMethod::NamedTuple
    )::Vector

# Arguments
  1. `model`: stage model
# usage: 
    Generate Lagrangian cuts

"""
function lagrangian_cut_generation(
    model::Model, 
    state::Vector{Float64},
    param::NamedTuple,
    param_cut::NamedTuple,
    param_LevelMethod::NamedTuple
)::NamedTuple

    (level_set_method_param, x₀) = setup_levelset_param(
        model, 
        state,
        param,
        param_cut,
        param_LevelMethod,
    );

    RemoveNonAnticipative!(model);

    (cut_info, iter) = LevelSetMethod_optimization!(
        model, 
        x₀,
        level_set_method_param,
        param,
        param_cut,
        param_LevelMethod
    );

    return (
        cut_info = cut_info, 
        iter = iter
    )
end

"""
    function strengthened_benders_cut_generation(
        model::Model, 
        state::Vector{Float64},
        param::NamedTuple,
        param_cut::NamedTuple,
        param_LevelMethod::NamedTuple
    )::Vector

# Arguments
  1. `model`: stage model
# usage: 
    Generate strengthened Benders' cuts

"""
function strengthened_benders_cut_generation(
    model::Model, 
    state::Vector{Float64},
    param::NamedTuple,
    param_cut::NamedTuple,
    param_LevelMethod::NamedTuple
)::NamedTuple
    @objective(
        model, 
        Min, 
        model[:primal_objective_expression]
    );
    lp_model = relax_integrality(model); ## generate the LP relaxation
    optimize!(model);
    slope = nothing; cut_info = nothing;
    if has_duals(model)
        slope = Vector{Float64}(undef, length(model[:NonAnticipative]));
        for j in 1:length(model[:NonAnticipative])
            slope[j] = dual(model[:NonAnticipative][j])
        end
        cut_info =  [ 
            JuMP.objective_value(model) - slope'state, 
            slope
        ];
    else
        @warn("No Benders' Cut Has been Generated.")
    end

    ## recover the integrality
    lp_model();                                

    (level_set_method_param, x₀) = setup_levelset_param(
        model, 
        state,
        param,
        param_cut,
        param_LevelMethod,
    );
    RemoveNonAnticipative!(model);

    (cut_info, iter) = LevelSetMethod_optimization!(
        model, 
        slope,
        level_set_method_param,
        param,
        param_cut,
        param_LevelMethod
    );

    return (
        cut_info = cut_info, 
        iter = iter
    )
end

"""
    backwardPass(backward_node_info)

    Backward pass routine for parallel execution.
"""
function backwardPass(
    backward_node_info::Tuple;
    forward_info_list::Union{Nothing, Dict{Int, Model}} = nothing,
    scenario_realizations::Union{Nothing, Dict{Int64, Dict{Int64, RandomVariables}}} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
    param_cut::Union{Nothing, NamedTuple} = nothing,
    param_LevelMethod::Union{Nothing, NamedTuple} = nothing,
    prob_list::Union{Nothing, Dict{Int64, Vector{Float64}}} = nothing,
    sol_collection::Union{Nothing, Dict{Any, Any}} = nothing,
    state_override::Union{Nothing, Vector{Float64}} = nothing,
)::NamedTuple
    runtime_context = RuntimeContext.get_context(:generation_expansion)
    local_forward_info_list = isnothing(forward_info_list) ? runtime_context[:forward_info_list] : forward_info_list
    local_scenario_realizations = isnothing(scenario_realizations) ? runtime_context[:scenario_realizations] : scenario_realizations
    local_param = isnothing(param) ? runtime_context[:param] : param
    local_param_cut = isnothing(param_cut) ? runtime_context[:param_cut] : param_cut
    local_param_levelmethod = isnothing(param_LevelMethod) ? runtime_context[:param_level_method] : param_LevelMethod
    local_prob_list = isnothing(prob_list) ? runtime_context[:prob_list] : prob_list
    local_sol_collection = isnothing(sol_collection) ? Dict{Any, Any}() : sol_collection

    (iteration_index, stage_index, node_index, sample_path_id) = backward_node_info
    runtime_param = resolve_ge_runtime_params(local_param)
    cut_selection = runtime_param.cut_selection
    parent_state = isnothing(state_override) ? local_sol_collection[stage_index - 1, sample_path_id].stage_solution : state_override
    disjunction_iteration_limit = runtime_param.disjunction_iteration_limit

    if cut_selection ∈ [:PLC, :LC, :SMC]
        ModelModification!(
            local_forward_info_list[stage_index],
            local_scenario_realizations[stage_index][node_index].d,
            parent_state
        );
        (cut_info, iter) = lagrangian_cut_generation(
            local_forward_info_list[stage_index],
            parent_state,
            local_param,
            local_param_cut,
            local_param_levelmethod
        );
    elseif cut_selection == :SBC
        ModelModification!(
            local_forward_info_list[stage_index],
            local_scenario_realizations[stage_index][node_index].d,
            parent_state
        );
        (cut_info, iter) = strengthened_benders_cut_generation(
            local_forward_info_list[stage_index],
            parent_state,
            local_param,
            local_param_cut,
            local_param_levelmethod
        );                          
    elseif cut_selection ∈ [:DBC, :iDBC, :SPC, :BC]

        if cut_selection == :SPC
            MDCiter = 1
        elseif cut_selection == :BC
            # BC is the no-disjunction baseline in the disjunctive-CPT family.
            MDCiter = 0
        elseif disjunction_iteration_limit < 0 
            ## we increase the number of disjunctions by 1 at each iteration
            MDCiter = iteration_index;
        else
            ## we use the fixed number of disjunctions
            MDCiter = disjunction_iteration_limit;
        end
        ModelModification!(
            local_forward_info_list[stage_index],
            local_scenario_realizations[stage_index][node_index].d,
            parent_state
        );
        (cut_info, iter) = CPT_optimization!(
            local_forward_info_list[stage_index],
            parent_state, 
            node_index;
            param = local_param,
            param_cut = local_param_cut,
            param_LevelMethod = local_param_levelmethod,
            MDCiter = MDCiter
        );
        RemoveNonAnticipative!(local_forward_info_list[stage_index]);
    elseif cut_selection == :FC
        if disjunction_iteration_limit < 0 
            ## we increase the number of disjunctions by 1 at each iteration
            MDCiter = iteration_index;
        else
            ## we use the fixed number of disjunctions
            MDCiter = disjunction_iteration_limit;
        end
        ModelModification!(
            local_forward_info_list[stage_index],
            local_scenario_realizations[stage_index][node_index].d,
            parent_state
        );
        (cut_info, iter) = Fenchel_cut_optimization!(
            local_forward_info_list[stage_index],
            parent_state, 
            node_index;
            param = local_param,
            param_cut = local_param_cut,
            param_LevelMethod = local_param_levelmethod,
            MDCiter = MDCiter
        );
        RemoveNonAnticipative!(local_forward_info_list[stage_index]);
    else
        error("Unknown cut selection method: $(cut_selection)")
    end

    return (
        cut_info = cut_info, 
        iter = iter, 
        cut_collection = local_forward_info_list[stage_index][:cut_expression]
    )
end
