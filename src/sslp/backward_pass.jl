
if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end


#############################################################################################
###################################  function: backward pass ################################
#############################################################################################
"""
    function backward_model!(
        stage_data::StageData,
        random_variables::RandomVariables,
        ω::Int64,
        param::NamedTuple
    )::Model

# Arguments
  1. `stage_data`: stage data
  2. `random_variables`: random variable
  3. `ω`: scenario index
  2. `param`: Named tuple of parameters for the PLC algorithm

# usage: 
        Create a backward model for scenario ω
"""
function backward_model!(
    stage_data::StageData,
    random_variables::RandomVariables,
    ω::Int64,
    param::NamedTuple
)::Model
    runtime_param = resolve_sslp_runtime_params(param)
    enforce_binary_copies = runtime_param.enforce_binary_copies
    time_limit = runtime_param.time_limit

    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)); 
    MOI.set(model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(model, "TimeLimit", time_limit);
    set_optimizer_attribute(model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(model, "FeasibilityTol", param.feasibility_tol);

    @variable(model, 0 ≤ y[i = 1:param.I, j = 1:param.J] ≤ stage_data.v, Int); 
    @variable(model, y₀[j = 1:param.J] ≥ 0); 

    # auxiliary variable (copy variable)
    if enforce_binary_copies  
        @variable(model, z[j = 1:param.J], Bin);          
    else
        @variable(model, 0 ≤ z[j = 1:param.J] ≤ 1);                     
    end

    @constraint(
        model, 
        [j in 1:param.J],
        sum(stage_data.d[i, j] * y[i, j] for i in 1:param.I) - y₀[j] ≤ stage_data.w * z[j]
    );                       
    @constraint(
        model, 
        client_availability[i in 1:param.I],
        sum(y[i, j] for j in 1:param.J) == random_variables.h[i, ω]
    );   

    @expression(
        model, 
        primal_objective_expression, 
        sum(stage_data.qₒ[j] * y₀[j] for j in 1:param.J) - sum(sum(stage_data.d[i, j] * y[i, j] for j in 1:param.J) for i in 1:param.I)
    );
    @objective(
        model, 
        Min, 
        primal_objective_expression
    );
    
    model[:disjunctive_cuts] = Dict();
    model[:cut_expression] = Dict();

    return model
end

"""
    function nonanticipativity_constraint(
        model::Model, 
        state::Vector
    )::Nothing
"""
function nonanticipativity_constraint(
    model::Model, 
    state::Vector
)::Nothing

    if :NonAnticipative ∈ keys(model.obj_dict) 
        delete(
            model, 
            model[:NonAnticipative]
        );
        unregister(
            model, 
            :NonAnticipative
        );
    end
    
    @objective(
        model, 
        Min, 
        model[:primal_objective_expression]
    );

    @constraint(
        model, 
        NonAnticipative[j in 1:length(state)], 
        model[:z][j] == state[j]
    );
    return 
end

"""
    function add_nonanticipativity_constraint(
        model::Model
    )::Nothing
"""
function add_nonanticipativity_constraint(
    model::Model,
    state::Vector
)::Nothing

    if :NonAnticipative ∈ keys(model.obj_dict) 
        J = length(model[:NonAnticipative]);
        for j in 1:J
            delete(
                model, 
                model[:NonAnticipative][j]
            );
        end
        unregister(
            model, 
            :NonAnticipative
        );
    end

    @constraint(
        model, 
        NonAnticipative[j in 1:length(state)], 
        model[:z][j] == state[j]
    );

    return 
end

"""
    function remove_nonanticipativity_constraint(
        model::Model
    )::Nothing
"""
function remove_nonanticipativity_constraint(
    model::Model
)::Nothing

    if :NonAnticipative ∈ keys(model.obj_dict) 
        J = length(model[:NonAnticipative]);
        for j in 1:J
            delete(
                model, 
                model[:NonAnticipative][j]
            );
        end
        unregister(
            model, 
            :NonAnticipative
        );
    end

    return 
end

"""
    backwardPass(backward_node_info)

    Backward pass routine for parallel execution.
"""
function backwardPass(
    backward_node_info::Tuple,
    state::Vector,
    f_star_value::Float64;
    backward_info_list::Union{Nothing, Dict{Int64, Model}} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
    param_cut::Union{Nothing, NamedTuple} = nothing,
    param_LevelMethod::Union{Nothing, NamedTuple} = nothing,
)::NamedTuple
    runtime_context = RuntimeContext.get_context(:sslp)
    local_backward_info_list = isnothing(backward_info_list) ? runtime_context[:backward_info_list] : backward_info_list
    local_param = isnothing(param) ? runtime_context[:param] : param
    local_param_cut = isnothing(param_cut) ? runtime_context[:param_cut] : param_cut
    local_param_levelmethod = isnothing(param_LevelMethod) ? runtime_context[:param_level_method] : param_LevelMethod

    (iteration_index, scenario_index) = backward_node_info
    runtime_param = resolve_sslp_runtime_params(local_param)
    cut_selection = runtime_param.cut_selection
    disjunction_iteration_limit = runtime_param.disjunction_iteration_limit
    
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    existing_cut_keys = if inherit_disjunctive_cuts
        Set(keys(local_backward_info_list[scenario_index][:cut_expression]))
    else
        Set()
    end

    if disjunction_iteration_limit < 0 
        ## dynamically control the number of disjunctions
        MDCiter = minimum([15, iteration_index]);
    else
        ## we use the fixed number of disjunctions
        MDCiter = disjunction_iteration_limit;
    end

    if cut_selection ∈ [:LC, :SMC, :PLC]
        remove_nonanticipativity_constraint(local_backward_info_list[scenario_index]);

        (level_set_method_param, x₀) = setup_levelset_param(
            f_star_value,
            state,
            local_param,
            local_param_cut,
            local_param_levelmethod
        );

        backward_info = LevelSetMethod_optimization!(
            local_backward_info_list[scenario_index],
            x₀,
            level_set_method_param,
            local_param,
            local_param_cut,
            local_param_levelmethod
        ); 

    elseif cut_selection ∈ [:DBC, :iDBC, :SPC]
        split_disjunction_iter = cut_selection == :SPC ? 1 : MDCiter
        backward_info = CPT_optimization!(
            local_backward_info_list[scenario_index],
            state, 
            scenario_index;
            param = local_param,
            param_cut = local_param_cut,
            param_LevelMethod = local_param_levelmethod,
            MDCiter = split_disjunction_iter
        ); 
        remove_nonanticipativity_constraint(local_backward_info_list[scenario_index]);

    elseif cut_selection ∈ [:BC]
        nonanticipativity_constraint(
            local_backward_info_list[scenario_index],
            state
        );
        backward_info = CPT_optimization!(
            local_backward_info_list[scenario_index],
            state, 
            scenario_index;
            param = local_param,
            param_cut = local_param_cut,
            param_LevelMethod = local_param_levelmethod,
            MDCiter = 0.0
        ); 
        remove_nonanticipativity_constraint(local_backward_info_list[scenario_index]);

    elseif cut_selection ∈ [:SBC]
        nonanticipativity_constraint(
            local_backward_info_list[scenario_index],
            state
        );
        backward_info = CPT_optimization!(
            local_backward_info_list[scenario_index],
            state, 
            scenario_index;
            param = local_param,
            param_cut = local_param_cut,
            param_LevelMethod = local_param_levelmethod,
            MDCiter = 0.0
        ); 
        remove_nonanticipativity_constraint(local_backward_info_list[scenario_index]);
        (level_set_method_param, x₀) = setup_levelset_param(
            f_star_value,
            state,
            local_param,
            local_param_cut,
            local_param_levelmethod
        );

        backward_info = LevelSetMethod_optimization!(
            local_backward_info_list[scenario_index],
            backward_info.cut_info[2],
            level_set_method_param,
            local_param,
            local_param_cut,
            local_param_levelmethod
        );

    elseif cut_selection == :FC
        nonanticipativity_constraint(
            local_backward_info_list[scenario_index],
            state
        );
        backward_info = Fenchel_cut_optimization!(
            local_backward_info_list[scenario_index],
            state, 
            scenario_index;
            param = local_param,
            param_cut = local_param_cut,
            param_LevelMethod = local_param_levelmethod,
            MDCiter = MDCiter
        ); 
        remove_nonanticipativity_constraint(local_backward_info_list[scenario_index]);
    end

    new_cut_collection = Dict()
    if inherit_disjunctive_cuts
        scenario_cut_expression = local_backward_info_list[scenario_index][:cut_expression]
        for (key, cut) in scenario_cut_expression
            if !(key in existing_cut_keys)
                new_cut_collection[key] = cut
            end
        end
    end

    return (
        backward_info = backward_info, 
        cut_collection = new_cut_collection,
    )
end
