if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

"""
    function forwardModel!( 
        param_demand::ParamDemand, 
        param_opf::ParamOPF, 
        stageRealization::StageRealization;
        index_sets::IndexSets = index_sets, 
        para::NamedTuple = para
    )
# Arguments

    1. `index_sets::IndexSets` : index sets for the power network
    2. `param_demand::ParamDemand` : demand parameters
    3. `param_opf::ParamOPF` : OPF parameters
    4. `stageRealization::StageRealization` : realization of the stage
  
# Returns
    1. `forwardModel::Model` : a forward pass model of stage t
"""
function forwardModel!(
    stage_data::StageData,
    param::NamedTuple
)::Model
    runtime_param = resolve_ge_runtime_params(param)
    enforce_binary_copies = runtime_param.enforce_binary_copies
    time_limit = runtime_param.time_limit
                            
    ## construct forward problem (3.1)
    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
        ); 
    MOI.set(model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(model, "TimeLimit", time_limit);
    set_optimizer_attribute(model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(model, "FeasibilityTol", param.feasibility_tol);

    @variable(model, x[ i = 1:param.binary_info.d] ≥ 0, Int);    ## x is the vector of newly built generators
    @variable(model, y[ i = 1:param.binary_info.d] ≥ 0);         ## y is the vector of electricity produced by each type of generator per hour in this stage
    @variable(model, Lt[i = 1:param.binary_info.n], Bin);       ## stage variable, A * Lt represents the cumulative number of different types of generators built until this stage
    @variable(model, slack ≥ 0 );
    @variable(model, θ ≥ param.theta_lower);

    # auxiliary variable (copy variable)
    if enforce_binary_copies
        @variable(model, Lc[i = 1:param.binary_info.n], Bin);             
    else
        @variable(model, 0 ≤ Lc[i = 1:param.binary_info.n]≤ 1);                      
    end

    ## no more than max num of generators
    @constraint(
        model, 
        param.binary_info.A * Lc + x .≤ stage_data.ū 
    );  

    ## satisfy demand
    @constraint(
        model, 
        demand_constraint,
        sum(y) + slack .≥ 0
    );

    ## Capacity constraint             
    @constraint(
        model, 
        stage_data.h * stage_data.N * (param.binary_info.A * Lc + x + stage_data.s₀ ) .≥ y 
    );             

    ## to ensure pass a binary variable for next stage
    @constraint(
        model, 
        param.binary_info.A * Lc + x .== param.binary_info.A * Lt 
    );

    @expression(
        model, 
        primal_objective_expression, 
        stage_data.c1'* x + 
        stage_data.c2' * y + 
        stage_data.penalty * slack + 
        θ 
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
    function ModelModification!(
        backward_model::Model, 
        demand::Vector{Float64}
        state::Vector{Float64}
    )::Nothing

# Arguments
  1. `backward_model`: backward model
  2. `demand`: modify the backward model to adaptive the current state
  3. `state`: the state of the previous stage

# usage: 
        Adapt the model to the current state

"""
function ModelModification!(
    model::Model, 
    demand::Vector{Float64},
    state::Vector{Float64}
)::Nothing

    delete(
        model, 
        model[:demand_constraint]
    );
    unregister(
        model, 
        :demand_constraint
    );

    # satisfy demand
    @constraint(
        model, 
        demand_constraint, 
        sum(model[:y]) + model[:slack] .≥ demand 
    );

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
    forwardPass(k): function for forward pass in parallel computing

# Arguments

  1. `k`: A sampled scenario path

# Returns
  1. `sol_collection`: scenario path info

"""
function forwardPass(
    sample_path_id::Int64,
    sampled_paths;
    forward_info_list::Union{Nothing, Dict{Int, Model}} = nothing,
    scenario_realizations::Union{Nothing, Dict{Int64, Dict{Int64, RandomVariables}}} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::Dict
    runtime_context = RuntimeContext.get_context(:generation_expansion)
    local_forward_info_list = isnothing(forward_info_list) ? runtime_context[:forward_info_list] : forward_info_list
    local_scenario_realizations = isnothing(scenario_realizations) ? runtime_context[:scenario_realizations] : scenario_realizations
    local_param = isnothing(param) ? runtime_context[:param] : param

    sol_collection = Dict();                     # to store every iteration results
    state = [0.0 for i in 1:local_param.binary_info.n];
    for t in 1:local_param.T
        ## realization of sampled path at stage t
        ω = sampled_paths[sample_path_id][t];
        ## the following function is used to (1). change the problem coefficients for different node within the same stage t.
        ModelModification!(
            local_forward_info_list[t],
            local_scenario_realizations[t][ω].d,
            state
        );
        optimize!(local_forward_info_list[t]);

        sol_collection[t, sample_path_id] = ( 
            stage_solution = round.(JuMP.value.(local_forward_info_list[t][:Lt]), digits = 2),
            stage_value = JuMP.objective_value(local_forward_info_list[t]) - JuMP.value(local_forward_info_list[t][:θ]),
            OPT = JuMP.objective_value(local_forward_info_list[t])
        );

        state = sol_collection[t, sample_path_id].stage_solution;
    end

    return sol_collection  
end
