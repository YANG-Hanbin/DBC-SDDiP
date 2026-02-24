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
    runtime_param = resolve_sslp_runtime_params(param)
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

    @variable(model, x[j = 1:param.J], Bin);    
    @variable(model, θ ≥ param.theta_lower);

    @constraint(model, sum(x) ≤ stage_data.r); ## total number of servers

    @objective(
        model, 
        Min, 
        sum(stage_data.c[j] * x[j] for j in 1:param.J) + 
        θ 
    );
    return model
end


"""
forwardPass(ξ): function for forward pass in parallel computing

# Arguments

  1. `ξ`: A sampled scenario path

# Returns
  1. `scenario_solution_collection`: cut coefficients

"""
function forwardPass(
    scenario_index::Int64;
    state::Vector,
    backward_info_list::Union{Nothing, Dict{Int, Model}} = nothing,
)
    runtime_context = RuntimeContext.get_context(:sslp)
    local_backward_info_list = isnothing(backward_info_list) ? runtime_context[:backward_info_list] : backward_info_list
    nonanticipativity_constraint(
        local_backward_info_list[scenario_index], 
        state
    );
    optimize!(local_backward_info_list[scenario_index]);

    state_info = ( 
        stage_solution = round.(JuMP.value.(local_backward_info_list[scenario_index][:y]), digits = 2),
        OPT = JuMP.objective_value(local_backward_info_list[scenario_index])
    );
    return (
        state_info = state_info,
        ω = scenario_index    
    )
end
