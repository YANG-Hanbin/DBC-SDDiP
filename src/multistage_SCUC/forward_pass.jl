export forwardModel!, forwardPass
if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "common", "runtime_context.jl"))
end

function _resolve_scuc_forward_param(param::Union{Nothing, NamedTuple})::NamedTuple
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
    error("Missing SCUC `param` in forward_pass; pass explicitly or initialize runtime context.")
end

function _resolve_scuc_forward_index_sets(index_sets::Union{Nothing, IndexSets})::IndexSets
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
    error("Missing SCUC `index_sets` in forward_pass; pass explicitly or initialize runtime context.")
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
    1. `SDDPModel`
"""
function forwardModel!( 
    param_demand::ParamDemand, 
    param_opf::ParamOPF, 
    stageRealization::StageRealization;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::SDDPModel
    local_index_sets = _resolve_scuc_forward_index_sets(index_sets)
    local_param = _resolve_scuc_forward_param(param)

    runtime_param = resolve_scuc_runtime_params(local_param)
    enforce_binary_copies = runtime_param.enforce_binary_copies
    time_limit = runtime_param.time_limit

    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)); 
    MOI.set(model, MOI.Silent(), !local_param.verbose);
    set_optimizer_attribute(model, "MIPGap", local_param.mip_gap);
    set_optimizer_attribute(model, "TimeLimit", time_limit);
    set_optimizer_attribute(model, "MIPFocus", local_param.mip_focus);           
    set_optimizer_attribute(model, "FeasibilityTol", local_param.feasibility_tol);
                
    ## define variables
    @variable(model, θ_angle[index_sets.B])                                                ## phase angle of the bus i
    @variable(model, P[index_sets.L])                                                      ## real power flow on line l; elements in L is Tuple (i, j)
    @variable(model, 0 ≤ s[g in index_sets.G] ≤ param_opf.smax[g])                          ## real power generation at generator g
    @variable(model, 0 ≤ x[index_sets.D] ≤ 1)                                              ## load shedding

    @variable(model, y[index_sets.G], Bin)                                                 ## binary variable for generator commitment status
    @variable(model, v[index_sets.G], Bin)                                                 ## binary variable for generator startup decision
    @variable(model, w[index_sets.G], Bin)                                                 ## binary variable for generator shutdown decision

    @variable(model, h[index_sets.G] ≥ 0);                                                 ## production cost at generator g

    @variable(model, θ[keys(stageRealization.prob)] ≥ local_param.theta_lower)                             ## auxiliary variable for approximation of the value function

    if local_param.algorithm == :SDDPL
        ## augmented variables
        # define the augmented variables for cont. variables
        augmentVar = Dict(
            (g, k) => @variable(model, base_name = "augmentVar[$g, $k]", binary = true)
            for g in index_sets.G for k in 1:1
        );
        model[:augmentVar] = augmentVar;
        
        ## define copy variables
        @variable(model, 0 ≤ s_copy[g in index_sets.G] ≤ param_opf.smax[g])
        if enforce_binary_copies
            @variable(model, y_copy[index_sets.G], Bin)        
            augmentVar_copy = Dict(
                (g, k) => @variable(model, base_name = "augmentVar_copy[$g, $k]", binary = true)
                for g in index_sets.G for k in 1:1
            );
            model[:augmentVar_copy] = augmentVar_copy;     
        else
            @variable(model, 0 ≤ y_copy[index_sets.G] ≤ 1)  
            augmentVar_copy = Dict(
                (g, k) => @variable(model, base_name = "augmentVar_copy[$g, $k]", lower_bound = 0, upper_bound = 1)
                for g in index_sets.G for k in 1:1
            );
            model[:augmentVar_copy] = augmentVar_copy;     
        end

        # constraints for augmented variables: Choosing one leaf node
        @constraint(model, [g in index_sets.G, k in [1]], augmentVar[g, k] == 1)
        ContVarLeaf = Dict(
            :s => Dict{Any, Dict{Any, Dict{Symbol, Any}}}(
                        g => Dict(
                            k => Dict(
                                :lb => 0.0, 
                                :ub => param_opf.smax[g], 
                                :parent => nothing, 
                                :sibling => nothing, 
                                :var => augmentVar[g,1]) for k in 1:1
                                ) for g in index_sets.G
            )
        );
        ContVarBinaries = nothing;
    elseif local_param.algorithm == :SDDP 
        ## define copy variables
        @variable(model, 0 ≤ s_copy[g in index_sets.G] ≤ param_opf.smax[g])
        if enforce_binary_copies
            @variable(model, y_copy[index_sets.G], Bin)             
        else
            @variable(model, 0 ≤ y_copy[index_sets.G] ≤ 1)    
        end
        ContVarLeaf = nothing;
        ContVarBinaries = nothing;
    elseif local_param.algorithm == :SDDiP
        ## approximate the continuous state s[g], s[g] = ∑_{i=0}^{κ-1} 2ⁱ * λ[g, i] * ε, κ = log2(param_opf.smax[g] / ε) + 1
        @variable(model, λ[g in local_index_sets.G, i in 1:local_param.kappa[g]], Bin)
        @constraint(model, ContiApprox[g in local_index_sets.G], local_param.epsilon * sum(2^(i-1) * λ[g, i] for i in 1:local_param.kappa[g]) == s[g])
        ContVarLeaf = nothing;
        ContVarBinaries = Dict(
            :s => Dict{Any, Dict{Any, VariableRef}}(
                    g => Dict(i => λ[g, i] for i in 1:local_param.kappa[g]) for g in local_index_sets.G
                )
        );
        @variable(model, 0 ≤ s_copy[g in local_index_sets.G] ≤ param_opf.smax[g])
        if enforce_binary_copies
            @variable(model, λ_copy[g in local_index_sets.G, i in 1:local_param.kappa[g]], Bin)
            @variable(model, y_copy[local_index_sets.G], Bin)        
        else
            @variable(model, 0 ≤ λ_copy[g in local_index_sets.G, i in 1:local_param.kappa[g]] ≤ 1)
            @variable(model, 0 ≤ y_copy[local_index_sets.G] ≤ 1)       
        end
        @constraint(model, [g in local_index_sets.G], local_param.epsilon * sum(2^(i-1) * λ_copy[g, i] for i in 1:local_param.kappa[g]) == s_copy[g])
    end

    ## problem constraints:
    # power flow constraints
    for l in local_index_sets.L
        i = l[1]
        j = l[2]
        @constraint(model, P[l] ≤ - param_opf.b[l] * (θ_angle[i] - θ_angle[j]))
        @constraint(model, P[l] ≥ - param_opf.b[l] * (θ_angle[i] - θ_angle[j]))
    end
    
    # power flow limitation
    @constraint(model, [l in local_index_sets.L], P[l] ≥ - param_opf.W[l])
    @constraint(model, [l in local_index_sets.L], P[l] ≤   param_opf.W[l])
    # generator limitation
    @constraint(model, [g in local_index_sets.G], s[g] ≥ param_opf.smin[g] * y[g])
    @constraint(model, [g in local_index_sets.G], s[g] ≤ param_opf.smax[g] * y[g])

    # power balance constraints
    @constraint(
        model, 
        PowerBalance[i in local_index_sets.B],
        sum(s[g] for g in local_index_sets.Gᵢ[i]) -
        sum(P[(i, j)] for j in local_index_sets.out_L[i]) + 
        sum(P[(j, i)] for j in local_index_sets.in_L[i]) 
        .== sum(param_demand.demand[d] * x[d] for d in local_index_sets.Dᵢ[i]) 
    )
    
    # on/off status with startup and shutdown decision
    @constraint(
        model, 
        ShutUpDown[g in local_index_sets.G], 
        v[g] - w[g] == y[g] - y_copy[g]
    );
    @constraint(
        model, 
        Ramping1[g in local_index_sets.G], 
        s[g] - s_copy[g] <= param_opf.M[g] * y_copy[g] + param_opf.smin[g] * v[g]
    );
    @constraint(
        model, 
        Ramping2[g in local_index_sets.G], 
        s[g] - s_copy[g] >= - param_opf.M[g] * y[g] - param_opf.smin[g] * w[g]
    );

    # production cost
    @constraint(
        model, 
        production[g in local_index_sets.G, o in keys(param_opf.slope[g])], 
        h[g] ≥ param_opf.slope[g][o] * s[g] + param_opf.intercept[g][o] * y[g]
    );

    # objective function
    @expression(
        model, 
        primal_objective_expression, 
        sum(h[g] + param_opf.C_start[g] * v[g] + 
        param_opf.C_down[g] * w[g] for g in local_index_sets.G) + 
        sum(param_demand.w[d] * (1 - x[d]) for d in local_index_sets.D) + 
        sum(θ)
    );

    @objective(
        model, 
        Min, 
        primal_objective_expression
    );
    
    disjunctive_cuts = Dict();
    model[:disjunctive_cuts] = disjunctive_cuts;
    model[:cut_expression] = Dict();
    
    return SDDPModel(
        model, 
        Dict{Any, Dict{Any, VariableRef}}(:y => Dict{Any, VariableRef}(g => y[g] for g in local_index_sets.G)), 
        nothing, 
        Dict{Any, Dict{Any, VariableRef}}(:s => Dict{Any, VariableRef}(g => s[g] for g in local_index_sets.G)), 
        nothing, 
        ContVarLeaf,
        nothing,
        ContVarBinaries
    )
end

"""
ModelModification!(; model::Model = model)

# Arguments

    1. `model::Model` : a forward pass model of stage t
    2. `random_variables::RandomVariables` : random variables
    3. `param_demand::ParamDemand` : demand parameters
    4. `state_info::StateInfo` : the last stage decisions
  
# Modification
    1. Remove the other scenario's demand balance constraints
    2. Add the current scenario's demand balance constraints
    3. Update its last stage decision with
"""
function ModelModification!( 
    model::Model, 
    random_variables::RandomVariables,
    param_demand::ParamDemand,
    state_info::StateInfo;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    local_index_sets = _resolve_scuc_forward_index_sets(index_sets)
    local_param = _resolve_scuc_forward_param(param)
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

    # power balance constraints
    for i in local_index_sets.B
        delete(model, model[:PowerBalance][i])
    end
    unregister(model, :PowerBalance)
    @constraint(
        model, 
        PowerBalance[i in local_index_sets.B], 
        sum(model[:s][g]      for g in local_index_sets.Gᵢ[i]) -
        sum(model[:P][(i, j)] for j in local_index_sets.out_L[i]) + 
        sum(model[:P][(j, i)] for j in local_index_sets.in_L[i]) .==
        sum(param_demand.demand[d] * random_variables.deviation[d] * model[:x][d] for d in local_index_sets.Dᵢ[i])
    );

    @objective(
        model, 
        Min, 
        model[:primal_objective_expression]
    );
    return
end

"""
forwardPass(ξ): function for forward pass in parallel computing

# Arguments

  1. `ξ`: A sampled scenario path

# Returns
  1. `scenario_solution_collection`: cut coefficients

"""
function forwardPass(
    ξ::Dict{Int64, RandomVariables};
    ModelList::Union{Nothing, Dict{Int, SDDPModel}} = nothing,
    param_demand::Union{Nothing, ParamDemand} = nothing,
    param_opf::Union{Nothing, ParamOPF} = nothing,
    index_sets::Union{Nothing, IndexSets} = nothing,
    initial_state_info::Union{Nothing, StateInfo} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)
    runtime_context = RuntimeContext.get_context(:scuc)
    local_model_list = isnothing(ModelList) ? runtime_context[:model_list] : ModelList
    local_param_demand = isnothing(param_demand) ? runtime_context[:param_demand] : param_demand
    local_param_opf = isnothing(param_opf) ? runtime_context[:param_opf] : param_opf
    local_index_sets = isnothing(index_sets) ? runtime_context[:index_sets] : index_sets
    local_initial_state_info = isnothing(initial_state_info) ? runtime_context[:initial_state_info] : initial_state_info
    local_param = isnothing(param) ? runtime_context[:param] : param

    runtime_param = resolve_scuc_runtime_params(local_param)
    stateInfoList = Dict();
    stateInfoList[0] = deepcopy(local_initial_state_info);
    for t in 1:local_index_sets.T
        ModelModification!( 
            local_model_list[t].model,
            ξ[t],
            local_param_demand,
            stateInfoList[t-1];
            index_sets = local_index_sets,
            param = local_param
        )
        optimize!(local_model_list[t].model);

        # record the solution
        enforce_binary_copies = runtime_param.enforce_binary_copies
        is_sddip_tight = (local_param.algorithm == :SDDiP) && enforce_binary_copies
        BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
            g => begin
                value_y = JuMP.value(local_model_list[t].BinVar[:y][g])
                is_sddip_tight ? (value_y >= 0.5 ? 1.0 : 0.0) : round(value_y, digits = 2)
            end for g in local_index_sets.G)
        );
        ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
            g => JuMP.value(local_model_list[t].ContVar[:s][g]) for g in local_index_sets.G)
        );
        if local_param.algorithm == :SDDPL
            ContVarLeaf = Dict{Any, Dict{Any, Dict{Any, Dict{Symbol, Any}}}}(
                :s => Dict{Any, Dict{Any, Dict{Symbol, Any}}}(
                    g => Dict(
                        k => Dict(
                            :var => round(JuMP.value(local_model_list[t].ContVarLeaf[:s][g][k][:var]), digits = 2)
                        ) for k in keys(local_model_list[t].ContVarLeaf[:s][g]) if JuMP.value(local_model_list[t].ContVarLeaf[:s][g][k][:var]) > .5
                    ) for g in local_index_sets.G
                )
            );
            ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(:s => Dict{Any, Dict{Any, Any}}(g => Dict{Any, Any}() for g in local_index_sets.G));
            ContStateBin = nothing;
        elseif local_param.algorithm == :SDDiP
            ContVarLeaf  = nothing;
            ContAugState = nothing;
            ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => begin
                            value_lambda = JuMP.value(local_model_list[t].ContVarBinaries[:s][g][i])
                            projected_lambda = is_sddip_tight ? (value_lambda >= 0.5 ? 1.0 : 0.0) : round(value_lambda, digits = 2)
                            BinVar[:y][g] <= 0.5 ? 0.0 : projected_lambda
                        end for i in 1:local_param.kappa[g]
                    ) for g in local_index_sets.G
                )
            );
        elseif local_param.algorithm == :SDDP
            ContVarLeaf  = nothing;
            ContAugState = nothing;
            ContStateBin = nothing;
        end
        
        stage_value = JuMP.objective_value(local_model_list[t].model) - sum(JuMP.value.(local_model_list[t].model[:θ]));
        state_value = JuMP.objective_value(local_model_list[t].model);
        stateInfoList[t] = StateInfo(
            BinVar, 
            nothing, 
            ContVar, 
            nothing, 
            ContVarLeaf, 
            stage_value, 
            state_value, 
            nothing, 
            ContAugState,
            nothing,
            ContStateBin
        );
    end  
    return stateInfoList  
end
