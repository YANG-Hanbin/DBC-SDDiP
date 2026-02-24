"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::ParetoLagrangianCutGeneration,
        model::Model, 
        x₀::StateInfo, 
        state_info::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::ParetoLagrangianCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `x₀::StateInfo` : the dual information
    4. `state_info::StateInfo` : the last stage decision
  
# Returns
    1. `current_info::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::ParetoLagrangianCutGeneration,
    model::Model, 
    x₀::StateInfo, 
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum( 
            x₀.BinVar[:y][g] * (state_info.BinVar[:y][g] - model[:y_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    x₀.ContStateBin[:s][g][i] * (state_info.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.kappa[g]
                ) : x₀.ContVar[:s][g] * (state_info.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    x₀.ContAugState[:s][g][k] * (state_info.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in index_sets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    negative_∇F = StateInfo(
        Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => JuMP.value(model[:y_copy][g]) - state_info.BinVar[:y][g] for g in index_sets.G
            )
        ), 
        nothing, 
        Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Any}(
                g => JuMP.value(model[:s_copy][g]) - state_info.ContVar[:s][g] for g in index_sets.G
            )
        ), 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        k => JuMP.value(model[:augmentVar_copy][g, k]) - state_info.ContAugState[:s][g][k]
                        for k in keys(state_info.ContAugState[:s][g])
                    ) 
                for g in index_sets.G
            )
        ) : nothing,
        nothing,
        param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => JuMP.value(model[:λ_copy][g, i]) - state_info.ContStateBin[:s][g][i]
                        for i in 1:param.kappa[g]
                    ) 
                for g in index_sets.G
            )
        ) : nothing
    );
    current_info = CurrentInfo(  
        x₀, 
        - F - sum(
            (param.algorithm == :SDDiP ?
                sum(
                    x₀.ContStateBin[:s][g][i] * (CutGenerationInfo.core_point.ContStateBin[:s][g][i] - state_info.ContStateBin[:s][g][i]) for i in 1:param.kappa[g]
                ) : x₀.ContVar[:s][g] * (CutGenerationInfo.core_point.ContVar[:s][g] - state_info.ContVar[:s][g])
            ) +
            x₀.BinVar[:y][g] * (CutGenerationInfo.core_point.BinVar[:y][g] - state_info.BinVar[:y][g]) +
            (param.algorithm == :SDDPL ?
                sum(
                    x₀.ContAugState[:s][g][k] * (CutGenerationInfo.core_point.ContAugState[:s][g][k] - state_info.ContAugState[:s][g][k])
                    for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            )
            for g in index_sets.G
        ),                                                                                                                                                              ## obj function value
        Dict(
            1 => CutGenerationInfo.primal_bound - F - CutGenerationInfo.δ
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ? 
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => JuMP.value(model[:s_copy][g]) - CutGenerationInfo.core_point.ContVar[:s][g] for g in index_sets.G),
            :y => Dict(g => JuMP.value(model[:y_copy][g]) - CutGenerationInfo.core_point.BinVar[:y][g] for g in index_sets.G),
            :sur => Dict(
                g => Dict(
                    k => JuMP.value(model[:augmentVar_copy][g, k]) - CutGenerationInfo.core_point.ContAugState[:s][g][k] 
                            for k in keys(state_info.ContAugState[:s][g])
                ) for g in index_sets.G
            ),
            :λ => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(
                        i => JuMP.value(model[:λ_copy][g, i]) - CutGenerationInfo.core_point.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
                    ) : nothing
                ) for g in index_sets.G
            )
        ) :
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => JuMP.value(model[:s_copy][g]) - CutGenerationInfo.core_point.ContVar[:s][g] for g in index_sets.G),
            :y => Dict(g => JuMP.value(model[:y_copy][g]) - CutGenerationInfo.core_point.BinVar[:y][g] for g in index_sets.G),
            :λ => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(
                        i => JuMP.value(model[:λ_copy][g, i]) - CutGenerationInfo.core_point.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
                    ) : nothing
                ) for g in index_sets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => negative_∇F )                                                                                                                                         ## constraint gradient
    );
    return (current_info = current_info, current_info_f = F)
end    

"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::SquareMinimizationCutGeneration,
        model::Model, 
        x₀::StateInfo, 
        state_info::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::SquareMinimizationCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `x₀::StateInfo` : the dual information
    4. `state_info::StateInfo` : the last stage decision
  
# Returns
    1. `current_info::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::SquareMinimizationCutGeneration,
    model::Model, 
    x₀::StateInfo, 
    state_info::StateInfo;
    index_sets::IndexSets = index_sets
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            x₀.BinVar[:y][g] * (state_info.BinVar[:y][g] - model[:y_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    x₀.ContStateBin[:s][g][i] * (state_info.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.kappa[g]
                ) : x₀.ContVar[:s][g] * (state_info.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    x₀.ContAugState[:s][g][k] * (state_info.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in index_sets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    negative_∇F = StateInfo(
        Dict{Any, Dict{Any, Any}}(
            :y => Dict{Any, Any}(
                g => JuMP.value(model[:y_copy][g]) - state_info.BinVar[:y][g] for g in index_sets.G
            )
        ), 
        nothing, 
        Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Any}(
                g => JuMP.value(model[:s_copy][g]) - state_info.ContVar[:s][g] for g in index_sets.G
            )
        ), 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        k => JuMP.value(model[:augmentVar_copy][g, k]) - state_info.ContAugState[:s][g][k]
                        for k in keys(state_info.ContAugState[:s][g])
                    ) 
                for g in index_sets.G
            )
        ) : nothing,
        nothing,
        param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => JuMP.value(model[:λ_copy][g, i]) - state_info.ContStateBin[:s][g][i]
                        for i in 1:param.kappa[g]
                    ) 
                for g in index_sets.G
            )
        ) : nothing
    );
    current_info = CurrentInfo(  
        x₀, 
        1/2 * (param.algorithm == :SDDiP ?
                sum(
                    sum(x₀.ContStateBin[:s][g][i] * x₀.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) 
                    for g in index_sets.G
                ) : 
                sum(x₀.ContVar[:s][g] * x₀.ContVar[:s][g] for g in index_sets.G)
        ) +
        1/2 * sum(x₀.BinVar[:y][g] * x₀.BinVar[:y][g] for g in index_sets.G) + 
        (param.algorithm == :SDDPL ? 
        1/2 * sum(
                sum(
                    x₀.ContAugState[:s][g][k] * x₀.ContAugState[:s][g][k] for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) for g in index_sets.G
            ) : 0.0
        ),                                                                                                                                                              ## obj function value
        Dict(
            1 => CutGenerationInfo.primal_bound - F - CutGenerationInfo.δ
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ?
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => x₀.ContVar[:s][g] for g in index_sets.G),
            :y => Dict(g => x₀.BinVar[:y][g] for g in index_sets.G), 
            :sur => Dict(g => Dict(k => x₀.ContAugState[:s][g][k] for k in keys(state_info.ContAugState[:s][g])) for g in index_sets.G),
            :λ => Dict(
                g => (
                    param.algorithm == :SDDiP ?
                        Dict(i => x₀.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) : nothing
                ) for g in index_sets.G
            )
        ) : 
        Dict{Symbol, Dict{Int64, Any}}(
            :s => Dict(g => x₀.ContVar[:s][g] for g in index_sets.G),
            :y => Dict(g => x₀.BinVar[:y][g] for g in index_sets.G),
            :λ => Dict(
                g => (
                    param.algorithm == :SDDiP ?
                        Dict(i => x₀.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) : nothing
                ) for g in index_sets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => negative_∇F )                                                                                                                                         ## constraint gradient
    );
    return (current_info = current_info, current_info_f = F)
end    

"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::LagrangianCutGeneration,
        model::Model, 
        x₀::StateInfo, 
        state_info::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::LagrangianCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `x₀::StateInfo` : the dual information
    4. `state_info::StateInfo` : the last stage decision
  
# Returns
    1. `current_info::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::LagrangianCutGeneration,
    model::Model, 
    x₀::StateInfo, 
    state_info::StateInfo;
    index_sets::IndexSets = index_sets
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            x₀.BinVar[:y][g] * (state_info.BinVar[:y][g] - model[:y_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    x₀.ContStateBin[:s][g][i] * (state_info.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.kappa[g]
                ) : x₀.ContVar[:s][g] * (state_info.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    x₀.ContAugState[:s][g][k] * (state_info.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in index_sets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    current_info = CurrentInfo(  
        x₀, 
        - F,                                                                                                                                                            ## obj function value
        Dict(
            1 => 0.0
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ?                                                                                                                            
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - state_info.ContVar[:s][g] for g in index_sets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - state_info.BinVar[:y][g]  for g in index_sets.G), 
            :sur => Dict(g => Dict(
                k => JuMP.value(model[:augmentVar_copy][g, k]) - state_info.ContAugState[:s][g][k] for k in keys(state_info.ContAugState[:s][g])
                ) for g in index_sets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) : nothing
                ) for g in index_sets.G
            )
        ) : 
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - state_info.ContVar[:s][g] for g in index_sets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - state_info.BinVar[:y][g]  for g in index_sets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) : nothing
                ) for g in index_sets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => StateInfo(
            Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                g => 0.0 for g in index_sets.G)
            ), 
            nothing, 
            Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                g => 0.0 for g in index_sets.G)
            ), 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                            k => 0.0
                            for k in keys(state_info.ContAugState[:s][g])
                        ) 
                    for g in index_sets.G
                )
            ) : nothing,
            nothing,
            param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => 0.0
                        for i in 1:param.kappa[g]
                    ) 
                for g in index_sets.G
            )
        ) : nothing
            )
        )                                                                                                                                                               ## constraint gradient
    );
    return (current_info = current_info, current_info_f = F)
end    


"""
    function solve_inner_minimization_problem(
        CutGenerationInfo::LagrangianCutGeneration,
        model::Model, 
        x₀::StateInfo, 
        state_info::StateInfo
    )

# Arguments

    1. `CutGenerationInfo::StrengthenedBendersCutGeneration` : the information of the cut that will be generated information
    2. `model::Model` : the backward model
    3. `x₀::StateInfo` : the dual information
    4. `state_info::StateInfo` : the last stage decision
  
# Returns
    1. `current_info::CurrentInfo` : the current information
  
"""
function solve_inner_minimization_problem(
    CutGenerationInfo::StrengthenedBendersCutGeneration,
    model::Model, 
    x₀::StateInfo, 
    state_info::StateInfo;
    index_sets::IndexSets = index_sets
)
    @objective(
        model, 
        Min,  
        model[:primal_objective_expression] +
        sum(
            x₀.BinVar[:y][g] * (state_info.BinVar[:y][g] - model[:y_copy][g]) + 
            (param.algorithm == :SDDiP ?
                sum(
                    x₀.ContStateBin[:s][g][i] * (state_info.ContStateBin[:s][g][i] - model[:λ_copy][g, i]) for i in 1:param.kappa[g]
                ) : x₀.ContVar[:s][g] * (state_info.ContVar[:s][g] - model[:s_copy][g])
            ) +
            (param.algorithm == :SDDPL ?
                sum(
                    x₀.ContAugState[:s][g][k] * (state_info.ContAugState[:s][g][k] - model[:augmentVar_copy][g, k]) 
                    for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) : 0.0
            ) 
            for g in index_sets.G
        )
    );
    ## ==================================================== solve the model and display the result ==================================================== ##
    optimize!(model);
    F  = JuMP.objective_value(model);
    
    current_info = CurrentInfo(  
        x₀, 
        - F,                                                                                                                                                            ## obj function value
        Dict(
            1 => 0.0
        ),                                                                                                                                                              ## constraint value
        param.algorithm == :SDDPL ?                                                                                                                            
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - state_info.ContVar[:s][g] for g in index_sets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - state_info.BinVar[:y][g]  for g in index_sets.G), 
            :sur => Dict(g => Dict(
                k => JuMP.value(model[:augmentVar_copy][g, k]) - state_info.ContAugState[:s][g][k] for k in keys(state_info.ContAugState[:s][g])
                ) for g in index_sets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) : nothing
                ) for g in index_sets.G
            )
        ) : 
        Dict{Symbol, Dict{Int64, Any}}(
            :s   => Dict(g => JuMP.value(model[:s_copy][g]) - state_info.ContVar[:s][g] for g in index_sets.G),
            :y   => Dict(g => JuMP.value(model[:y_copy][g]) - state_info.BinVar[:y][g]  for g in index_sets.G),
            :λ   => Dict(
                g => (param.algorithm == :SDDiP ?
                    Dict(i => JuMP.value(model[:λ_copy][g, i]) - state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]) : nothing
                ) for g in index_sets.G
            )
        ),                                                                                                                                                              ## obj gradient
        Dict(1 => StateInfo(
            Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                g => 0.0 for g in index_sets.G)
            ), 
            nothing, 
            Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                g => 0.0 for g in index_sets.G)
            ), 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            param.algorithm == :SDDPL ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                            k => 0.0
                            for k in keys(state_info.ContAugState[:s][g])
                        ) 
                    for g in index_sets.G
                )
            ) : nothing,
            nothing,
            param.algorithm == :SDDiP ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                        i => 0.0
                        for i in 1:param.kappa[g]
                    ) 
                for g in index_sets.G
            )
        ) : nothing
            )
        )                                                                                                                                                               ## constraint gradient
    );
    return (current_info = current_info, current_info_f = F)
end    


"""
    function StrengthedBendersCut_optimization!(
        model::Model, 
        levelsetmethod_oracle_param::LevelSetMethodOracleParam, 
        state_info::StateInfo,
        CutGenerationInfo::CutGeneration;
        index_sets::IndexSets = index_sets, 
        param_demand::ParamDemand = param_demand, 
        param_opf::ParamOPF = param_opf, 
        param::NamedTuple, param_levelsetmethod::NamedTuple
    )  
# Arguments

    1. `stage_decision::Dict{Symbol, Dict{Int64, Float64}}` : the decision of the last stage
    2. `f_star_value::Float64` : the optimal value of the current approximate value function
    3. `x_interior::Union{Dict{Symbol, Dict{Int64, Float64}}, Nothing}` : an interior point
    4. `x₀::Dict{Symbol, Dict{Int64, Float64}}` : the initial point of the lagrangian dual variables
    5. `model::Model` : backward model
  
# Returns
    1. `cut_info::Array{Any,1}` : the cut information
  
"""
function StrengthedBendersCut_optimization!(
    model::Model, 
    state_info::StateInfo,
    CutGenerationInfo::CutGeneration;
    index_sets::IndexSets = index_sets, 
    param_demand::ParamDemand = param_demand, 
    param_opf::ParamOPF = param_opf, 
    param::NamedTuple, param_levelsetmethod::NamedTuple
)    
    ## ==================================================== Level-set Method ============================================== ##    
    (D, G, L, B) = (index_sets.D, index_sets.G, index_sets.L, index_sets.B);
    iter = 1;

    # trajectory
    current_info, current_info_f = solve_inner_minimization_problem(
        CutGenerationInfo,
        model, 
        CutGenerationInfo.x₀, 
        state_info;
        index_sets = index_sets
    );

    cut_info = [
        current_info_f - 
        sum(
            (param.algorithm == :SDDiP ?
                sum(
                    current_info.x.ContStateBin[:s][g][i] * state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
                ) : current_info.x.ContVar[:s][g] * state_info.ContVar[:s][g]
            )
             + 
            current_info.x.BinVar[:y][g] * state_info.BinVar[:y][g] for g in G
        ) - 
        (param.algorithm == :SDDPL ? 
            sum(
                sum(
                    current_info.x.ContAugState[:s][g][k] * state_info.ContAugState[:s][g][k] 
                    for k in keys(state_info.ContAugState[:s][g]); init = 0.0
                ) for g in G
            ) : 0.0),
        current_info.x
    ];

    return cut_info
end
