if !isdefined(Main, :RuntimeContext)
    include(joinpath(@__DIR__, "..", "..", "common", "runtime_context.jl"))
end

"""
    _resolve_scuc_levelset_arg(key, explicit, legacy_symbols)

# Purpose
    Resolve optional SCUC level-set keyword arguments through compatibility fallbacks.

# Resolution order
    1. Explicit keyword argument.
    2. `RuntimeContext(:scuc)[key]` if available.
    3. Legacy `Main` global aliases.
"""
@inline function _resolve_scuc_levelset_arg(
    key::Symbol,
    explicit_value,
    legacy_symbols::Tuple{Vararg{Symbol}},
)
    if !isnothing(explicit_value)
        return explicit_value
    end
    if RuntimeContext.has_context(:scuc)
        context = RuntimeContext.get_context(:scuc)
        if haskey(context, key)
            return context[key]
        end
    end
    for legacy_symbol in legacy_symbols
        if isdefined(Main, legacy_symbol)
            return getfield(Main, legacy_symbol)
        end
    end
    error("Missing required SCUC level-set argument `$(key)`.")
end

"""
    function SetupLevelSetMethodOracleParam(
        param::NamedTuple
    )::LevelSetMethodOracleParam

# Arguments

    1. `param::NamedTuple` : the parameters of the level set method

# Returns

    1. `LevelSetMethodOracleParam` : the parameters of the level set method

"""
function SetupLevelSetMethodOracleParam(
    state_info::StateInfo;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
    param_levelsetmethod::Union{Nothing, NamedTuple} = nothing,
)::LevelSetMethodOracleParam
    local_index_sets = _resolve_scuc_levelset_arg(:index_sets, index_sets, (:index_sets, :indexSets))
    local_param = _resolve_scuc_levelset_arg(:param, param, (:param,))
    local_param_levelsetmethod = _resolve_scuc_levelset_arg(
        :param_level_method,
        param_levelsetmethod,
        (:param_levelsetmethod, :param_LevelMethod),
    )
    
    μ             = get(local_param_levelsetmethod, :μ, 0.9);
    λ             = get(local_param_levelsetmethod, :λ, 0.5);
    threshold     = get(local_param_levelsetmethod, :threshold, nothing);
    nxt_bound     = get(local_param_levelsetmethod, :nxt_bound, 1e10);
    MaxIter       = get(local_param_levelsetmethod, :MaxIter, 200);
    verbose       = get(local_param_levelsetmethod, :levelsetmethod_verbose, true);

    return LevelSetMethodOracleParam(
        μ, 
        λ, 
        threshold, 
        nxt_bound, 
        MaxIter, 
        verbose, 
        setup_initial_point(
            state_info;
            index_sets = local_index_sets,
            param = local_param,
        )
    )
end

"""
   function Δ_model_formulation(
        function_history::FunctionHistory, 
        f_star::Float64, 
        iter::Int64; 
        para::NamedTuple = para
    )

# Arguments

    1. `function_history::FunctionHistory` : the history of the function
    2. `f_star::Float64` : the optimal value of the current approximate value function
    3. `iter::Int64` : the iteration number
    4. `para::NamedTuple` : the parameters of the level set method

# Returns

    1. `Dict{Int64, Float64}` : the value of the gap and the bounds of alpha
"""
function Δ_model_formulation(
    function_history::FunctionHistory, 
    f_star::Float64, 
    iter::Int64; 
    param::NamedTuple
)
    
    alpha_model = Model(optimizer_with_attributes(()->Gurobi.Optimizer(GRB_ENV), 
                                                    "Threads" => 0)); 
    MOI.set(alpha_model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(alpha_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(alpha_model, "TimeLimit", param.time_limit);

    @variable(alpha_model, z);
    @variable(alpha_model, 0 ≤ α ≤ 1);
    @constraint(alpha_model, con[j = 1:iter], z ≤  α * ( function_history.f_his[j] - f_star) + (1 - α) * function_history.G_max_his[j] );
    
    # we first compute gap Δ
    @objective(alpha_model, Max, z);
    optimize!(alpha_model);
    st = termination_status(alpha_model);
    Δ = JuMP.objective_value(alpha_model);

    
    ## then we modify above model to compute alpha
    # alpha_min
    @constraint(alpha_model, z .≥ 0);
    @objective(alpha_model, Min, α);
    optimize!(alpha_model);
    a_min = JuMP.value(α);

    # alpha_max
    @objective(alpha_model, Max, α);
    optimize!(alpha_model);
    a_max = JuMP.value(α);

    return Dict(1 => Δ, 2 => a_min, 3 => a_max)

end


"""
    function add_constraint(
        current_info::CurrentInfo,
        model_info::ModelInfo;
        index_sets::Union{Nothing, IndexSets} = nothing
    )::Nothing

# Arguments

    1. `current_info::CurrentInfo` : the current information
    2. `model_info::ModelInfo` : the model information
    3. `index_sets::IndexSets` : the index sets
    
# Returns
    This function is to add constraints for the model f_star and nxt pt.
"""
function add_constraint(
    current_info::CurrentInfo,
    model_info::ModelInfo;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
)::Nothing
    local_index_sets = _resolve_scuc_levelset_arg(:index_sets, index_sets, (:index_sets, :indexSets))
    local_param = _resolve_scuc_levelset_arg(:param, param, (:param,))
    m = length(current_info.G)

    # add constraints     
    @constraint(
        model_info.model, 
        model_info.z .≥ current_info.f + 
        (local_param.algorithm == :SDDiP ?  
            sum(
                sum(current_info.df[:λ][g][i] * (model_info.sur[g, i] - current_info.x.ContStateBin[:s][g][i]) for i in 1:local_param.kappa[g]; init = 0.0) for g in local_index_sets.G
            ) : sum(current_info.df[:s][g] * (model_info.xs[g] - current_info.x.ContVar[:s][g]) for g in local_index_sets.G)
        ) + 
        sum(current_info.df[:y][g] * (model_info.xy[g] - current_info.x.BinVar[:y][g]) for g in local_index_sets.G) + 
        (local_param.algorithm == :SDDPL ?  
            sum(
                sum(current_info.df[:sur][g][k] * (model_info.sur[g, k] - current_info.x.ContAugState[:s][g][k]) for k in keys(current_info.df[:sur][g]); init = 0.0) for g in local_index_sets.G
            ) : 0.0
        )
    );

    @constraint(
        model_info.model, 
        [k = 1:m], 
        model_info.y .≥ current_info.G[k] + 
        (local_param.algorithm == :SDDiP ?  
            sum(
                sum(current_info.dG[k].ContStateBin[:s][g][j] * (model_info.sur[g, j] .- current_info.x.ContStateBin[:s][g][j]) for j in 1:local_param.kappa[g]; init = 0.0) for g in local_index_sets.G
            ) : sum(current_info.dG[k].ContVar[:s][g] * (model_info.xs[g] .- current_info.x.ContVar[:s][g]) for g in keys(current_info.df[:s]))
        ) + 
        sum(current_info.dG[k].BinVar[:y][g] * (model_info.xy[g] .- current_info.x.BinVar[:y][g]) for g in keys(current_info.df[:y])) + 
        (local_param.algorithm == :SDDPL ?  
            sum(
                sum(current_info.dG[k].ContAugState[:s][g][j] * (model_info.sur[g, j] .- current_info.x.ContAugState[:s][g][j]) for j in keys(current_info.df[:sur][g]); init = 0.0) for g in local_index_sets.G
            ) : 0.0
        )
    );

    return                                                                              
end

"""
    function LevelSetMethod_optimization!(
        model::Model, 
        levelsetmethod_oracle_param::LevelSetMethodOracleParam, 
        state_info::StateInfo,
        CutGenerationInfo::CutGeneration;
        index_sets::Union{Nothing, IndexSets} = nothing,
        param_demand::Union{Nothing, ParamDemand} = nothing,
        param_opf::Union{Nothing, ParamOPF} = nothing,
        param::Union{Nothing, NamedTuple} = nothing,
        param_levelsetmethod::Union{Nothing, NamedTuple} = nothing
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
function LevelSetMethod_optimization!(
    model::Model, 
    levelsetmethod_oracle_param::LevelSetMethodOracleParam, 
    state_info::StateInfo,
    CutGenerationInfo::CutGeneration;
    index_sets::Union{Nothing, IndexSets} = nothing,
    param_demand::Union{Nothing, ParamDemand} = nothing,
    param_opf::Union{Nothing, ParamOPF} = nothing,
    param::Union{Nothing, NamedTuple} = nothing,
    param_levelsetmethod::Union{Nothing, NamedTuple} = nothing,
)    
    local_index_sets = _resolve_scuc_levelset_arg(:index_sets, index_sets, (:index_sets, :indexSets))
    local_param_demand = _resolve_scuc_levelset_arg(:param_demand, param_demand, (:param_demand, :paramDemand))
    local_param_opf = _resolve_scuc_levelset_arg(:param_opf, param_opf, (:param_opf, :paramOPF))
    local_param = _resolve_scuc_levelset_arg(:param, param, (:param,))
    local_param_levelsetmethod = _resolve_scuc_levelset_arg(
        :param_level_method,
        param_levelsetmethod,
        (:param_levelsetmethod, :param_LevelMethod),
    )

    # Keep local aliases explicit so downstream formulas remain unchanged.
    index_sets = local_index_sets
    param = local_param
    param_levelsetmethod = local_param_levelsetmethod
    _ = local_param_demand
    _ = local_param_opf

    ## ==================================================== Level-set Method ============================================== ##    
    (D, G, L, B) = (index_sets.D, index_sets.G, index_sets.L, index_sets.B);
    iter = 1;
    α = 1/2;
    Δ = Inf; 

    # trajectory
    current_info, current_info_f = solve_inner_minimization_problem(
        CutGenerationInfo,
        model, 
        levelsetmethod_oracle_param.x₀, 
        state_info;
        index_sets = index_sets
    );

    function_history = FunctionHistory(  
        Dict(1 => current_info.f), 
        Dict(1 => maximum(current_info.G[k] for k in keys(current_info.G)) )
    );

    # model for oracle
    oracle_model = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
    ); 
    MOI.set(oracle_model, MOI.Silent(), true);
    set_optimizer_attribute(oracle_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(oracle_model, "TimeLimit", param.time_limit);

    ## ==================================================== Levelset Method ============================================== ##
    para_oracle_bound = abs(current_info.f);
    z_rhs = 5 * 10^(ceil(log10(para_oracle_bound)));
    @variable(oracle_model, z ≥ - z_rhs);
    @variable(oracle_model, xs_oracle[G]);
    @variable(oracle_model, xy_oracle[G]);
    if param.algorithm == :SDDPL
        @variable(oracle_model, sur_oracle[g in G, k in keys(state_info.ContAugState[:s][g])]);
    elseif param.algorithm == :SDDP
        sur_oracle = nothing;
    elseif param.algorithm == :SDDiP
        @variable(oracle_model, sur_oracle[g in G, i in 1:param.kappa[g]]);
    end
    @variable(oracle_model, y ≤ 0);

    @objective(oracle_model, Min, z);
    oracle_info = ModelInfo(oracle_model, xs_oracle, xy_oracle, sur_oracle, y, z);


    next_model = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
    ); 
    MOI.set(next_model, MOI.Silent(), true);
    set_optimizer_attribute(next_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(next_model, "TimeLimit", param.time_limit);

    @variable(next_model, xs[G]);
    @variable(next_model, xy[G]);
    if param.algorithm == :SDDPL
        @variable(next_model, sur[g in G, k in keys(state_info.ContAugState[:s][g])]);
    elseif param.algorithm == :SDDP
        sur = nothing;
    elseif param.algorithm == :SDDiP
        @variable(next_model, sur[g in G, i in 1:param.kappa[g]])
    end
    @variable(next_model, z1);
    @variable(next_model, y1);
    next_info = ModelInfo(next_model, xs, xy, sur, y1, z1);

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

    while true
        add_constraint(current_info, oracle_info; index_sets = index_sets, param = param);
        optimize!(oracle_model);
        st = termination_status(oracle_model);
        if termination_status(oracle_model) == MOI.OPTIMAL
            f_star = JuMP.objective_value(oracle_model);
        else 
            # @info "Oracle Model is $(st)!"
            return (cut_info = cut_info, iter = iter)
        end

        # formulate alpha model
        result = Δ_model_formulation(function_history, f_star, iter; param = param);
        previousΔ = copy.(Δ);
        Δ, a_min, a_max = result[1], result[2], result[3];

        if param_levelsetmethod.verbose # && (iter % 30 == 0)
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, - current_info.f, current_info.G[1])
        end

        x₀ = current_info.x;
        if (round(previousΔ) > round(Δ)) || ((current_info.G[1] ≤ 0.0))
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
        end

        # update α
        if param_levelsetmethod.μ/2 ≤ (α-a_min)/(a_max-a_min) .≤ 1 - param_levelsetmethod.μ/2
            α = α;
        else
            α = round.((a_min+a_max)/2, digits = 6);
        end

        # update level
        w = α * f_star;
        W = minimum( α * function_history.f_his[j] + (1-α) * function_history.G_max_his[j] for j in 1:iter);
        
        level = w + param_levelsetmethod.λ * (W - w);
        
        ## ==================================================== next iteration point ============================================== ##
        # obtain the next iteration point
        if iter == 1
            @constraint(next_model, level_constraint, α * z1 + (1 - α) * y1 ≤ level);
        else 
            delete(next_model, next_model[:level_constraint]);
            unregister(next_model, :level_constraint);
            @constraint(next_model, level_constraint, α * z1 + (1 - α) * y1 ≤ level);
        end
        add_constraint(current_info, next_info; index_sets = index_sets, param = param);
        @objective(
            next_model, 
            Min, 
            sum(
                (param.algorithm == :SDDiP ?  
                    sum((sur[g, i] - x₀.ContStateBin[:s][g][i]) * (sur[g, i] - x₀.ContStateBin[:s][g][i]) for i in 1:param.kappa[g]; init = 0.0
                    ) : (xs[g] - x₀.ContVar[:s][g]) * (xs[g] - x₀.ContVar[:s][g])
                ) + 
                (xy[g] - x₀.BinVar[:y][g]) * (xy[g] - x₀.BinVar[:y][g]) + 
                (param.algorithm == :SDDPL ?  
                    sum((sur[g, k] - x₀.ContAugState[:s][g][k]) * (sur[g, k] - x₀.ContAugState[:s][g][k]) for k in keys(x₀.ContAugState[:s][g]); init = 0.0
                    ) : 0.0 
                )
                for g in G
            ) 
        );
        optimize!(next_model);
        st = termination_status(next_model);
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                g => JuMP.value(xy[g]) for g in index_sets.G)
            );
            ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                g => JuMP.value(xs[g]) for g in index_sets.G)
            );
            if param.algorithm == :SDDP
                ContAugState = nothing; 
                ContStateBin = nothing;
            elseif param.algorithm == :SDDPL
                ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                    :s => Dict{Any, Dict{Any, Any}}(
                        g => Dict{Any, Any}(
                            k => JuMP.value(sur[g, k]) for k in keys(state_info.ContAugState[:s][g])
                        ) for g in index_sets.G
                    )
                );
                ContStateBin = nothing;
            elseif param.algorithm == :SDDiP
                ContAugState = nothing;
                ContStateBin = Dict{Any, Dict{Any, Any}}(
                    :s => Dict{Any, Dict{Any, Any}}(
                        g => Dict{Any, Any}(
                            i => JuMP.value(sur[g, i]) for i in 1:param.kappa[g]
                        ) for g in index_sets.G
                    )
                );
            end

            x_nxt = StateInfo(
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
        elseif st == MOI.NUMERICAL_ERROR ## need to figure out why this case happened and fix it
            # @info "Numerical Error occurs -- Build a new next_model"
            next_model = Model(optimizer_with_attributes(
                ()->Gurobi.Optimizer(GRB_ENV), 
                "Threads" => 0)
            ); 
            MOI.set(next_model, MOI.Silent(), true);
            set_optimizer_attribute(next_model, "MIPGap", param.mip_gap);
            set_optimizer_attribute(next_model, "TimeLimit", param.time_limit);
            @variable(next_model, xs[G]);
            @variable(next_model, xy[G]);
            if param.algorithm == :SDDPL
                @variable(next_model, sur[g in G, k in keys(state_info.ContAugState[:s][g])]);
            elseif param.algorithm == :SDDP
                sur = nothing;
            elseif param.algorithm == :SDDiP
                @variable(next_model, sur[g in G, k in 1:param.kappa[g]])
            end
            @variable(next_model, z1);
            @variable(next_model, y1);
            next_info = ModelInfo(next_model, xs, xy, sur, y1, z1);
            @constraint(next_model, level_constraint, α * z1 + (1 - α) * y1 ≤ level);
            add_constraint(current_info, next_info; index_sets = index_sets, param = param);
            @objective(
                next_model, 
                Min, 
                sum(
                    (param.algorithm == :SDDiP ?  
                        sum((sur[g, i] - x₀.ContStateBin[:s][g][i]) * (sur[g, i] - x₀.ContStateBin[:s][g][i]) for i in 1:param.kappa[g]; init = 0.0
                        ) : (xs[g] - x₀.ContVar[:s][g]) * (xs[g] - x₀.ContVar[:s][g])
                    ) + 
                    (xy[g] - x₀.BinVar[:y][g]) * (xy[g] - x₀.BinVar[:y][g]) + 
                    (param.algorithm == :SDDPL ?  
                        sum((sur[g, k] - x₀.ContAugState[:s][g][k]) * (sur[g, k] - x₀.ContAugState[:s][g][k]) for k in keys(x₀.ContAugState[:s][g]); init = 0.0
                        ) : 0.0 
                    )
                    for g in G
                ) 
            );
            optimize!(next_model);
            st = termination_status(next_model);
            if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
                BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                    g => JuMP.value(xy[g]) for g in index_sets.G)
                );
                ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                    g => JuMP.value(xs[g]) for g in index_sets.G)
                );
                if param.algorithm == :SDDP
                    ContAugState = nothing; 
                    ContStateBin = nothing;
                elseif param.algorithm == :SDDPL
                    ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                        :s => Dict{Any, Dict{Any, Any}}(
                            g => Dict{Any, Any}(
                                k => JuMP.value(sur[g, k]) for k in keys(state_info.ContAugState[:s][g])
                            ) for g in index_sets.G
                        )
                    );
                    ContStateBin = nothing;
                elseif param.algorithm == :SDDiP
                    ContAugState = nothing;
                    ContStateBin = Dict{Any, Dict{Any, Any}}(
                        :s => Dict{Any, Dict{Any, Any}}(
                            g => Dict{Any, Any}(
                                i => JuMP.value(sur[g, i]) for i in 1:param.kappa[g]
                            ) for g in index_sets.G
                        )
                    );
                end

                x_nxt = StateInfo(
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
            else
                return (cut_info = cut_info, iter = iter)
            end
        else
            # @info "Re-compute Next Iteration Point -- change to a safe level!"
            set_normalized_rhs( level_constraint, w + .99 * (W - w))
            optimize!(next_model)
            st = termination_status(next_model);
            if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
                BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                    g => JuMP.value(xy[g]) for g in index_sets.G)
                );
                ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                    g => JuMP.value(xs[g]) for g in index_sets.G)
                );
                BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                    g => JuMP.value(xy[g]) for g in index_sets.G)
                );
                ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
                    g => JuMP.value(xs[g]) for g in index_sets.G)
                );
                if param.algorithm == :SDDP
                    ContAugState = nothing; 
                    ContStateBin = nothing;
                elseif param.algorithm == :SDDPL
                    ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                        :s => Dict{Any, Dict{Any, Any}}(
                            g => Dict{Any, Any}(
                                k => JuMP.value(sur[g, k]) for k in keys(state_info.ContAugState[:s][g])
                            ) for g in index_sets.G
                        )
                    );
                    ContStateBin = nothing;
                elseif param.algorithm == :SDDiP
                    ContAugState = nothing;
                    ContStateBin = Dict{Any, Dict{Any, Any}}(
                        :s => Dict{Any, Dict{Any, Any}}(
                            g => Dict{Any, Any}(
                                i => JuMP.value(sur[g, i]) for i in 1:param.kappa[g]
                            ) for g in index_sets.G
                        )
                    );
                end

                x_nxt = StateInfo(
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
            else
                return (cut_info = cut_info, iter = iter)
            end
        end

        ## stop rules
        if Δ ≤ param_levelsetmethod.threshold * CutGenerationInfo.primal_bound || iter > param_levelsetmethod.MaxIter
            return (cut_info = cut_info, iter = iter)
        end
        ## ==================================================== end ============================================== ##
        ## save the trajectory
        current_info, current_info_f = solve_inner_minimization_problem(
            CutGenerationInfo,
            model, 
            x_nxt, 
            state_info;
            index_sets = index_sets
        );
        iter = iter + 1;
        function_history.f_his[iter] = current_info.f;
        function_history.G_max_his[iter] = maximum(current_info.G[k] for k in keys(current_info.G));
    end
end
