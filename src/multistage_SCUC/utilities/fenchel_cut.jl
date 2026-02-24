"""
    function AddDisjunctiveCuts(
        model::Model,
        j::Int64
    )::Nothing

# Arguments
  1. `model`: stage model
  2. `n`: index of the node
  3. `variables`: corresponding to order of the columns in the matrix form

# usage: 
        Add previous generated MDCs

"""
function AddDisjunctiveCuts(
    model::Model,
    n::Int64,
    variables::Vector{VariableRef}
)::Nothing

    ## add the previous MDCs
    for ((k1, k2), cut_coefficient) in model[:cut_expression]
        if k1 == n
            model[:disjunctive_cuts][k2] = @constraint(
                model, 
                sum(cut_coefficient.π_value[i] * variables[i] for i in 1:length(variables)) ≥ cut_coefficient.π_0_value
            );
        end
    end
    
    return 
end


"""
    function CPT_optimization!(
        tightenLPModel::Model, 
        state_info::StateInfo,
        index_sets::IndexSets = index_sets, 
        param_demand::ParamDemand = param_demand, 
        param_opf::ParamOPF = param_opf, 
        param::NamedTuple;
        MDCiter::Int64 = 10
    )

# usage: 
        The cutting-plane-tree algorithm
"""
# Define a helper function to compute deviation from integer values
function compute_deviation(dict, dev)
    for (idx, val) in dict
        if !is_almost_integer(val)
            dev[idx] = minimum([floor(val) - val + 1, val - floor(val)])
        end
    end
end
is_almost_integer(x) = isinteger(round(x, digits=12));

"""
    function Fenchel_cut_optimization!(...)

# Purpose
    Run Fenchel-cut-based subproblem strengthening and return the resulting Benders cut data.

# Arguments
    1. A backward model plus anchor state and runtime cut parameters.

# Returns
    1. Cut coefficients/intercept and iteration metadata for the backward pass.
"""
function Fenchel_cut_optimization!(
    tightenLPModel::Model, 
    state_info::StateInfo,
    n::Int64;
    index_sets::IndexSets = index_sets, 
    param_demand::ParamDemand = param_demand, 
    param_opf::ParamOPF = param_opf, 
    param::NamedTuple,
    param_levelsetmethod::Union{Nothing, NamedTuple} = nothing,
    MDCiter::Real = 10
)
    d = 0; 
    runtime_param = resolve_scuc_runtime_params(param)
    time_limit = runtime_param.time_limit
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    levelset_param = isnothing(param_levelsetmethod) ? (
        isdefined(Main, :param_levelsetmethod) ? Main.param_levelsetmethod :
        (μ = 0.9, λ = 0.5, threshold = 1e-4, nxt_bound = 1e10, MaxIter = 200, verbose = false)
    ) : param_levelsetmethod
    variables = all_variables(tightenLPModel);
    columns = Dict(
        var => i for (i, var) in enumerate(variables)
    );

    ## add the previous MDCs
    if inherit_disjunctive_cuts && MDCiter > 0
        AddDisjunctiveCuts(
            tightenLPModel,
            n,
            variables
        );
    end

    x̂ = nothing; st = nothing; x_fractional = nothing; Z = nothing; slope = nothing; 
    while d < MDCiter
        lp_model = relax_integrality(tightenLPModel); optimize!(tightenLPModel); st = termination_status(tightenLPModel);
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            # the point to be cut off             
            x̂_v = Dict(
                tightenLPModel[:v][i] => value.(tightenLPModel[:v][i]) for i in keys(tightenLPModel[:v])
            );
            x̂_y = Dict(
                tightenLPModel[:y][i] => value.(tightenLPModel[:y][i]) for i in keys(tightenLPModel[:y])
            );
            x̂_w = Dict(
                tightenLPModel[:w][i] => value.(tightenLPModel[:w][i]) for i in keys(tightenLPModel[:w])
            );

            x̂ = (
                v = x̂_v, 
                y = x̂_y, 
                w = x̂_w
            );
        else
            @warn("tightened Model is infeasible! Return the Benders' Cut.")
            for i in keys(tightenLPModel[:disjunctive_cuts])
                delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
            end
            unregister(tightenLPModel, :disjunctive_cuts);
            tightenLPModel[:disjunctive_cuts] = Dict();
            tightenLPModel[:cut_expression] = Dict();
            
            lp_model();  
            return CPT_optimization!(
                tightenLPModel, 
                state_info,
                n;
                index_sets = index_sets, 
                param_demand = param_demand, 
                param_opf = param_opf, 
                param = param,
                MDCiter = 0
            )
        end
        all_integer_v = all(is_almost_integer.(values(x̂_v)));
        all_integer_y = all(is_almost_integer.(values(x̂_y)));
        all_integer_w = all(is_almost_integer.(values(x̂_w)));

        if all_integer_v && all_integer_y && all_integer_w
            Z = JuMP.objective_value(tightenLPModel); 
            BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                g => dual(tightenLPModel[:BinVarNonAnticipative][g]) for g in index_sets.G)
            );
            ContStateBin = Dict{Any, Dict{Any, Any}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => dual(tightenLPModel[:BinarizationNonAnticipative][g, i]) for i in 1:param.kappa[g]
                    ) for g in index_sets.G
                )
            );
            slope = StateInfo(
                BinVar, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing,
                nothing,
                ContStateBin
            );

            cut_info = [ 
                Z - sum(
                    sum(
                        ContStateBin[:s][g][i] * state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
                    ) + 
                    BinVar[:y][g] * state_info.BinVar[:y][g] for g in index_sets.G
                ),
                slope
            ];
            for i in keys(tightenLPModel[:disjunctive_cuts])
                delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
            end
            unregister(tightenLPModel, :disjunctive_cuts);
            tightenLPModel[:disjunctive_cuts] = Dict();
            
            lp_model();  
            return (cut_info = cut_info, iter = d)
        end
        fractional_point = value.(variables);

        lp_model(); 
        ## formulate CGLP and generate a disjunctive cut
        RemoveContVarNonAnticipative!(
            tightenLPModel, 
            index_sets = index_sets, 
            param = param
        );
        (π_value, π_0_value, st) = LevelSetMethod_FC_optimization!( 
            tightenLPModel,  
            variables,
            fractional_point,
            param,
            levelset_param
        );
        AddContVarNonAnticipative!( 
            tightenLPModel, 
            state_info;
            index_sets = index_sets,
            param = param
        );
        @objective(
            tightenLPModel, 
            Min, 
            tightenLPModel[:primal_objective_expression]
        );

        if st == MOI.OPTIMAL
            num_of_MDC = length(tightenLPModel[:disjunctive_cuts]); 
            if inherit_disjunctive_cuts
                tightenLPModel[:cut_expression][n, num_of_MDC + 1] = (
                    π_value = π_value, 
                    π_0_value = π_0_value
                );
            end
            tightenLPModel[:disjunctive_cuts][num_of_MDC + 1] = @constraint(
                tightenLPModel, 
                sum(π_value[i] * variables[i] for i in 1:length(columns)) ≥ π_0_value
            );
            
            d = d + 1;
        else
            optimize!(tightenLPModel); 
            Z = JuMP.objective_value(tightenLPModel);     
            BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
                g => dual(tightenLPModel[:BinVarNonAnticipative][g]) for g in index_sets.G)
            );
            ContStateBin = Dict{Any, Dict{Any, Any}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => dual(tightenLPModel[:BinarizationNonAnticipative][g, i]) for i in 1:param.kappa[g]
                    ) for g in index_sets.G
                )
            );
            slope = StateInfo(
                BinVar, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing, 
                nothing,
                nothing,
                ContStateBin
            );

            cut_info = [ 
                Z - sum(
                    sum(
                        ContStateBin[:s][g][i] * state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
                    ) + 
                    BinVar[:y][g] * state_info.BinVar[:y][g] for g in index_sets.G
                ),
                slope
            ];
            
            for i in keys(tightenLPModel[:disjunctive_cuts])
                delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
            end
            unregister(tightenLPModel, :disjunctive_cuts);
            tightenLPModel[:disjunctive_cuts] = Dict();

            lp_model();  
            return (cut_info = cut_info, iter = d)
        end
    end 
    # generate Benders cuts
    lp_model = relax_integrality(tightenLPModel); 
    optimize!(tightenLPModel); 
    st = termination_status(tightenLPModel);
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
        Z = JuMP.objective_value(tightenLPModel); 
    else
        @warn("tightened Model is infeasible! Return the Benders' Cut.")
        for i in keys(tightenLPModel[:disjunctive_cuts])
            delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
        end
        unregister(tightenLPModel, :disjunctive_cuts);
        tightenLPModel[:disjunctive_cuts] = Dict();
        tightenLPModel[:cut_expression] = Dict();
        
        optimize!(tightenLPModel); 
        Z = JuMP.objective_value(tightenLPModel); 
        BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
            g => dual(tightenLPModel[:BinVarNonAnticipative][g]) for g in index_sets.G)
        );
        ContStateBin = Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => dual(tightenLPModel[:BinarizationNonAnticipative][g, i]) for i in 1:param.kappa[g]
                ) for g in index_sets.G
            )
        );
        slope = StateInfo(
            BinVar, 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            nothing, 
            nothing,
            nothing,
            ContStateBin
        );

        cut_info = [ 
            Z - sum(
                sum(
                    ContStateBin[:s][g][i] * state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
                ) + 
                BinVar[:y][g] * state_info.BinVar[:y][g] for g in index_sets.G
            ),
            slope
        ];
        for i in keys(tightenLPModel[:disjunctive_cuts])
            delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
        end
        unregister(tightenLPModel, :disjunctive_cuts);
        tightenLPModel[:disjunctive_cuts] = Dict();
        lp_model();  
        
        return (cut_info = cut_info, iter = d)
    end

    BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
        g => dual(tightenLPModel[:BinVarNonAnticipative][g]) for g in index_sets.G)
    );
    ContStateBin = Dict{Any, Dict{Any, Any}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => Dict{Any, Any}(
                i => dual(tightenLPModel[:BinarizationNonAnticipative][g, i]) for i in 1:param.kappa[g]
            ) for g in index_sets.G
        )
    );
    slope = StateInfo(
        BinVar, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing,
        nothing,
        ContStateBin
    );

    cut_info = [ 
        Z - sum(
            sum(
                ContStateBin[:s][g][i] * state_info.ContStateBin[:s][g][i] for i in 1:param.kappa[g]
            ) + 
            BinVar[:y][g] * state_info.BinVar[:y][g] for g in index_sets.G
        ),
        slope
    ];
    for i in keys(tightenLPModel[:disjunctive_cuts])
        delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
    end
    unregister(tightenLPModel, :disjunctive_cuts);
    tightenLPModel[:disjunctive_cuts] = Dict();
    lp_model();  
    return (cut_info = cut_info, iter = d)

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
    iter::Int64,
    param::NamedTuple
)::Dict
    runtime_param = resolve_scuc_runtime_params(param)
    time_limit = runtime_param.time_limit
    
    alpha_model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
        ); 
    MOI.set(alpha_model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(alpha_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(alpha_model, "TimeLimit", time_limit);

    @variable(alpha_model, z)
    @variable(alpha_model, 0 ≤ α ≤ 1)
    @constraint(alpha_model, con[j = 1:iter], z ≤  α * ( function_history.f_his[j] - f_star) + (1 - α) * function_history.G_max_his[j] )
    
    # we first compute gap Δ
    @objective(alpha_model, Max, z)
    optimize!(alpha_model)
    st = termination_status(alpha_model)
    Δ = JuMP.objective_value(alpha_model)

    
    ## then we modify above model to compute alpha
    # alpha_min
    @constraint(alpha_model, z .≥ 0)
    @objective(alpha_model, Min, α)
    optimize!(alpha_model)
    a_min = JuMP.value(α)

    # alpha_max
    @objective(alpha_model, Max, α)
    optimize!(alpha_model)
    a_max = JuMP.value(α)

    return Dict(
        1 => Δ, 
        2 => a_min, 
        3 => a_max
    )

end

"""
    function add_constraint(
        current_info::FenchelInfo,
        model_info::FCModelInfo;
        index_sets::IndexSets = index_sets
    )::Nothing

# Arguments

    1. `current_info::FenchelInfo` : the current information
    2. `model_info::FCModelInfo` : the model information
    3. `index_sets::IndexSets` : the index sets
    
# Returns
    This function is to add constraints for the model f_star and nxt pt.
"""
function add_constraint(
    current_info::FenchelInfo, 
    model_info::FCModelInfo
)::Nothing
    m = length(current_info.G)

    xⱼ = current_info.x
    # add constraints     
    @constraint(
        model_info.model, 
        model_info.z .≥ current_info.f + current_info.df' * (model_info.x .- xⱼ) 
        );
    @constraint(
        model_info.model, 
        [k = 1:m], 
        model_info.y .≥ current_info.G[k] + sum(current_info.dG[k] .* (model_info.x .- xⱼ))
        );
    return                                                                           
end

"""
    function LevelSetMethod_FC_optimization!( 
        model                           ::Model, 
        level_set_method_param             ::LevelSetMethodParam, 
        param                           ::NamedTuple,
        param_cut                       ::NamedTuple,
        param_levelsetmethod            ::NamedTuple
    )

# Arguments

    1. `model::Model` : the model to be optimized
    2. `fractional_point::Vector{Float64}` : the fractional point to be optimized
    3. `param::NamedTuple` : parameters for the optimization
    4. `param_levelsetmethod::NamedTuple` : parameters for the level set method

# Usage
        This function implements the level set method for Fenchel cut optimization.
"""
function LevelSetMethod_FC_optimization!( 
    model                           ::Model, 
    variables                       ::Vector{VariableRef},
    fractional_point                ::Vector{Float64},
    param                           ::NamedTuple,
    param_levelsetmethod            ::NamedTuple
)::NamedTuple
    runtime_param = resolve_scuc_runtime_params(param)
    time_limit = runtime_param.time_limit

    (μ, λ, max_iter) = (0.95, 0.5, 5);

    n = length(fractional_point);    
    ## ==================================================== Levelset Method ============================================== ##
    cut_info = nothing; 
    iter = 1;
    α = 1/2;
    ## trajectory
    x₀ = ones(n)./n; x_nxt = ones(n)./n;

    @objective(
        model, 
        Min, 
        x₀' * variables
    );
    optimize!(model);
    current_info = FenchelInfo(
        x₀, 
        - JuMP.objective_value(model) + x₀' * fractional_point, 
        Dict(1 => 1/2 * sum(x₀ .* x₀) - 1/2),
        fractional_point - JuMP.value.(variables), 
        Dict(1 => x₀), 
        JuMP.objective_value(model)
    );
    st = termination_status(model);
    cut_info =  (
        π_value = x₀, 
        π_0_value = current_info.L_at_x̂,
        st = st
    );

    function_history = FunctionHistory(  
        Dict(1 => current_info.f), 
        Dict(1 => maximum(current_info.G[k] for k in keys(current_info.G)))
    );

    ## model for oracle
    oracle_model = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
    ); 
    MOI.set(oracle_model, MOI.Silent(), true);
    set_optimizer_attribute(oracle_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(oracle_model, "TimeLimit", time_limit);
    set_optimizer_attribute(oracle_model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(oracle_model, "FeasibilityTol", param.feasibility_tol);

    para_oracle_bound = abs(current_info.f);
    z_rhs = 5 * 10^(ceil(log10(para_oracle_bound)));
    @variable(oracle_model, z  ≥  - z_rhs);
    @variable(oracle_model, x[i = 1:n]);
    @variable(oracle_model, y ≤ 0);

    @objective(oracle_model, Min, z);
    oracle_info = FCModelInfo(
        oracle_model, 
        x, 
        y, 
        z
    );

    next_model = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
    ); 
    MOI.set(next_model, MOI.Silent(), true);
    set_optimizer_attribute(next_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(next_model, "TimeLimit", time_limit);
    set_optimizer_attribute(next_model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(next_model, "FeasibilityTol", param.feasibility_tol);

    @variable(next_model, x1[i = 1:n]);
    @variable(next_model, z1 );
    @variable(next_model, y1 );
    next_info = FCModelInfo(
        next_model, 
        x1, 
        y1, 
        z1
    );

    Δ = Inf; τₖ = 1; τₘ = .5; μₖ = 1;

    while true
        add_constraint(
            current_info, 
            oracle_info
        );
        optimize!(oracle_model);

        st = termination_status(oracle_model)
        # @info "oracle, $st, grad = $(current_info.dG), G = $(current_info.G)"
        
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            f_star = JuMP.objective_value(oracle_model)
        else
            return cut_info
        end

        f_star = JuMP.objective_value(oracle_model)

        # formulate alpha model
        result = Δ_model_formulation(
            function_history, 
            f_star, 
            iter, 
            param
        );
        previousΔ = Δ;
        Δ, a_min, a_max = result[1], result[2], result[3];
        

        if param_levelsetmethod.verbose # && (iter % 30 == 0)
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, - current_info.f, current_info.G[1])
        end

        # push!(gap_list, Δ);
        x₀ = current_info.x;
        if round(previousΔ) > round(Δ)
            cut_info =  (
                π_value = x₀, 
                π_0_value = current_info.L_at_x̂,
                st = st
            );
        end
        
        ## update α
        if μ/2 ≤ (α-a_min)/(a_max-a_min) .≤ 1-μ/2
            α = α
        else
            α = (a_min+a_max)/2
        end

        ## update level
        w = α * f_star;
        W = minimum( α * function_history.f_his[j] + (1-α) * function_history.G_max_his[j] for j in 1:iter);

        λ = iter ≤ 10 ? 0.05 : 0.1;
        λ = iter ≥ 20 ? 0.2 : λ;
        λ = iter ≥ 40 ? 0.3 : λ;
        λ = iter ≥ 50 ? 0.4 : λ;
        λ = iter ≥ 60 ? 0.5 : λ;
        λ = iter ≥ 70 ? 0.6 : λ;
        λ = iter ≥ 80 ? 0.7 : λ;
        λ = iter ≥ 90 ? 0.8 : λ;
        
        level = w + λ * (W - w);
        
        ## ==================================================== next iteration point ============================================== ##
        # obtain the next iteration point
        if iter == 1
            @constraint(
                next_model, 
                level_constraint, 
                α * z1 + (1 - α) * y1 ≤ level
            );
        else 
            delete(next_model, next_model[:level_constraint]);
            unregister(next_model, :level_constraint);
            @constraint(
                next_model, 
                level_constraint, 
                α * z1 + (1 - α) * y1 ≤ level
            );
        end

        add_constraint(current_info, next_info);
        @objective(
            next_model, 
            Min, 
            sum((x1 .- x₀) .* (x1 .- x₀))
        );
        optimize!(next_model)
        st = termination_status(next_model)
        # @info "$st"
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            x_nxt = JuMP.value.(x1);
            λₖ = abs(dual(level_constraint)); μₖ = λₖ + 1; 
        elseif st == MOI.NUMERICAL_ERROR || st == MOI.INFEASIBLE_OR_UNBOUNDED
            next_model = Model(optimizer_with_attributes(
                ()->Gurobi.Optimizer(GRB_ENV), 
                "Threads" => 0)
            ); 
            MOI.set(next_model, MOI.Silent(), true);
            set_optimizer_attribute(next_model, "MIPGap", param.mip_gap);
            set_optimizer_attribute(next_model, "TimeLimit", time_limit);
            set_optimizer_attribute(next_model, "MIPFocus", param.mip_focus);           
            set_optimizer_attribute(next_model, "FeasibilityTol", param.feasibility_tol);

            @variable(next_model, x1[i = 1:n]);
            @variable(next_model, z1 );
            @variable(next_model, y1 );
            next_info = FCModelInfo(
                next_model, 
                x1, 
                y1, 
                z1
            );
            @constraint(
                next_model, 
                level_constraint, 
                α * z1 + (1 - α) * y1 ≤ level
            );
                @objective(
                next_model, 
                Min, 
                sum((x1 .- x₀) .* (x1 .- x₀))
            );
            optimize!(next_model)
            st = termination_status(next_model)
            if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
                x_nxt = JuMP.value.(x1);
                λₖ = abs(dual(level_constraint)); μₖ = λₖ + 1; 
            else
                return cut_info
            end
        end

        ## stop rule: gap ≤ .07 * function-value && constraint ≤ 0.05 * LagrangianFunction
        if Δ ≤ param_levelsetmethod.threshold || iter > max_iter
            # @info "yes"
            return cut_info
        end

        ## ==================================================== end ============================================== ##
        ## save the trajectory
        @objective(
            model, 
            Min, 
            x_nxt' * variables
        );
        optimize!(model);
        current_info = FenchelInfo(
            x₀, 
            - JuMP.objective_value(model) + x₀' * fractional_point, 
            Dict(1 => 1/2 * sum(x₀ .* x₀) - 1/2),
            fractional_point - JuMP.value.(variables), 
            Dict(1 => x₀), 
            JuMP.objective_value(model)
        );

        iter = iter + 1;
        
        function_history.f_his[iter] = current_info.f;
        function_history.G_max_his[iter] = maximum(current_info.G[k] for k in keys(current_info.G));

    end

end