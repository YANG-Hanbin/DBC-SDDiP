#############################################################################################
###########################  auxiliary functions for level set method #######################
#############################################################################################

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
    
    alpha_model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
        ); 
    MOI.set(alpha_model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(alpha_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(alpha_model, "TimeLimit", param.time_limit);

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
    model_info::ModelInfo
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
    This function is to collect the information from the objecive f, and constraints G
"""
function FuncInfo_LevelSetMethod(
    x₀::Vector{Float64},
    model::Model,
    cut_selection::Symbol,
    f_star_value::Union{Float64, Nothing},
    state::Vector{Float64},
    core_point::Union{Vector{Float64}, Nothing},
    ϵ::Float64,
)::CurrentInfo

    if cut_selection == :PLC
        @objective(
            model, 
            Min, 
            model[:primal_objective_expression] + 
            x₀' * ( state .- model[:Lc])
        );
        optimize!(model);
        F_solution = ( 
            F = JuMP.objective_value(model), 
            ∇F = state .- JuMP.value.(model[:Lc])
        );

        current_info  = CurrentInfo(
            x₀, 
            - F_solution.F - x₀' * ( core_point .-  state), 
            Dict(1 => f_star_value - F_solution.F - ϵ),
            - F_solution.∇F - ( core_point .-  state), 
            Dict(1 => - F_solution.∇F), 
            F_solution.F
        );                                        
    elseif cut_selection ∈ [:LC, :SBC]
        @objective(
            model, 
            Min, 
            model[:primal_objective_expression] - 
            x₀' * model[:Lc] 
        );
        optimize!(model);
        F_solution = (
            F = JuMP.objective_value(model), 
            ∇F = - JuMP.value.(model[:Lc])
        );

        current_info  = CurrentInfo(
            x₀, 
            - F_solution.F - x₀' *  state, 
            Dict(1 => 0.0), 
            - F_solution.∇F -  state, 
            Dict(1 => - F_solution.∇F * 0), 
            F_solution.F
        );
    elseif cut_selection == :SMC
        @objective(
            model, 
            Min, 
            model[:primal_objective_expression] + 
            x₀' * ( state .- model[:Lc])
        );
        optimize!(model);
        F_solution = ( 
            F = JuMP.objective_value(model), 
            ∇F = state .- JuMP.value.(model[:Lc])
        );

        current_info  = CurrentInfo(
            x₀, 
            1/2 * sum(x₀ .* x₀), 
            Dict(1 => f_star_value - F_solution.F - ϵ),
            x₀,
            Dict(1 => - F_solution.∇F), 
            F_solution.F 
        );

    end

    return current_info
end

"""
    function LevelSetMethod_optimization!( 
        backward_model                   ::Model, 
        stage_data                       ::StageData, 
        x₀                              ::Vector{Float64},
        level_set_method_param             ::LevelSetMethodParam, 
        param                           ::NamedTuple,
        param_cut                       ::NamedTuple,
        param_LevelMethod            ::NamedTuple
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
    model                           ::Model, 
    x₀                              ::Vector{Float64},
    level_set_method_param             ::LevelSetMethodParam, 
    param                           ::NamedTuple,
    param_cut                       ::NamedTuple,
    param_LevelMethod               ::NamedTuple
)::NamedTuple

    (μ, λ, threshold, nxt_bound, max_iter) = (
        level_set_method_param.μ, 
        level_set_method_param.λ, 
        level_set_method_param.threshold, 
        level_set_method_param.nxt_bound, 
        level_set_method_param.max_iter
    );

    (state, cut_selection, core_point, f_star_value) = (
        level_set_method_param.state, 
        level_set_method_param.cut_selection, 
        level_set_method_param.core_point, 
        level_set_method_param.f_star_value
    );
    
    (A, n, d) = (
        param.binary_info.A, 
        param.binary_info.n, 
        param.binary_info.d
    );
    
    ## ==================================================== Levelset Method ============================================== ##
    iter = 1;
    α = 1/2;

    ## trajectory
    current_info = FuncInfo_LevelSetMethod(
        x₀,
        model,
        cut_selection,
        f_star_value,
        state,
        core_point,
        param_cut.ϵ,
    );

    cut_info = nothing; 
    
    if cut_selection == :PLC 
        cut_info =  [ 
            - current_info.f - current_info.x' *  core_point, 
            current_info.x
        ];
    elseif cut_selection == :LC
        cut_info = [ 
            - current_info.f - current_info.x' *  state,  
            current_info.x
        ]; 
    elseif cut_selection == :SMC
        cut_info = [ 
            current_info.L_at_x̂ - current_info.x' *  state,  
            current_info.x
        ]; 
    elseif cut_selection == :SBC
        cut_info = [ 
            - current_info.f - current_info.x' *  state,  
            current_info.x
        ]; 
        return (cut_info = cut_info, iter = iter)
    end 

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
    set_optimizer_attribute(oracle_model, "TimeLimit", param.time_limit);
    set_optimizer_attribute(oracle_model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(oracle_model, "FeasibilityTol", param.feasibility_tol);

    para_oracle_bound = abs(current_info.f);
    z_rhs = 200 * 10^(ceil(log10(para_oracle_bound)));
    @variable(oracle_model, z  ≥  - z_rhs);
    @variable(oracle_model, x[i = 1:n]);
    @variable(oracle_model, y ≤ 0);

    @objective(oracle_model, Min, z);
    oracle_info = ModelInfo(oracle_model, x, y, z);

    next_model = Model(optimizer_with_attributes(
        ()->Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
    ); 
    MOI.set(next_model, MOI.Silent(), true);
    set_optimizer_attribute(next_model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(next_model, "TimeLimit", param.time_limit);
    set_optimizer_attribute(next_model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(next_model, "FeasibilityTol", param.feasibility_tol);

    @variable(next_model, x1[i = 1:n]);
    @variable(next_model, z1 );
    @variable(next_model, y1 );
    next_info = ModelInfo(
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
            return (cut_info = cut_info, iter = iter)
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
        

        if param_LevelMethod.verbose # && (iter % 30 == 0)
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, - current_info.f, current_info.G[1])
        end

        # push!(gap_list, Δ);
        x₀ = current_info.x;
        if round(previousΔ) > round(Δ)
            x₀ = current_info.x; τₖ = μₖ * τₖ;
            if cut_selection == :PLC
                cut_info =  [ 
                    - current_info.f - current_info.x' *  core_point,  
                    current_info.x
                ];
            elseif cut_selection == :LC
                cut_info = [ 
                    - current_info.f - current_info.x' *  state,  
                    current_info.x
                ];
            elseif cut_selection == :SMC
                cut_info = [ 
                    current_info.L_at_x̂ - current_info.x' *  state,  
                    current_info.x
                ];
            end 
        else
            τₖ = (τₖ + τₘ) / 2;
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
            set_optimizer_attribute(next_model, "TimeLimit", param.time_limit);
            set_optimizer_attribute(next_model, "MIPFocus", param.mip_focus);           
            set_optimizer_attribute(next_model, "FeasibilityTol", param.feasibility_tol);

            @variable(next_model, x1[i = 1:n]);
            @variable(next_model, z1 );
            @variable(next_model, y1 );
            next_info = ModelInfo(
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
            elseif st == MOI.NUMERICAL_ERROR || st == MOI.INFEASIBLE_OR_UNBOUNDED
                return (cut_info = cut_info, iter = iter)
            end
        end

        ## stop rule: gap ≤ .07 * function-value && constraint ≤ 0.05 * LagrangianFunction
        if Δ ≤ param_LevelMethod.threshold * f_star_value || iter > max_iter
            # @info "yes"
            return (cut_info = cut_info, iter = iter)
        end

        ## ==================================================== end ============================================== ##
        ## save the trajectory
        current_info = FuncInfo_LevelSetMethod(
            x_nxt,
            model,
            cut_selection,
            f_star_value,
            state,
            core_point,
            param_cut.ϵ,
        );
        iter = iter + 1;
        
        function_history.f_his[iter] = current_info.f;
        function_history.G_max_his[iter] = maximum(current_info.G[k] for k in keys(current_info.G));

    end

end
