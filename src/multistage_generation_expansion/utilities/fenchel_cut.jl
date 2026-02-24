"""
    function AddDisjunctiveCuts(
        model::Model,
        j::Int64
    )::Nothing

# Arguments
  1. `model`: stage model
  2. `ω`: index of the realization
  3. `variables`: corresponding to order of the columns in the matrix form

# usage: 
        Add previous generated MDCs

"""
function AddDisjunctiveCuts(
    model::Model,
    ω::Int64,
    variables::Vector{VariableRef}
)::Nothing

    ## add the previous MDCs
    for ((k1, k2), cut_coefficient) in model[:cut_expression]
        if k1 == ω
            model[:disjunctive_cuts][k2] = @constraint(
                model, 
                sum(cut_coefficient.π_value[i] * variables[i] for i in 1:length(variables)) ≥ cut_coefficient.π_0_value
            );
        end
    end
    
    return 
end

"""
    function Fenchel_cut_optimization!(
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
    state::Vector, 
    ω::Int64;
    param::NamedTuple, 
    param_cut::NamedTuple, 
    param_LevelMethod::NamedTuple,
    MDCiter::Real = 10
)::NamedTuple
    d = 0; 
    runtime_param = resolve_ge_runtime_params(param)
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    variables = all_variables(tightenLPModel);
    ## add the previous MDCs
    if inherit_disjunctive_cuts && MDCiter > 0
        AddDisjunctiveCuts(
            tightenLPModel,
            ω,
            variables
        );
    end
    x̂ = nothing; st = nothing; Z = nothing; slope = nothing; 

    level_set_method_param = LevelSetMethodParam(
        0.95, 
        param_LevelMethod.λ, 
        1.0, 
        1e14, 
        1e2,  
        state, 
        runtime_param.cut_selection, 
        nothing, 
        0.0
    );
    while d < MDCiter
        lp_model = relax_integrality(tightenLPModel); optimize!(tightenLPModel); st = termination_status(tightenLPModel);
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            # the point to be cut off       
            x̂_x = Dict(
                tightenLPModel[:x][i] => value.(tightenLPModel[:x][i]) for i in 1:param.binary_info.d
            );
        end
        all_integer = all(is_almost_integer.(values(x̂_x)));
        if all_integer
            Z = JuMP.objective_value(tightenLPModel); 
            slope = Vector{Float64}(undef, param.binary_info.n);
            for j in 1:param.binary_info.n
                slope[j] = dual(tightenLPModel[:NonAnticipative][j])
            end
            cut_info =  [ 
                Z - slope'state, 
                slope
            ];
            for i in keys(tightenLPModel[:disjunctive_cuts])
                delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
            end
            unregister(tightenLPModel, :disjunctive_cuts);
            tightenLPModel[:disjunctive_cuts] = Dict();
            
            lp_model();  
            return (
                cut_info = cut_info, 
                iter = d
            )
        end
        fractional_point = value.(variables);
        
        lp_model(); 
        ## formulate CGLP and generate a disjunctive cut
        RemoveNonAnticipative!(tightenLPModel);
        (β_value, β_0_value, st) = LevelSetMethod_FC_optimization!( 
            tightenLPModel, 
            level_set_method_param, 
            fractional_point,
            param,
            param_LevelMethod
        );
        AddNonAnticipative!(tightenLPModel, state);
        
        if st == MOI.OPTIMAL 
            num_of_MDC = length(tightenLPModel[:disjunctive_cuts]); 
            if inherit_disjunctive_cuts
                tightenLPModel[:cut_expression][ω, num_of_MDC + 1] = (
                    π_value = β_value, 
                    π_0_value = β_0_value
                );
            end
            tightenLPModel[:disjunctive_cuts][num_of_MDC + 1] = @constraint(
                tightenLPModel, 
                β_value' * variables ≥ β_0_value
            );
            @objective(
                tightenLPModel, 
                Min, 
                tightenLPModel[:primal_objective_expression]
            );
        end
        d = d + 1;
    end 
    # generate Benders cuts
    lp_model = relax_integrality(tightenLPModel); 
    optimize!(tightenLPModel); 
    st = termination_status(tightenLPModel);
    if st == MOI.OPTIMAL 
        Z = JuMP.objective_value(tightenLPModel); 

        slope = Vector{Float64}(undef, param.binary_info.n);
        for j in 1:param.binary_info.n
            slope[j] = dual(tightenLPModel[:NonAnticipative][j])
        end
        cut_info =  [ 
            Z - slope'state, 
            slope
        ];
        for i in keys(tightenLPModel[:disjunctive_cuts])
            delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
        end
        unregister(tightenLPModel, :disjunctive_cuts);
        tightenLPModel[:disjunctive_cuts] = Dict();

        lp_model();  
        return (
            cut_info = cut_info, 
            iter = d
        )

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
            state,
            ω;
            param = param, 
            param_cut = param_cut, 
            param_LevelMethod = param_LevelMethod,
            MDCiter = 0
        )
    end
end


"""
    This function is to collect the information from the objective f, and constraints G
"""
function FuncInfo_LevelSetMethod(
    x₀::Vector{Float64},
    model::Model,
    fractional_point::Vector{Float64}
)::CurrentInfo

    variables = all_variables(model);

    @objective(
        model, 
        Min, 
        x₀' * variables
    );
    optimize!(model);
    F_solution = ( 
        F = JuMP.objective_value(model), 
        ∇F = JuMP.value.(variables)
    );

    current_info  = CurrentInfo(
        x₀, 
        - F_solution.F + x₀' * fractional_point, 
        Dict(1 => 1/2 * sum(x₀ .* x₀) - 1/2),
        fractional_point - F_solution.∇F, 
        Dict(1 => x₀), 
        F_solution.F
    );   

    return current_info
end

"""
    function LevelSetMethod_FC_optimization!( 
        model                           ::Model, 
        level_set_method_param             ::LevelSetMethodParam, 
        fractional_point                ::Vector{Float64},
        param                           ::NamedTuple,
        param_cut                       ::NamedTuple,
        param_LevelMethod               ::NamedTuple
    )

# Arguments

    1. `model::Model` : the model to be optimized
    2. `level_set_method_param::LevelSetMethodParam` : parameters for the level set method
    3. `fractional_point::Vector{Float64}` : the fractional point to be optimized
    4. `param::NamedTuple` : parameters for the optimization
    5. `param_LevelMethod::NamedTuple` : parameters for the level set method

# Usage
        This function implements the level set method for Fenchel cut optimization.
"""
function LevelSetMethod_FC_optimization!( 
    model                           ::Model, 
    level_set_method_param             ::LevelSetMethodParam, 
    fractional_point                ::Vector{Float64},
    param                           ::NamedTuple,
    param_LevelMethod               ::NamedTuple
)::NamedTuple
    runtime_param = resolve_ge_runtime_params(param)
    time_limit = runtime_param.time_limit

    (μ, λ, threshold, nxt_bound, max_iter) = (
        level_set_method_param.μ, 
        level_set_method_param.λ, 
        level_set_method_param.threshold, 
        level_set_method_param.nxt_bound, 
        level_set_method_param.max_iter
    );

    n = length(fractional_point);    
    ## ==================================================== Levelset Method ============================================== ##
    cut_info = nothing; 
    iter = 1;
    α = 1/2;
    ## trajectory
    x₀ = ones(n)./n; x_nxt = ones(n)./n;
    current_info = FuncInfo_LevelSetMethod(
        x₀, ## initial point
        model, 
        fractional_point,
    );
    st = termination_status(model);
    if st == MOI.OPTIMAL
        cut_info =  (
            π_value = x₀, 
            π_0_value = current_info.L_at_x̂,
            st = st
        )
    else
        cut_info =  (
            π_value = [0.0 for i in 1:n], 
            π_0_value = 0.0,
            st = st
        )
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
    set_optimizer_attribute(oracle_model, "TimeLimit", time_limit);
    set_optimizer_attribute(oracle_model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(oracle_model, "FeasibilityTol", param.feasibility_tol);

    para_oracle_bound = abs(current_info.f);
    z_rhs = 5 * 10^(ceil(log10(para_oracle_bound)));
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
    set_optimizer_attribute(next_model, "TimeLimit", time_limit);
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
            else
                return cut_info
            end
        end

        ## stop rule: gap ≤ .07 * function-value && constraint ≤ 0.05 * LagrangianFunction
        if Δ ≤ param_LevelMethod.threshold || iter > max_iter * 2
            # @info "yes"
            return cut_info
        end

        ## ==================================================== end ============================================== ##
        ## save the trajectory
        current_info = FuncInfo_LevelSetMethod(
            x_nxt,
            model,
            fractional_point
        );
        iter = iter + 1;
        
        function_history.f_his[iter] = current_info.f;
        function_history.G_max_his[iter] = maximum(current_info.G[k] for k in keys(current_info.G));

    end

end