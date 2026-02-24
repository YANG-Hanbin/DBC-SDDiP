"""
    function setupLevelmetPara_benders(...)

# Purpose
    Prepare level-method parameters specialized for Benders-cut refinement.

# Arguments
    1. Current slope estimate, incumbent value, and level-method hyperparameters.

# Returns
    1. Configured level-method parameter tuple.
"""
function setupLevelmetPara_benders(L̂::Vector{Float64}, f_star_value::Float64;
                                    cutSelection::String = "ELC", 
                                    Output_Gap::Bool = false,
                                    λ::Union{Float64, Nothing} = .3, ℓ1::Real = .0, ℓ2::Real = 0.8, threshold::Float64 = 1e-8)
    if cutSelection == "ELC"
        L̃ = L̂ .* ℓ2 .+ (1 - ℓ2)/2;
        Output = 0; threshold = threshold; 
        levelSetMethodParam = LevelSetMethodParam(0.95, λ, threshold, 
                                                            1e14, 2e2, Output, Output_Gap,
                                                                        L̂, cutSelection, L̃, f_star_value)

    elseif cutSelection == "ShrinkageLC" 
        Output = 0; threshold = threshold; 
        levelSetMethodParam = LevelSetMethodParam(0.95, λ, threshold, 
                                                            1e14, 1e2, Output, Output_Gap,
                                                                        L̂,  cutSelection, nothing, f_star_value)

    elseif cutSelection == "ELCwithoutConstraint" 
        L̃ = L̂ .* ℓ2 .+ (1 - ℓ2)/2;
        Output = 0; threshold = threshold; f_star_value = 0.0;
        levelSetMethodParam = LevelSetMethodParam(0.95, λ, threshold, 
                                                            1e14, 1e2, Output, Output_Gap,
                                                                        L̂,  cutSelection, L̃, f_star_value)

    elseif cutSelection == "LC" 
        L̃ = nothing; f_star_value = 0.0;
        Output = 0; threshold = threshold; 
        levelSetMethodParam = LevelSetMethodParam(0.95, λ, threshold, 
                                                            1e14, 1e2, Output, Output_Gap,
                                                                        L̂,  cutSelection, L̃, f_star_value)
    end

    x₀ =  L̂ .* f_star_value .* ℓ1 .- f_star_value * (ℓ1 ./ 2);

    return (levelSetMethodParam = levelSetMethodParam, x₀ = x₀)
end

"""
    This function is to collect the information from the objecive f, and constraints G
"""
function FuncInfo_LevelMethod(x₀::Vector{Float64}; 
                                        bendersModel::Model = bendersModel,
                                            cutSelection::String = cutSelection, 
                                                f_star_value::Union{Float64, Nothing} = f_star_value, 
                                                    stageData::StageData = stageData, 
                                                        L̂::Vector{Float64} =  L̂, L̃::Union{Vector{Float64}, Nothing} =  L̃, 
                                                            ϵ::Float64 = ϵ )

    if cutSelection == "ELC"
        @objective(bendersModel, Min, stageData.c1' * bendersModel[:x] + stageData.c2' * bendersModel[:y] + bendersModel[:θ] + stageData.penalty * bendersModel[:slack] + 
                                                            x₀' * ( L̂ .- bendersModel[:Lc]) );
        optimize!(bendersModel);
        F_solution = ( F = JuMP.objective_value(bendersModel), 
                            ∇F = L̂ .- JuMP.value.(bendersModel[:Lc]) );

        currentInfo  = CurrentInfo(x₀, 
                                    - F_solution.F - x₀' * ( L̃ .-  L̂),
                                    Dict(1 => (1 - ϵ) * f_star_value - F_solution.F),
                                    - F_solution.∇F - ( L̃ .-  L̂),
                                    Dict(1 => - F_solution.∇F), 
                                    F_solution.F
                                    )                                        
    elseif cutSelection == "LC"
        @objective(bendersModel, Min, stageData.c1' * bendersModel[:x] + stageData.c2' * bendersModel[:y] + bendersModel[:θ] + stageData.penalty * bendersModel[:slack] - 
                                                            x₀' * bendersModel[:Lc] );
        optimize!(bendersModel);
        F_solution = (F = JuMP.objective_value(bendersModel), 
                                ∇F = - JuMP.value.(bendersModel[:Lc]) );

        currentInfo  = CurrentInfo(x₀, 
                                    - F_solution.F - x₀' *  L̂, 
                                    Dict(1 => 0.0),
                                    - F_solution.∇F -  L̂,
                                    Dict(1 => - F_solution.∇F * 0), 
                                    F_solution.F
                                    );
    elseif cutSelection == "ELCwithoutConstraint"
        @objective(bendersModel, Min, stageData.c1' * bendersModel[:x] + stageData.c2' * bendersModel[:y] + bendersModel[:θ] + stageData.penalty * bendersModel[:slack] + 
                                                            x₀' * ( L̂ .- bendersModel[:Lc]) );
        optimize!(bendersModel);
        F_solution = ( F = JuMP.objective_value(bendersModel), 
                        ∇F = L̂ .- JuMP.value.(bendersModel[:Lc]) );

        currentInfo  = CurrentInfo(x₀, 
                                    - F_solution.F - x₀' * ( L̃ .-  L̂),
                                    Dict(1 => 0.0 ),
                                    - F_solution.∇F - ( L̃ .-  L̂),
                                    Dict(1 => - F_solution.∇F * 0), 
                                    F_solution.F
                                    ) 
    elseif cutSelection == "ShrinkageLC"
        @objective(bendersModel, Min, stageData.c1' * bendersModel[:x] + stageData.c2' * bendersModel[:y] + bendersModel[:θ] + stageData.penalty * bendersModel[:slack] + 
                                                            x₀' * ( L̂ .- bendersModel[:Lc]) );
        optimize!(bendersModel);
        F_solution = ( F = JuMP.objective_value(bendersModel), 
                        ∇F = L̂ .- JuMP.value.(bendersModel[:Lc]) );

        currentInfo  = CurrentInfo(x₀, 
                                    1/2 * sum(x₀ .* x₀),
                                    Dict(1 => (1 - ϵ) * f_star_value - F_solution.F),
                                    x₀,
                                    Dict(1 => - F_solution.∇F), 
                                    F_solution.F
                                    ) 

    end

    return currentInfo
end

######################################################################################################################
## -------------------------------------------------- Main Function -------------------------------------------- ##
######################################################################################################################
# bendersModel = copy(tightenLPModel)
"""
    function bendersCut_levelmethod!(...)

# Purpose
    Refine Benders cuts through level-method iterations on the backward model.

# Arguments
    1. Benders model, initial slope point, and level-method parameters.

# Returns
    1. Updated cut coefficients and iteration metadata.
"""
function bendersCut_levelmethod!( bendersModel::Model, x₀::Vector{Float64}; 
                                        levelSetMethodParam::LevelSetMethodParam = levelSetMethodParam, 
                                        stageData::StageData = stageData, 
                                        ϵ::Float64 = 1e-4, 
                                        binaryInfo::BinaryInfo = binaryInfo) 
    delete(bendersModel, bendersModel[:bindingConstraint])
    unregister(bendersModel, :bindingConstraint)
    set_optimizer(bendersModel, () -> Gurobi.Optimizer(GRB_ENV))
    set_attribute(bendersModel, "OutputFlag", 0)
    ######################################################################################################################
    ###############################   auxiliary function for function information   ######################################
    ######################################################################################################################
    ##  μ larger is better
    (μ, λ, threshold, nxt_bound, max_iter, Output, Output_Gap) = (levelSetMethodParam.μ, levelSetMethodParam.λ, levelSetMethodParam.threshold, levelSetMethodParam.nxt_bound, levelSetMethodParam.max_iter, levelSetMethodParam.Output,levelSetMethodParam.Output_Gap);
    (L̂, cutSelection, L̃, f_star_value) = (levelSetMethodParam.L̂, levelSetMethodParam.cutSelection, levelSetMethodParam.L̃, levelSetMethodParam.f_star_value)
    (A, n, d) = (binaryInfo.A, binaryInfo.n, binaryInfo.d);
    
    ## ==================================================== Levelset Method ============================================== ##
    iter = 1;
    α = 1/2;

    ## trajectory
    currentInfo = FuncInfo_LevelMethod(x₀, cutSelection = cutSelection, bendersModel = bendersModel, f_star_value = f_star_value, stageData = stageData, L̂ = L̂, L̃ = L̃, ϵ = ϵ);

    functionHistory = FunctionHistory(  Dict(1 => currentInfo.f), 
                                            Dict(1 => maximum(currentInfo.G[k] for k in keys(currentInfo.G)) )
                                            );

    ## model for oracle
    oracleModel = Model(
        optimizer_with_attributes(
            ()->Gurobi.Optimizer(GRB_ENV), 
            "OutputFlag" => Output, 
            "Threads" => 0)
            );

    para_oracle_bound = abs(currentInfo.f);
    z_rhs = 10 * 10^(ceil(log10(para_oracle_bound)));
    @variable(oracleModel, z  ≥  - z_rhs);
    @variable(oracleModel, x[i = 1:n]);
    @variable(oracleModel, y ≤ 0);

    @objective(oracleModel, Min, z);
    oracleInfo = ModelInfo(oracleModel, x, y, z);

    nxtModel = Model(
        optimizer_with_attributes(()->Gurobi.Optimizer(GRB_ENV), 
        "OutputFlag" => Output, 
        "Threads" => 0)
        );

    @variable(nxtModel, x1[i = 1:n]);
    @variable(nxtModel, z1 );
    @variable(nxtModel, y1 );
    nxtInfo = ModelInfo(nxtModel, x1, y1, z1);


    Δ = Inf; τₖ = 1; τₘ = .5; μₖ = 1;

    if cutSelection == "ELC" || cutSelection == "ELCwithoutConstraint"
        cutInfo =  [ - currentInfo.f - currentInfo.x' *  L̃,  
                                                                    currentInfo.x] 
    elseif cutSelection == "LC"
        cutInfo = [ - currentInfo.f - currentInfo.x' *  L̂,  
                                                                    currentInfo.x] 
    elseif cutSelection == "ShrinkageLC"
        cutInfo = [ currentInfo.L_at_x̂ - currentInfo.x' *  L̂,  
                                                                    currentInfo.x] 
    end 

    while true
        add_constraint(currentInfo, oracleInfo);
        optimize!(oracleModel);

        st = termination_status(oracleModel)
        # @info "oracle, $st, grad = $(currentInfo.dG), G = $(currentInfo.G)"
        
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            f_star = JuMP.objective_value(oracleModel)
        else
            oracleModel = Model(
            optimizer_with_attributes(
                ()->Gurobi.Optimizer(GRB_ENV), 
                "OutputFlag" => Output, 
                "Threads" => 0)
                );

            para_oracle_bound = abs(currentInfo.f);
            z_rhs = 10 * 10^(ceil(log10(para_oracle_bound)));
            @variable(oracleModel, z  ≥  - z_rhs);
            @variable(oracleModel, x[i = 1:n]);
            @variable(oracleModel, y ≤ 0);

            @objective(oracleModel, Min, z);
            oracleInfo = ModelInfo(oracleModel, x, y, z);
            add_constraint(currentInfo, oracleInfo);
            optimize!(oracleModel);
            f_star = JuMP.objective_value(oracleModel)
        end

        f_star = JuMP.objective_value(oracleModel)

        # formulate alpha model
        result = Δ_model_formulation(functionHistory, f_star, iter, Output = Output);
        previousΔ = Δ;
        Δ, a_min, a_max = result[1], result[2], result[3];
        

        if Output_Gap # && (iter % 30 == 0)
            if iter == 1
                println("------------------------------------ Iteration Info --------------------------------------")
                println("Iter |   Gap                              Objective                             Constraint")
            end
            @printf("%3d  |   %5.3g                         %5.3g                              %5.3g\n", iter, Δ, - currentInfo.f, currentInfo.G[1])
        end

        # push!(gap_list, Δ);
        x₀ = currentInfo.x;
        if round(previousΔ) > round(Δ)
            x₀ = currentInfo.x; τₖ = μₖ * τₖ;
            if cutSelection == "ELC" || cutSelection == "ELCwithoutConstraint"
                cutInfo =  [ - currentInfo.f - currentInfo.x' *  L̃,  
                                                                            currentInfo.x];
            elseif cutSelection == "LC"
                cutInfo = [ - currentInfo.f - currentInfo.x' *  L̂,  
                                                                            currentInfo.x];
            elseif cutSelection == "ShrinkageLC"
                cutInfo = [ currentInfo.L_at_x̂ - currentInfo.x' *  L̂,  
                                                                            currentInfo.x];
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
        W = minimum( α * functionHistory.f_his[j] + (1-α) * functionHistory.G_max_his[j] for j in 1:iter);

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
            @constraint(nxtModel, level_constraint, α * z1 + (1 - α) * y1 ≤ level);
        else 
            delete(nxtModel, nxtModel[:level_constraint]);
            unregister(nxtModel, :level_constraint);
            @constraint(nxtModel, level_constraint, α * z1 + (1 - α) * y1 ≤ level);
        end

        add_constraint(currentInfo, nxtInfo);
        @objective(nxtModel, Min, sum((x1 .- x₀) .* (x1 .- x₀)));
        optimize!(nxtModel)
        st = termination_status(nxtModel)
        # @info "$st"
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            x_nxt = JuMP.value.(x1);
            λₖ = abs(dual(level_constraint)); μₖ = λₖ + 1; 
        elseif st == MOI.NUMERICAL_ERROR || st == MOI.INFEASIBLE_OR_UNBOUNDED
            # @info "Numerical Error occures! -- Build a new nxtModel"
            nxtModel = Model(
                optimizer_with_attributes(()->Gurobi.Optimizer(GRB_ENV), 
                "OutputFlag" => Output, 
                "Threads" => 0)
                );
        
            @variable(nxtModel, x1[i = 1:n]);
            @variable(nxtModel, z1 );
            @variable(nxtModel, y1 );
            nxtInfo = ModelInfo(nxtModel, x1, y1, z1);
            @constraint(nxtModel, level_constraint, α * z1 + (1 - α) * y1 ≤ level);
            add_constraint(currentInfo, nxtInfo);
            @objective(nxtModel, Min, (x1 - x₀)' * (x1 - x₀));
            optimize!(nxtModel);
            x_nxt = JuMP.value.(x1);
            λₖ = abs(dual(level_constraint)); μₖ = λₖ + 1; 
        else
            set_normalized_rhs( level_constraint, w + .1 * (W - w));
            optimize!(nxtModel);
            x_nxt = JuMP.value.(x1);
        end

        ## stop rule: gap ≤ .07 * function-value && constraint ≤ 0.05 * LagrangianFunction
        if Δ ≤ (abs(f_star_value) * threshold) || iter > max_iter
            # @info "yes"
            return cutInfo
        end

        ## ==================================================== end ============================================== ##
        ## save the trajectory
        currentInfo = FuncInfo_LevelMethod(x_nxt, cutSelection = cutSelection, bendersModel = bendersModel, f_star_value = f_star_value, stageData = stageData, L̂ = L̂, L̃ = L̃, ϵ = ϵ);
        iter = iter + 1;
        
        functionHistory.f_his[iter] = currentInfo.f;
        functionHistory.G_max_his[iter] = maximum(currentInfo.G[k] for k in keys(currentInfo.G));

    end

end