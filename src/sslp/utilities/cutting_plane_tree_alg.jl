if !isdefined(Main, :select_branch_variable)
    include(joinpath(@__DIR__, "..", "..", "common", "branching_policy.jl"))
end

"""
    find_node: given a branch tree, and a x̂, find the deepest node which x̂ located in
"""
function find_node(
    node, 
    Tree, 
    columns,
    point_value_map::Union{Nothing, Dict{Any, Float64}} = nothing,
)::Int
    var_ = Tree[node][:var]
    if typeof(var_) == Nothing                                       # return a leaf node
        return node
    end

    var_value = if !isnothing(point_value_map) && haskey(point_value_map, var_)
        point_value_map[var_]
    else
        try
            Float64(value(var_))
        catch err
            if err isa JuMP.OptimizeNotCalled
                return node
            end
            rethrow(err)
        end
    end
    if round(var_value, digits = 6) ≤ Tree[node][:lb]
        find_node(Tree[node][:l_child], Tree, columns, point_value_map)
    elseif round(var_value, digits = 6) ≥ Tree[node][:ub]
        find_node(Tree[node][:r_child], Tree, columns, point_value_map)                         # arrive a non-leaf node
    else
        return node                                                # return a non-leaf node
    end
end

"""
    update_cutting_plane_tree(;
        Tree::Dict{Any, Any} = Tree, 
        Leaf::Vector{Int64} = Leaf,
        branch_var::VariableRef = branch_var,
        value::Float64 = value(branch_var),
        columns::Dict{VariableRef, Int64} = columns
    )::NamedTuple
# Arguments
    1. `Tree`: branch tree
    2. `Leaf`: leaf nodes
    3. `branch_var`: branching variable
    4. `value`: branching variable value
    5. `columns`: variable index

# usage: 
        Given a branch tree, and a x̂, update the tree by adding a new node
"""
function update_cutting_plane_tree(;
    Tree::Dict{Any, Any} = Tree, 
    Leaf::Vector{Int64} = Leaf,
    branch_var::VariableRef = branch_var, 
    value::Float64 = value(branch_var),
    columns::Dict{VariableRef, Int64} = columns,
    point_value_map::Union{Nothing, Dict{Any, Float64}} = nothing,
)::NamedTuple

    σ = find_node(1, Tree, columns, point_value_map);
    if σ ∈ Leaf                                  # σ is a leaf node
        tot_node = length(keys(Tree));
        Tree[σ][:l_child] = tot_node + 1; Tree[σ][:r_child] = tot_node + 2; 
        Tree[σ][:var] = branch_var; 
        # Tree[σ][:index] = var_index;
        Tree[σ][:lb] = floor(value);
        Tree[σ][:ub] = Tree[σ][:lb] + 1;
    
        Tree[tot_node + 1] = Dict{Symbol, Any}(:parent =>σ, :l_child => nothing, :r_child => nothing, :var => nothing, :lb => nothing, :ub => nothing);
        Tree[tot_node + 1][:track] = copy(Tree[σ][:track]); push!(Tree[tot_node + 1][:track], (branch_var, :upper_bound, Tree[σ][:lb]));
    
        Tree[tot_node + 2] = Dict{Symbol, Any}(:parent =>σ, :l_child => nothing, :r_child => nothing, :var => nothing, :lb => nothing, :ub => nothing);
        Tree[tot_node + 2][:track] = copy(Tree[σ][:track]); push!(Tree[tot_node + 2][:track], (branch_var, :lower_bound, Tree[σ][:ub]));
        deleteat!(Leaf, findall(x -> x == σ, Leaf));
        push!(Leaf, tot_node + 1, tot_node + 2);
    end

    return (
        Tree = Tree, 
        Leaf = Leaf
    )
end


"""
    lp_matrix_data(model::GenericModel{T})

Given a JuMP model of a linear program, return an [`LPMatrixData{T}`](@ref)
struct storing data for an equivalent linear program in the form:
```math
\\begin{aligned}
\\min & c^\\top x + c_0 \\\\
      & b \\le A x \\le d \\\\
      & x_1 \\le x \\le x_2
\\end{aligned}
```
where elements in `x` may be continuous, integer, or binary variables.

## Fields

The struct returned by [`lp_matrix_data`](@ref) has the fields:

 * `A::SparseArrays.SparseMatrixCSC{T,Int}`: the constraint matrix in sparse
   matrix form.
 * `b_lower::Vector{T}`: the dense vector of row lower bounds. If missing, the
   value of `typemin(T)` is used.
 * `b_upper::Vector{T}`: the dense vector of row upper bounds. If missing, the
   value of `typemax(T)` is used.
 * `x_lower::Vector{T}`: the dense vector of variable lower bounds. If missing,
   the value of `typemin(T)` is used.
 * `x_upper::Vector{T}`: the dense vector of variable upper bounds. If missing,
   the value of `typemax(T)` is used.
 * `c::Vector{T}`: the dense vector of linear objective coefficients
 * `c_offset::T`: the constant term in the objective function.
 * `sense::MOI.OptimizationSense`: the objective sense of the model.
 * `integers::Vector{Int}`: the sorted list of column indices that are integer
   variables.
 * `binaries::Vector{Int}`: the sorted list of column indices that are binary
   variables.
 * `variables::Vector{GenericVariableRef{T}}`: a vector of [`GenericVariableRef`](@ref),
   corresponding to order of the columns in the matrix form.
 * `affine_constraints::Vector{ConstraintRef}`: a vector of [`ConstraintRef`](@ref),
   corresponding to the order of rows in the matrix form.

    retrieve_coef_CGLP: get the constraint matrix and necessary data from "relaxed_convex_hull_region" model
"""
function retrieve_coef_CGLP(
    tightenLPModel::Model
)::NamedTuple
    standard_form = JuMP.lp_matrix_data(tightenLPModel);
    # Use JuMP's LP matrix column order directly to keep matrix/cut indices consistent.
    variables = standard_form.variables;
    columns = Dict(
        var => i for (i, var) in enumerate(variables)
    );

    A = standard_form.A;
    b = standard_form.b_lower;
    d = standard_form.b_upper;
    l = standard_form.x_lower;
    u = standard_form.x_upper;

    return (
        A = A, 
        b = b, 
        d = d,
        l = l, 
        u = u,
        columns = columns,
        variables = variables
    )
end

"""
    function cut_generation_lp(
        A, 
        b::Vector{Float64}, 
        x::Vector{VariableRef}, 
        columns::Dict{VariableRef, Int64}, 
        x_fractional::Vector{Float64}, 
        param::NamedTuple; 
        u::Vector{Float64} = u, 
        l::Vector{Float64} = l, 
        Tree::Dict{Any, Any} = Tree, 
        Leaf::Vector{Int64} = Leaf
    )::NamedTuple

# usage: 
        The MDC cut-generation linear program
"""   
function cut_generation_lp(
    A, 
    b::Vector{Float64}, 
    b̄::Vector{Float64},
    l::Vector{Float64},
    u::Vector{Float64},
    Tree::Dict{Any, Any},
    Leaf::Vector{Int64},
    columns::Dict{VariableRef, Int64}, 
    x_fractional::Dict{Int64, Float64},
    param::NamedTuple
)::NamedTuple

    model = Model(optimizer_with_attributes(
        () -> Gurobi.Optimizer(GRB_ENV), 
        "Threads" => 0)
        ); 
    MOI.set(model, MOI.Silent(), !param.verbose);
    set_optimizer_attribute(model, "MIPGap", param.mip_gap);
    set_optimizer_attribute(model, "TimeLimit", param.time_limit);
    set_optimizer_attribute(model, "MIPFocus", param.mip_focus);           
    set_optimizer_attribute(model, "FeasibilityTol", param.feasibility_tol);
    set_optimizer_attribute(model, "NumericFocus", param.numeric_focus);

    lbset = Dict()
    ubset = Dict()
    # update lower/upper bounds according to leaf nodes
    for leaf in Leaf
        lbset[leaf] = copy(l)
        ubset[leaf] = copy(u)
        for track in Tree[leaf][:track]
            if track[2] == :upper_bound
                if track[3] < ubset[leaf][columns[track[1]]]
                    ubset[leaf][columns[track[1]]] = track[3]
                end
            elseif track[2] == :lower_bound
                if track[3] > lbset[leaf][columns[track[1]]]
                    lbset[leaf][columns[track[1]]] = track[3]
                end
            end
        end
    end
    (m, n) = size(A)

    if param.normalization == :Regular
        # variables
        @variable(model, λ[1:m, Leaf] ≥ 0);
        @variable(model, μ[1:n, Leaf] ≥ 0);
        @variable(model, ν[1:n, Leaf] ≥ 0);
        @variable(model, γ[1:m, Leaf] ≥ 0);
        # normalization
        @variable(model, -1 ≤ π_[1:n] ≤ 1);
        @variable(model, -1 ≤ π_0 ≤ 1);
        # constraints
        @constraint(model, [τ in Leaf], π_' .== (λ[:, τ] - γ[:, τ])'A .+ μ[:, τ]' .- ν[:, τ]');
        # @constraint(model, [τ in Leaf], π_0 ≤ b'λ[:, τ] - b̄'γ[:, τ] + lbset[τ]'μ[:, τ] - ubset[τ]'ν[:, τ]);
        ex = Dict{Int, Any}();
        for τ in Leaf
            ex[τ] = @expression(model, π_0)
            for i in 1:m
                if b[i] == -Inf
                    @constraint(model, λ[i, τ] == 0)
                else 
                    ex[τ] -= b[i] * λ[i, τ]
                end

                if b̄[i] == Inf
                    @constraint(model, γ[i, τ] == 0)
                else 
                    ex[τ] += b̄[i] * γ[i, τ]
                end
            end
            for i in 1:n
                if lbset[τ][i] == -Inf
                    @constraint(model, μ[i, τ] == 0)
                else 
                    ex[τ] -= lbset[τ][i] * μ[i, τ]
                end

                if ubset[τ][i] == Inf
                    @constraint(model, ν[i, τ] == 0)
                else 
                    ex[τ] += ubset[τ][i] * ν[i, τ]
                end
            end
        end
        @constraint(model, [τ in Leaf], ex[τ] ≤ 0.0);


        # Construct the expression for the objective function
        obj_expression = @expression(model, π_0 - sum(x_fractional[i] * π_[i] for i in keys(x_fractional)))
        # @objective(model, Max, π_0 - x_fractional'π_);
        @objective(model, Max, obj_expression);
        optimize!(model); 
        st = termination_status(model);
        if st == MOI.OPTIMAL
            π_value = value.(model[:π_]); 
            π_0_value = value.(model[:π_0]);
        else
            π_value = [0.0 for i in 1:n]; 
            π_0_value = 0.0;
        end
    ## ------------------------------------------------------------------------------------------------------------- ##
    elseif param.normalization == :L1norm
        error("CPT CGLP normalization ':L1norm' is not implemented for SSLP.")
    elseif param.normalization == :Facet
        error("CPT CGLP normalization ':Facet' is not implemented for SSLP.")
    elseif param.normalization == :α
        # variables
        @variable(model, λ[1:m, Leaf] ≥ 0);
        @variable(model, μ[1:n, Leaf] ≥ 0);
        @variable(model, ν[1:n, Leaf] ≥ 0);
        @variable(model, γ[1:m, Leaf] ≥ 0);

        @variable(model, π_[1:n]);
        @variable(model, π_0);
        # normalization
        @variable(model, z ≤ 1);
        @constraint(model, [i in 1:n], z ≥ π_[i]);
        @constraint(model, [i in 1:n], z ≥ -π_[i]);
        # constraints
        @constraint(model, [τ in Leaf], π_' .== (λ[:, τ] - γ[:, τ])'A .+ μ[:, τ]' .- ν[:, τ]');
        # @constraint(model, [τ in Leaf], π_0 ≤ b'λ[:, τ] - b̄'γ[:, τ] + lbset[τ]'μ[:, τ] - ubset[τ]'ν[:, τ]);
        ex = Dict{Int, Any}();
        for τ in Leaf
            ex[τ] = @expression(model, π_0)
            for i in 1:m
                if b[i] == -Inf
                    @constraint(model, λ[i, τ] == 0)
                else 
                    ex[τ] -= b[i] * λ[i, τ]
                end

                if b̄[i] == Inf
                    @constraint(model, γ[i, τ] == 0)
                else 
                    ex[τ] += b̄[i] * γ[i, τ]
                end
            end
            for i in 1:n
                if lbset[τ][i] == -Inf
                    @constraint(model, μ[i, τ] == 0)
                else 
                    ex[τ] -= lbset[τ][i] * μ[i, τ]
                end

                if ubset[τ][i] == Inf
                    @constraint(model, ν[i, τ] == 0)
                else 
                    ex[τ] += ubset[τ][i] * ν[i, τ]
                end
            end
        end
        @constraint(model, [τ in Leaf], ex[τ] ≤ 0.0);


        # Construct the expression for the objective function
        obj_expression = @expression(model, π_0 - sum(x_fractional[i] * π_[i] for i in keys(x_fractional)))
        # @objective(model, Max, π_0 - x_fractional'π_);
        @objective(model, Max, obj_expression);
        optimize!(model); 
        st = termination_status(model);
        if st == MOI.OPTIMAL
            π_value = value.(model[:π_]); 
            π_0_value = value.(model[:π_0]);
        else
            π_value = [0.0 for i in 1:n]; 
            π_0_value = 0.0;
        end
    elseif param.normalization == :SNC 
        # variables
        @variable(model, λ[1:m, Leaf] ≥ 0);
        @variable(model, μ[1:n, Leaf] ≥ 0);
        @variable(model, ν[1:n, Leaf] ≥ 0);
        @variable(model, γ[1:m, Leaf] ≥ 0);

        @variable(model, π_[1:n]);
        @variable(model, π_0);
        # normalization
        @constraint(model, sum(λ) + sum(μ) + sum(ν) + sum(γ) == 1);
        # constraints
        @constraint(model, [τ in Leaf], π_' .== (λ[:, τ] - γ[:, τ])'A .+ μ[:, τ]' .- ν[:, τ]');
        # @constraint(model, [τ in Leaf], π_0 ≤ b'λ[:, τ] - b̄'γ[:, τ] + lbset[τ]'μ[:, τ] - ubset[τ]'ν[:, τ]);
        ex = Dict{Int, Any}();
        for τ in Leaf
            ex[τ] = @expression(model, π_0)
            for i in 1:m
                if b[i] == -Inf
                    @constraint(model, λ[i, τ] == 0)
                else 
                    ex[τ] -= b[i] * λ[i, τ]
                end

                if b̄[i] == Inf
                    @constraint(model, γ[i, τ] == 0)
                else 
                    ex[τ] += b̄[i] * γ[i, τ]
                end
            end
            for i in 1:n
                if lbset[τ][i] == -Inf
                    @constraint(model, μ[i, τ] == 0)
                else 
                    ex[τ] -= lbset[τ][i] * μ[i, τ]
                end

                if ubset[τ][i] == Inf
                    @constraint(model, ν[i, τ] == 0)
                else 
                    ex[τ] += ubset[τ][i] * ν[i, τ]
                end
            end
        end
        @constraint(model, [τ in Leaf], ex[τ] ≤ 0.0);


        # Construct the expression for the objective function
        obj_expression = @expression(model, π_0 - sum(x_fractional[i] * π_[i] for i in keys(x_fractional)))
        # @objective(model, Max, π_0 - x_fractional'π_);
        @objective(model, Max, obj_expression);
        optimize!(model); 
        st = termination_status(model);
        if st == MOI.OPTIMAL
            π_value = value.(model[:π_]); 
            π_0_value = value.(model[:π_0]);
        else
            π_value = [0.0 for i in 1:n]; 
            π_0_value = 0.0;
        end

    elseif param.normalization == :Trivial
        # variables
        @variable(model, λ[1:m, Leaf] ≥ 0);
        @variable(model, μ[1:n, Leaf] ≥ 0);
        @variable(model, ν[1:n, Leaf] ≥ 0);
        @variable(model, γ[1:m, Leaf] ≥ 0);

        @variable(model, π_[1:n]);
        @variable(model, π_0);
        # normalization
        @constraint(model, sum(μ) + sum(ν) == 1);
        # constraints
        @constraint(model, [τ in Leaf], π_' .== (λ[:, τ] - γ[:, τ])'A .+ μ[:, τ]' .- ν[:, τ]');
        # @constraint(model, [τ in Leaf], π_0 ≤ b'λ[:, τ] - b̄'γ[:, τ] + lbset[τ]'μ[:, τ] - ubset[τ]'ν[:, τ]);
        ex = Dict{Int, Any}();
        for τ in Leaf
            ex[τ] = @expression(model, π_0)
            for i in 1:m
                if b[i] == -Inf
                    @constraint(model, λ[i, τ] == 0)
                else 
                    ex[τ] -= b[i] * λ[i, τ]
                end

                if b̄[i] == Inf
                    @constraint(model, γ[i, τ] == 0)
                else 
                    ex[τ] += b̄[i] * γ[i, τ]
                end
            end
            for i in 1:n
                if lbset[τ][i] == -Inf
                    @constraint(model, μ[i, τ] == 0)
                else 
                    ex[τ] -= lbset[τ][i] * μ[i, τ]
                end

                if ubset[τ][i] == Inf
                    @constraint(model, ν[i, τ] == 0)
                else 
                    ex[τ] += ubset[τ][i] * ν[i, τ]
                end
            end
        end
        @constraint(model, [τ in Leaf], ex[τ] ≤ 0.0);


        # Construct the expression for the objective function
        obj_expression = @expression(model, π_0 - sum(x_fractional[i] * π_[i] for i in keys(x_fractional)))
        # @objective(model, Max, π_0 - x_fractional'π_);
        @objective(model, Max, obj_expression);
        optimize!(model); 
        st = termination_status(model);
        if st == MOI.OPTIMAL
            π_value = value.(model[:π_]); 
            π_0_value = value.(model[:π_0]);
        else
            π_value = [0.0 for i in 1:n]; 
            π_0_value = 0.0;
        end
    end
    
    return (
        π_value = π_value, 
        π_0_value = π_0_value,
        st = st
    )
end

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
    function _normalize_cut_selection(...)

# Purpose
    Validate that cut symbol uses canonical runtime naming.

# Arguments
    1. Raw cut symbol from config or call site.

# Returns
    1. Canonical cut symbol used by the algorithm switch logic.
"""
function _normalize_cut_selection(sym::Symbol)::Symbol
    if sym in (:MDC, :iMDC, :Split, :SplitCut)
        error("Legacy cut symbol `$sym` is no longer supported. Use canonical symbols: :DBC, :iDBC, :SPC.")
    end
    return sym
end

"""
    function _sslp_remove_nonanticipativity_constraints!(...)

# Purpose
    Remove SSLP nonanticipativity constraints so the model matches the CGLP relaxation region.

# Arguments
    1. SSLP backward model.

# Returns
    1. In-place model update with anchor nonanticipativity removed.
"""
function _sslp_remove_nonanticipativity_constraints!(model::Model)::Nothing
    if :NonAnticipative ∈ keys(model.obj_dict)
        for j in eachindex(model[:NonAnticipative])
            delete(model, model[:NonAnticipative][j])
        end
        unregister(model, :NonAnticipative)
    end
    return
end

"""
    function _sslp_remove_surrogate_constraints!(...)

# Purpose
    Clear SSLP surrogate-Delta helper constraints from prior candidate evaluations.

# Arguments
    1. SSLP backward model.

# Returns
    1. In-place cleanup of surrogate alpha and surrogate equalities.
"""
function _sslp_remove_surrogate_constraints!(model::Model)::Nothing
    if :SurrogateNonAnticipative ∈ keys(model.obj_dict)
        for constraint_ref in values(model[:SurrogateNonAnticipative])
            delete(model, constraint_ref)
        end
        unregister(model, :SurrogateNonAnticipative)
    end
    if :SurrogateAlpha ∈ keys(model.obj_dict)
        delete(model, model[:SurrogateAlpha])
        unregister(model, :SurrogateAlpha)
    end
    return
end

"""
    function _sslp_restore_anchor_nonanticipativity!(...)

# Purpose
    Restore anchor nonanticipativity equalities in SSLP after surrogate candidate checks.

# Arguments
    1. SSLP backward model and anchor state vector.

# Returns
    1. In-place model reset to anchored form.
"""
function _sslp_restore_anchor_nonanticipativity!(
    model::Model,
    state::Vector{Float64},
)::Nothing
    _sslp_remove_surrogate_constraints!(model)
    _sslp_remove_nonanticipativity_constraints!(model)
    add_nonanticipativity_constraint(model, state)
    return
end

"""
    function _sslp_add_surrogate_edge_constraints!(...)

# Purpose
    Add SSLP surrogate interpolation constraints for one copy-variable split candidate.

# Arguments
    1. Model, anchor state, copy-variable array, and selected branch position.

# Returns
    1. In-place model mutation defining surrogate edge constraints.
"""
function _sslp_add_surrogate_edge_constraints!(
    model::Model,
    state::Vector{Float64},
    copy_vars::Vector{VariableRef},
    branch_position::Int64,
)::Nothing
    _sslp_remove_nonanticipativity_constraints!(model)
    _sslp_remove_surrogate_constraints!(model)

    alpha = @variable(
        model,
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "surrogate_alpha",
    )
    model[:SurrogateAlpha] = alpha

    constraints = Dict{Int64, ConstraintRef}()
    for idx in eachindex(copy_vars)
        if idx == branch_position
            endpoint_value = 1.0 - state[idx]
            constraints[idx] = @constraint(
                model,
                copy_vars[idx] == alpha * state[idx] + (1.0 - alpha) * endpoint_value,
            )
        else
            constraints[idx] = @constraint(model, copy_vars[idx] == state[idx])
        end
    end
    model[:SurrogateNonAnticipative] = constraints
    return
end

"""
    function _sslp_endpoint_objective!(...)

# Purpose
    Solve SSLP model at one endpoint state and return endpoint objective when feasible.

# Arguments
    1. Model and endpoint state vector.

# Returns
    1. Endpoint objective value or `nothing` if infeasible.
"""
function _sslp_endpoint_objective!(
    model::Model,
    endpoint_state::Vector{Float64},
)::Union{Nothing, Float64}
    _sslp_restore_anchor_nonanticipativity!(model, endpoint_state)
    optimize!(model)
    st = termination_status(model)
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
        return JuMP.objective_value(model)
    end
    return nothing
end

"""
    function _sslp_evaluate_surrogate_candidate!(...)

# Purpose
    Compute surrogate Delta and branch data for one SSLP copy-variable candidate.

# Arguments
    1. Model, anchor state, copy-variable list, candidate index, matrix columns, and anchor objective.

# Returns
    1. Named tuple with feasibility, Delta, branch value, and fractional-point cache.
"""
function _sslp_evaluate_surrogate_candidate!(
    model::Model,
    state::Vector{Float64},
    copy_vars::Vector{VariableRef},
    branch_position::Int64,
    columns::Dict{VariableRef, Int64},
    anchor_value::Float64,
)::NamedTuple
    endpoint_state = copy(state)
    endpoint_state[branch_position] = 1.0 - endpoint_state[branch_position]
    endpoint_value = _sslp_endpoint_objective!(model, endpoint_state)
    if isnothing(endpoint_value)
        _sslp_restore_anchor_nonanticipativity!(model, state)
        return (
            feasible = false,
            delta = -Inf,
            branch_value = 0.5,
            x_fractional = Dict{Int64, Float64}(),
        )
    end

    _sslp_add_surrogate_edge_constraints!(model, state, copy_vars, branch_position)
    optimize!(model)
    st = termination_status(model)
    if !(st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED)
        _sslp_restore_anchor_nonanticipativity!(model, state)
        return (
            feasible = false,
            delta = -Inf,
            branch_value = 0.5,
            x_fractional = Dict{Int64, Float64}(),
        )
    end

    alpha = value(model[:SurrogateAlpha])
    objective_value = JuMP.objective_value(model)
    delta = anchor_value * alpha + endpoint_value * (1.0 - alpha) - objective_value
    x_fractional = Dict(index => value(var) for (var, index) in columns)
    branch_value = value(copy_vars[branch_position])

    _sslp_restore_anchor_nonanticipativity!(model, state)
    return (
        feasible = true,
        delta = delta,
        branch_value = branch_value,
        x_fractional = x_fractional,
    )
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
function compute_deviation(
    x̂::Dict
)::Dict
    dev = Dict();
    for (idx, val) in x̂
        if !is_almost_integer(val)
            dev[idx] = minimum([floor(val) - val + 1, val - floor(val)])
        end
    end
    return dev
end
is_almost_integer(x) = isinteger(round(x, digits=12));

"""
    function CPT_optimization!(...)

# Purpose
    Execute one CPT-based disjunctive strengthening routine for a backward subproblem (DBC/iDBC/SPC path).

# Arguments
    1. Tightened backward model, anchor state info, scenario/node id, and runtime parameter bundles.

# Returns
    1. Named tuple with generated Benders cut information and number of disjunctive rounds used.
"""
function CPT_optimization!(
    tightenLPModel::Model, 
    state::Vector,
    ω::Int64;
    param::NamedTuple, 
    param_cut::NamedTuple, 
    param_LevelMethod::NamedTuple,
    MDCiter::Real = 10
)::NamedTuple
    runtime_param = resolve_sslp_runtime_params(param)
    cut_selection = runtime_param.cut_selection
    enable_surrogate_copy_split = runtime_param.enable_surrogate_copy_split
    copy_split_strategy = runtime_param.copy_split_strategy
    copy_split_delta_tol = runtime_param.copy_split_delta_tol
    copy_branch_policy = runtime_param.copy_branch_policy
    copy_branch_min_deviation = runtime_param.copy_branch_min_deviation
    copy_branch_dominance_ratio = runtime_param.copy_branch_dominance_ratio
    copy_branch_mean_ratio = runtime_param.copy_branch_mean_ratio
    copy_branch_boost = runtime_param.copy_branch_boost
    use_copy_branching = runtime_param.use_copy_branching

    d = 0; 
    Tree = Dict();
    # root node "1"
    Tree[1] = Dict{Symbol, Any}(
        :parent  =>nothing, 
        :l_child => nothing, 
        :r_child => nothing, 
        :var     => nothing, 
        :lb      => nothing,
        :ub      => nothing,
        :track   => [] 
    ); 
    Leaf = [1];

    # relaxed_region = relaxed_convex_hull_region(stage_data, demand);
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    remove_nonanticipativity_constraint(tightenLPModel);
    (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(
        tightenLPModel
    );
    ## add the previous MDCs
    if inherit_disjunctive_cuts && MDCiter > 0
        AddDisjunctiveCuts(
            tightenLPModel,
            ω,
            variables
        );
        (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(
            tightenLPModel
        );
    end
    add_nonanticipativity_constraint(
        tightenLPModel,
        state
    );
    x̂ = nothing; st = nothing; x_fractional = nothing; Z = nothing; slope = nothing; 
    lp_model = relax_integrality(tightenLPModel);
    while d < MDCiter
        optimize!(tightenLPModel); st = termination_status(tightenLPModel);
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
            # the point to be cut off
            x_fractional = Dict(
                index => value(var) for (var, index) in columns
            );
            xy = Dict(tightenLPModel[:y][i, j] => value(tightenLPModel[:y][i, j]) for i in 1:param.I, j in 1:param.J)
            xz = Dict(tightenLPModel[:z][j] => value(tightenLPModel[:z][j]) for j in 1:param.J)
            x̂ = xy
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
        
        surrogate_cut_added = false
        if use_copy_branching && enable_surrogate_copy_split && copy_split_strategy == :surrogate_delta
            anchor_value = JuMP.objective_value(tightenLPModel)
            copy_vars = [tightenLPModel[:z][j] for j in 1:param.J]
            best_delta = -Inf
            best_position = 0
            best_branch_value = 0.5
            best_x_fractional = Dict{Int64, Float64}()
            for branch_position in eachindex(copy_vars)
                candidate = _sslp_evaluate_surrogate_candidate!(
                    tightenLPModel,
                    state,
                    copy_vars,
                    branch_position,
                    columns,
                    anchor_value,
                )
                if candidate.feasible && candidate.delta > best_delta
                    best_delta = candidate.delta
                    best_position = branch_position
                    best_branch_value = candidate.branch_value
                    best_x_fractional = candidate.x_fractional
                end
            end

            if best_position > 0 && best_delta > copy_split_delta_tol
                surrogate_point_value_map = Dict{Any, Float64}()
                for (var, idx) in columns
                    if haskey(best_x_fractional, idx)
                        surrogate_point_value_map[var] = Float64(best_x_fractional[idx])
                    end
                end
                (Tree, Leaf) = update_cutting_plane_tree(
                    Tree = Tree,
                    Leaf = Leaf,
                    branch_var = copy_vars[best_position],
                    value = best_branch_value,
                    columns = columns,
                    point_value_map = surrogate_point_value_map,
                )
                (π_value, π_0_value, surrogate_status) = cut_generation_lp(
                    A,
                    b,
                    b̄,
                    l,
                    u,
                    Tree,
                    Leaf,
                    columns,
                    best_x_fractional,
                    param,
                )
                if surrogate_status == MOI.OPTIMAL
                    num_of_MDC = length(tightenLPModel[:disjunctive_cuts])
                    if inherit_disjunctive_cuts
                        tightenLPModel[:cut_expression][ω, num_of_MDC + 1] = (
                            π_value = π_value,
                            π_0_value = π_0_value,
                        )
                    end
                    tightenLPModel[:disjunctive_cuts][num_of_MDC + 1] = @constraint(
                        tightenLPModel,
                        sum(π_value[index] * var for (var, index) in columns) ≥ π_0_value,
                    )
                    remove_nonanticipativity_constraint(tightenLPModel)
                    (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(tightenLPModel)
                    add_nonanticipativity_constraint(tightenLPModel, state)
                    surrogate_cut_added = true
                end
            end
        end

        if surrogate_cut_added
            optimize!(tightenLPModel)
            st = termination_status(tightenLPModel)
            if !(st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED)
                @warn("tightened Model is infeasible after surrogate copy split. Return the Benders' Cut.")
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
                    MDCiter = 0,
                )
            end
            x_fractional = Dict(index => value(var) for (var, index) in columns)
            xy = Dict(tightenLPModel[:y][i, j] => value(tightenLPModel[:y][i, j]) for i in 1:param.I, j in 1:param.J)
            xz = Dict(tightenLPModel[:z][j] => value(tightenLPModel[:z][j]) for j in 1:param.J)
            x̂ = xy
            all_integer = all(is_almost_integer.(values(x̂)))
            if all_integer
                Z = JuMP.objective_value(tightenLPModel)
                if has_duals(tightenLPModel)
                    slope = Vector{Float64}(undef, param.J)
                    for j in 1:param.J
                        slope[j] = dual(tightenLPModel[:NonAnticipative][j])
                    end
                    cut_info = [
                        Z - slope' * state,
                        slope,
                    ]
                    for i in keys(tightenLPModel[:disjunctive_cuts])
                        delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                    end
                    unregister(tightenLPModel, :disjunctive_cuts);
                    tightenLPModel[:disjunctive_cuts] = Dict();

                    lp_model();
                    return (
                        cut_info = cut_info,
                        iter = d,
                    )
                end
            end
        end

        # Algorithm 3 order: try structure-aware copy split first, then
        # check integrality and extract the dual-based Benders cut.
        if !surrogate_cut_added
            if use_copy_branching && enable_surrogate_copy_split && copy_split_strategy == :surrogate_delta
                optimize!(tightenLPModel)
                st = termination_status(tightenLPModel)
                if !(st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED)
                    @warn("tightened Model is infeasible after surrogate evaluation. Return the Benders' Cut.")
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
                x_fractional = Dict(index => value(var) for (var, index) in columns)
                xy = Dict(tightenLPModel[:y][i, j] => value(tightenLPModel[:y][i, j]) for i in 1:param.I, j in 1:param.J)
                xz = Dict(tightenLPModel[:z][j] => value(tightenLPModel[:z][j]) for j in 1:param.J)
                x̂ = xy
            end
            all_integer = all(is_almost_integer.(values(x̂)))
            if all_integer
                Z = JuMP.objective_value(tightenLPModel); 
                if has_duals(tightenLPModel)
                    slope = Vector{Float64}(undef, param.J);
                    for j in 1:param.J
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
                    @warn("No Cut Has been Generated. Return the Benders' Cut")
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
        end

        primary_dev_raw = compute_deviation(xy)
        primary_dev = Dict{Any, Float64}(var => Float64(dev) for (var, dev) in primary_dev_raw)
        primary_value_map = Dict{Any, Float64}(var => Float64(val) for (var, val) in xy)

        copy_dev = Dict{Any, Float64}()
        copy_value_map = Dict{Any, Float64}()
        if use_copy_branching
            copy_dev_raw = compute_deviation(xz)
            copy_dev = Dict{Any, Float64}(var => Float64(dev) for (var, dev) in copy_dev_raw)
            copy_value_map = Dict{Any, Float64}(var => Float64(val) for (var, val) in xz)
        end

        branch_pool = select_branch_candidate_pool(
            primary_dev,
            copy_dev;
            copy_policy = copy_branch_policy,
            min_copy_deviation = copy_branch_min_deviation,
            dominance_ratio = copy_branch_dominance_ratio,
            mean_dominance_ratio = copy_branch_mean_ratio,
            copy_boost = copy_branch_boost,
        )
        dev = branch_pool.deviation
        branch_value_map = if branch_pool.source == :copy
            copy_value_map
        elseif branch_pool.source == :mixed
            merged = copy(primary_value_map)
            for (var, val) in copy_value_map
                merged[var] = val
            end
            merged
        else
            primary_value_map
        end

        ## choose the branching variable
        branch_strategy = runtime_param.branch_selection_strategy
        ml_weights = runtime_param.branching_ml_weights
        branch_var = select_branch_variable(
            dev;
            strategy = branch_strategy,
            fallback = :MFV,
            ml_weights = ml_weights,
        );
        if isnothing(branch_var)
            @warn("No fractional branching candidate found in CPT; returning fallback Benders cut.")
            lp_model()
            return CPT_optimization!(
                tightenLPModel,
                state,
                ω;
                param = param,
                param_cut = param_cut,
                param_LevelMethod = param_LevelMethod,
                MDCiter = 0,
            )
        end
        if !haskey(branch_value_map, branch_var)
            @warn("Selected branching variable has no branching value; using model value fallback.")
            branch_value_map[branch_var] = Float64(value(branch_var))
        end

        ## re-start a cutting-plane tree if we use native CPT algorithm
        if param.algorithm == :NaiveCPT
            Tree = Dict();
            # root node "1"
            Tree[1] = Dict{Symbol, Any}(
                :parent  =>nothing, 
                :l_child => nothing, 
                :r_child => nothing, 
                :var     => nothing, 
                :lb      => nothing,
                :ub      => nothing,
                :track   => [] 
            ); 
            Leaf = [1];
        end

        (Tree, Leaf) = update_cutting_plane_tree(
            Tree = Tree, 
            Leaf = Leaf,
            branch_var = branch_var,
            value = branch_value_map[branch_var],
            columns = columns,
            point_value_map = branch_value_map,
        );

        ## formulate CGLP and generate a disjunctive cut
        (π_value, π_0_value, st) = cut_generation_lp(
            A, 
            b, 
            b̄,
            l,
            u,
            Tree,
            Leaf,
            columns, 
            x_fractional, 
            param
        );
        if st == MOI.OPTIMAL 
            # you need to add the cut to two model: tightenLPModel & relaxed_region
            num_of_MDC = length(tightenLPModel[:disjunctive_cuts]); 
            if inherit_disjunctive_cuts
                tightenLPModel[:cut_expression][ω, num_of_MDC + 1] = (
                    π_value = π_value, 
                    π_0_value = π_0_value
                );
            end
            tightenLPModel[:disjunctive_cuts][num_of_MDC + 1] = @constraint(
                tightenLPModel, 
                sum(π_value[index] * var for (var, index) in columns) ≥ π_0_value
            );
            remove_nonanticipativity_constraint(tightenLPModel);
            (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(
                tightenLPModel
            );
            add_nonanticipativity_constraint(
                tightenLPModel,
                state
            );
            d = d + 1;
        else
            if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED   ## local solution
                Z = JuMP.objective_value(tightenLPModel); 
                if has_duals(tightenLPModel)
                    slope = Vector{Float64}(undef, param.J);
                    for j in 1:param.J
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
                    return (cut_info = cut_info, iter = d)
                else
                    @warn("No Cut Has been Generated. Return the Benders' Cut")
                    lp_model();  
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
    end 
    # generate Benders cuts
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

    if has_duals(tightenLPModel)
        slope = Vector{Float64}(undef, param.J);
        for j in 1:param.J
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
        @warn("No Cut Has been Generated. Return the Benders' Cut")
        lp_model();  
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
