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
    4. `value`: value of the branching variable
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
        st = termination_status(model)
        if st == MOI.OPTIMAL
            π_value = value.(model[:π_]); 
            π_0_value = value.(model[:π_0]);
        else
            π_value = [0.0 for i in 1:n]; 
            π_0_value = 0.0;
        end
    ## ------------------------------------------------------------------------------------------------------------- ##
    elseif param.normalization == :L1norm
        error("CPT CGLP normalization ':L1norm' is not implemented for SCUC.")
    elseif param.normalization == :Facet
        error("CPT CGLP normalization ':Facet' is not implemented for SCUC.")
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
        st = termination_status(model)
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
        st = termination_status(model)
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
        st = termination_status(model)
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
    return sym == :CPT ? :DBC : sym
end

"""
    function _uc_remove_nonanticipative_constraints!(...)

# Purpose
    Remove SCUC nonanticipativity constraints so the model can be used in relaxed/CGLP form.

# Arguments
    1. SCUC subproblem model and indexing metadata.

# Returns
    1. In-place model update with anchor constraints removed.
"""
function _uc_remove_nonanticipative_constraints!(
    model::Model;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Nothing
    if :ContVarNonAnticipative ∈ keys(model.obj_dict)
        for g in index_sets.G
            delete(model, model[:ContVarNonAnticipative][g])
        end
        unregister(model, :ContVarNonAnticipative)
    end
    if :BinVarNonAnticipative ∈ keys(model.obj_dict)
        for g in index_sets.G
            delete(model, model[:BinVarNonAnticipative][g])
        end
        unregister(model, :BinVarNonAnticipative)
    end
    if :BinarizationNonAnticipative ∈ keys(model.obj_dict)
        for g in index_sets.G, i in 1:param.kappa[g]
            delete(model, model[:BinarizationNonAnticipative][g, i])
        end
        unregister(model, :BinarizationNonAnticipative)
    end
    return
end

"""
    function _uc_remove_surrogate_constraints!(...)

# Purpose
    Clear previously added surrogate-Delta constraints from the SCUC model.

# Arguments
    1. SCUC subproblem model.

# Returns
    1. In-place cleanup of surrogate alpha and related constraints.
"""
function _uc_remove_surrogate_constraints!(model::Model)::Nothing
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
    function _uc_restore_anchor_nonanticipative!(...)

# Purpose
    Restore anchor-state nonanticipativity in SCUC after surrogate evaluation.

# Arguments
    1. SCUC model and anchor `state_info`.

# Returns
    1. In-place model reset to anchored backward subproblem.
"""
function _uc_restore_anchor_nonanticipative!(
    model::Model,
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Nothing
    _uc_remove_surrogate_constraints!(model)
    _uc_remove_nonanticipative_constraints!(model; index_sets = index_sets, param = param)
    AddContVarNonAnticipative!(
        model,
        state_info;
        index_sets = index_sets,
        param = param,
    )
    return
end

"""
    function _uc_collect_copy_components(...)

# Purpose
    Collect SCUC copy-variable components eligible for surrogate-Delta evaluation.

# Arguments
    1. SCUC model and parent stage state information.

# Returns
    1. Vector of component descriptors (variable reference plus anchor value metadata).
"""
function _uc_collect_copy_components(
    model::Model,
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Vector{NamedTuple}
    components = NamedTuple[]
    if :y_copy ∈ keys(model.obj_dict)
        for g in index_sets.G
            push!(components, (
                var = model[:y_copy][g],
                kind = :y_copy,
                g = g,
                i = 0,
                state_value = Float64(state_info.BinVar[:y][g]),
            ))
        end
    end
    return components
end

"""
    function _uc_make_endpoint_state(...)

# Purpose
    Construct an endpoint state by flipping the selected copy component for Delta evaluation.

# Arguments
    1. Anchor `state_info` and one copy-component descriptor.

# Returns
    1. New `StateInfo` endpoint used in surrogate objective checks.
"""
function _uc_make_endpoint_state(
    state_info::StateInfo,
    component::NamedTuple,
)::StateInfo
    endpoint_state = deepcopy(state_info)
    if component.kind == :y_copy
        endpoint_state.BinVar[:y][component.g] = 1.0 - component.state_value
    elseif component.kind == :lambda_copy && endpoint_state.ContStateBin !== nothing
        endpoint_state.ContStateBin[:s][component.g][component.i] = 1.0 - component.state_value
    end
    return endpoint_state
end

"""
    function _uc_add_surrogate_edge_constraints!(...)

# Purpose
    Inject surrogate edge interpolation constraints for one SCUC copy split candidate.

# Arguments
    1. SCUC model, copy-component list, and selected branch position.

# Returns
    1. In-place model update that replaces exact anchor equalities with surrogate interpolation equalities.
"""
function _uc_add_surrogate_edge_constraints!(
    model::Model,
    components::Vector{NamedTuple},
    branch_position::Int64;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Nothing
    _uc_remove_nonanticipative_constraints!(model; index_sets = index_sets, param = param)
    _uc_remove_surrogate_constraints!(model)
    alpha = @variable(
        model,
        lower_bound = 0.0,
        upper_bound = 1.0,
        base_name = "surrogate_alpha",
    )
    model[:SurrogateAlpha] = alpha
    constraints = Dict{Int64, ConstraintRef}()
    for idx in eachindex(components)
        component = components[idx]
        if idx == branch_position
            endpoint_value = 1.0 - component.state_value
            constraints[idx] = @constraint(
                model,
                component.var == alpha * component.state_value + (1.0 - alpha) * endpoint_value,
            )
        else
            constraints[idx] = @constraint(model, component.var == component.state_value)
        end
    end
    model[:SurrogateNonAnticipative] = constraints
    return
end

"""
    function _uc_endpoint_objective!(...)

# Purpose
    Solve SCUC model at an endpoint anchor state and return objective value if feasible.

# Arguments
    1. SCUC model and endpoint `StateInfo`.

# Returns
    1. `Float64` objective value, or `nothing` when endpoint solve is infeasible.
"""
function _uc_endpoint_objective!(
    model::Model,
    endpoint_state::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Union{Nothing, Float64}
    _uc_restore_anchor_nonanticipative!(
        model,
        endpoint_state;
        index_sets = index_sets,
        param = param,
    )
    optimize!(model)
    st = termination_status(model)
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED
        return JuMP.objective_value(model)
    end
    return nothing
end

"""
    function _uc_evaluate_surrogate_candidate!(...)

# Purpose
    Evaluate one SCUC copy candidate by computing surrogate Delta and branch-point data.

# Arguments
    1. SCUC model, anchor state, candidate index, matrix columns, and anchor objective.

# Returns
    1. Named tuple with feasibility flag, Delta value, branch value, and cached fractional point.
"""
function _uc_evaluate_surrogate_candidate!(
    model::Model,
    state_info::StateInfo,
    components::Vector{NamedTuple},
    branch_position::Int64,
    columns::Dict{VariableRef, Int64},
    anchor_value::Float64;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::NamedTuple
    endpoint_state = _uc_make_endpoint_state(state_info, components[branch_position])
    endpoint_value = _uc_endpoint_objective!(
        model,
        endpoint_state;
        index_sets = index_sets,
        param = param,
    )
    if isnothing(endpoint_value)
        _uc_restore_anchor_nonanticipative!(
            model,
            state_info;
            index_sets = index_sets,
            param = param,
        )
        return (
            feasible = false,
            delta = -Inf,
            branch_value = 0.5,
            x_fractional = Dict{Int64, Float64}(),
        )
    end

    _uc_add_surrogate_edge_constraints!(
        model,
        components,
        branch_position;
        index_sets = index_sets,
        param = param,
    )
    optimize!(model)
    st = termination_status(model)
    if !(st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED)
        _uc_restore_anchor_nonanticipative!(
            model,
            state_info;
            index_sets = index_sets,
            param = param,
        )
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
    x_fractional = Dict(i => value(var) for (var, i) in columns)
    branch_value = value(components[branch_position].var)

    _uc_restore_anchor_nonanticipative!(
        model,
        state_info;
        index_sets = index_sets,
        param = param,
    )
    return (
        feasible = true,
        delta = delta,
        branch_value = branch_value,
        x_fractional = x_fractional,
    )
end

"""
    function _uc_neutral_cut_info(...)

# Purpose
    Build a shape-compatible zero-slope cut payload used as a safe neutral fallback.

# Arguments
    1. `state_info::StateInfo`: Anchor state used to infer slope structure.
    2. `index_sets::IndexSets`: Problem indices (mainly generator index set).
    3. `param::NamedTuple`: Runtime parameter bundle.

# Returns
    1. `Vector{Any}` payload in `[intercept, slope]` form.
"""
function _uc_neutral_cut_info(
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Vector{Any}
    zero_bin = Dict{Any, Dict{Any, Any}}(
        :y => Dict{Any, Any}(g => 0.0 for g in index_sets.G),
    )
    zero_cont = param.algorithm == :SDDiP ? nothing :
                Dict{Any, Dict{Any, Any}}(
                    :s => Dict{Any, Any}(g => 0.0 for g in index_sets.G),
                )
    zero_cont_aug = (
        param.algorithm == :SDDPL && state_info.ContAugState !== nothing
    ) ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => Dict{Any, Any}(
                k => 0.0 for k in keys(state_info.ContAugState[:s][g])
            ) for g in index_sets.G
        ),
    ) : nothing
    zero_cont_state_bin = (
        param.algorithm == :SDDiP && state_info.ContStateBin !== nothing
    ) ? Dict{Any, Dict{Any, Dict{Any, Any}}}(
        :s => Dict{Any, Dict{Any, Any}}(
            g => Dict{Any, Any}(
                i => 0.0 for i in 1:param.kappa[g]
            ) for g in index_sets.G
        ),
    ) : nothing
    zero_slope = StateInfo(
        zero_bin,
        nothing,
        zero_cont,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        zero_cont_aug,
        nothing,
        zero_cont_state_bin,
    )
    return [0.0, zero_slope]
end

"""
    function _uc_fallback_to_neutral_cut(...)

# Purpose
    Package a neutral fallback cut into the standard CPT return format.

# Arguments
    1. `iter_count::Int64`: Number of disjunctive rounds completed.
    2. `state_info::StateInfo`: Anchor state used to build neutral slope shape.
    3. `index_sets::IndexSets`: Problem indices.
    4. `param::NamedTuple`: Runtime parameter bundle.

# Returns
    1. Named tuple with fields `cut_info` and `iter`.
"""
function _uc_fallback_to_neutral_cut(
    iter_count::Int64,
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::NamedTuple
    @warn("CPT fallback with MDCiter=0 remained infeasible; returning a neutral cut to avoid recursion.")
    return (
        cut_info = _uc_neutral_cut_info(
            state_info;
            index_sets = index_sets,
            param = param,
        ),
        iter = iter_count,
    )
end

"""
    function _uc_has_primal_feasible_point(...)

# Purpose
    Check whether a solved model still provides a usable primal point.

# Arguments
    1. `model::Model`: Solved JuMP model.

# Returns
    1. `true` when primal status is feasible or nearly feasible.
"""
function _uc_has_primal_feasible_point(model::Model)::Bool
    status = primal_status(model)
    return status == MOI.FEASIBLE_POINT || status == MOI.NEARLY_FEASIBLE_POINT
end

"""
    function _uc_build_cut_info_from_duals(...)

# Purpose
    Build the Benders intercept/slope payload from nonanticipativity dual multipliers.

# Arguments
    1. `model::Model`: Solved backward model.
    2. `state_info::StateInfo`: Anchor state where dual-based cut is linearized.
    3. `index_sets::IndexSets`: Problem indices.
    4. `param::NamedTuple`: Runtime parameter bundle.

# Returns
    1. Cut payload `[intercept, slope]`, or `nothing` when required duals are unavailable.
"""
function _uc_build_cut_info_from_duals(
    model::Model,
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::Union{Nothing, Vector{Any}}
    if !has_duals(model) || (:BinVarNonAnticipative ∉ keys(model.obj_dict))
        return nothing
    end
    objective_value = JuMP.objective_value(model)
    bin_slope = Dict{Any, Dict{Any, Any}}(
        :y => Dict{Any, Any}(g => dual(model[:BinVarNonAnticipative][g]) for g in index_sets.G)
    )
    has_binarization = (:BinarizationNonAnticipative ∈ keys(model.obj_dict)) &&
                       (state_info.ContStateBin !== nothing)
    if has_binarization
        cont_state_bin_slope = Dict{Any, Dict{Any, Any}}(
            :s => Dict{Any, Dict{Any, Any}}(
                g => Dict{Any, Any}(
                    i => dual(model[:BinarizationNonAnticipative][g, i]) for i in 1:get(param.kappa, g, 0)
                ) for g in index_sets.G
            )
        )
        slope = StateInfo(
            bin_slope,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            cont_state_bin_slope,
        )
        intercept = objective_value - sum(
            bin_slope[:y][g] * state_info.BinVar[:y][g] +
            sum(
                get(cont_state_bin_slope[:s][g], i, 0.0) * state_info.ContStateBin[:s][g][i] for i in keys(state_info.ContStateBin[:s][g])
            ) for g in index_sets.G
        )
        return [intercept, slope]
    end

    cont_slope = if (:ContVarNonAnticipative ∈ keys(model.obj_dict)) && (state_info.ContVar !== nothing)
        Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(g => dual(model[:ContVarNonAnticipative][g]) for g in index_sets.G))
    else
        nothing
    end
    slope = StateInfo(
        bin_slope,
        nothing,
        cont_slope,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
    intercept = objective_value - sum(
        bin_slope[:y][g] * state_info.BinVar[:y][g] +
        (isnothing(cont_slope) ? 0.0 : cont_slope[:s][g] * state_info.ContVar[:s][g]) for g in index_sets.G
    )
    return [intercept, slope]
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
    state_info::StateInfo, 
    n::Int64;
    index_sets::IndexSets = index_sets, 
    param_demand::ParamDemand = param_demand, 
    param_opf::ParamOPF = param_opf, 
    param::NamedTuple,
    MDCiter::Real = 10
)
    runtime_param = resolve_scuc_runtime_params(param)
    cut_selection = runtime_param.cut_selection
    inherit_disjunctive_cuts = runtime_param.inherit_disjunctive_cuts
    enable_surrogate_copy_split = runtime_param.enable_surrogate_copy_split
    copy_split_strategy = runtime_param.copy_split_strategy
    copy_split_delta_tol = runtime_param.copy_split_delta_tol
    copy_split_max_candidates = runtime_param.copy_split_max_candidates
    copy_split_min_violation = runtime_param.copy_split_min_violation
    use_copy_branching = runtime_param.use_copy_branching
    copy_branch_policy = runtime_param.copy_branch_policy
    copy_branch_min_deviation = runtime_param.copy_branch_min_deviation
    copy_branch_dominance_ratio = runtime_param.copy_branch_dominance_ratio
    copy_branch_mean_ratio = runtime_param.copy_branch_mean_ratio
    copy_branch_boost = runtime_param.copy_branch_boost

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
    RemoveContVarNonAnticipative!(
        tightenLPModel, 
        index_sets = index_sets, 
        param = param
    );
    (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(
        tightenLPModel
    );
     ## add the previous MDCs
     if inherit_disjunctive_cuts && MDCiter > 0
        AddDisjunctiveCuts(
            tightenLPModel,
            n,
            variables
        );
        (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(
            tightenLPModel
        );
    end
    AddContVarNonAnticipative!( 
        tightenLPModel, 
        state_info;
        index_sets = index_sets,
        param = param
    );
    x̂ = nothing; st = nothing; x_fractional = nothing; Z = nothing; slope = nothing; var_index = nothing;
    lp_model = relax_integrality(tightenLPModel);
    # lp_model();  
    while d < MDCiter
        optimize!(tightenLPModel); st = termination_status(tightenLPModel);
        if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED || _uc_has_primal_feasible_point(tightenLPModel)
            # the point to be cut off
            x_fractional = Dict(
                i => value(variables[i]) for i in 1:length(columns)
            ); 
                        
            x̂_v = Dict(
                tightenLPModel[:v][i] => value.(tightenLPModel[:v][i]) for i in keys(tightenLPModel[:v])
            );
            x̂_y = Dict(
                tightenLPModel[:y][i] => value.(tightenLPModel[:y][i]) for i in keys(tightenLPModel[:y])
            );
            x̂_w = Dict(
                tightenLPModel[:w][i] => value.(tightenLPModel[:w][i]) for i in keys(tightenLPModel[:w])
            );
            all_integer_v = all(is_almost_integer.(values(x̂_v)));
            all_integer_y = all(is_almost_integer.(values(x̂_y)));
            all_integer_w = all(is_almost_integer.(values(x̂_w)));
            x̂_λ_copy = Dict{Any, Float64}()
            x̂_y_copy = Dict{Any, Float64}()
            if use_copy_branching
                if :λ_copy ∈ keys(tightenLPModel.obj_dict)
                    for g in index_sets.G, i in 1:param.kappa[g]
                        x̂_λ_copy[tightenLPModel[:λ_copy][g, i]] = Float64(value(tightenLPModel[:λ_copy][g, i]))
                    end
                end
                if :y_copy ∈ keys(tightenLPModel.obj_dict)
                    for g in index_sets.G
                        x̂_y_copy[tightenLPModel[:y_copy][g]] = Float64(value(tightenLPModel[:y_copy][g]))
                    end
                end
            end
        else
            @warn("tightened model has no primal feasible point; fallback to Benders cut." * " status=$(st), primal_status=$(primal_status(tightenLPModel))")
            for i in keys(tightenLPModel[:disjunctive_cuts])
                delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
            end
            unregister(tightenLPModel, :disjunctive_cuts);
            tightenLPModel[:disjunctive_cuts] = Dict();
            tightenLPModel[:cut_expression] = Dict();
            
            lp_model();  
            if MDCiter == 0
                return _uc_fallback_to_neutral_cut(
                    d,
                    state_info;
                    index_sets = index_sets,
                    param = param,
                )
            end
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

        surrogate_cut_added = false
        if use_copy_branching && enable_surrogate_copy_split && copy_split_strategy == :surrogate_delta
            anchor_value = JuMP.objective_value(tightenLPModel)
            copy_components = _uc_collect_copy_components(
                tightenLPModel,
                state_info;
                index_sets = index_sets,
                param = param,
            )
            if !isempty(copy_components)
                candidate_positions = Int64[]
                candidate_scores = Float64[]
                for idx in eachindex(copy_components)
                    copy_value = value(copy_components[idx].var)
                    deviation = minimum([abs(copy_value - floor(copy_value)), abs(ceil(copy_value) - copy_value)])
                    if deviation >= copy_split_min_violation
                        push!(candidate_positions, idx)
                        push!(candidate_scores, deviation)
                    end
                end
                if isempty(candidate_positions)
                    candidate_positions = collect(eachindex(copy_components))
                    candidate_scores = zeros(Float64, length(candidate_positions))
                end
                if copy_split_max_candidates > 0 && length(candidate_positions) > copy_split_max_candidates
                    selected_order = sortperm(candidate_scores; rev = true)[1:copy_split_max_candidates]
                    candidate_positions = [candidate_positions[idx] for idx in selected_order]
                end

                best_delta = -Inf
                best_position = 0
                best_branch_value = 0.5
                best_x_fractional = Dict{Int64, Float64}()
                for branch_position in candidate_positions
                    candidate = _uc_evaluate_surrogate_candidate!(
                        tightenLPModel,
                        state_info,
                        copy_components,
                        branch_position,
                        columns,
                        anchor_value;
                        index_sets = index_sets,
                        param = param,
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
                        branch_var = copy_components[best_position].var,
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
                            tightenLPModel[:cut_expression][n, num_of_MDC + 1] = (
                                π_value = π_value,
                                π_0_value = π_0_value,
                            )
                        end
                        tightenLPModel[:disjunctive_cuts][num_of_MDC + 1] = @constraint(
                            tightenLPModel,
                            sum(π_value[i] * variables[i] for i in 1:length(columns)) ≥ π_0_value,
                        )
                        RemoveContVarNonAnticipative!(
                            tightenLPModel;
                            index_sets = index_sets,
                            param = param,
                        )
                        (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(tightenLPModel)
                        AddContVarNonAnticipative!(
                            tightenLPModel,
                            state_info;
                            index_sets = index_sets,
                            param = param,
                        )
                        surrogate_cut_added = true
                    end
                end
            end
        end

        # Algorithm 3 order: try structure-aware copy split first, then
        # check integrality and extract the dual-based Benders cut.
        if !surrogate_cut_added &&
           !(use_copy_branching && enable_surrogate_copy_split && copy_split_strategy == :surrogate_delta) &&
           all_integer_v && all_integer_y && all_integer_w
            cut_info = _uc_build_cut_info_from_duals(
                tightenLPModel,
                state_info;
                index_sets = index_sets,
                param = param,
            )
            if !isnothing(cut_info)
                for i in keys(tightenLPModel[:disjunctive_cuts])
                    delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                end
                unregister(tightenLPModel, :disjunctive_cuts);
                tightenLPModel[:disjunctive_cuts] = Dict();
                
                lp_model();  
                return (cut_info = cut_info, iter = d)
            else
                @warn("No Cut Has been Generated. Return the Benders' Cut")
                for i in keys(tightenLPModel[:disjunctive_cuts])
                    delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                end
                unregister(tightenLPModel, :disjunctive_cuts);
                tightenLPModel[:disjunctive_cuts] = Dict();

                lp_model();  
                if MDCiter == 0
                    return _uc_fallback_to_neutral_cut(
                        d,
                        state_info;
                        index_sets = index_sets,
                        param = param,
                    )
                end
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
        end
        if !surrogate_cut_added && use_copy_branching && enable_surrogate_copy_split && copy_split_strategy == :surrogate_delta
            optimize!(tightenLPModel)
            st = termination_status(tightenLPModel)
            if !(st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED || _uc_has_primal_feasible_point(tightenLPModel))
                @warn("tightened model has no primal feasible point after surrogate evaluation; fallback to Benders cut." * " status=$(st), primal_status=$(primal_status(tightenLPModel))")
                for i in keys(tightenLPModel[:disjunctive_cuts])
                    delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                end
                unregister(tightenLPModel, :disjunctive_cuts);
                tightenLPModel[:disjunctive_cuts] = Dict();
                tightenLPModel[:cut_expression] = Dict();

                lp_model();
                if MDCiter == 0
                    return _uc_fallback_to_neutral_cut(
                        d,
                        state_info;
                        index_sets = index_sets,
                        param = param,
                    )
                end
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
            x_fractional = Dict(i => value(variables[i]) for i in 1:length(columns))
            x̂_v = Dict(tightenLPModel[:v][i] => value.(tightenLPModel[:v][i]) for i in keys(tightenLPModel[:v]))
            x̂_y = Dict(tightenLPModel[:y][i] => value.(tightenLPModel[:y][i]) for i in keys(tightenLPModel[:y]))
            x̂_w = Dict(tightenLPModel[:w][i] => value.(tightenLPModel[:w][i]) for i in keys(tightenLPModel[:w]))
            x̂_λ_copy = Dict{Any, Float64}()
            x̂_y_copy = Dict{Any, Float64}()
            if use_copy_branching
                if :λ_copy ∈ keys(tightenLPModel.obj_dict)
                    for g in index_sets.G, i in 1:param.kappa[g]
                        x̂_λ_copy[tightenLPModel[:λ_copy][g, i]] = Float64(value(tightenLPModel[:λ_copy][g, i]))
                    end
                end
                if :y_copy ∈ keys(tightenLPModel.obj_dict)
                    for g in index_sets.G
                        x̂_y_copy[tightenLPModel[:y_copy][g]] = Float64(value(tightenLPModel[:y_copy][g]))
                    end
                end
            end
            all_integer_v = all(is_almost_integer.(values(x̂_v)))
            all_integer_y = all(is_almost_integer.(values(x̂_y)))
            all_integer_w = all(is_almost_integer.(values(x̂_w)))
            if all_integer_v && all_integer_y && all_integer_w
                cut_info = _uc_build_cut_info_from_duals(
                    tightenLPModel,
                    state_info;
                    index_sets = index_sets,
                    param = param,
                )
                if !isnothing(cut_info)
                    for i in keys(tightenLPModel[:disjunctive_cuts])
                        delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                    end
                    unregister(tightenLPModel, :disjunctive_cuts);
                    tightenLPModel[:disjunctive_cuts] = Dict();

                    lp_model();
                    return (cut_info = cut_info, iter = d)
                end
            end
        end

        if surrogate_cut_added
            optimize!(tightenLPModel)
            st = termination_status(tightenLPModel)
            if !(st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED || _uc_has_primal_feasible_point(tightenLPModel))
                @warn("tightened Model is infeasible after surrogate copy split. Return the Benders' Cut.")
                for i in keys(tightenLPModel[:disjunctive_cuts])
                    delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i])
                end
                unregister(tightenLPModel, :disjunctive_cuts)
                tightenLPModel[:disjunctive_cuts] = Dict()
                tightenLPModel[:cut_expression] = Dict()
                lp_model()
                if MDCiter == 0
                    return _uc_fallback_to_neutral_cut(
                        d,
                        state_info;
                        index_sets = index_sets,
                        param = param,
                    )
                end
                return CPT_optimization!(
                    tightenLPModel,
                    state_info,
                    n;
                    index_sets = index_sets,
                    param_demand = param_demand,
                    param_opf = param_opf,
                    param = param,
                    MDCiter = 0,
                )
            end
            x_fractional = Dict(i => value(variables[i]) for i in 1:length(columns))
            x̂_v = Dict(tightenLPModel[:v][i] => value.(tightenLPModel[:v][i]) for i in keys(tightenLPModel[:v]))
            x̂_y = Dict(tightenLPModel[:y][i] => value.(tightenLPModel[:y][i]) for i in keys(tightenLPModel[:y]))
            x̂_w = Dict(tightenLPModel[:w][i] => value.(tightenLPModel[:w][i]) for i in keys(tightenLPModel[:w]))
            x̂_λ_copy = Dict{Any, Float64}()
            x̂_y_copy = Dict{Any, Float64}()
            if use_copy_branching
                if :λ_copy ∈ keys(tightenLPModel.obj_dict)
                    for g in index_sets.G, i in 1:param.kappa[g]
                        x̂_λ_copy[tightenLPModel[:λ_copy][g, i]] = Float64(value(tightenLPModel[:λ_copy][g, i]))
                    end
                end
                if :y_copy ∈ keys(tightenLPModel.obj_dict)
                    for g in index_sets.G
                        x̂_y_copy[tightenLPModel[:y_copy][g]] = Float64(value(tightenLPModel[:y_copy][g]))
                    end
                end
            end
            all_integer_v = all(is_almost_integer.(values(x̂_v)))
            all_integer_y = all(is_almost_integer.(values(x̂_y)))
            all_integer_w = all(is_almost_integer.(values(x̂_w)))
            if all_integer_v && all_integer_y && all_integer_w
                cut_info = _uc_build_cut_info_from_duals(
                    tightenLPModel,
                    state_info;
                    index_sets = index_sets,
                    param = param,
                )
                if !isnothing(cut_info)
                    for i in keys(tightenLPModel[:disjunctive_cuts])
                        delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i])
                    end
                    unregister(tightenLPModel, :disjunctive_cuts)
                    tightenLPModel[:disjunctive_cuts] = Dict()
                    lp_model()
                    return (cut_info = cut_info, iter = d)
                end
            end
        end

        primary_dev = Dict{Any, Float64}()
        compute_deviation(x̂_v, primary_dev);
        compute_deviation(x̂_y, primary_dev);
        compute_deviation(x̂_w, primary_dev);
        primary_value_map = Dict{Any, Float64}()
        for (var, val) in x̂_v
            primary_value_map[var] = Float64(val)
        end
        for (var, val) in x̂_y
            primary_value_map[var] = Float64(val)
        end
        for (var, val) in x̂_w
            primary_value_map[var] = Float64(val)
        end
        copy_dev = Dict{Any, Float64}()
        copy_value_map = Dict{Any, Float64}()
        if use_copy_branching
            compute_deviation(x̂_y_copy, copy_dev);
            for (var, val) in x̂_y_copy
                copy_value_map[var] = Float64(val)
            end
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
            @warn("No fractional branching candidate found in CPT; returning a fallback Benders cut.")
            lp_model()
            if MDCiter == 0
                return _uc_fallback_to_neutral_cut(
                    d,
                    state_info;
                    index_sets = index_sets,
                    param = param,
                )
            end
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
        if !haskey(branch_value_map, branch_var)
            @warn("Selected branching variable has no branching value; using model value fallback.")
            branch_value_map[branch_var] = Float64(value(branch_var))
        end

        ## re-start a cutting-plane tree if we use native CPT algorithm
        if param.cpt_method == :NaiveCPT
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
            RemoveContVarNonAnticipative!(
                tightenLPModel, 
                index_sets = index_sets, 
                param = param
            );
            (A, b, b̄, l, u, columns, variables) = retrieve_coef_CGLP(
                tightenLPModel
            );
            AddContVarNonAnticipative!( 
                tightenLPModel, 
                state_info;
                index_sets = index_sets,
                param = param
            );
            
            d = d + 1;
        else
            optimize!(tightenLPModel); 
            cut_info = _uc_build_cut_info_from_duals(
                tightenLPModel,
                state_info;
                index_sets = index_sets,
                param = param,
            )
            if !isnothing(cut_info)
                
                for i in keys(tightenLPModel[:disjunctive_cuts])
                    delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                end
                unregister(tightenLPModel, :disjunctive_cuts);
                tightenLPModel[:disjunctive_cuts] = Dict();

                lp_model();  
                return (cut_info = cut_info, iter = d)
            else
                @warn("No Cut Has been Generated. Return the Benders' Cut")
                for i in keys(tightenLPModel[:disjunctive_cuts])
                    delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
                end
                unregister(tightenLPModel, :disjunctive_cuts);
                tightenLPModel[:disjunctive_cuts] = Dict();
                
                lp_model();  
                if MDCiter == 0
                    return _uc_fallback_to_neutral_cut(
                        d,
                        state_info;
                        index_sets = index_sets,
                        param = param,
                    )
                end
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
        end
    end 
    # generate Benders cuts
    optimize!(tightenLPModel); 
    st = termination_status(tightenLPModel);
    if st == MOI.OPTIMAL || st == MOI.LOCALLY_SOLVED || _uc_has_primal_feasible_point(tightenLPModel)
        Z = JuMP.objective_value(tightenLPModel); 
    else
        @warn("tightened model has no primal feasible point; fallback to Benders cut." * " status=$(st), primal_status=$(primal_status(tightenLPModel))")
        for i in keys(tightenLPModel[:disjunctive_cuts])
            delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
        end
        unregister(tightenLPModel, :disjunctive_cuts);
        tightenLPModel[:disjunctive_cuts] = Dict();
        tightenLPModel[:cut_expression] = Dict();

        lp_model();  
        if MDCiter == 0
            return _uc_fallback_to_neutral_cut(
                d,
                state_info;
                index_sets = index_sets,
                param = param,
            )
        end
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

    cut_info = _uc_build_cut_info_from_duals(
        tightenLPModel,
        state_info;
        index_sets = index_sets,
        param = param,
    )
    if !isnothing(cut_info)
        for i in keys(tightenLPModel[:disjunctive_cuts])
            delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
        end
        unregister(tightenLPModel, :disjunctive_cuts);
        tightenLPModel[:disjunctive_cuts] = Dict();
        lp_model();  
        return (cut_info = cut_info, iter = d)
    else
        @warn("No Cut Has been Generated. Return the Benders' Cut")
        for i in keys(tightenLPModel[:disjunctive_cuts])
            delete(tightenLPModel, tightenLPModel[:disjunctive_cuts][i]);
        end
        unregister(tightenLPModel, :disjunctive_cuts);
        tightenLPModel[:disjunctive_cuts] = Dict();
        
        lp_model();  
        if MDCiter == 0
            return _uc_fallback_to_neutral_cut(
                d,
                state_info;
                index_sets = index_sets,
                param = param,
            )
        end
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
end
