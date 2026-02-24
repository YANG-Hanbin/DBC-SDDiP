"""
function sample_scenarios(; 
    num_scenarios::Int64 = 10, 
    scenario_tree::ScenarioTree = scenario_tree
)

# Arguments

    1. `num_scenarios`: The number of scenarios will be sampled
    2. `scenario_tree`: A scenario tree

# Returns
    1. `Ξ`: A subset of scenarios.
"""
function sample_scenarios(; 
    num_scenarios::Int64 = 10, 
    scenario_tree::ScenarioTree = scenario_tree
)
    Ξ = Dict{Int64, Dict{Int64, RandomVariables}}()
    for ω in 1:num_scenarios
        ξ = Dict{Int64, RandomVariables}()
        ξ[1] = scenario_tree.tree[1].nodes[1]
        n = wsample(
            collect(keys(scenario_tree.tree[1].prob)), 
            collect(values(scenario_tree.tree[1].prob)), 
            1
        )[1]
        for t in 2:length(keys(scenario_tree.tree))
            ξ[t] = scenario_tree.tree[t].nodes[n]
            n = wsample(
                collect(keys(scenario_tree.tree[t].prob)), 
                collect(values(scenario_tree.tree[t].prob)), 
            1)[1]
        end
        Ξ[ω] = ξ
    end
    return Ξ
end

"""

function print_iteration_info(
    i::Int, 
    LB::Float64, 
    UB::Float64,
    gap::Float64, 
    iter_time::Float64, 
    LM_iter::Int, 
    total_time::Float64
)

# Arguments

    1. `i`: The current iteration number
    2. `LB`: The lower bound
    3. `UB`: The upper bound
    4. `gap`: The gap between the lower and upper bounds
    5. `iter_time`: The time spent on the current iteration
    6. `LM_iter`: The number of Lagrangian multipliers updated in the current iteration
    7. `total_time`: The total time spent on the algorithm

# Prints

"""
function print_iteration_info(
    i::Int64, 
    LB::Float64, 
    UB::Float64,
    gap::Float64, 
    iter_time::Float64, 
    LM_iter::Int, 
    total_Time::Float64
)::Nothing
    @printf("%4d | %12.2f     | %12.2f     | %9.2f%%     | %9.2f s     | %6d     | %10.2f s     \n", 
                i, LB, UB, gap, iter_time, LM_iter, total_Time); 
    return 
end

"""
    function print_iteration_info_bar(...)

# Purpose
    Print the standard iteration table header used by runtime logs.

# Arguments
    1. No required positional arguments.

# Returns
    1. `nothing`; side effect is console output.
"""
function print_iteration_info_bar()::Nothing
    println("------------------------------------------ Iteration Info ------------------------------------------------")
    println("Iter |        LB        |        UB        |       Gap      |      i-time     |    #LM     |     T-Time")
    println("----------------------------------------------------------------------------------------------------------")
    return 
end

const SCUC_PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

@inline function _sanitize_token(value)::String
    return replace(string(value), r"[^A-Za-z0-9_.=-]+" => "_")
end

@inline function _kv_token(key::AbstractString, value)::String
    return string(key, "=", _sanitize_token(value))
end

"""
    function _build_scuc_results_path(...)

# Purpose
    Build canonical SCUC result file path from effective runtime parameters.

# Arguments
    1. SCUC runtime parameter bundle.

# Returns
    1. Absolute result file path used for persistence.
"""
function _build_scuc_results_path(
    param::NamedTuple,
    param_cut::NamedTuple,
)::String
    results_root = hasproperty(param, :results_root) ? param.results_root : joinpath(SCUC_PROJECT_ROOT, "results")
    experiment_tag = hasproperty(param, :experiment_tag) ? param.experiment_tag : "default"
    cut_selection = param.cut_selection
    disjunction_iter = cut_selection == :SPC ? 1 : (param.disjunction_iteration_limit < 0 ? "Inf" : param.disjunction_iteration_limit)
    normalization = cut_selection ∈ [:DBC, :iDBC, :SPC, :SC, :FC, :CPT] ? param.normalization : "na"
    inherit_disjunctive_cuts = param.inherit_disjunctive_cuts
    enforce_binary_copies = param.enforce_binary_copies
    epsilon_scale = Int(round(1 / param.epsilon))

    dataset_id = join(
        [
            _kv_token("case", param.case),
            _kv_token("T", param.T),
            _kv_token("R", param.num),
        ],
        "__",
    )
    run_id = join(
        [
            _kv_token("tag", experiment_tag),
            _kv_token("alg", param.algorithm),
            _kv_token("cut", cut_selection),
            _kv_token("disj", disjunction_iter),
            _kv_token("norm", normalization),
            _kv_token("inherit", Int(inherit_disjunctive_cuts)),
            _kv_token("copybin", Int(enforce_binary_copies)),
            _kv_token("eps", epsilon_scale),
            _kv_token("cpt", param.cpt_method),
            _kv_token("core", param_cut.core_point_strategy),
        ],
        "__",
    )
    return joinpath(results_root, "scuc", dataset_id, "$run_id.jld2")
end

"""
    function save_sddp_results(...)

# Purpose
    Persist SCUC SDDP results to the canonical result location.

# Arguments
    1. Runtime parameter bundle and serialized result payload.

# Returns
    1. `nothing`; writes JLD2 output to disk.
"""
function save_sddp_results(
    path::AbstractString,
    sddp_results::Dict;
    run_meta::Union{Nothing, Dict} = nothing,
)::Nothing
    mkpath(dirname(path))
    if isnothing(run_meta)
        save(path, "sddp_results", sddp_results)
    else
        save(path, "sddp_results", sddp_results, "run_meta", run_meta)
    end
    return nothing
end

"""
    function save_info(...)

# Purpose
    Persist per-run histories (solution/gap metadata) to the configured output location.

# Arguments
    1. Runtime parameters and data payload to save.

# Returns
    1. `nothing`; writes result artifacts when logging is enabled.
"""
function save_info(
    param::NamedTuple, 
    param_cut::NamedTuple,
    sddp_results::Dict;
    logger_save::Bool = true
)::Nothing
    if !logger_save
        return nothing
    end

    case = param.case
    cut_selection = param.cut_selection
    num = param.num
    T = param.T
    algorithm = param.algorithm
    med_method = param.med_method
    epsilon_scale = Int(round(1 / param.epsilon))
    ℓ = param_cut.ℓ
    disjunction_iter = cut_selection == :SPC ? 1 : (param.disjunction_iteration_limit < 0 ? Inf : param.disjunction_iteration_limit)
    normalization = param.normalization
    cpt_method = param.cpt_method
    legacy_logger_paths = hasproperty(param, :legacy_logger_paths) ? param.legacy_logger_paths : false

    new_save_path = _build_scuc_results_path(param, param_cut)
    run_meta = Dict(
        :project => "scuc",
        :algorithm => algorithm,
        :cut_selection => cut_selection,
        :dataset => Dict(:case => case, :T => T, :num => num),
        :normalization => normalization,
        :disjunction_iter => disjunction_iter,
        :inherit_disjunctive_cuts => param.inherit_disjunctive_cuts,
        :legacy_logger_paths => legacy_logger_paths,
    )
    save_sddp_results(new_save_path, sddp_results; run_meta = run_meta)

    if !legacy_logger_paths
        return nothing
    end

    scuc_dir = joinpath(SCUC_PROJECT_ROOT, "src", "multistage_SCUC")
    save_path = nothing

    if algorithm == :SDDPL
        target_dir = joinpath(scuc_dir, "new_logger", "numerical_results-$case", "Periods$T-Real$num")
        filename = cut_selection == :PLC ?
            "$algorithm-$cut_selection-$med_method-$ℓ.jld2" :
            "$algorithm-$cut_selection-$med_method.jld2"
        save_path = joinpath(target_dir, filename)
    elseif algorithm == :SDDP
        target_dir = joinpath(scuc_dir, "new_logger", "numerical_results-$case", "Periods$T-Real$num")
        save_path = joinpath(target_dir, "$algorithm-$cut_selection.jld2")
    elseif algorithm == :SDDiP
        if cut_selection ∈ [:DBC, :iDBC, :SPC, :SC, :FC, :CPT]
            inheritance = param.inherit_disjunctive_cuts ? "cut_inheritance" : "without_cut_inheritance"
            target_dir = joinpath(scuc_dir, "logger", inheritance, "Periods$T-Real$num")
            filename = cut_selection ∈ [:DBC, :iDBC, :SPC, :SC, :CPT] ?
                "$cpt_method-$epsilon_scale-$cut_selection-$disjunction_iter-$normalization.jld2" :
                "$algorithm-$epsilon_scale-$cut_selection.jld2"
            save_path = joinpath(target_dir, filename)
        elseif cut_selection ∈ [:BC, :SMC, :LC, :SBC]
            target_dir = joinpath(scuc_dir, "logger", "without_cut_inheritance", "Periods$T-Real$num")
            save_path = joinpath(target_dir, "$algorithm-$epsilon_scale-$cut_selection.jld2")
        end
    end

    if !isnothing(save_path)
        save_sddp_results(save_path, sddp_results)
    end

    return nothing
end

"""
    function param_setup(...)

# Purpose
    Assemble the canonical SCUC runtime parameter bundle.
"""
function param_setup(;
    terminate_time::Float64 = 3600.0,
    time_limit::Float64 = 10.0,
    terminate_threshold::Float64 = 1e-3,
    feasibility_tol::Float64 = 1e-6,
    mip_focus::Int64 = 3,
    numeric_focus::Int64 = 3,
    mip_gap::Float64 = 1e-4,
    epsilon::Float64 = 0.125,
    max_iter::Real = Inf,
    enforce_binary_copies::Bool = false,
    inherit_disjunctive_cuts::Bool = true,
    num_scenarios::Int64 = 3,
    lift_iter_threshold::Int64 = 10,
    num_backward_scenarios::Int64 = 1,
    num_partition_scenarios::Int64 = 1,
    branch_threshold::Float64 = 1e-3,
    branch_variable::Symbol = :ALL,
    cut_selection::Symbol = :PLC,
    algorithm::Symbol = :SDDPL,
    cpt_method::Symbol = :CPT,
    T::Int64 = 12,
    num::Int64 = 10,
    normalization::Symbol = :SNC,
    med_method::Symbol = :ExactPoint,
    case::String = "case30pwl",
    logger_save::Bool = true,
    branch_selection_strategy::Symbol = :MFV,
    copy_branch_policy::Symbol = :adaptive,
    copy_branch_min_deviation::Float64 = 1e-6,
    copy_branch_dominance_ratio::Float64 = 1.0,
    copy_branch_mean_ratio::Float64 = 0.9,
    copy_branch_boost::Float64 = 1.0,
    enable_copy_branching::Bool = true,
    enable_surrogate_copy_split::Bool = true,
    copy_split_strategy::Symbol = :surrogate_delta,
    copy_split_delta_tol::Float64 = 1e-6,
    copy_split_max_candidates::Int64 = 16,
    copy_split_min_violation::Float64 = 1e-6,
    imdc_state_projection::Bool = true,
    idbc_warm_pass::Bool = false,
    results_root::Union{Nothing, String} = nothing,
    legacy_logger_paths::Bool = false,
    experiment_tag::String = "default",
    branching_ml_weights::NamedTuple = (
        intercept = 0.0,
        deviation = 1.0,
        inverse_deviation = 0.15,
        name_hash = 0.02,
        bias = 1.0,
    ),
    disjunction_iteration_limit::Int64 = 10,
)::NamedTuple
    results_root_value = isnothing(results_root) ? joinpath(SCUC_PROJECT_ROOT, "results") : results_root
    effective_cut = cut_selection == :iDBC && !enable_copy_branching ? :DBC : cut_selection

    return (
        verbose = false,
        mip_gap = mip_gap,
        time_limit = time_limit,
        terminate_time = terminate_time,
        terminate_threshold = terminate_threshold,
        feasibility_tol = feasibility_tol,
        mip_focus = mip_focus,
        numeric_focus = numeric_focus,
        max_iter = max_iter,
        theta_lower = 0.0,
        opt = 0.0,
        epsilon = epsilon,
        kappa = Dict{Int64, Int64}(),
        enforce_binary_copies = enforce_binary_copies,
        inherit_disjunctive_cuts = inherit_disjunctive_cuts,
        num_scenarios = num_scenarios,
        lift_iter_threshold = lift_iter_threshold,
        num_backward_scenarios = num_backward_scenarios,
        num_partition_scenarios = num_partition_scenarios,
        branch_threshold = branch_threshold,
        branch_variable = branch_variable,
        med_method = med_method,
        cut_selection = effective_cut,
        algorithm = algorithm,
        cpt_method = cpt_method,
        T = T,
        num = num,
        case = case,
        logger_save = logger_save,
        branch_selection_strategy = branch_selection_strategy,
        copy_branch_policy = copy_branch_policy,
        copy_branch_min_deviation = copy_branch_min_deviation,
        copy_branch_dominance_ratio = copy_branch_dominance_ratio,
        copy_branch_mean_ratio = copy_branch_mean_ratio,
        copy_branch_boost = copy_branch_boost,
        enable_copy_branching = enable_copy_branching,
        enable_surrogate_copy_split = enable_surrogate_copy_split,
        copy_split_strategy = copy_split_strategy,
        copy_split_delta_tol = copy_split_delta_tol,
        copy_split_max_candidates = copy_split_max_candidates,
        copy_split_min_violation = copy_split_min_violation,
        idbc_warm_pass = idbc_warm_pass,
        imdc_state_projection = imdc_state_projection,
        results_root = results_root_value,
        legacy_logger_paths = legacy_logger_paths,
        experiment_tag = experiment_tag,
        branching_ml_weights = branching_ml_weights,
        disjunction_iteration_limit = disjunction_iteration_limit,
        normalization = normalization,
    )
end

"""
    resolve_scuc_runtime_params(param::NamedTuple)::NamedTuple

# Purpose
    Return the canonical runtime-view used by SCUC components.
"""
function resolve_scuc_runtime_params(param::NamedTuple)::NamedTuple
    return (
        num_scenarios = param.num_scenarios,
        num_partition_scenarios = param.num_partition_scenarios,
        num_backward_scenarios = param.num_backward_scenarios,
        lift_iter_threshold = param.lift_iter_threshold,
        branch_threshold = param.branch_threshold,
        branch_variable = param.branch_variable,
        max_iter = param.max_iter,
        time_limit = param.time_limit,
        enforce_binary_copies = param.enforce_binary_copies,
        disjunction_iteration_limit = param.disjunction_iteration_limit,
        cut_selection = param.cut_selection,
        inherit_disjunctive_cuts = param.inherit_disjunctive_cuts,
        idbc_warm_pass = param.idbc_warm_pass,
        imdc_state_projection = param.imdc_state_projection,
        enable_copy_branching = param.enable_copy_branching,
        enable_surrogate_copy_split = param.enable_surrogate_copy_split,
        copy_split_strategy = param.copy_split_strategy,
        copy_split_delta_tol = param.copy_split_delta_tol,
        copy_split_max_candidates = param.copy_split_max_candidates,
        copy_split_min_violation = param.copy_split_min_violation,
        use_copy_branching = (param.cut_selection == :iDBC) && param.enable_copy_branching,
        copy_branch_policy = param.copy_branch_policy,
        copy_branch_min_deviation = param.copy_branch_min_deviation,
        copy_branch_dominance_ratio = param.copy_branch_dominance_ratio,
        copy_branch_mean_ratio = param.copy_branch_mean_ratio,
        copy_branch_boost = param.copy_branch_boost,
        branch_selection_strategy = param.branch_selection_strategy,
        branching_ml_weights = param.branching_ml_weights,
    )
end


"""
    function param_levelsetmethod_setup(...)

# Purpose
    Build SCUC level-set method parameter bundle with canonical aliases.

# Arguments
    1. Level-set tuning hyperparameters.

# Returns
    1. Named tuple passed to level-set oracle/solver routines.
"""
function param_levelsetmethod_setup(;
    μ::Float64 = 0.9,
    λ::Float64 = 0.5,
    mu::Union{Nothing, Float64} = nothing,
    lambda::Union{Nothing, Float64} = nothing,
    threshold::Float64 = 1e-4,
    nxt_bound::Float64 = 1e10,
    next_bound::Union{Nothing, Float64} = nothing,
    MaxIter::Int64 = 200,
    max_iter::Union{Nothing, Int64} = nothing,
    verbose::Bool = false
)::NamedTuple
    mu_value = isnothing(mu) ? μ : mu
    lambda_value = isnothing(lambda) ? λ : lambda
    next_bound_value = isnothing(next_bound) ? nxt_bound : next_bound
    max_iter_value = isnothing(max_iter) ? MaxIter : max_iter

    return (
        μ             = mu_value,
        λ             = lambda_value,
        threshold     = threshold,
        nxt_bound     = next_bound_value,
        MaxIter       = max_iter_value,
        verbose       = verbose,
    )
end


"""
    function param_cut_setup(...)

# Purpose
    Build cut-generation hyperparameter bundle with alias handling.

# Arguments
    1. Cut hyperparameters such as delta, ell/ell1/ell2, and epsilon.

# Returns
    1. Named tuple used by CPT/Fenchel/level-set components.
"""
function param_cut_setup(;
    core_point_strategy::String = "Eps", # "Mid", "Eps"
    δ::Float64 = 1e-3,
    delta::Union{Nothing, Float64} = nothing,
    ℓ::Float64 = .0,
    ell::Union{Nothing, Float64} = nothing,
)::NamedTuple
    delta_value = isnothing(delta) ? δ : delta
    ell_value = isnothing(ell) ? ℓ : ell

    return (
        core_point_strategy = core_point_strategy, # "Mid", "Eps"
        δ                   = delta_value,
        ℓ                   = ell_value,
    )
end
