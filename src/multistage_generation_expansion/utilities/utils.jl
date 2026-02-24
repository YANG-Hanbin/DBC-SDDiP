## sampling function 
"""
    function DrawSamples(...)

# Purpose
    Sample one scenario path by drawing one realization per stage.

# Arguments
    1. Stage-wise scenario dictionary.

# Returns
    1. A sampled scenario sequence.
"""
function DrawSamples(scenario_sequence::Dict{Int64, Dict{Int64, Any}})
    # draw f, A, B, C, b from Ωₜ according to distribution P
    P = Vector{Float64}()
    for key in keys(scenario_sequence)
        push!(P, scenario_sequence[key][2])
    end
    items = [i for i in keys(scenario_sequence)]
    weights = Weights(P)
    j = sample(items, weights)
    return j
end


## form scenarios
"""
    function SampleScenarios(...)

# Purpose
    Generate a batch of sampled scenario paths for Monte Carlo forward simulation.

# Arguments
    1. Scenario dictionary, probabilities, and requested sample count.

# Returns
    1. Dictionary of sampled paths indexed by sample id.
"""
function SampleScenarios(
    scenario_sequence::Dict{Int64, Dict{Int64, Any}}; 
    M::Int64 = 30
)
    ## a dict to store realization for each stage t in scenario k
    scenarios = Dict{Int64, Int64}()
    for k in 1:M
          scenarios[k] = DrawSamples(scenario_sequence)
    end
    return scenarios
end

## rounding data
"""
    function round!(...)

# Purpose
    Numerically stabilize floating-point values by rounding tiny noise.

# Arguments
    1. Single floating-point value to normalize.

# Returns
    1. Rounded/stabilized floating-point value.
"""
function round!(a::Float64)               ## a = 1.3333e10
    b = floor(log10(a))                   ## b = 10
    c = round(a/10^b,digits = 2)          ## c = 1.33
    d = c * 10^b                          ## d = 1.33e10
    return [b, c, d]
end

"""
    function SampleScenarios(...)

# Purpose
    Generate a batch of sampled scenario paths for Monte Carlo forward simulation.

# Arguments
    1. Scenario dictionary, probabilities, and requested sample count.

# Returns
    1. Dictionary of sampled paths indexed by sample id.
"""
function SampleScenarios(
    Ω::Dict{Int64, Dict{Int64, RandomVariables}}, 
    prob_list::Dict{Int64, Vector{Float64}}; 
    M::Int64 = 30
)
    ## a dict to store realization for each stage t in scenario k
    T = length(keys(Ω));
    scenarios = Dict{Int64, Vector{Int64}}()
    for k in 1:M
        scenario_path = [1];
        for t in 2:T
            items = [i for i in keys(Ω[t])]
            weights = Weights(prob_list[t])
            j = sample(items, weights)
            push!(scenario_path, j)
        end
        scenarios[k] = scenario_path
    end
    return scenarios
end

"""
    function print_iteration_info(...)

# Purpose
    Print one iteration row of solver progress (LB/UB/gap/time/cut stats).

# Arguments
    1. Iteration metrics already computed by the outer loop.

# Returns
    1. `nothing`; side effect is formatted console output.
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
    println("Iter |        LB        |        UB        |       Gap      |      i-time     |    #D.     |     T-Time")
    println("----------------------------------------------------------------------------------------------------------")
    return 
end

const GE_PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

@inline function _sanitize_token(value)::String
    return replace(string(value), r"[^A-Za-z0-9_.=-]+" => "_")
end

@inline function _kv_token(key::AbstractString, value)::String
    return string(key, "=", _sanitize_token(value))
end

"""
    function _build_ge_results_path(...)

# Purpose
    Build canonical generation-expansion result path from effective runtime parameters.

# Arguments
    1. Generation-expansion runtime parameter bundle.

# Returns
    1. Absolute output file path.
"""
function _build_ge_results_path(param::NamedTuple)::String
    results_root = hasproperty(param, :results_root) ? param.results_root : joinpath(GE_PROJECT_ROOT, "results")
    experiment_tag = hasproperty(param, :experiment_tag) ? param.experiment_tag : "default"
    cut_selection = param.cut_selection
    disjunction_iter = cut_selection == :SPC ? 1 : (param.disjunction_iteration_limit >= 0 ? param.disjunction_iteration_limit : "Inf")
    normalization = cut_selection ∈ [:DBC, :iDBC, :SPC, :SC, :FC] ? param.normalization : "na"
    inherit_disjunctive_cuts = param.inherit_disjunctive_cuts
    enforce_binary_copies = param.enforce_binary_copies

    dataset_id = join(
        [
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
        ],
        "__",
    )
    return joinpath(results_root, "generation_expansion", dataset_id, "$run_id.jld2")
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
    sddp_results::Dict;
    logger_save::Bool = true
)::Nothing
    if !logger_save
        return nothing
    end

    cut_selection = param.cut_selection
    num, T = param.num, param.T
    algorithm = param.algorithm
    D = cut_selection == :SPC ? 1 : (param.disjunction_iteration_limit >= 0 ? param.disjunction_iteration_limit : Inf)
    normalization = param.normalization
    inheritance = param.inherit_disjunctive_cuts
    legacy_logger_paths = hasproperty(param, :legacy_logger_paths) ? param.legacy_logger_paths : false

    new_save_path = _build_ge_results_path(param)
    run_meta = Dict(
        :project => "generation_expansion",
        :algorithm => algorithm,
        :cut_selection => cut_selection,
        :dataset => Dict(:T => T, :num => num),
        :normalization => normalization,
        :disjunction_iter => D,
        :inherit_disjunctive_cuts => inheritance,
        :legacy_logger_paths => legacy_logger_paths,
    )
    mkpath(dirname(new_save_path))
    save(new_save_path, "sddp_results", sddp_results, "run_meta", run_meta)

    if !legacy_logger_paths
        return nothing
    end

    base_path = joinpath(GE_PROJECT_ROOT, "src", "multistage_generation_expansion", "new_logger")

    if cut_selection ∈ [:DBC, :iDBC, :SPC, :SC, :FC]
        if inheritance == true
            prefix = "with_cut_inheritance/Periods$T-Real$num"
            filename = "$algorithm-$cut_selection-$D-$normalization"
        else
            prefix = "without_cut_inheritance/Periods$T-Real$num"
            filename = "$algorithm-$cut_selection-$D-$normalization"
        end
    elseif cut_selection ∈ [:BC, :SBC, :LC, :SMC]
        prefix = "without_cut_inheritance/Periods$T-Real$num"
        filename = "$cut_selection"
    else
        return nothing
    end

    save_path = joinpath(base_path, prefix, "$filename.jld2")
    mkpath(dirname(save_path))
    save(save_path, "sddp_results", sddp_results)
    return nothing
end

"""
    function param_setup(...)

# Purpose
    Assemble the canonical generation-expansion runtime parameter bundle.
"""
function param_setup(;
    verbose::Bool = false,
    mip_gap::Float64 = 1e-4,
    time_limit::Float64 = 10.0,
    mip_focus::Int64 = 1,
    feasibility_tol::Float64 = 1e-6,
    numeric_focus::Int64 = 3,
    terminate_time::Float64 = 3600.0,
    terminate_threshold::Float64 = 1e-3,
    max_iter::Real = Inf,
    opt::Float64 = 0.0,
    theta_lower::Float64 = -1e4,
    sample_count::Int64 = 5,
    num_backward_samples::Int64 = 1,
    disjunction_iteration_limit::Int64 = 4,
    lagrangian_cut::Bool = true,
    enhancement::Bool = false,
    binary_info::BinaryInfo = binaryInfo,
    count_benders_cut::Bool = true,
    epsilon::Float64 = 0.125,
    enforce_binary_copies::Bool = true,
    inherit_disjunctive_cuts::Bool = true,
    cut_selection::Symbol = :PLC,
    normalization::Symbol = :SNC,
    algorithm::Symbol = :CPT,
    branch_selection_strategy::Symbol = :Random,
    copy_branch_policy::Symbol = :adaptive,
    copy_branch_min_deviation::Float64 = 1e-6,
    copy_branch_dominance_ratio::Float64 = 1.0,
    copy_branch_mean_ratio::Float64 = 0.9,
    copy_branch_boost::Float64 = 1.0,
    enable_copy_branching::Bool = true,
    enable_surrogate_copy_split::Bool = true,
    copy_split_strategy::Symbol = :surrogate_delta,
    copy_split_delta_tol::Float64 = 1e-6,
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
    idbc_warm_pass::Bool = false,
    improvement::Bool = true,
    T::Int64 = 12,
    num::Int64 = 10,
    case::String = "case30pwl",
    logger_save::Bool = true,
)::NamedTuple
    results_root_value = isnothing(results_root) ? joinpath(GE_PROJECT_ROOT, "results") : results_root
    effective_cut = cut_selection == :iDBC && !enable_copy_branching ? :DBC : cut_selection

    return (
        verbose = verbose,
        mip_gap = mip_gap,
        time_limit = time_limit,
        mip_focus = mip_focus,
        feasibility_tol = feasibility_tol,
        numeric_focus = numeric_focus,
        terminate_time = terminate_time,
        terminate_threshold = terminate_threshold,
        max_iter = max_iter,
        opt = opt,
        theta_lower = theta_lower,
        sample_count = sample_count,
        num_backward_samples = num_backward_samples,
        disjunction_iteration_limit = disjunction_iteration_limit,
        lagrangian_cut = lagrangian_cut,
        enhancement = enhancement,
        binary_info = binary_info,
        count_benders_cut = count_benders_cut,
        epsilon = epsilon,
        enforce_binary_copies = enforce_binary_copies,
        inherit_disjunctive_cuts = inherit_disjunctive_cuts,
        cut_selection = effective_cut,
        normalization = normalization,
        algorithm = algorithm,
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
        results_root = results_root_value,
        legacy_logger_paths = legacy_logger_paths,
        experiment_tag = experiment_tag,
        branching_ml_weights = branching_ml_weights,
        idbc_warm_pass = idbc_warm_pass,
        improvement = improvement,
        T = T,
        num = num,
        case = case,
        logger_save = logger_save,
    )
end

"""
    resolve_ge_runtime_params(param::NamedTuple)::NamedTuple

# Purpose
    Return the canonical runtime-view used by generation-expansion components.
"""
function resolve_ge_runtime_params(param::NamedTuple)::NamedTuple
    return (
        sample_count = param.sample_count,
        num_backward_samples = param.num_backward_samples,
        count_benders_cut = param.count_benders_cut,
        max_iter = param.max_iter,
        time_limit = param.time_limit,
        enforce_binary_copies = param.enforce_binary_copies,
        disjunction_iteration_limit = param.disjunction_iteration_limit,
        cut_selection = param.cut_selection,
        inherit_disjunctive_cuts = param.inherit_disjunctive_cuts,
        idbc_warm_pass = param.idbc_warm_pass,
        enable_copy_branching = param.enable_copy_branching,
        enable_surrogate_copy_split = param.enable_surrogate_copy_split,
        copy_split_strategy = param.copy_split_strategy,
        copy_split_delta_tol = param.copy_split_delta_tol,
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
    function param_LevelMethod_setup(...)

# Purpose
    Build level-set method parameter bundle (SSLP/GEP variants) with canonical aliases.

# Arguments
    1. Level-set tuning hyperparameters.

# Returns
    1. Named tuple passed to level-set routines.
"""
function param_LevelMethod_setup(;
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
    ℓ1::Float64 = .0,
    ell1::Union{Nothing, Float64} = nothing,
    ℓ2::Float64 = .0,
    ell2::Union{Nothing, Float64} = nothing,
    ϵ::Float64 = 1e-4
    ,
    epsilon_cut::Union{Nothing, Float64} = nothing
)::NamedTuple
    delta_value = isnothing(delta) ? δ : delta
    ell1_value = isnothing(ell1) ? ℓ1 : ell1
    ell2_value = isnothing(ell2) ? ℓ2 : ell2
    epsilon_cut_value = isnothing(epsilon_cut) ? ϵ : epsilon_cut

    return (
        core_point_strategy = core_point_strategy, # "Mid", "Eps"
        δ                   = delta_value,
        ℓ1                  = ell1_value,
        ℓ2                  = ell2_value,
        ϵ                   = epsilon_cut_value
    )
end
