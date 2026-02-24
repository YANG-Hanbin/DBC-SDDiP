Base.@kwdef struct SslpStaticConfig
    opt::Float64 = 0.0
    feasibility_tol::Float64 = 1e-9
    mip_focus::Int = 0
    numeric_focus::Int = 3
    mip_gap::Float64 = 1e-4
    branch_selection_strategy::Symbol = :MFV
    copy_branch_policy::Symbol = :adaptive
    copy_branch_min_deviation::Float64 = 1e-6
    copy_branch_dominance_ratio::Float64 = 1.0
    copy_branch_mean_ratio::Float64 = 0.9
    copy_branch_boost::Float64 = 1.0
    enable_copy_branching::Bool = true
    enable_surrogate_copy_split::Bool = true
    copy_split_strategy::Symbol = :surrogate_delta
    copy_split_delta_tol::Float64 = 1e-6
    branching_ml_weights::NamedTuple = (
        intercept = 0.0,
        deviation = 1.0,
        inverse_deviation = 0.15,
        name_hash = 0.02,
        bias = 1.0,
    )
    normalization::Symbol = :SNC
    algorithm::Symbol = :CPT
    idbc_warm_pass::Bool = false
    inherit_disjunctive_cuts::Bool = false
    enforce_binary_copies::Bool = true
    logger_save::Bool = true
    terminate_time::Float64 = 3600.0
    time_limit::Float64 = 20.0
    terminate_threshold::Float64 = 1e-3
    max_iter::Real = Inf
    theta_lower::Float64 = -4000.0
    epsilon::Float64 = 0.125
    results_root::Union{Nothing, String} = nothing
    legacy_logger_paths::Bool = false
    experiment_tag::String = "default"
end

Base.@kwdef struct SslpSweepConfig
    omegas::Vector{Int} = Int[100, 200]
    problem_sizes::Vector{Tuple{Int, Int}} = Tuple{Int, Int}[(5, 15), (10, 20), (10, 35), (15, 25), (15, 30)]
    base_cuts::Vector{Symbol} = Symbol[:BC, :LC, :SMC, :SBC]
    mdc_iters::Vector{Int} = Int[5, 10]
    mdc_cuts::Vector{Symbol} = Symbol[:DBC, :iDBC, :SPC, :FC]
    normalization_candidates::Vector{Symbol} = Symbol[:Regular, :α]
    normalization_cut::Symbol = :iDBC
    normalization_mdc_iter::Int = 10
end

Base.@kwdef struct SslpCutConfig
    delta::Float64 = 1e-4
    ell1::Float64 = 0.0
    ell2::Float64 = 0.9
    epsilon::Float64 = 0.1
end

Base.@kwdef struct SslpLevelSetConfig
    mu::Float64 = 0.9
    lambda::Float64 = 0.5
    threshold::Float64 = 1e-4
    next_bound::Float64 = 1e10
    max_iter::Real = 200
    verbose::Bool = false
end

Base.@kwdef struct SslpExperimentConfig
    num_workers::Int = 5
    dry_run::Bool = false
    static::SslpStaticConfig = SslpStaticConfig()
    sweep::SslpSweepConfig = SslpSweepConfig()
    cut::SslpCutConfig = SslpCutConfig()
    level_set::SslpLevelSetConfig = SslpLevelSetConfig()
end

"""
    function build_sslp_run_grid(...)

# Purpose
    Expand SSLP sweep configuration into concrete benchmark cases.

# Arguments
    1. `config` defines cut family, problem sizes, omega values, and disjunction schedules.

# Returns
    1. Vector of SSLP run-case named tuples.
"""
function build_sslp_run_grid(config::SslpExperimentConfig)
    runs = NamedTuple[]
    static = config.static
    sweep = config.sweep

    for cut_selection in sweep.base_cuts
        for omega in sweep.omegas
            for (J, I) in sweep.problem_sizes
                push!(runs, (
                    cut_selection = cut_selection,
                    omega = omega,
                    J = J,
                    I = I,
                    disjunction_iteration_limit = 0,
                    normalization = static.normalization,
                ))
            end
        end
    end

    for (iter_idx, disjunction_iteration_limit) in enumerate(sweep.mdc_iters)
        for cut_selection in sweep.mdc_cuts
            normalized_cut = cut_selection
            if normalized_cut == :SPC && iter_idx > 1
                continue
            end
            effective_mdc_iter = normalized_cut == :SPC ? 1 : disjunction_iteration_limit
            for omega in sweep.omegas
                for (J, I) in sweep.problem_sizes
                    push!(runs, (
                        cut_selection = normalized_cut,
                        omega = omega,
                        J = J,
                        I = I,
                        disjunction_iteration_limit = effective_mdc_iter,
                        normalization = static.normalization,
                    ))
                end
            end
        end
    end

    for normalization in sweep.normalization_candidates
        for omega in sweep.omegas
            for (J, I) in sweep.problem_sizes
                push!(runs, (
                    cut_selection = sweep.normalization_cut,
                    omega = omega,
                    J = J,
                    I = I,
                    disjunction_iteration_limit = sweep.normalization_mdc_iter,
                    normalization = normalization,
                ))
            end
        end
    end

    return runs
end

"""
    function build_sslp_solver_params(...)

# Purpose
    Translate one SSLP run case into runtime parameter bundles.

# Arguments
    1. Experiment config and selected run-case tuple.

# Returns
    1. Named tuple with `param`, `param_cut`, and `param_levelset`.
"""
function build_sslp_solver_params(config::SslpExperimentConfig, run_case::NamedTuple)
    static = config.static

    param = param_setup(
        verbose = false,
        mip_gap = static.mip_gap,
        time_limit = static.time_limit,
        mip_focus = static.mip_focus,
        feasibility_tol = static.feasibility_tol,
        numeric_focus = static.numeric_focus,
        terminate_time = static.terminate_time,
        theta_lower = static.theta_lower,
        terminate_threshold = static.terminate_threshold,
        max_iter = static.max_iter,
        disjunction_iteration_limit = run_case.disjunction_iteration_limit,
        epsilon = static.epsilon,
        enforce_binary_copies = static.enforce_binary_copies,
        inherit_disjunctive_cuts = static.inherit_disjunctive_cuts,
        opt = static.opt,
        cut_selection = run_case.cut_selection,
        branch_selection_strategy = static.branch_selection_strategy,
        copy_branch_policy = static.copy_branch_policy,
        copy_branch_min_deviation = static.copy_branch_min_deviation,
        copy_branch_dominance_ratio = static.copy_branch_dominance_ratio,
        copy_branch_mean_ratio = static.copy_branch_mean_ratio,
        copy_branch_boost = static.copy_branch_boost,
        enable_copy_branching = static.enable_copy_branching,
        enable_surrogate_copy_split = static.enable_surrogate_copy_split,
        copy_split_strategy = static.copy_split_strategy,
        copy_split_delta_tol = static.copy_split_delta_tol,
        branching_ml_weights = static.branching_ml_weights,
        algorithm = static.algorithm,
        normalization = run_case.normalization,
        idbc_warm_pass = static.idbc_warm_pass,
        logger_save = static.logger_save,
        J = run_case.J,
        I = run_case.I,
        omega_count = run_case.omega,
        results_root = static.results_root,
        legacy_logger_paths = static.legacy_logger_paths,
        experiment_tag = static.experiment_tag,
    )

    param_cut = param_cut_setup(
        delta = config.cut.delta,
        ell1 = config.cut.ell1,
        ell2 = config.cut.ell2,
        epsilon_cut = config.cut.epsilon,
    )

    param_levelset = param_LevelMethod_setup(
        mu = config.level_set.mu,
        lambda = config.level_set.lambda,
        threshold = config.level_set.threshold,
        next_bound = config.level_set.next_bound,
        max_iter = config.level_set.max_iter,
        verbose = config.level_set.verbose,
    )

    return (
        param = param,
        param_cut = param_cut,
        param_levelset = param_levelset,
    )
end

"""
    function load_sslp_experiment_data(...)

# Purpose
    Load SSLP stage/random-variable data for one `(J, I, Omega)` case.

# Arguments
    1. Repository root and run-case descriptors.

# Returns
    1. Named tuple with stage data and scenario random variables.
"""
function load_sslp_experiment_data(
    repo_root::AbstractString,
    run_case::NamedTuple,
)
    data_dir = joinpath(
        repo_root,
        "src",
        "sslp",
        "testData",
        "J$(run_case.J)-I$(run_case.I)-Ω$(run_case.omega)",
    )
    return (
        stageData = load(joinpath(data_dir, "stageData.jld2"))["stageData"],
        randomVariables = load(joinpath(data_dir, "randomVariables.jld2"))["randomVariables"],
    )
end
