Base.@kwdef struct GeStaticConfig
    enhancement::Bool = false
    enforce_binary_copies::Bool = true
    sample_count::Int = 500
    num_backward_samples::Int = -1
    count_benders_cut::Bool = false
    opt::Float64 = 0.0
    logger_save::Bool = true
    feasibility_tol::Float64 = 1e-9
    mip_focus::Int = 0
    numeric_focus::Int = 3
    mip_gap::Float64 = 1e-6
    disjunction_iteration_limit::Int = -1
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
    idbc_warm_pass::Bool = false
    branching_ml_weights::NamedTuple = (
        intercept = 0.0,
        deviation = 1.0,
        inverse_deviation = 0.15,
        name_hash = 0.02,
        bias = 1.0,
    )
    algorithm::Symbol = :modifiedCPT
    normalization::Symbol = :SNC
    inherit_disjunctive_cuts::Bool = true
    improvement::Bool = false
    terminate_time::Float64 = 600.0
    terminate_threshold::Float64 = 1e-3
    max_iter::Real = Inf
    time_limit::Float64 = 20.0
    theta_lower::Float64 = 0.0
    epsilon::Float64 = 0.125
    results_root::Union{Nothing, String} = nothing
    legacy_logger_paths::Bool = false
    experiment_tag::String = "default"
end

Base.@kwdef struct GeSweepConfig
    cuts::Vector{Symbol} = Symbol[:SMC, :LC, :DBC, :iDBC, :SPC, :SBC, :FC]
    realizations::Vector{Int} = Int[6, 9]
    periods::Vector{Int} = Int[6, 8]
end

Base.@kwdef struct GeCutConfig
    delta::Float64 = 1e-2
    ell1::Float64 = 0.0
    ell2::Float64 = 0.0
    epsilon::Float64 = 0.1
end

Base.@kwdef struct GeLevelSetConfig
    mu::Float64 = 0.9
    lambda::Float64 = 0.5
    threshold::Float64 = 1e-4
    next_bound::Float64 = 1e10
    max_iter::Int = 200
    verbose::Bool = false
end

Base.@kwdef struct GeExperimentConfig
    num_workers::Int = 5
    dry_run::Bool = false
    static::GeStaticConfig = GeStaticConfig()
    sweep::GeSweepConfig = GeSweepConfig()
    cut::GeCutConfig = GeCutConfig()
    level_set::GeLevelSetConfig = GeLevelSetConfig()
end

"""
    function build_ge_run_grid(...)

# Purpose
    Expand generation-expansion sweep settings into concrete run cases.

# Arguments
    1. `config` holds static and sweep sections used to enumerate `(cut, periods, realizations)`.

# Returns
    1. Vector of run-case named tuples.
"""
function build_ge_run_grid(config::GeExperimentConfig)
    runs = NamedTuple[]
    for cut_selection in config.sweep.cuts
        for realizations in config.sweep.realizations
            for periods in config.sweep.periods
                push!(runs, (
                    cut_selection = cut_selection,
                    realizations = realizations,
                    periods = periods,
                ))
            end
        end
    end
    return runs
end

"""
    function build_ge_solver_params(...)

# Purpose
    Convert one generation-expansion run case into runtime parameter bundles consumed by solver code.

# Arguments
    1. Experiment config, selected run case, and binary-variable metadata from data files.

# Returns
    1. Named tuple with `param`, `param_cut`, and `param_levelset`.
"""
function build_ge_solver_params(
    config::GeExperimentConfig,
    run_case::NamedTuple,
    binary_info,
)
    static = config.static
    disjunction_iteration_limit = static.disjunction_iteration_limit

    param = param_setup(
        verbose = false,
        mip_gap = static.mip_gap,
        time_limit = static.time_limit,
        mip_focus = static.mip_focus,
        feasibility_tol = static.feasibility_tol,
        numeric_focus = static.numeric_focus,
        terminate_time = static.terminate_time,
        terminate_threshold = static.terminate_threshold,
        max_iter = static.max_iter,
        theta_lower = static.theta_lower,
        sample_count = static.sample_count,
        num_backward_samples = static.num_backward_samples,
        disjunction_iteration_limit = disjunction_iteration_limit,
        enhancement = static.enhancement,
        epsilon = static.epsilon,
        enforce_binary_copies = static.enforce_binary_copies,
        inherit_disjunctive_cuts = static.inherit_disjunctive_cuts,
        binary_info = binary_info,
        opt = static.opt,
        count_benders_cut = static.count_benders_cut,
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
        idbc_warm_pass = static.idbc_warm_pass,
        branching_ml_weights = static.branching_ml_weights,
        algorithm = static.algorithm,
        normalization = static.normalization,
        improvement = static.improvement,
        T = run_case.periods,
        num = run_case.realizations,
        logger_save = static.logger_save,
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
    function load_ge_experiment_data(...)

# Purpose
    Load pre-generated generation-expansion dataset artifacts for one run case.

# Arguments
    1. Repository root path and run-case descriptors (periods/realizations).

# Returns
    1. Named tuple with stage data, scenario realizations, binary metadata, and probabilities.
"""
function load_ge_experiment_data(
    repo_root::AbstractString,
    run_case::NamedTuple,
)
    data_dir = joinpath(
        repo_root,
        "src",
        "multistage_generation_expansion",
        "testData",
        "stage($(run_case.periods))real($(run_case.realizations))",
    )

    return (
        stageDataList = load(joinpath(data_dir, "stageDataList.jld2"))["stageDataList"],
        Ω = load(joinpath(data_dir, "Ω.jld2"))["Ω"],
        binaryInfo = load(joinpath(data_dir, "binaryInfo.jld2"))["binaryInfo"],
        probList = load(joinpath(data_dir, "probList.jld2"))["probList"],
    )
end
