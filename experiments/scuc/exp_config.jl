Base.@kwdef struct ScucStaticConfig
    case_name::String = "case_RTS_GMLC"
    enforce_binary_copies::Bool = true
    inherit_disjunctive_cuts::Bool = true
    imdc_state_projection::Bool = true
    algorithm_cpt_method::Symbol = :modifiedCPT
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
    copy_split_max_candidates::Int = 16
    copy_split_min_violation::Float64 = 1e-6
    idbc_warm_pass::Bool = false
    branching_ml_weights::NamedTuple = (
        intercept = 0.0,
        deviation = 1.0,
        inverse_deviation = 0.15,
        name_hash = 0.02,
        bias = 1.0,
    )
    branch_variable::Symbol = :ALL
    branch_threshold::Float64 = 1e-6
    normalization::Symbol = :α
    disjunction_iteration_limit::Int = -1
    logger_save::Bool = true
    results_root::Union{Nothing, String} = nothing
    legacy_logger_paths::Bool = false
    experiment_tag::String = "default"
    num_scenarios::Int = 50
    num_backward_scenarios::Int = 1
    num_partition_scenarios::Int = 1
    epsilon::Float64 = 1 / 64
    terminate_time::Float64 = 3600.0
    time_limit::Float64 = 10.0
    terminate_threshold::Float64 = 1e-3
    max_iter::Real = Inf
    lift_iter_threshold::Int = 2
    mip_focus::Int = 3
    numeric_focus::Int = 3
    feasibility_tol::Float64 = 1e-8
    mip_gap::Float64 = 1e-4
end

Base.@kwdef struct ScucSweepConfig
    algorithms::Vector{Symbol} = Symbol[:SDDPL, :SDDP, :SDDiP]
    cuts::Vector{Symbol} = Symbol[:LC, :DBC, :iDBC, :SPC, :SMC, :SBC, :FC]
    periods::Vector{Int} = Int[6, 8, 12]
    realizations::Vector{Int} = Int[5, 10]
end

Base.@kwdef struct ScucCutConfig
    core_point_strategy::String = "Eps"
    delta::Float64 = 1e-4
    ell::Float64 = 0.0
end

Base.@kwdef struct ScucLevelSetConfig
    mu::Float64 = 0.9
    lambda::Float64 = 0.5
    threshold::Float64 = 1e-4
    next_bound::Float64 = 1e10
    max_iter::Int = 200
    verbose::Bool = false
end

Base.@kwdef struct ScucExperimentConfig
    num_workers::Int = 5
    dry_run::Bool = false
    static::ScucStaticConfig = ScucStaticConfig()
    sweep::ScucSweepConfig = ScucSweepConfig()
    cut::ScucCutConfig = ScucCutConfig()
    level_set::ScucLevelSetConfig = ScucLevelSetConfig()
end

"""
    function build_scuc_run_grid(...)

# Purpose
    Expand SCUC sweep settings into concrete `(algorithm, cut, periods, realizations)` cases.

# Arguments
    1. `config` defines the algorithm/cut/time-horizon grid.

# Returns
    1. Vector of SCUC run-case named tuples.
"""
function build_scuc_run_grid(config::ScucExperimentConfig)
    grid = NamedTuple[]
    for algorithm in config.sweep.algorithms
        for cut in config.sweep.cuts
            for realizations in config.sweep.realizations
                for periods in config.sweep.periods
                    push!(grid, (
                        algorithm = algorithm,
                        cut = cut,
                        realizations = realizations,
                        periods = periods,
                    ))
                end
            end
        end
    end
    return grid
end

"""
    function build_scuc_solver_params(...)

# Purpose
    Build SCUC runtime parameter bundles for a single run case.

# Arguments
    1. Experiment config and one run-case tuple.

# Returns
    1. Named tuple with `param`, `param_cut`, and `param_levelset`.
"""
function build_scuc_solver_params(config::ScucExperimentConfig, run_case::NamedTuple)
    static = config.static
    disjunction_iteration_limit = static.disjunction_iteration_limit

    param = param_setup(
        terminate_time = static.terminate_time,
        time_limit = static.time_limit,
        terminate_threshold = static.terminate_threshold,
        max_iter = static.max_iter,
        feasibility_tol = static.feasibility_tol,
        mip_gap = static.mip_gap,
        mip_focus = static.mip_focus,
        numeric_focus = static.numeric_focus,
        epsilon = static.epsilon,
        num_scenarios = static.num_scenarios,
        num_backward_scenarios = static.num_backward_scenarios,
        num_partition_scenarios = static.num_partition_scenarios,
        lift_iter_threshold = static.lift_iter_threshold,
        cut_selection = run_case.cut,
        algorithm = run_case.algorithm,
        cpt_method = static.algorithm_cpt_method,
        branch_threshold = static.branch_threshold,
        branch_variable = static.branch_variable,
        T = run_case.periods,
        num = run_case.realizations,
        enforce_binary_copies = static.enforce_binary_copies,
        inherit_disjunctive_cuts = static.inherit_disjunctive_cuts,
        imdc_state_projection = static.imdc_state_projection,
        case = static.case_name,
        logger_save = static.logger_save,
        results_root = static.results_root,
        legacy_logger_paths = static.legacy_logger_paths,
        experiment_tag = static.experiment_tag,
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
        copy_split_max_candidates = static.copy_split_max_candidates,
        copy_split_min_violation = static.copy_split_min_violation,
        idbc_warm_pass = static.idbc_warm_pass,
        branching_ml_weights = static.branching_ml_weights,
        disjunction_iteration_limit = disjunction_iteration_limit,
        normalization = static.normalization,
    )

    param_cut = param_cut_setup(
        core_point_strategy = config.cut.core_point_strategy,
        delta = config.cut.delta,
        ell = config.cut.ell,
    )

    param_levelset = param_levelsetmethod_setup(
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
    function load_scuc_experiment_data(...)

# Purpose
    Load SCUC test instance data and initial state for a selected run case.

# Arguments
    1. Repository root, experiment config, and run-case descriptors.

# Returns
    1. Named tuple containing index sets, OPF/demand parameters, scenario tree, and initial state.
"""
function load_scuc_experiment_data(
    repo_root::AbstractString,
    config::ScucExperimentConfig,
    run_case::NamedTuple,
)
    case_dir = joinpath(
        repo_root,
        "src",
        "multistage_SCUC",
        "experiment_$(config.static.case_name)",
    )
    instance_dir = joinpath(
        case_dir,
        "stage($(run_case.periods))real($(run_case.realizations))",
    )

    return (
        indexSets = load(joinpath(instance_dir, "indexSets.jld2"))["indexSets"],
        paramOPF = load(joinpath(instance_dir, "paramOPF.jld2"))["paramOPF"],
        paramDemand = load(joinpath(instance_dir, "paramDemand.jld2"))["paramDemand"],
        scenarioTree = load(joinpath(instance_dir, "scenarioTree.jld2"))["scenarioTree"],
        initialStateInfo = load(joinpath(case_dir, "initialStateInfo.jld2"))["initialStateInfo"],
    )
end
