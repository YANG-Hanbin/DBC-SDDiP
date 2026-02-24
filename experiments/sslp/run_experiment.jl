using Pkg
using Distributed

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO_ROOT)

using JLD2
using FileIO

include(joinpath(@__DIR__, "exp_config.jl"))

"""
    function _resolve_config_path(...)

# Purpose
    Resolve which experiment config file should be loaded from CLI arguments.

# Arguments
    1. `args` is the raw argument vector passed to the run script.

# Returns
    1. Absolute config-file path and remaining CLI flags (via script-specific convention).
"""
function _resolve_config_path(args::Vector{String})
    filtered = filter(arg -> arg != "--dry-run", args)
    if isempty(filtered)
        return joinpath(@__DIR__, "configs", "default.jl")
    end
    return abspath(filtered[1])
end

"""
    function _with_dry_run(...)

# Purpose
    Return a config copy with `dry_run=true` when command-line dry-run flag is present.

# Arguments
    1. Experiment config struct instance.

# Returns
    1. Config struct with effective dry-run behavior.
"""
function _with_dry_run(config::SslpExperimentConfig)
    return SslpExperimentConfig(
        num_workers = config.num_workers,
        dry_run = true,
        static = config.static,
        sweep = config.sweep,
        cut = config.cut,
        level_set = config.level_set,
    )
end

config_path = _resolve_config_path(copy(ARGS))
include(config_path)

if !isdefined(Main, :EXPERIMENT_CONFIG)
    error("Config file must define `EXPERIMENT_CONFIG::SslpExperimentConfig`.")
end

config = EXPERIMENT_CONFIG
if !(config isa SslpExperimentConfig)
    error("`EXPERIMENT_CONFIG` must be `SslpExperimentConfig`, got $(typeof(config)).")
end
if any(arg -> arg == "--dry-run", ARGS)
    config = _with_dry_run(config)
end

if nworkers() < config.num_workers
    addprocs(config.num_workers - nworkers())
end

@everywhere begin
    using JuMP
    using Gurobi
    using ParallelDataTransfer
    using Distributions
    using Statistics
    using StatsBase
    using Distributed
    using Random
    using Test
    using Dates
    using Printf
    using CSV
    using DataFrames
    using JLD2
    using FileIO
    using SparseArrays

    const GRB_ENV = Gurobi.Env()
end

source_files = [
    joinpath(REPO_ROOT, "src", "common", "branching_policy.jl"),
    joinpath(REPO_ROOT, "src", "common", "runtime_context.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "utilities", "structs.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "utilities", "cutting_plane_tree_alg.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "utilities", "fenchel_cut.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "utilities", "setting.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "utilities", "utils.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "backward_pass.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "forward_pass.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "level_set_method.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "runtime_context.jl"),
    joinpath(REPO_ROOT, "src", "sslp", "algorithm.jl"),
]

for file in source_files
    @everywhere include($file)
end

run_grid = build_sslp_run_grid(config)
println("Loaded config: $(config_path)")
println("Total run cases: $(length(run_grid))")
println("Dry run mode: $(config.dry_run)")

for run_case in run_grid
    data_bundle = load_sslp_experiment_data(REPO_ROOT, run_case)
    params_bundle = build_sslp_solver_params(config, run_case)
    effective_cut = params_bundle.param.cut_selection
    println(
        "[SSLP] cut=$(effective_cut) J=$(run_case.J) I=$(run_case.I) ",
        "Ω=$(run_case.omega) disj_iter=$(run_case.disjunction_iteration_limit) norm=$(run_case.normalization)",
    )

    if config.dry_run
        continue
    end

    stage_data_bundle = data_bundle.stageData
    random_variables_bundle = data_bundle.randomVariables
    run_param = params_bundle.param
    run_param_cut = params_bundle.param_cut
    run_param_levelmethod = params_bundle.param_levelset

    stochastic_dual_dynamic_programming_algorithm(
        stage_data_bundle,
        random_variables_bundle;
        param = run_param,
        param_cut = run_param_cut,
        param_LevelMethod = run_param_levelmethod,
    )

    @everywhere GC.gc()
end
