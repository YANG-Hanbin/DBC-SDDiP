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
function _with_dry_run(config::GeExperimentConfig)
    return GeExperimentConfig(
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
    error("Config file must define `EXPERIMENT_CONFIG::GeExperimentConfig`.")
end

config = EXPERIMENT_CONFIG
if !(config isa GeExperimentConfig)
    error("`EXPERIMENT_CONFIG` must be `GeExperimentConfig`, got $(typeof(config)).")
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
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "structs.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "ext_form.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "setting.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "utils.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "cutting_plane_tree_alg.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "fenchel_cut.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "backward_pass.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "forward_pass.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "level_set_method.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "runtime_context.jl"),
    joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "algorithm.jl"),
]

for file in source_files
    @everywhere include($file)
end

run_grid = build_ge_run_grid(config)
println("Loaded config: $(config_path)")
println("Total run cases: $(length(run_grid))")
println("Dry run mode: $(config.dry_run)")

for run_case in run_grid
    data_bundle = load_ge_experiment_data(REPO_ROOT, run_case)
    params_bundle = build_ge_solver_params(config, run_case, data_bundle.binaryInfo)
    effective_cut = params_bundle.param.cut_selection
    println(
        "[GEN-EXP] cut=$(effective_cut) periods=$(run_case.periods) ",
        "realizations=$(run_case.realizations)",
    )

    if config.dry_run
        continue
    end

    stageDataList = data_bundle.stageDataList
    scenario_realizations = data_bundle.Ω
    binaryInfo = data_bundle.binaryInfo
    probList = data_bundle.probList

    run_param = params_bundle.param
    run_param_cut = params_bundle.param_cut
    run_param_levelmethod = params_bundle.param_levelset

    stochastic_dual_dynamic_programming_algorithm(
        scenario_realizations,
        probList,
        stageDataList;
        param = run_param,
        param_cut = run_param_cut,
        param_LevelMethod = run_param_levelmethod,
    )

    @everywhere GC.gc()
end
