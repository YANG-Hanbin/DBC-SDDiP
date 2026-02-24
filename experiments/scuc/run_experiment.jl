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
function _with_dry_run(config::ScucExperimentConfig)
    return ScucExperimentConfig(
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
    error("Config file must define `EXPERIMENT_CONFIG::ScucExperimentConfig`.")
end

config = EXPERIMENT_CONFIG
if !(config isa ScucExperimentConfig)
    error("`EXPERIMENT_CONFIG` must be `ScucExperimentConfig`, got $(typeof(config)).")
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
    using PowerModels
    using Statistics
    using StatsBase
    using Random
    using Dates
    using Distributions
    using Distributed
    using ParallelDataTransfer
    using CSV
    using DataFrames
    using Printf
    using JLD2
    using FileIO

    const GRB_ENV = Gurobi.Env()
end

source_files = [
    joinpath(REPO_ROOT, "src", "common", "branching_policy.jl"),
    joinpath(REPO_ROOT, "src", "common", "runtime_context.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utilities", "structs.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utilities", "auxiliary.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utilities", "level_set_method.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utilities", "fenchel_cut.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utilities", "cutting_plane_tree_alg.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utilities", "cut_variants.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "utils.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "forward_pass.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "backward_pass.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "partition_tree.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "runtime_context.jl"),
    joinpath(REPO_ROOT, "src", "multistage_SCUC", "sddp.jl"),
]

for file in source_files
    @everywhere include($file)
end

run_grid = build_scuc_run_grid(config)
println("Loaded config: $(config_path)")
println("Total run cases: $(length(run_grid))")
println("Dry run mode: $(config.dry_run)")

for run_case in run_grid
    data_bundle = load_scuc_experiment_data(REPO_ROOT, config, run_case)
    params_bundle = build_scuc_solver_params(config, run_case)
    effective_cut = params_bundle.param.cut_selection
    println(
        "[SCUC] algorithm=$(run_case.algorithm) cut=$(effective_cut) ",
        "periods=$(run_case.periods) realizations=$(run_case.realizations)",
    )

    if config.dry_run
        continue
    end

    index_sets_data = data_bundle.indexSets
    param_opf_data = data_bundle.paramOPF
    param_demand_data = data_bundle.paramDemand
    scenario_tree_data = data_bundle.scenarioTree
    initial_state_info_data = data_bundle.initialStateInfo

    run_param = params_bundle.param
    run_param_cut = params_bundle.param_cut
    run_param_levelsetmethod = params_bundle.param_levelset

    stochastic_dual_dynamic_programming_algorithm(
        scenario_tree_data,
        index_sets_data,
        param_demand_data,
        param_opf_data;
        initialStateInfo = initial_state_info_data,
        param_cut = run_param_cut,
        param_levelsetmethod = run_param_levelsetmethod,
        param = run_param,
    )

    @everywhere GC.gc()
end
# julia --project=. experiments/scuc/run_experiment.jl experiments/scuc/configs/default_01.jl
