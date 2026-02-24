using Pkg

"""
    function find_repo_root(...)

# Purpose
    Resolve repository root by walking parent directories until project markers are found.

# Arguments
    1. Optional start directory; defaults to current working directory.

# Returns
    1. Absolute path to repository root.
"""
function find_repo_root(start::AbstractString = pwd())
    dir = abspath(start)
    while true
        has_project = isfile(joinpath(dir, "Project.toml"))
        has_entry = isfile(joinpath(dir, "experiments", "generation_expansion", "run_experiment.jl"))
        if has_project && has_entry
            return dir
        end
        parent = dirname(dir)
        parent == dir && error("Could not locate repository root from $(start).")
        dir = parent
    end
end

repo_root = find_repo_root()
cd(repo_root)
Pkg.activate(repo_root)

project_name = "generation_expansion"
run_script = joinpath("experiments", project_name, "run_experiment.jl")
smoke_config = joinpath("experiments", project_name, "configs", "smoke.jl")
default_config = joinpath("experiments", project_name, "configs", "default.jl")
custom_config = joinpath("experiments", project_name, "configs", "notebook_ge_custom.jl")

src = joinpath(repo_root, default_config)
dst = joinpath(repo_root, custom_config)
write(dst, read(src, String))
println("Copied default config to: ", dst)

num_workers = 5
dry_run = false
cut = :DBC
periods = [6, 8]
realizations = [6]
sample_count = 100
num_backward_samples = 1
terminate_time = 600.0
time_limit = 20.0
results_root = nothing
experiment_tag = "ge_nb"
legacy_logger_paths = false

as_symbol_vector(values::Vector{Symbol}) = "Symbol[" * join(repr.(values), ", ") * "]"
as_int_vector(values::Vector{Int}) = "Int[" * join(values, ", ") * "]"

config_text = """
EXPERIMENT_CONFIG = GeExperimentConfig(
    num_workers = $(num_workers),
    dry_run = $(dry_run),
    static = GeStaticConfig(
        sample_count = $(sample_count),
        num_backward_samples = $(num_backward_samples),
        terminate_time = $(terminate_time),
        time_limit = $(time_limit),
        results_root = $(repr(results_root)),
        experiment_tag = $(repr(experiment_tag)),
        legacy_logger_paths = $(legacy_logger_paths),
    ),
    sweep = GeSweepConfig(
        cuts = $(as_symbol_vector([cut])),
        realizations = $(as_int_vector(realizations)),
        periods = $(as_int_vector(periods)),
    ),
)
"""

dst = joinpath(repo_root, custom_config)
write(dst, config_text)

cd(repo_root) do
    run(`julia --project=. $run_script $custom_config`)
end