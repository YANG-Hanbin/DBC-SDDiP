using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(REPO_ROOT)
using JuMP, Gurobi, ParallelDataTransfer
using Distributions, Statistics, StatsBase, Distributed, Random
using Test, Dates, Printf
using CSV, DataFrames
using JLD2, FileIO
using SparseArrays #, LinearAlgebra

const GRB_ENV = Gurobi.Env()

project_root = REPO_ROOT

include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "structs.jl"))
include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "ext_form.jl"))
# include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "generation_data.jl"))
include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "setting.jl"))
include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "utils.jl"))

include(joinpath(project_root, "src", "multistage_generation_expansion", "algorithm.jl"))
include(joinpath(project_root, "src", "multistage_generation_expansion", "backward_pass.jl"))
include(joinpath(project_root, "src", "multistage_generation_expansion", "forward_pass.jl"))
include(joinpath(project_root, "src", "multistage_generation_expansion", "level_set_method.jl"))
