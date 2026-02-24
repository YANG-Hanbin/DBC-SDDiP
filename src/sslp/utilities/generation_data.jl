using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(REPO_ROOT)
using JuMP, Gurobi, ParallelDataTransfer
using Distributions, Statistics, StatsBase, Distributed, Random
using Test, Dates, Printf
using CSV, DataFrames
using JLD2, FileIO
using SparseArrays #, LinearAlgebra

# const GRB_ENV = Gurobi.Env()

include(joinpath(REPO_ROOT, "src", "sslp", "utilities", "structs.jl"))
include(joinpath(REPO_ROOT, "src", "sslp", "utilities", "setting.jl"))
include(joinpath(REPO_ROOT, "src", "sslp", "utilities", "utils.jl"))

## --------------------------------------------------------------------------------- ##
## --------------------------------- Save the data --------------------------------- ##
## --------------------------------------------------------------------------------- ##
for Ω in [100, 200]
    for (J, I) in [(5, 15), (10, 20), (15, 25)]
        (stageData, randomVariables) = dataGeneration(J, I, Ω)
        save(joinpath(REPO_ROOT, "src", "sslp", "testData", "J$J-I$I-Ω$Ω", "stageData.jld2"), "stageData", stageData);
        save(joinpath(REPO_ROOT, "src", "sslp", "testData", "J$J-I$I-Ω$Ω", "randomVariables.jld2"), "randomVariables", randomVariables);
    end
end

