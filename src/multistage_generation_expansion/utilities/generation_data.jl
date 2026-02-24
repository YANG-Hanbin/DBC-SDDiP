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

include(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "structs.jl"))
include(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "setting.jl"))
include(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "utilities", "utils.jl"))
project_root = @__DIR__


"""
    The data is provided by Jin2011.
"""
# Constants
r = 0.08  # Annualized interest rate
T = 10  # Planning horizon (10 years)

# Generator capacity matrix (Table 6)
N = [
    1130.0 0 0 0 0 0;
    0 390 0 0 0 0; 
    0 0 380 0 0 0; 
    0 0 0 1180 0 0;
    0 0 0 0 175 0; 
    0 0 0 0 0 560
]

# Cost per MW to build generator type g (Table 4)
c = [1.446, 0.795, 0.575, 1.613, 1.650, 1.671] .* 1e6

# Installed capacity (Table 6)
mg = [1200, 400, 400, 1200, 500, 600]

# Fuel price, heat rate, efficiency, O&M cost (Table 5)
fuel_price = [3.37, 9.11, 9.11, 0.93e-3, 0, 3.37]
heat_rate = [8844, 7196, 10842, 10400, 0, 8613]
eff = [0.4, 0.56, 0.4, 0.45, 1, 0.48]
om_cost = [4.7, 2.11, 3.66, 0.51, 5.00, 2.98]

# Initial number of generators (s0), assuming all start at 0
s₀ = [0, 0, 0, 0, 0, 0]   

# Construction limits for each type of generator (ū from Table 7)
ū = [4., 10, 10, 1, 45, 4]

# Penalty for unserved load (pu)
penalty = 100000.  # $/MWh

# Average hourly demand for 2008 (d0 / 8760 hours in a year)
initial_demand = 5.685e8
total_hours = 8760.

s₀ = [1, 0, 0, 0, 0, 0];

## --------------------------------------------------------------------------------- ##
## --------------------------------- Save the data --------------------------------- ##
## --------------------------------------------------------------------------------- ##
for T in [10, 15]
    for numRealization in [5, 10]
        (probList, stageDataList, Ω, binaryInfo) = dataGeneration(
            T = T , 
            initial_demand = initial_demand/100, 
            total_hours = total_hours/100,
            seed = 12345, 
            num_Ω = numRealization, 
            penalty = penalty, 
            r = r,
            N = N,
            c = c,
            mg = mg,
            fuel_price = fuel_price,
            heat_rate = heat_rate,
            eff = eff,
            om_cost = om_cost,
            s₀ = s₀,
            ū = ū
        );
        # scenario_sequence = Dict{Int64, Dict{Int64, Any}}();  ## the first index is for scenario index, the second one is for stage
        # pathList = Vector{Int64}();
        # push!(pathList, 1);
        # P = 1.0;

        # recursion_scenario_tree(
        #     pathList, 
        #     P, 
        #     scenario_sequence, 
        #     2, 
        #     T = T, 
        #     prob = probList
        # );
        # scenario_tree = scenario_sequence;

        save(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "testData", "stage($T)real($numRealization)", "stageDataList.jld2"), "stageDataList", stageDataList);
        save(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "testData", "stage($T)real($numRealization)", "Ω.jld2"), "Ω", Ω);
        save(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "testData", "stage($T)real($numRealization)", "binaryInfo.jld2"), "binaryInfo", binaryInfo);
        # save("src/multistage_generation_expansion/testData/stage($T)real($numRealization)/scenario_sequence.jld2", "scenario_sequence", scenario_sequence);
        save(joinpath(REPO_ROOT, "src", "multistage_generation_expansion", "testData", "stage($T)real($numRealization)", "probList.jld2"), "probList", probList);
    end
end
