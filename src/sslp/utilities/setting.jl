# This file contains the data generation function for the Stochastic Server Location Sizing (SSLS).
"""
    function dataGeneration(...)

# Purpose
    Generate synthetic/preprocessed experiment data and write it to test-data files.

# Arguments
    1. Problem-size settings and generation controls defined in the signature.

# Returns
    1. Generated dataset artifacts on disk for subsequent experiments.
"""
function dataGeneration(
    J::Int, 
    I::Int, 
    Ω::Int; 
    seed::Int=1234,
    prob::Float64=0.5
)::NamedTuple
    Random.seed!(seed)
    
    # Probability values
    p = Dict(ω => 1.0 / Ω for ω in 1:Ω)

    # Cost values
    c = Dict(j => rand(Uniform(40, 80)) for j in 1:J)

    # Demand values (continuous uniform in [0, 25])
    d = Dict((i, j) => rand(Uniform(0, 25)) for i in 1:I, j in 1:J)

    # Shortage penalty values
    qₒ = Dict(j => 1000.0 for j in 1:J)

    # Parameters
    v = 5  # Upper bound of possible number of clients
    w = (prob * v) * sum(d[i, j] for i in 1:I, j in 1:J) / J  # Budget upper bound
    r = floor(J / 2)  # Upper bound on total number of servers

    # Client availability h[i, ω] ~ Binomial(v, 1/2)
    h = Dict((i, ω) => rand(Binomial(v, prob)) for i in 1:I, ω in 1:Ω)
    
    return (
        stage_data = StageData(p, c, d, qₒ, w, v, r),
        random_variables = RandomVariables(h)
    )
end