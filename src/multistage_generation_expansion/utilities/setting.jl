## binarize stage variable, x = A * L, where L ∈ {0, 1}ⁿ
"""
    function integer_var_binary(...)

# Purpose
    Encode integer variables into binary expansion structures for multistage generation-expansion formulations.

# Arguments
    1. Integer bounds and encoding metadata from the call signature.

# Returns
    1. Binary-encoding metadata used by model builders.
"""
function integer_var_binary(
    ū::Vector{Float64}
)
    row_num = size(ū)[1];
    var_num = floor.(Int, log.(2,ū)) .+ 1; 
    col_num = sum(Int, var_num);
    A = zeros(Int64, row_num, col_num)
    for i in 1:row_num
        l = sum(var_num[l] for l in 1:i)
        for j in (l+1-var_num[i]):(l+var_num[i]-var_num[i])
            A[i,j] = 2^(j-(l+1-var_num[i]))
        end
    end
    return BinaryInfo(
        A, 
        col_num, 
        row_num
    )
end

################### non-anticipativity for multistage problem #######################
"""
    function recursion_scenario_tree(...)

# Purpose
    Recursively generate scenario-tree nodes and transition probabilities.

# Arguments
    1. Scenario generation settings, stage depth, and random-variable definitions.

# Returns
    1. Scenario-tree structure used by forward/backward simulation.
"""
function recursion_scenario_tree(
    path_list::Vector{Int64}, 
    P::Float64, 
    scenario_sequence::Dict{Int64, Dict{Int64, Any}}, 
    t::Int64; 
    Ω::Dict{Int64,Dict{Int64,RandomVariables}} = Ω, 
    prob::Dict{Int64,Vector{Float64}} = prob, 
    T::Int64 = 2
)

    if t ≤ T
        for ω_key in keys(Ω[t])

            path_list_copy = copy(path_list)
            P_copy = copy(P)

            push!(path_list_copy, ω_key)
            P_copy = P_copy * prob[t][ω_key]

            recursion_scenario_tree(path_list_copy, P_copy, scenario_sequence, t+1, Ω = Ω, prob = prob, T = T)
        end
    else
        if haskey(scenario_sequence, 1)
            scenario_sequence[maximum(keys(scenario_sequence))+1] = Dict(1 => path_list, 2 => P)
        else
            scenario_sequence[1] = Dict(1 => path_list, 2 => P)
        end
        return scenario_sequence
    end

end

## setup coefficients
"""
    function dataGeneration(...)

# Purpose
    Generate synthetic/preprocessed experiment data and write it to test-data files.

# Arguments
    1. Problem-size settings and generation controls defined in the signature.

# Returns
    1. Generated dataset artifacts on disk for subsequent experiments.
"""
function dataGeneration(;   
    T::Int64 = 2, num_Ω::Int64 = num_Ω, seed::Int64 = 1234,
    r::Float64 = r, ## the annualized interest rate
    N::Matrix{Float64} = N, ## Generator rating
    ū::Vector{Float64} = ū, ## maximum number of each type of generators
    c::Vector{Float64} = c, # c_g from table 4, cost/MW to build a generator of type g
    mg::Vector{Int64} = mg,
    fuel_price::Vector{Float64} = fuel_price,
    heat_rate::Vector{Int64} = heat_rate,
    eff::Vector{Float64} = eff,
    om_cost::Vector{Float64} = om_cost, 
    s₀::Vector{Int64} = s₀,
    penalty::Float64 = penalty, 
    total_hours::Float64 = 8760., ## total hours in a year
    initial_demand::Float64 = initial_demand
)::NamedTuple

    binary_info = integer_var_binary(ū)

    # Compute c1 (investment cost per MW)
    # 1e5 is used to scale the cost to a reasonable range
    c1 = [[c[i] * mg[i] / (1 + r)^j for i in 1:6] for j in 1:T]./1e5                                                            

    # Compute c2 (generation cost per MWh)
    c2 = [[(fuel_price[i] * heat_rate[i] * 1e-3 / eff[i]) * (1.02)^j + om_cost[i] * (1.03)^j for i in 1:6] for j in 1:T]./1e5

    stage_data_list = Dict{Int64,StageData}()
    for t in 1:T 
        stage_data_list[t] = StageData(c1[t], c2[t], ū, total_hours, N, s₀, penalty/1e5)
    end

    ##########################################################################################
    ############################  To generate random variable  ###############################
    ##########################################################################################
    N_rv = Vector{Int64}()  # the number of realization of each stage
    N_rv = [num_Ω for t in 1:T] 
    # N_rv = round.(rand(T) * 10) 

    Random.seed!(seed)
    Ω = Dict{Int64,Dict{Int64,RandomVariables}}()   # each stage t, its node are included in Ω[t]
    for t in 1:T 
        Ω[t] = Dict{Int64,RandomVariables}()
        for i in 1:N_rv[t]
            if t == 1
                Ω[t][i]= RandomVariables([initial_demand])
            else
                # Ω[t][i]= RandomVariables( rand(Uniform(1.05, 1.2))*Ω[t-1][i].d )
                Ω[t][i]= RandomVariables( 1.05^t * rand(Uniform(1.0, 1.2))*Ω[1][1].d )
            end
        end
    end

    prob_list = Dict{Int64,Vector{Float64}}()  # P(node in t-1 --> node in t ) = prob[t]
    for t in 1:T 
        # random_vector = round.(rand(N_rv[t]),digits = 2)
        # prob_list[t] = round.(random_vector/sum(random_vector),digits = 2)
        prob_list[t] = [1/N_rv[t] for i in 1:N_rv[t]]
    end

    return (
        prob_list = prob_list, 
        stage_data_list = stage_data_list, 
        Ω = Ω, 
        binary_info = binary_info
    )
end