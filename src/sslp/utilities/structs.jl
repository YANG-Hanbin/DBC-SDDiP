#############################################################################################
####################################   Data Structure   #####################################
#############################################################################################
struct CutCoefficient
    v               ::Dict{Int64,Dict{Int64, Float64}} # [i][k] where i is the iteration index, and k is the scenario index
    π               ::Dict{Int64,Dict{Int64, Vector{Float64}}}  # [[1.,2.,3.],[1.,2.,3.]]  -- push!(Π, π) to add a new element
end

struct RandomVariables
    h :: Dict{Tuple{Int64, Int64}, Int64}  # h[i, ω], availability of client i in scenario ω
end

struct StageData
    p  :: Dict{Int64, Float64}                     # p[ω], probability
    c  :: Dict{Int64, Float64}                     # c[j], cost of allocation servers to site j
    d  :: Dict{Tuple{Int64, Int64}, Float64}       # d[i,j], client i chooses d[i,j] unit of resource if served at site j 
    qₒ :: Dict{Int64, Float64}                     # qₒ[j], penalty for shortage at site j
    w  :: Float64                                  # budget upper bound
    v  :: Int64                                    # upper bound of possible number of clients
    r  :: Int64                                    # upper bound on total number of servers
end


struct BackwardModelInfo
    model           :: Model
    x               :: Vector{VariableRef} ## for current state, x is the number of generators
    Lt              :: Vector{VariableRef} ## stage variable, A * Lt is total number of generators built
    Lc              :: Vector{VariableRef} ## local cooy variable
    y               :: Vector{VariableRef} ## amount of electricity
    θ               :: VariableRef
    # demand          :: Vector{Float64}
    slack           :: VariableRef
    # sum_generator   :: Vector{Float64}
end

## data structure for levelset method
mutable struct FunctionHistory
    f_his        :: Dict{Int64, Float64}          ## record f(x_j)     
    G_max_his    :: Dict{Int64, Float64}          ## record max(g[k] for k in 1:m)(x_j)   
end

mutable struct CurrentInfo
    x            :: Vector{Float64}                 ## record x point
    f            :: Float64                         ## record f(x_j)
    G            :: Dict{Int64, Float64} 
    df           :: Vector{Float64}
    dG           :: Dict{Int64, Vector{Float64}}    ## actually is a matrix.  But we use dict to store it
    L_at_x̂       :: Float64                         ## only for solving dual problem
end

struct ModelInfo
    model :: Model
    x     :: Vector{VariableRef}
    y     :: VariableRef
    z     :: VariableRef
end

mutable struct LevelSetMethodParam
    μ             ::Float64                         ## param for adjust α
    λ             ::Union{Float64, Nothing}         ## param for adjust level
    threshold     ::Float64                         ## threshold for Δ
    nxt_bound     ::Float64                         ## lower bound for solving next iteration point π
    MaxIter      ::Int64     
    state         ::Vector{Float64}                 ## first stage solution
    cut_selection  ::Symbol
    core_point    ::Union{Vector{Float64}, Nothing} ## interior point
    f_star_value  ::Union{Float64, Nothing}         ## subproblem optimal value
end