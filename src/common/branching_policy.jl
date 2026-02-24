using Random
using Statistics

"""
Select a branching variable from a fractional-deviation dictionary.

Supported strategies:
- `:MFV`: most fractional variable (max deviation)
- `:Random`: uniform random over candidates
- `:First`: first inserted candidate
- `:LFV`: least fractional variable (min deviation)
- `:ML`: lightweight feature-ranking model with safe fallback

Returns `nothing` if `deviation` is empty.
"""
const DEFAULT_BRANCHING_ML_WEIGHTS = (
    intercept = 0.0,
    deviation = 1.0,
    inverse_deviation = 0.15,
    name_hash = 0.02,
    bias = 1.0,
)

@inline function _max_deviation(dev::AbstractDict)::Float64
    return isempty(dev) ? 0.0 : maximum(Float64(v) for v in values(dev))
end

@inline function _mean_deviation(dev::AbstractDict)::Float64
    return isempty(dev) ? 0.0 : mean(Float64(v) for v in values(dev))
end

"""
Decide whether copy/auxiliary variables should be used as branching candidates.

Supported policies:
- `:always`: use copy-variable candidates whenever available.
- `:never`: never use copy-variable candidates.
- `:fallback_only`: use copy-variable candidates only when primary candidates are empty.
- `:dominant`: use copy-variable candidates only when their max deviation dominates primary.
- `:adaptive`: use copy-variable candidates when they are either dominant in max deviation or competitive in mean deviation.
"""
function should_branch_on_copy_variables(
    primary_deviation::AbstractDict,
    copy_deviation::AbstractDict;
    policy::Symbol = :adaptive,
    min_copy_deviation::Float64 = 1e-6,
    dominance_ratio::Float64 = 1.0,
    mean_dominance_ratio::Float64 = 0.9,
)::Bool
    if isempty(copy_deviation)
        return false
    end

    max_primary = _max_deviation(primary_deviation)
    max_copy = _max_deviation(copy_deviation)
    mean_primary = _mean_deviation(primary_deviation)
    mean_copy = _mean_deviation(copy_deviation)
    copy_is_active = max_copy >= min_copy_deviation

    if policy == :always
        return true
    elseif policy == :never
        return false
    elseif policy == :fallback_only
        return isempty(primary_deviation) && copy_is_active
    elseif policy == :dominant
        return copy_is_active && (isempty(primary_deviation) || max_copy >= dominance_ratio * max_primary)
    elseif policy == :adaptive
        return copy_is_active && (
            isempty(primary_deviation) ||
            max_copy >= dominance_ratio * max_primary ||
            mean_copy >= mean_dominance_ratio * mean_primary
        )
    elseif policy == :blend
        return copy_is_active
    else
        @warn "Unknown copy branching policy '$policy'. Falling back to ':adaptive'."
        return should_branch_on_copy_variables(
            primary_deviation,
            copy_deviation;
            policy = :adaptive,
            min_copy_deviation = min_copy_deviation,
            dominance_ratio = dominance_ratio,
            mean_dominance_ratio = mean_dominance_ratio,
        )
    end
end

"""
Select the deviation pool used by branching-variable selection.

Returns a NamedTuple with:
- `deviation`: selected candidate deviation dictionary.
- `source`: one of `:primary`, `:copy`, or `:mixed`.
- `max_primary`, `max_copy`, `mean_primary`, `mean_copy`.
"""
function select_branch_candidate_pool(
    primary_deviation::AbstractDict,
    copy_deviation::AbstractDict;
    copy_policy::Symbol = :adaptive,
    min_copy_deviation::Float64 = 1e-6,
    dominance_ratio::Float64 = 1.0,
    mean_dominance_ratio::Float64 = 0.9,
    copy_boost::Float64 = 1.0,
)::NamedTuple
    primary_pool = Dict{Any, Float64}(k => Float64(v) for (k, v) in primary_deviation)
    copy_pool = Dict{Any, Float64}(k => Float64(v) for (k, v) in copy_deviation)

    use_copy = should_branch_on_copy_variables(
        primary_pool,
        copy_pool;
        policy = copy_policy,
        min_copy_deviation = min_copy_deviation,
        dominance_ratio = dominance_ratio,
        mean_dominance_ratio = mean_dominance_ratio,
    )

    selected_pool = primary_pool
    source = :primary

    if use_copy
        if copy_policy == :blend
            selected_pool = copy(primary_pool)
            for (var, dev) in copy_pool
                selected_pool[var] = copy_boost * dev
            end
            source = isempty(primary_pool) ? :copy : :mixed
        else
            selected_pool = copy_pool
            source = :copy
        end
    end

    if isempty(selected_pool)
        if !isempty(primary_pool)
            selected_pool = primary_pool
            source = :primary
        elseif !isempty(copy_pool)
            selected_pool = copy_pool
            source = :copy
        end
    end

    return (
        deviation = selected_pool,
        source = source,
        max_primary = _max_deviation(primary_pool),
        max_copy = _max_deviation(copy_pool),
        mean_primary = _mean_deviation(primary_pool),
        mean_copy = _mean_deviation(copy_pool),
    )
end

@inline function _get_ml_weight(weights::NamedTuple, key::Symbol, default_value::Float64)::Float64
    return hasproperty(weights, key) ? Float64(getproperty(weights, key)) : default_value
end

@inline function _stable_name_hash(var)::Float64
    # Stable hash feature for deterministic tie-breaking across runs.
    code_sum = sum(codeunits(string(var)))
    return Float64(code_sum % 997) / 997.0
end

@inline function _ml_score(
    var,
    dev::Float64;
    weights::NamedTuple,
    ml_bias::AbstractDict,
)::Float64
    score = 0.0
    score += _get_ml_weight(weights, :intercept, 0.0)
    score += _get_ml_weight(weights, :deviation, 1.0) * dev
    score += _get_ml_weight(weights, :inverse_deviation, 0.0) * (1.0 - dev)
    score += _get_ml_weight(weights, :name_hash, 0.0) * _stable_name_hash(var)
    score += _get_ml_weight(weights, :bias, 0.0) * Float64(get(ml_bias, var, 0.0))
    return score
end

"""
    function select_branch_variable(...)

# Purpose
    Select one branching candidate from fractional deviations using the configured policy (MFV/LFV/Random/First/ML).

# Arguments
    1. `deviation` is the candidate-to-deviation map; keyword options control strategy, fallback, and optional ML scoring.

# Returns
    1. Returns the selected branching variable key, or `nothing` when no candidate exists.
"""
function select_branch_variable(
    deviation::AbstractDict;
    strategy::Symbol = :MFV,
    fallback::Symbol = :MFV,
    rng::AbstractRNG = Random.default_rng(),
    ml_weights::Union{Nothing, NamedTuple} = nothing,
    ml_bias::AbstractDict = Dict{Any, Float64}(),
)
    if isempty(deviation)
        return nothing
    end

    if strategy == :MFV
        max_dev = maximum(values(deviation))
        candidates = [var for (var, dev) in deviation if dev == max_dev]
        return first(candidates)
    elseif strategy == :LFV
        min_dev = minimum(values(deviation))
        candidates = [var for (var, dev) in deviation if dev == min_dev]
        return first(candidates)
    elseif strategy == :Random
        return rand(rng, collect(keys(deviation)))
    elseif strategy == :First
        return first(keys(deviation))
    elseif strategy == :ML
        weights = isnothing(ml_weights) ? DEFAULT_BRANCHING_ML_WEIGHTS : ml_weights
        best_var = nothing
        best_score = -Inf
        best_dev = -Inf
        for (var, dev_raw) in deviation
            dev = Float64(dev_raw)
            score = _ml_score(var, dev; weights = weights, ml_bias = ml_bias)
            if score > best_score || (score == best_score && dev > best_dev)
                best_var = var
                best_score = score
                best_dev = dev
            end
        end
        if isnothing(best_var) || !isfinite(best_score)
            @warn "ML branching produced no valid candidate. Falling back to '$fallback'."
            return select_branch_variable(
                deviation;
                strategy = fallback,
                fallback = :MFV,
                rng = rng,
                ml_weights = ml_weights,
                ml_bias = ml_bias,
            )
        end
        return best_var
    else
        @warn "Unknown branching strategy '$strategy'. Falling back to '$fallback'."
        return select_branch_variable(
            deviation;
            strategy = fallback,
            fallback = :MFV,
            rng = rng,
            ml_weights = ml_weights,
            ml_bias = ml_bias,
        )
    end
end