""" 
    function setup_core_point(
        state_info::StateInfo;
        index_sets::IndexSets = index_sets,
        param_opf::ParamOPF = param_opf, 
        param_cut::NamedTuple   
    )::StateInfo
"""
function setup_core_point(
    state_info::StateInfo;
    index_sets::IndexSets = index_sets,
    param_opf::ParamOPF = param_opf, 
    param::NamedTuple,
    param_cut::NamedTuple   
)::StateInfo
    if param_cut.core_point_strategy == "Mid"
        BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
            g => 0.5 for g in index_sets.G)
        );
        ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
            g => (param_opf.smin[g] + param_opf.smax[g])/2 for g in index_sets.G)
        );
        if state_info.ContAugState == nothing
            ContAugState = nothing
        else
            ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        k => .5 for k in keys(state_info.ContAugState[:s][g])
                    ) for g in index_sets.G
                )
            );
        end

        if state_info.ContStateBin == nothing
            ContStateBin = nothing
        else
            ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => .5 for i in 1:param.kappa[g]
                    ) for g in index_sets.G
                )
            );
        end
    elseif param_cut.core_point_strategy == "Eps"
        BinVar = Dict{Any, Dict{Any, Any}}(:y => Dict{Any, Any}(
            g => state_info.BinVar[:y][g] * param_cut.ℓ + (1 - param_cut.ℓ)/2 for g in index_sets.G)
        );
        ContVar = Dict{Any, Dict{Any, Any}}(:s => Dict{Any, Any}(
            g => state_info.ContVar[:s][g] * param_cut.ℓ + (1 - param_cut.ℓ) * (param_opf.smin[g] + param_opf.smax[g])/2 for g in index_sets.G)
        );
        if state_info.ContAugState == nothing
            ContAugState = nothing
        else
            ContAugState = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        k => state_info.ContAugState[:s][g][k] * param_cut.ℓ + (1 - param_cut.ℓ)/2 for k in keys(state_info.ContAugState[:s][g])
                    ) for g in index_sets.G
                )
            );
        end

        if state_info.ContStateBin == nothing
            ContStateBin = nothing
        else
            ContStateBin = Dict{Any, Dict{Any, Dict{Any, Any}}}(
                :s => Dict{Any, Dict{Any, Any}}(
                    g => Dict{Any, Any}(
                        i => state_info.ContStateBin[:s][g][i] * param_cut.ℓ + (1 - param_cut.ℓ)/2 for i in 1:param.kappa[g]
                    ) for g in index_sets.G
                )
            );
        end
    end

    return StateInfo(
        BinVar, 
        nothing, 
        ContVar, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        nothing, 
        ContAugState,
        nothing,
        ContStateBin
    );
end

""" 
    function binarize_continuous_variable(
        state::Float64, 
        smax::Float64, 
        param::NamedTuple  
    )::StateInfo
"""
function binarize_continuous_variable(
    state::Float64, 
    smax::Float64, 
    param::NamedTuple  
)::Vector
    if smax == 0
        return [0]  
    end
    num_binary_vars = floor(Int, log2(smax / param.epsilon)) + 1  # the number of binary variables needed
    max_integer = floor(Int, state / param.epsilon)  # 将连续变量映射到整数
    binary_representation = digits(max_integer, base=2, pad=num_binary_vars)  # 获取二进制表示（低位在前）
    return binary_representation
end

"""
    sanitize_imdc_copy_state(
        state_info::StateInfo;
        enforce_binary::Bool = false,
        zero_copy_when_off::Bool = true,
        index_sets::IndexSets = index_sets,
        param::NamedTuple,
    )::StateInfo

Create a sanitized copy of `state_info` for iDBC backward passes so that
copy-state nonanticipativity remains compatible with UC physics:
- all copied binary-like states are clamped to `[0, 1]`;
- optionally projected to `{0, 1}` when `enforce_binary=true`;
- if `y_copy[g] == 0`, then all copied binarization bits `λ_copy[g, i]` are forced to 0.
"""
function sanitize_imdc_copy_state(
    state_info::StateInfo;
    enforce_binary::Bool = false,
    zero_copy_when_off::Bool = true,
    index_sets::IndexSets = index_sets,
    param::NamedTuple,
)::StateInfo
    sanitized_state = deepcopy(state_info)

    if sanitized_state.BinVar !== nothing && haskey(sanitized_state.BinVar, :y)
        for g in index_sets.G
            raw_y = sanitized_state.BinVar[:y][g]
            clipped_y = clamp(Float64(raw_y), 0.0, 1.0)
            sanitized_state.BinVar[:y][g] = enforce_binary ? (clipped_y >= 0.5 ? 1.0 : 0.0) : clipped_y
        end
    end

    if sanitized_state.ContStateBin !== nothing && haskey(sanitized_state.ContStateBin, :s)
        for g in index_sets.G
            y_state = (sanitized_state.BinVar === nothing || !haskey(sanitized_state.BinVar, :y)) ?
                      1.0 : sanitized_state.BinVar[:y][g]
            for i in 1:param.kappa[g]
                raw_lambda = sanitized_state.ContStateBin[:s][g][i]
                clipped_lambda = clamp(Float64(raw_lambda), 0.0, 1.0)
                projected_lambda = enforce_binary ? (clipped_lambda >= 0.5 ? 1.0 : 0.0) : clipped_lambda
                if zero_copy_when_off && y_state <= 0.5
                    projected_lambda = 0.0
                end
                sanitized_state.ContStateBin[:s][g][i] = projected_lambda
            end
        end
    end

    return sanitized_state
end
