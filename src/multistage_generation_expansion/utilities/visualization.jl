using Pkg
Pkg.activate(".")
using JuMP, Gurobi
using Statistics, StatsBase, Random, Dates, Distributions
using Distributed, ParallelDataTransfer
using CSV, DataFrames, Printf
using JLD2, FileIO
using StatsPlots, PlotThemes
using PrettyTables

project_root = @__DIR__;
include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "structs.jl"))



theme(:default)
## ================================================================ gap vs. Time ================================================================ ##
# lens!([0, 10], [15000, 20000], inset = (1, bbox(0.75, 0.05, 0.2, 0.2)), framestyle = :none)
# 处理单个 JLD2 文件
"""
    function process_and_save(...)

# Purpose
    Post-process raw experiment outputs and write aggregated summaries to disk.

# Arguments
    1. Input result file path.

# Returns
    1. Saved processed artifact(s) and summary payload.
"""
function process_and_save(filepath)
    if isfile(filepath)
        # 加载数据
        data = load(filepath)
        
        # 确保 "sddpResults" 存在
        if haskey(data, "sddpResults")
            sddpResults = data["sddpResults"]

            # 确保 "solHistory" 存在
            if haskey(sddpResults, :solHistory)
                df = sddpResults[:solHistory]

                # 确保 df 是 DataFrame
                if df isa DataFrame
                    # 计算递减的 Upper Bound
                    df.UB_new = accumulate(min, df.UB)

                    # 重新计算 gap
                    df.gap_new = string.(((df.UB_new .- df.LB) ./ df.UB_new) .* 100, "%")

                    # 更新 solHistory
                    sddpResults[:solHistory] = df

                    # 重新保存 JLD2 文件
                    save(filepath, Dict("sddpResults" => sddpResults))
                    println("✅ Processed and saved: $filepath")
                else
                    println("⚠️ Warning: solHistory is not a DataFrame in $filepath")
                end
            else
                println("⚠️ Warning: 'solHistory' not found in sddpResults for $filepath")
            end
        else
            println("⚠️ Warning: 'sddpResults' not found in $filepath")
        end
    else
        println("⚠️ Warning: File not found: $filepath")
    end
end

# 遍历所有数据集
for num in [3, 6, 9]
    for T in [6, 8, 10]
        for method in ["CPT-MDC-0", "CPT-MDC-1", "CPT-MDC-2", "CPT-MDC-3", "CPT-MDC-4", "LC", "SBC", "SMC"]
            filepath = joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "$method.jld2")
            process_and_save(filepath)
        end
    end
end
## ------------------------------------------------- TABLE TO LATEX CODE ------------------------------------------------- ##
final_results = DataFrame(
    MDCiter = Int[],
    T = Int[],  
    num = Int[],  
    Best_LB = Float64[],
    Best_UB = Float64[],
    Best_Gap = Float64[],
    Total_Iter = Int[],
    Avg_Iter_Time = Float64[],
    Iter_to_Best = Int[],
    Time_to_Best = Float64[],
)

"""
    function summarize_sddp_results(...)

# Purpose
    Aggregate multiple SDDP result files into summary statistics tables.

# Arguments
    1. Collection of run outputs and summary options.

# Returns
    1. Summary data structure suitable for reporting/plotting.
"""
function summarize_sddp_results(
    solHistory, 
    T, 
    num
)::DataFrame
    df = DataFrame(solHistory)

    if eltype(df[!, :gap_new]) <: AbstractString
        df[!, :gap_new] .= parse.(Float64, replace.(df[!, :gap_new], "%" => ""))
    end

    if eltype(df[!, :LB]) <: AbstractString
        df[!, :LB] .= parse.(Float64, df[!, :LB])
    end

    MDCiter = df[end, :MDCiter]
    best_idx = argmax(df.LB)
    Best_LB = df[best_idx, :LB]
    Best_UB = df[best_idx, :UB_new]
    Best_Gap = df[best_idx, :gap_new]
    Iter_to_Best = df[best_idx, :Iter]
    Time_to_Best = df[best_idx, :Time]

    Avg_Iter_Time = round(df[end,:Time]/100, digits = 1)
    Total_Iter = floor(Int, 3600 / Avg_Iter_Time)
    
    return DataFrame(
        MDCiter = MDCiter,
        T = T, num = num, 
        Best_LB = Best_LB, 
        Best_UB = Best_UB,
        Best_Gap = Best_Gap,
        Total_Iter = Total_Iter,
        Avg_Iter_Time = Avg_Iter_Time,
        Iter_to_Best = Iter_to_Best, 
        Time_to_Best = Time_to_Best,
    )
end



for i in 0:3  
    for T in [6, 8, 10]
        for num in [3, 6, 9]
       
            filename = joinpath("src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "CPT-MDC-i.jld2")
            sddpResults = load(filename)["sddpResults"][:solHistory]
            result = summarize_sddp_results(sddpResults, T, num)
            append!(final_results, result)  # 添加到最终 DataFrame
        end
    end
end

formatter = ft_printf("%0.1f")  # 设定格式化规则，保留 2 位小数

latex_table = pretty_table(
    String, 
    final_results, 
    backend=Val(:latex),
    formatters=formatter
)
println(latex_table);


final_results = DataFrame(
    MDCiter = Int[],
    T = Int[],  
    num = Int[],  
    Best_LB = Float64[],
    Best_UB = Float64[],
    Best_Gap = Float64[],
    Total_Iter = Int[],
    Avg_Iter_Time = Float64[],
    Iter_to_Best = Int[],
    Time_to_Best = Float64[],
)

for T in [6, 8, 10]
    for num in [3, 6, 9]
            sddpResults = load(joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "LC.jld2"))["sddpResults"][:solHistory]
            result = summarize_sddp_results(sddpResults, T, num)
            append!(final_results, result)  # 添加到最终 DataFrame
    end
end

for T in [6, 8, 10]
    for num in [3, 6, 9]
            sddpResults = load(joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "SBC.jld2"))["sddpResults"][:solHistory]
            result = summarize_sddp_results(sddpResults, T, num)
            append!(final_results, result)  # 添加到最终 DataFrame
    end
end

for T in [6, 8, 10]
    for num in [3, 6, 9]
            sddpResults = load(joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "SMC.jld2"))["sddpResults"][:solHistory]
            result = summarize_sddp_results(sddpResults, T, num)
            append!(final_results, result)  # 添加到最终 DataFrame
    end
end

formatter = ft_printf("%0.1f")  # 设定格式化规则，保留 2 位小数

latex_table = pretty_table(
    String, 
    final_results, 
    backend=Val(:latex),
    formatters=formatter
)
println(latex_table);


for T in [6, 8, 10] 
    for num in [3, 6, 9]
        snc = load(joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "CPT-MDC-1.jld2"))["sddpResults"][:solHistory]
        trivial = load(joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "CPT-MDC-1-false-Trivial.jld2"))["sddpResults"][:solHistory]
        regular = load(joinpath(project_root, "src", "multistage_generation_expansion", "logger", "numericalResults", "Periods$T-Real$num", "CPT-MDC-1-false-Regular.jld2"))["sddpResults"][:solHistory]

        UB = snc[100,2] + 1000
        TimeUB = snc[34, 8]
        IterUB = 50


        iter_gap = @df snc plot(
            :Iter, 
            :LB, 
            label="SNC", 
            # title = "Lower Bounds vs. Iteration", 
            xlab = "Iteration", 
            xlim = [0, IterUB], 
            ylim = [2000, UB], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:xcross, 2, 1.), 
            color=:goldenrod,       # 使用蓝色
            yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            legend=:outertop,  # legend 在顶部
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=3,  # legend 列数减少，使其松散
            legend_spacing=6,  # 控制 legend 之间的间距
            linestyle=:solid        # 实线
        )  
        @df trivial plot!(:Iter, :LB, marker=(:vline, 2, 1.), label="Trivial", linestyle=:solid, color=:blue)
        @df regular plot!(:Iter, :LB, marker=(:plus, 2, 1.), label="Regular", linestyle=:solid, color=:lightslategray)

        iter_gap |> save("src/alg/logger/numericalResults/Periods$T-Real$num/normalization-lower_bound_iteration_Period$T-Real$num.pdf")


        time_gap = @df snc plot(
            :Time, 
            :LB, 
            label="SNC", 
            # title = "Lower Bounds vs. Time", 
            xlab = "Time", 
            xlim = [0, TimeUB], 
            ylim = [10000, UB], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:xcross, 2, 1.), 
            color=:goldenrod,       # 使用蓝色
            yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            legend=:outertop,  # legend 在顶部
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=3,  # legend 列数减少，使其松散
            legend_spacing=6,  # 控制 legend 之间的间距
            linestyle=:solid        # 实线
        )  
        @df trivial plot!(:Time, :LB, marker=(:vline, 2, 1.), label="Trivial", linestyle=:solid, color=:blue)
        @df regular plot!(:Time, :LB, marker=(:plus, 2, 1.), label="Regular", linestyle=:solid, color=:lightslategray)

        time_gap |> save("src/alg/logger/numericalResults/Periods$T-Real$num/normalization-lower_bound_Time_Period$T-Real$num.pdf")
    end
end