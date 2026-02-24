using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(REPO_ROOT)
using JuMP, Gurobi
using Statistics, StatsBase, Random, Dates, Distributions
using Distributed, ParallelDataTransfer
using CSV, DataFrames, Printf
using JLD2, FileIO
using StatsPlots, PlotThemes
using PrettyTables
using VegaLite, VegaDatasets;

project_root = REPO_ROOT
include(joinpath(project_root, "src", "multistage_generation_expansion", "utilities", "structs.jl"))

theme(:default)
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------- TO GENERATE TABLES ---------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##

## ----------------------------------------------------------------------------- MDC iter ---------------------------------------------------------------------------------------------- ##
T = 6; # 10, 15
num = 6; # 5, 10
cut = :BC
normalization = :SNC
result_df = DataFrame(
    D=[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    # gap_under_1_time=Union{Missing, Float64}[],
    # gap_under_1_iter=Union{Missing, Int}[]
);

for i in [1, 2, 3]
    for T in [6, 8, 10]
        for num in [6, 9]
            try
                file_path = "src/multistage_generation_expansion/logger/numericalResults/Periods$T-Real$num/CPT-MDC-$i-false-$normalization.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 将 gap 列（字符串）转换为 Float64 含义的百分数
                gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                # 找到 gap 第一次小于 1.0 的位置
                below1_idx = findfirst(<(1), gap_vals)

                # 初始化默认值
                gap_under_1_iter = missing
                gap_under_1_time = missing

                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.Iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 添加到DataFrame
                push!(result_df, (
                    i, T, num, best_LB, final_gap, total_iter, 
                    avg_iter_time, 
                    best_LB_time, best_LB_iter,
                    # gap_under_1_time, gap_under_1_iter
                    )
                );
            catch e
                @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
            end
        end
    end
end

# 定义格式化函数，保留一位小数
column_formatter = function(x, i, j)
    if x isa Float64
        return @sprintf("%.1f", x)  # 保留一位小数
    elseif x isa Tuple  # 处理 iter_range 之类的元组数据
        return "$(x[1])--$(x[2])"
    else
        return string(x)  # 其他数据类型转换为字符串
    end
end

# 生成 LaTeX 表格
latex_table = pretty_table(
    String, 
    result_df, 
    backend=Val(:latex),
    formatters=(column_formatter,)
)

# 输出 LaTeX 代码
println(latex_table)

## ---------------------------------------------------------------------------- Cut Families --------------------------------------------------------------------------------------- ##
T = 10; # 10, 15
num = 5; # 5, 10
cut = :BC
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    # gap_under_1_time=Union{Missing, Float64}[],
    # gap_under_1_iter=Union{Missing, Int}[]
);

for cut in [:BC, :LC, :SBC, :SMC]
    for T in [6, 8, 10]
        for num in [6, 9]
            try
                file_path = "src/multistage_generation_expansion/logger/without_cut_inheritance/Periods$T-Real$num/$cut.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 将 gap 列（字符串）转换为 Float64 含义的百分数
                gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                # 找到 gap 第一次小于 1.0 的位置
                below1_idx = findfirst(<(1), gap_vals)

                # 初始化默认值
                gap_under_1_iter = missing
                gap_under_1_time = missing

                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.Iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end

                # 添加到DataFrame
                push!(result_df, (
                    cut, T, num, best_LB, final_gap, total_iter, 
                    avg_iter_time, 
                    best_LB_time, best_LB_iter,
                    # gap_under_1_time, gap_under_1_iter
                    )
                );
            catch e
                @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
            end
        end
    end
end

# 定义格式化函数，保留一位小数
column_formatter = function(x, i, j)
    if x isa Float64
        return @sprintf("%.1f", x)  # 保留一位小数
    elseif x isa Tuple  # 处理 iter_range 之类的元组数据
        return "$(x[1])--$(x[2])"
    else
        return string(x)  # 其他数据类型转换为字符串
    end
end

# 生成 LaTeX 表格
latex_table = pretty_table(
    String, 
    result_df, 
    backend=Val(:latex),
    formatters=(column_formatter,)
)

# 输出 LaTeX 代码
println(latex_table)

## ------------------------- Normalization ---------------------------------------------------------------------------------- ##
T = 6; # 10, 15
num = 6; # 5, 10
cut = :BC
normalization = :SNC
result_df = DataFrame(
    D=[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    avg_MDC=String[],
    # gap_under_1_time=Union{Missing, Float64}[],
    # gap_under_1_iter=Union{Missing, Int}[]
);

for normalization in [:SNC, :Regular, :α]
    for T in [6, 8, 10]
        for num in [6, 9]
            try
                file_path = "src/multistage_generation_expansion/logger/with_cut_inheritance/Periods$T-Real$num/CPT-MDC-Inf-$normalization.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 将 gap 列（字符串）转换为 Float64 含义的百分数
                gap_vals = parse.(Float64, replace.(solHistory.gap, "%" => ""))

                # 找到 gap 第一次小于 1.0 的位置
                below1_idx = findfirst(<(1), gap_vals)

                # 初始化默认值
                gap_under_1_iter = missing
                gap_under_1_time = missing

                if below1_idx !== nothing
                    gap_under_1_iter = solHistory.Iter[below1_idx]
                    gap_under_1_time = solHistory.Time[below1_idx]
                end
                MDCiter = mean(solHistory.MDCiter)
                std_MDCiter = std(solHistory.MDCiter)
                avg_MDC = @sprintf "%.1f (%.1f)" MDCiter std_MDCiter  # 格式化字符串

                # 添加到DataFrame
                push!(result_df, (
                    normalization, T, num, best_LB, final_gap, total_iter, 
                    avg_iter_time, 
                    best_LB_time, best_LB_iter,
                    avg_MDC
                    # gap_under_1_time, gap_under_1_iter
                    )
                );
            catch e
                @warn "Error processing file: $file_path" exception=(e, catch_backtrace())
            end
        end
    end
end

# 定义格式化函数，保留一位小数
column_formatter = function(x, i, j)
    if x isa Float64
        return @sprintf("%.1f", x)  # 保留一位小数
    elseif x isa Tuple  # 处理 iter_range 之类的元组数据
        return "$(x[1])--$(x[2])"
    else
        return string(x)  # 其他数据类型转换为字符串
    end
end

# 生成 LaTeX 表格
latex_table = pretty_table(
    String, 
    result_df, 
    backend=Val(:latex),
    formatters=(column_formatter,)
)

# 输出 LaTeX 代码
println(latex_table)

# --------------------------------------------------- Normalization --------------------------------------------------- #
for T in [6, 8, 10] 
    for num in [6, 9]
        cutInheritance = :without_cut_inheritance # :with_cut_inheritance, :without_cut_inheritance
        snc = load("src/multistage_generation_expansion/logger/$cutInheritance/Periods$T-Real$num/CPT-MDC-Inf-SNC.jld2")["sddpResults"][:solHistory]
        regular = load("src/multistage_generation_expansion/logger/$cutInheritance/Periods$T-Real$num/CPT-MDC-Inf-regular.jld2")["sddpResults"][:solHistory]
        alpha = load("src/multistage_generation_expansion/logger/$cutInheritance/Periods$T-Real$num/CPT-MDC-Inf-α.jld2")["sddpResults"][:solHistory]

        # snc = filter(row -> row.Iter % 2 == 1, snc)
        # regular = filter(row -> row.Iter % 2 == 1, regular)
        # alpha = filter(row -> row.Iter % 2 == 1, alpha)

        time_gap = @df snc plot(
            :Time, 
            :LB, 
            label="SNC", 
            xlab = "Time (sec.)", 
            # xlim = [0, TimeUB], 
            ylim = [10000, 20000], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:none, 2, 1.), 
            color=:purple,       
            yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            # legend=:outertop,  # legend 在顶部
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            # legend_column=3,  # legend 列数减少，使其松散
            # legend_spacing=6,  # 控制 legend 之间的间距
            linestyle=:dashdotdot,        # 实线
            linewidth=1.5     # 线条加粗
        )  
        @df regular plot!(:Time, :LB, marker=(:vline, 2, 1.), label="Regular", linestyle=:dot, color="#ED4043", linewidth=1.5)
        @df alpha plot!(:Time, :LB, marker=(:plus, 2, 1.), label="π", linestyle=:dash, color="#47AF79", linewidth=1.5)

        time_gap |> save("src/multistage_generation_expansion/logger/$cutInheritance/Periods$T-Real$num/normalization.pdf")
    end
end

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- the same instance with Benchmarks -------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 
for num in [6, 9]
    for T in [6, 8]
        sddpResultsSC = load("src/multistage_generation_expansion/logger/with_cut_inheritance/Periods$T-Real$num/CPT-MDC-1-SNC.jld2")["sddpResults"][:solHistory]
        sddpResultsFC = load("src/multistage_generation_expansion/logger/with_cut_inheritance/Periods$T-Real$num/FC.jld2")["sddpResults"][:solHistory]
        sddpResultsBC = load("src/multistage_generation_expansion/logger/without_cut_inheritance/Periods$T-Real$num/BC.jld2")["sddpResults"][:solHistory]
        sddpResultsLC = load("src/multistage_generation_expansion/logger/without_cut_inheritance/Periods$T-Real$num/LC.jld2")["sddpResults"][:solHistory]
        sddpResultsSBC = load("src/multistage_generation_expansion/logger/without_cut_inheritance/Periods$T-Real$num/SBC.jld2")["sddpResults"][:solHistory]
        sddpResultsSMC = load("src/multistage_generation_expansion/logger/without_cut_inheritance/Periods$T-Real$num/SMC.jld2")["sddpResults"][:solHistory]

        sddpResultsSNC = load("src/multistage_generation_expansion/logger/with_cut_inheritance/Periods$T-Real$num/CPT-MDC-Inf-SNC.jld2")["sddpResults"][:solHistory]


        UB = sddpResultsSNC[end,2] + 2000

        sddpResultsBC = filter(row -> row.Iter % 3 == 1, sddpResultsBC)
        sddpResultsFC = filter(row -> row.Iter % 3 == 1, sddpResultsFC)
        sddpResultsSC = filter(row -> row.Iter % 3 == 1, sddpResultsSC)
        sddpResultsLC = filter(row -> row.Iter % 3 == 1, sddpResultsLC)
        sddpResultsSNC = filter(row -> row.Iter % 3 == 1, sddpResultsSNC)
        sddpResultsSMC = filter(row -> row.Iter % 3 == 1, sddpResultsSMC)

        
        time_LB = @df sddpResultsBC plot(
            :Time, 
            :LB, 
            label="BC", 
            # title = "Lower Bounds vs. Time", 
            xlab = "Time (sec.)", 
            xlim = [0, 600], 
            # ylim = [5000, UB], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:xcross, 2, 1.), 
            color="#3B5387",       # 使用蓝色
            yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            # legend=:outertop,  # legend 在顶部
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            # legend_column=3,  # legend 列数减少，使其松散
            # legend_spacing=6,  # 控制 legend 之间的间距
            linewidth=1.5,
            linestyle=:solid        # 实线
        )  
        @df sddpResultsLC plot!(:Time, :LB, marker=(:circle, 2, 1.), label="LC", linestyle=:solid, color="#E5637B", linewidth=1.5)
        @df sddpResultsSBC plot!(:Time, :LB, marker=(:x, 2, 1.), label="SBC", linestyle=:solid, color="#47AF79", linewidth=1.5)
        @df sddpResultsSMC plot!(:Time, :LB, marker=(:xcross, 2, 1.), label="SMC", linestyle=:solid, color=:purple, linewidth=1.5)

        @df sddpResultsFC plot!(:Time, :LB, marker=(:vline, 2, 1.), label="FC", linestyle=:dash, color="#ED4043", linewidth=1.5)
        @df sddpResultsSC plot!(:Time, :LB, marker=(:plus, 2, 1.), label="SC", linestyle=:dot, color="#6A5DC4", linewidth=1.5)
        
        @df sddpResultsSNC plot!(:Time, :LB, marker=(:none, 2, 1.), label="DBC", linestyle=:dashdotdot, color=:orange, linewidth=1.5)

        time_LB |> save("src/multistage_generation_expansion/logger/with_cut_inheritance/Periods$T-Real$num/benchmarks.pdf")

    end
end
