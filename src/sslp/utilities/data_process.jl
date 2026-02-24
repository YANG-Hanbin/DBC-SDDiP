using Pkg
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
Pkg.activate(REPO_ROOT)
using JuMP, Gurobi, PowerModels
using Statistics, StatsBase, Random, Dates, Distributions
using Distributed, ParallelDataTransfer
using CSV, DataFrames, Printf
using JLD2, FileIO
using StatsPlots, PlotThemes
using PrettyTables;
using VegaLite, VegaDatasets;

project_root = REPO_ROOT
include(joinpath(project_root, "src", "sslp", "utilities", "structs.jl"))
theme(:default)


cutSelection = :SMC; 
Ω = 200; (J, I)  = (15, 30);

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------- TO GENERATE TABLES ---------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------- MDCiter --------------------- ---------------------------------------- ##
# 初始化DataFrame
result_df = DataFrame(
    D=[], 
    Ω=Int[], 
    num=Tuple[], 
    best_LB=Float64[], 
    final_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],                       
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    avg_MDC = String[]
);
inherit_disjunctive_cuts = false
i = Inf;
for i in [1, 2]
    for Ω in [100, 200]
        for (J, I) in [(5, 15), (10, 20), (15, 25), (10, 35), (15, 30)]
            try
                # 加载数据
                file_path = "$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/NaiveCPT-MDC-$inherit_disjunctive_cuts-SNC.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                MDCiter = mean(solHistory.MDCiter)
                std_MDCiter = std(solHistory.MDCiter)
                avg_MDC = @sprintf "%.1f (%.1f)" MDCiter std_MDCiter  # 格式化字符串

                # 添加到DataFrame
                push!(
                    result_df, 
                    (i, Ω, (J, I), round(best_LB, digits = 1), round(final_gap, digits = 1), total_iter, avg_iter_time, round(best_LB_time, digits = 1), best_LB_iter, avg_MDC))
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
    # elseif x isa Tuple  # 处理 iter_range 之类的元组数据
    #     return "$(x[1])--$(x[2])"
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

## ---------------------------------------- Cut Families with different Ω ---------------------------------------- ##
# 初始化DataFrame
result_df = DataFrame(cut=Symbol[], Ω=Int[], num=Tuple[], best_LB=Float64[], 
                      final_gap=Float64[], total_iter=Int[], avg_iter_time=String[], 
                      best_LB_time=Float64[], best_LB_iter=Int[])

for cut in [:BC, :LC, :SBC, :SMC]
    for Ω in [100, 200]
        for (J, I) in [(5, 15), (10, 20), (15, 25), (10, 35), (15, 30)]
            try
                # 加载数据
                
                file_path = "$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/$cut-true.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 添加到DataFrame
                push!(result_df, (cut, Ω, (J, I), round(best_LB, digits = 1), round(final_gap, digits = 1), total_iter, avg_iter_time, round(best_LB_time, digits = 1), best_LB_iter))
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
    # elseif x isa Tuple  # 处理 iter_range 之类的元组数据
    #     return "$(x[1])--$(x[2])"
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

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## --------------------------------------------------------------------- the same instance with different benchmarks ---------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 
for Ω in [50, 100]
    for (J, I) in [(20, 70), (30, 60), (40, 50), (50, 40)]
        # 读取数据
        enforce_binary_copies = false
        sddpResultsBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/BC.jld2")["sddpResults"][:solHistory]
        sddpResultsLC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/LC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddpResultsSBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SBC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddpResultsSMC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SMC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        MDCfalse = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/CPT-MDC-$enforce_binary_copies-SNC.jld2")["sddpResults"][:solHistory]
        
        # 处理 gap 数据
        sddpResultsBC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsBC.gap, "%" => "")) 
        sddpResultsLC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsLC.gap, "%" => "")) 
        sddpResultsSBC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsSBC.gap, "%" => ""))
        sddpResultsSMC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsSMC.gap, "%" => ""))
        MDCfalse.gap_float = 100 .- parse.(Float64, replace.(MDCfalse.gap, "%" => ""))
        
        time_LB = @df sddpResultsBC plot(
            :Time, 
            :gap_float, 
            label="BC", 
            ylab = "Closed gap (%)",
            xlab = "Time (sec.)", 
            xlim = [0, 200], 
            ylim = [5, 40], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:none, 1, 1.), 
            color="#3B5387",
            yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            legend=:outertop,
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=5, 
            linestyle=:dot,
            linewidth=1.5     # 线条加粗
        )  
        @df sddpResultsLC plot!(:Time, :gap_float, marker=(:none, 1, 1.), label="LC", linestyle=:dash, color="#E5637B", linewidth=1.5)
        @df sddpResultsSBC plot!(:Time, :gap_float, marker=(:none, 1, 1.), label="SBC", linestyle=:dashdotdot, color="#47AF79", linewidth=1.5)
        @df sddpResultsSMC plot!(:Time, :gap_float, marker=(:o, 1, 1.), label="SMC", linestyle=:dot, color=:purple, linewidth=1.5)
        @df MDCfalse plot!(:Time, :gap_float, marker=(:x, 1, 1.), label="DBC", linestyle=:dashdotdot, color="#1f77b4", linewidth=1.5)

        time_LB |> save("src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/benchmarks-$enforce_binary_copies.pdf")
    end
end

for Ω in [50, 100]
    for (J, I) in [(20, 70), (30, 60), (40, 50), (50, 40)]
        # 读取数据
        enforce_binary_copies = true
        sddpResultsBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/BC.jld2")["sddpResults"][:solHistory]
        sddpResultsLC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/LC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddpResultsSBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SBC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddpResultsSMC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SMC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        MDCfalse = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/CPT-iMDC-true-SNC.jld2")["sddpResults"][:solHistory]
        
        # 处理 gap 数据
        sddpResultsBC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsBC.gap, "%" => "")) 
        sddpResultsLC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsLC.gap, "%" => "")) 
        sddpResultsSBC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsSBC.gap, "%" => ""))
        sddpResultsSMC.gap_float = 100 .- parse.(Float64, replace.(sddpResultsSMC.gap, "%" => ""))
        MDCfalse.gap_float = 100 .- parse.(Float64, replace.(MDCfalse.gap, "%" => ""))
        
        time_LB = @df sddpResultsBC plot(
            :Time, 
            :gap_float, 
            label="BC", 
            # title = "Lower Bounds vs. Time", 
            xlab = "Time (sec.)", 
            ylab = "Closed gap (%)",
            xlim = [0, 200], 
            ylim = [5, 100], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:none, 2, 1.), 
            color="#3B5387",       # 使用蓝色
            yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            legend=:outertop,  # legend 在顶部
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=5,  # legend 列数减少，使其松散
            # legend_spacing=6,  # 控制 legend 之间的间距
            linewidth=1.5,     # 线条加粗
            linestyle=:dot        # 实线
        )  
        @df sddpResultsLC plot!(:Time, :gap_float, marker=(:none, 2, 1.), label="LC", linestyle=:dash, color="#E5637B", linewidth=1.5)
        @df sddpResultsSBC plot!(:Time, :gap_float, marker=(:none, 2, 1.), label="SBC", linestyle=:dashdotdot, color="#47AF79", linewidth=1.5)
        @df sddpResultsSMC plot!(:Time, :gap_float, marker=(:o, 2, 1.), label="SMC", linestyle=:dot, color=:purple, linewidth=1.5)
        @df MDCfalse plot!(:Time, :gap_float, marker=(:x, 2, 1.), label="DBC", linestyle=:dashdotdot, color="#1f77b4", linewidth=1.5)

        time_LB |> save("src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/benchmarks-$enforce_binary_copies.pdf")
    end
end
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- the same instance with different cuts -------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 

for Ω in [50, 100]
    for (J, I) in [(20, 70), (30, 60), (40, 50), (50, 40)]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#FA8072"]  # 蓝色、橙色、绿色
        # 读取数据
        sddlpResultBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/BC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddlpResultLC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/LC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddlpResultSBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SBC-$enforce_binary_copies.jld2")["sddpResults"][:solHistory]
        sddlpResultSMC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SMC-true.jld2")["sddpResults"][:solHistory]

        # 处理 gap 数据
        sddlpResultBC.gap_float = parse.(Float64, replace.(sddlpResultBC.gap, "%" => "")) 
        sddlpResultLC.gap_float = parse.(Float64, replace.(sddlpResultLC.gap, "%" => "")) 
        sddlpResultSBC.gap_float = parse.(Float64, replace.(sddlpResultSBC.gap, "%" => ""))
        sddlpResultSMC.gap_float = parse.(Float64, replace.(sddlpResultSMC.gap, "%" => ""))

        # 统一数据格式
        df_BC = DataFrame(Iter=sddlpResultBC.Iter, Time=sddlpResultBC.Time, LB=sddlpResultBC.LB ./ 10^3, Cut="BC")
        df_LC = DataFrame(Iter=sddlpResultLC.Iter, Time=sddlpResultLC.Time, LB=sddlpResultLC.LB ./ 10^3, Cut="LC")
        df_SBC = DataFrame(Iter=sddlpResultSBC.Iter, Time=sddlpResultSBC.Time, LB=sddlpResultSBC.LB ./ 10^3, Cut="SBC")
        df_SMC = DataFrame(Iter=sddlpResultSMC.Iter, Time=sddlpResultSMC.Time, LB=sddlpResultSMC.LB ./ 10^3, Cut="SMC")

        df_BC = filter(row -> row.Time < 500, df_BC)
        df_LC = filter(row -> row.Time < 500, df_LC)
        df_SBC = filter(row -> row.Time < 500, df_SBC)
        df_SMC = filter(row -> row.Time < 500, df_SMC)

        # 合并数据
        df = vcat(df_BC, df_LC, df_SBC, df_SMC)

        df |> @vlplot(
            :line,
            x={:Time, axis={title="Time (sec.)", titleFontSize=25, labelFontSize=25,}},
            y={:LB, axis={title="Lower bounds (× 10³)", titleFontSize=25, labelFontSize=25}, 
            scale={domain=[-5, -3.5]}},
            color={
                :Cut, 
                legend={title=nothing, orient="top", columns=4}, 
                scale={domain=["BC", "LC", "SBC", "SMC"],  # 这里显式定义颜色顺序
                    range=colors}  # 绑定对应颜色
            },  
            strokeDash={
                :Cut, 
                scale={domain=["BC", "LC", "SBC", "SMC"], 
                    range=[[5, 3], [15, 3], [10, 2], [10, 5, 2, 5]]}  # 虚线样式
            },  
            shape={
                :Cut, 
                scale={domain=["BC", "LC", "SBC", "SMC"], 
                    range=["circle", "diamond", "cross", "circle"]}  # 形状
            },  
            strokeWidth={
                :Cut, 
                scale={domain=["BC", "LC", "SBC", "SMC"], 
                    range=[1, 1, 1, 1]}  # LC 粗，PLC 中等，SMC 细
            },  
            width=500,
            height=350,
            config={ 
                axis={
                    labelFont="Times New Roman", 
                    titleFont="Times New Roman"
                    }, 
                legend={
                    labelFont="Times New Roman", 
                    titleFont="Times New Roman",
                    labelFontSize=25,   # 调整 legend 标签字体大小
                    symbolSize=150,      # 增大 legend 符号大小
                    symbolStrokeWidth=3  # 增加 legend 线条粗细
                }, 
                title={font="Times New Roman"} 
            }
        ) |> save("$(homedir())/StoCutDyProg/src/sslp/logger/numericalResults-$case/Periods$T-Real$num/lower_bound_Time_Period$T-Real$num.pdf")

        sddlpResultBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/BC.jld2")["sddpResults"][:solHistory]
        sddlpResultLC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/LC.jld2")["sddpResults"][:solHistory]
        sddlpResultSBC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SBC.jld2")["sddpResults"][:solHistory]
        sddlpResultSMC = load("$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/SMC.jld2")["sddpResults"][:solHistory]

        # 处理 gap 数据
        sddlpResultBC.gap_float = parse.(Float64, replace.(sddlpResultBC.gap, "%" => "")) 
        sddlpResultLC.gap_float = parse.(Float64, replace.(sddlpResultLC.gap, "%" => "")) 
        sddlpResultSBC.gap_float = parse.(Float64, replace.(sddlpResultSBC.gap, "%" => ""))
        sddlpResultSMC.gap_float = parse.(Float64, replace.(sddlpResultSMC.gap, "%" => ""))

        # 统一数据格式
        df_BC = DataFrame(Iter=sddlpResultBC.Iter, Time=sddlpResultBC.Time, LB=sddlpResultBC.LB ./ 10^3, Cut="BC")
        df_LC = DataFrame(Iter=sddlpResultLC.Iter, Time=sddlpResultLC.Time, LB=sddlpResultLC.LB ./ 10^3, Cut="LC")
        df_SBC = DataFrame(Iter=sddlpResultSBC.Iter, Time=sddlpResultSBC.Time, LB=sddlpResultSBC.LB ./ 10^3, Cut="SBC")
        df_SMC = DataFrame(Iter=sddlpResultSMC.Iter, Time=sddlpResultSMC.Time, LB=sddlpResultSMC.LB ./ 10^3, Cut="SMC")

        df_BC = filter(row -> row.Iter < 30, df_BC)
        df_LC = filter(row -> row.Iter < 30, df_LC)
        df_SBC = filter(row -> row.Iter < 30, df_SBC)
        df_SMC = filter(row -> row.Iter < 30, df_SMC)

        # 合并数据
        df = vcat(df_BC, df_LC, df_SBC, df_SMC)

        df |> @vlplot(
            :line,
            x={:Iter, axis={title="Iteration", titleFontSize=25, labelFontSize=25}},
            y={:LB, axis={title="Lower bounds (× 10³)", titleFontSize=25, labelFontSize=25}, scale={domain=[-5, -3.5]}},
            color={
                :Cut, 
                legend={title=nothing, orient="top", columns=4}, 
                scale={domain=["BC", "LC", "SBC", "SMC"],  # 这里显式定义颜色顺序
                    range=colors}  # 绑定对应颜色
            },  
            strokeDash={
                :Cut, 
                scale={domain=["BC", "LC", "SBC", "SMC"], 
                    range=[[5, 3], [15, 3], [10, 2], [10, 5, 2, 5]]}  # 虚线样式
            },  
            shape={
                :Cut, 
                scale={domain=["BC", "LC", "SBC", "SMC"], 
                    range=["circle", "diamond", "cross", "circle"]}  # 形状
            },  
            strokeWidth={
                :Cut, 
                scale={domain=["BC", "LC", "SBC", "SMC"], 
                    range=[1, 1, 1, 1]}  # LC 粗，PLC 中等，SMC 细
            },  
            width=500,
            height=350,
            config={ 
                axis={
                    labelFont="Times New Roman", 
                    titleFont="Times New Roman"
                    }, 
                legend={
                    labelFont="Times New Roman", 
                    titleFont="Times New Roman",
                    labelFontSize=25,   # 调整 legend 标签字体大小
                    symbolSize=150,      # 增大 legend 符号大小
                    symbolStrokeWidth=3  # 增加 legend 线条粗细
                }, 
                title={font="Times New Roman"} 
            }
        ) |> save("$(homedir())/StoCutDyProg/src/sslp/logger/numericalResults-$case/Periods$T-Real$num/lower_bound_Iter_Period$T-Real$num.pdf")
    end
end



## ---------------------------------------------------------------------------------------------------------------------------------------- ##
## ------------------------------------------------------------  Bar Chart  --------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------- ##
results = DataFrame(T = Int[], num = Int[], method = String[], avg_time = Float64[], std_time = Float64[], avg_LM_iter = Float64[], std_LM_iter = Float64[])

# 遍历不同的 (T, num) 组合
for T in [6, 8, 12]
    for num in [3, 5, 10]
        for method in ["LC", "PLC", "SMC"]
            # 读取数据
            data = load("$(REPO_ROOT)/src/sslp/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-$method.jld2")["sddpResults"][:solHistory]

            df = DataFrame(Iter=data.Iter, time=data.time, LM_iters=data.LM_iter, Cut=method)
            
            # 计算平均值和标准差
            avg_time = mean(df.time)
            std_time = std(df.time)
            avg_LM_iter = mean(df.LM_iters)
            std_LM_iter = std(df.LM_iters)

            # 存入 DataFrame
            push!(results, (T, num, method, avg_time, std_time, avg_LM_iter, std_LM_iter))
        end
    end
end


# 颜色方案
# colors = ["#1E90FF", "#DC143C", "#006400"]  
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 蓝色、橙色、绿色
# 画第一个 Grouped Bar Chart（平均迭代时间）
results |>
@vlplot(
    :bar,
    x={"T:n", title="T", axis={labelFont="Times New Roman", labelFontSize=25, titleFontSize=25, labelAngle=0}},
    xOffset={"method:n", title="Cut"},
    y={"avg_time:q", title="Average Iteration Time", axis={labelFontSize=25, titleFontSize=25}},
    color={"method:n", scale={range=colors}, title=nothing, labelFontSize=25, titleFontSize=25},
    column={"num:n", title="R", 
            header={labelFont="Times New Roman", titleFont="Times New Roman", 
                    labelFontSize=25, titleFontSize=25}},  # 确保 R 也是 Times New Roman
    tooltip=[{ "T:n"}, {"num:n"}, {"method:n"}, {"avg_time:q"}, {"std_time:q"}],
    width=300, height=250,
    config={ 
        axis={labelFont="Times New Roman", titleFont="Times New Roman"}, 
        legend={
            labelFont="Times New Roman", titleFont="Times New Roman",
            labelFontSize=25, symbolSize=150, symbolStrokeWidth=3
        }, 
        title={font="Times New Roman"},
        bar={width=20}
    }
) + 
@vlplot(
    :errorbar,
    x={"T:n"},
    y={"avg_time:q", scale={zero=false}},
    yError={"std_time:q"}
) |> save("$(homedir())/StoCutDyProg/src/sslp/logger/numericalResults-$case/$algorithm-AverageIterTime.pdf")

# 画第二个 Grouped Bar Chart（平均 LM_iter 次数）
results |>
@vlplot(
    :bar,
    x={"T:n", title="T", axis={labelFont="Times New Roman", labelFontSize=25, titleFontSize=25, labelAngle=0}},
    xOffset={"method:n", title="Cut"},
    y={"avg_LM_iter:q", title="Average Iteration Counts", axis={labelFontSize=25, titleFontSize=25}},
    color={"method:n", scale={range=colors}, title=nothing, labelFontSize=25, titleFontSize=25},
    column={"num:n", title="R", 
            header={labelFont="Times New Roman", titleFont="Times New Roman", 
                    labelFontSize=25, titleFontSize=25}},  # 确保 R 也是 Times New Roman
    tooltip=[{ "T:n"}, {"num:n"}, {"method:n"}, {"avg_LM_iter:q"}, {"std_LM_iter:q"}],
    width=300, height=250,
    config={ 
        axis={labelFont="Times New Roman", titleFont="Times New Roman"}, 
        legend={
            labelFont="Times New Roman", titleFont="Times New Roman",
            labelFontSize=25, symbolSize=150, symbolStrokeWidth=3
        }, 
        title={font="Times New Roman"},
        bar={width=20}
    }
) + 
@vlplot(
    :errorbar,
    x={"T:n"},
    y={"avg_LM_iter:q", scale={zero=false}},
    yError={"std_LM_iter:q"}
) |> save("$(homedir())/StoCutDyProg/src/sslp/logger/numericalResults-$case/$algorithm-LMiter.pdf")

## ---------------------------------------- Cut Families with different M ---------------------------------------- ##
# 初始化DataFrame
result_df = DataFrame(cut=Symbol[], Ω=Int[], num=Tuple[], best_LB=Float64[], 
                      final_gap=Float64[], total_iter=Int[], avg_iter_time=String[], 
                      best_LB_time=Float64[], best_LB_iter=Int[])

for cut in [:iMDC]
    for Ω in [100, 200]
        for (J, I) in [(5, 15), (10, 20), (15, 25), (10, 35), (15, 30)]
            try
                # 加载数据
                
                file_path = "$(REPO_ROOT)/src/sslp/logger/numericalResults/J$J-I$I-Ω$Ω/CPT-$cut-true-SNC.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                total_iter = solHistory.Iter[end]  # 总迭代数
                iter_times = diff(solHistory.Time)  # 计算每次迭代的时间间隔
                avg_time = mean(iter_times)  # 计算平均迭代时间
                std_time = std(iter_times)   # 计算标准差
                avg_iter_time = @sprintf "%.1f (%.1f)" avg_time std_time  # 格式化字符串
                best_LB_time = solHistory.Time[best_LB_idx]  # 到达best LB的时间
                best_LB_iter = solHistory.Iter[best_LB_idx]  # 到达best LB的迭代数

                # 添加到DataFrame
                push!(result_df, (cut, Ω, (J, I), round(best_LB, digits = 1), round(final_gap, digits = 1), total_iter, avg_iter_time, round(best_LB_time, digits = 1), best_LB_iter))
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
    # elseif x isa Tuple  # 处理 iter_range 之类的元组数据
    #     return "$(x[1])--$(x[2])"
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
