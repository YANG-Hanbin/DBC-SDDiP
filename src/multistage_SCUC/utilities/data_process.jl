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
include(joinpath(project_root, "src", "multistage_SCUC", "utilities", "structs.jl"))
theme(:default)

case = "case_RTS_GMLC"; #"case_RTS_GMLC", "case30"
enforce_binary_copies = true; 
cutSelection = :SMC; 
num = 10; T = 12; 
algorithm = :SDDiP;

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------- TO GENERATE TABLES ---------------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##

## ----------------------------------------------------------------------------------- MDC iter ----------------------------------------------------------------------------------------- ##
Normalization = :α
# 初始化DataFrame
result_df = DataFrame(
    D=[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    min_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    # gap_under_1_time=Union{Missing, Float64}[],
    # gap_under_1_iter=Union{Missing, Int}[]
);

for i in [1, 5, 10, 15, Inf]
    for T in [6, 8, 12]
        for num in [5, 10]
            try
                # 加载数据
                file_path = "$(REPO_ROOT)/src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/$algorithm-64-CPT-$i-$Normalization.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                gap_float = parse.(Float64, replace.(solHistory.gap, "%" => ""))
                min_gap = minimum(gap_float)
                min_index = argmin(gap_float)
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
                    i, T, num, best_LB, min_gap, total_iter, 
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

## ----------------------------------------------------------------------------------- Normalization ----------------------------------------------------------------------------------------- ##
# 初始化DataFrame
result_df = DataFrame(
    Normalization=[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    min_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    avg_MDC=String[],
    # gap_under_1_time=Union{Missing, Float64}[],
    # gap_under_1_iter=Union{Missing, Int}[]
);

for Normalization in [:SNC, :Regular, :α]
    for T in [6, 8, 12]
        for num in [5, 10]
            try
                # 加载数据
                file_path = "$(REPO_ROOT)/src/multistage_SCUC/new_logger/numericalResults-$case/Periods$T-Real$num/$algorithm-64-CPT-Inf-$Normalization.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                gap_float = parse.(Float64, replace.(solHistory.gap, "%" => ""))
                min_gap = minimum(gap_float)
                min_index = argmin(gap_float)
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
                
                MDCiter = mean(solHistory.LM_iter)
                std_MDCiter = std(solHistory.LM_iter)
                avg_MDC = @sprintf "%.1f (%.1f)" MDCiter std_MDCiter  # 格式化字符串

                # 添加到DataFrame
                push!(result_df, (
                    Normalization, T, num, best_LB, min_gap, total_iter, 
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


## ----------------------------------------------------------------------------------- Cut Families ----------------------------------------------------------------------------------------- ##
# 初始化DataFrame
enforce_binary_copies = false
result_df = DataFrame(
    cut=Symbol[], 
    T=Int[], 
    num=Int[], 
    best_LB=Float64[],         
    min_gap=Float64[], 
    total_iter=Int[], 
    avg_iter_time=String[],         
    best_LB_time=Float64[], 
    best_LB_iter=Int[],
    # gap_under_1_time=Union{Missing, Float64}[],
    # gap_under_1_iter=Union{Missing, Int}[]
);

for T in [6, 8, 12]
    for num in [5, 10]
        try
            # 加载数据
            file_path = "$(REPO_ROOT)/src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/$algorithm-64-$cut.jld2"
            solHistory = load(file_path)["sddpResults"][:solHistory]

            # 计算所需的统计数据
            best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
            final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
            gap_float = parse.(Float64, replace.(solHistory.gap, "%" => ""))
            min_gap = minimum(gap_float)
            min_index = argmin(gap_float)
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
                cut, T, num, best_LB, min_gap, total_iter, 
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

for cut in [:LC, :SBC, :SMC]
    for T in [6, 8, 12]
        for num in [5, 10]
            try
                # 加载数据
                file_path = "$(REPO_ROOT)/src/multistage_SCUC/new_logger/numericalResults-$case/Periods$T-Real$num/$algorithm-64-$cut-$enforce_binary_copies.jld2"
                solHistory = load(file_path)["sddpResults"][:solHistory]

                # 计算所需的统计数据
                best_LB, best_LB_idx = findmax(solHistory.LB)  # 最优LB及其索引
                final_gap = parse(Float64, replace(solHistory.gap[end], "%" => ""))  # 最终gap
                gap_float = parse.(Float64, replace.(solHistory.gap, "%" => ""))
                min_gap = minimum(gap_float)
                min_index = argmin(gap_float)
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
                    cut, T, num, best_LB, min_gap, total_iter, 
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

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- the same instance with different cuts -------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 

for T in [6, 8, 12]
    for num in [5, 10]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#FA8072"]  # 蓝色、橙色、绿色
        # 读取数据
        sddlpResultBC = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-BC.jld2")["sddpResults"][:solHistory]
        sddlpResultLC = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-LC.jld2")["sddpResults"][:solHistory]
        sddlpResultSBC = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-SBC.jld2")["sddpResults"][:solHistory]
        sddlpResultSMC = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-SMC.jld2")["sddpResults"][:solHistory]

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

        # 合并数据
        df = vcat(df_BC, df_LC, df_SBC, df_SMC)

        df |> @vlplot(
            :line,
            x={:Time, axis={title="Time (sec.)", titleFontSize=25, labelFontSize=25,}},
            y={:LB, axis={title="Lower bounds (× 10³)", titleFontSize=25, labelFontSize=25}},
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
        ) |> save("$(homedir())/StoCutDyProg/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/lower_bound_Time_Period$T-Real$num.pdf")


        df |> @vlplot(
            :line,
            x={:Iter, axis={title="Iteration", titleFontSize=25, labelFontSize=25}},
            y={:LB, axis={title="Lower bounds (× 10³)", titleFontSize=25, labelFontSize=25}},
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
                    range=["circle", "diamond", "cross"]}  # 形状
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
        ) |> save("$(homedir())/StoCutDyProg/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/lower_bound_Iter_Period$T-Real$num.pdf")
    end
end

for T in [6, 8, 12]
    for num in [5, 10]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#FA8072"]  # 蓝色、橙色、绿色
        # 读取数据
        sddlpResult0 = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-BC.jld2")["sddpResults"][:solHistory]
        sddlpResult1 = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-CPT-1-$Normalization.jld2")["sddpResults"][:solHistory]
        sddlpResult2 = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-CPT-2-$Normalization.jld2")["sddpResults"][:solHistory]
        # sddlpResult3 = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-CPT-3-$Normalization.jld2")["sddpResults"][:solHistory]
        sddlpResultInf = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-256-CPT-Inf-$Normalization.jld2")["sddpResults"][:solHistory]

        # 处理 gap 数据
        sddlpResult0.gap_float = parse.(Float64, replace.(sddlpResult0.gap, "%" => "")) 
        sddlpResult1.gap_float = parse.(Float64, replace.(sddlpResult1.gap, "%" => "")) 
        sddlpResult2.gap_float = parse.(Float64, replace.(sddlpResult2.gap, "%" => "")) 
        # sddlpResult3.gap_float = parse.(Float64, replace.(sddlpResult3.gap, "%" => ""))
        sddlpResultInf.gap_float = parse.(Float64, replace.(sddlpResultInf.gap, "%" => ""))

        # 统一数据格式
        df_0 = DataFrame(Iter=sddlpResult0.Iter, Time=sddlpResult0.Time, LB=sddlpResult0.LB ./ 10^3, Cut="D=0")
        df_1 = DataFrame(Iter=sddlpResult1.Iter, Time=sddlpResult1.Time, LB=sddlpResult1.LB ./ 10^3, Cut="D=1")
        df_2 = DataFrame(Iter=sddlpResult2.Iter, Time=sddlpResult2.Time, LB=sddlpResult2.LB ./ 10^3, Cut="D=2")
        # df_3 = DataFrame(Iter=sddlpResult3.Iter, Time=sddlpResult3.Time, LB=sddlpResult3.LB ./ 10^3, Cut="D=3")
        df_Inf = DataFrame(Iter=sddlpResultInf.Iter, Time=sddlpResultInf.Time, LB=sddlpResultInf.LB ./ 10^3, Cut="D=i")

        # 合并数据
        df = vcat(df_0, df_1, df_2, df_Inf)

        df |> @vlplot(
            :line,
            x={:Time, axis={title="Time (sec.)", titleFontSize=25, labelFontSize=25,}},
            y={:LB, axis={title="Lower bounds (× 10³)", titleFontSize=25, labelFontSize=25}},
            color={
                :Cut, 
                legend={title=nothing, orient="top", columns=4}, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"],  # 这里显式定义颜色顺序
                    range=colors}  # 绑定对应颜色
            },  
            strokeDash={
                :Cut, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"], 
                    range=[[5, 3], [15, 3], [10, 2], [10, 5, 2, 5]]}  # 虚线样式
            },  
            shape={
                :Cut, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"], 
                    range=["circle", "diamond", "cross", "circle"]}  # 形状
            },  
            strokeWidth={
                :Cut, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"], 
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
        ) |> save("$(homedir())/StoCutDyProg/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/cptlower_bound_Time_Period$T-Real$num.pdf")


        df |> @vlplot(
            :line,
            x={:Iter, axis={title="Iteration", titleFontSize=25, labelFontSize=25}},
            y={:LB, axis={title="Lower bounds (× 10³)", titleFontSize=25, labelFontSize=25}},
            color={
                :Cut, 
                legend={title=nothing, orient="top", columns=4}, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"],  # 这里显式定义颜色顺序
                    range=colors}  # 绑定对应颜色
            },  
            strokeDash={
                :Cut, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"], 
                    range=[[5, 3], [15, 3], [10, 2], [10, 5, 2, 5]]}  # 虚线样式
            },  
            shape={
                :Cut, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"], 
                    range=["circle", "diamond", "cross"]}  # 形状
            },  
            strokeWidth={
                :Cut, 
                scale={domain=["D=0", "D=1", "D=2", "D=i"], 
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
        ) |> save("$(homedir())/StoCutDyProg/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/cptlower_bound_Iter_Period$T-Real$num.pdf")
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
            data = load("$(REPO_ROOT)/src/multistage_SCUC/logger/numericalResults-$case/Periods$T-Real$num/$algorithm-$method.jld2")["sddpResults"][:solHistory]

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
) |> save("$(homedir())/StoCutDyProg/src/multistage_SCUC/logger/numericalResults-$case/$algorithm-AverageIterTime.pdf")

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
) |> save("$(homedir())/StoCutDyProg/src/multistage_SCUC/logger/numericalResults-$case/$algorithm-LMiter.pdf")

for T in [6, 8, 12] 
    for num in [5, 10]
        cutInheritance = :without_cut_inheritance # :cut_inheritance, :without_cut_inheritance
        snc = load("src/multistage_SCUC/logger/$cutInheritance/Periods$T-Real$num/SDDiP-64-CPT-Inf-SNC.jld2")["sddpResults"][:solHistory]
        regular = load("src/multistage_SCUC/logger/$cutInheritance/Periods$T-Real$num/SDDiP-64-CPT-Inf-regular.jld2")["sddpResults"][:solHistory]
        alpha = load("src/multistage_SCUC/logger/$cutInheritance/Periods$T-Real$num/SDDiP-64-CPT-Inf-α.jld2")["sddpResults"][:solHistory]

        # snc = filter(row -> row.Iter % 2 == 1, snc)
        # regular = filter(row -> row.Iter % 2 == 1, regular)
        # alpha = filter(row -> row.Iter % 2 == 1, alpha)

        time_gap = @df snc plot(
            :Time, 
            :LB, 
            label="SNC", 
            xlab = "Time (sec.)", 
            xlim = [0, 3800], 
            ylim = [100000, 250000], 
            yformatter=:scientific,   # <--- 科学计数法
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:none, 2, 1.), 
            color=:purple,       
            # yformatter=y->string(Int(y)),
            tickfont=font("Computer Modern"),
            legend=:outertop,  # legend 在顶部
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=3,  # legend 列数减少，使其松散
            legend_spacing=6,  # 控制 legend 之间的间距
            linestyle=:dashdotdot,        # 实线
            linewidth=1.5     # 线条加粗
        )  
        @df regular plot!(:Time, :LB, marker=(:vline, 2, 1.), label="Regular", linestyle=:dot, color="#ED4043", linewidth=1.5)
        @df alpha plot!(:Time, :LB, marker=(:plus, 2, 1.), label="π", linestyle=:dash, color="#47AF79", linewidth=1.5)

        time_gap |> save("src/multistage_SCUC/logger/$cutInheritance/Periods$T-Real$num/normalization.pdf")
    end
end

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- the same instance with Benchmarks -------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 
for num in [5, 10]
    for T in [6, 8, 12]
        sddpResultsSC = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-CPT-1-α.jld2")["sddpResults"][:solHistory]
        sddpResultsFC = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-FC.jld2")["sddpResults"][:solHistory]
        sddpResultsBC = load("src/multistage_SCUC/logger/without_cut_inheritance/Periods$T-Real$num/SDDiP-64-BC.jld2")["sddpResults"][:solHistory]
        sddpResultsLC = load("src/multistage_SCUC/logger/without_cut_inheritance/Periods$T-Real$num/SDDiP-64-LC.jld2")["sddpResults"][:solHistory]
        sddpResultsSBC = load("src/multistage_SCUC/logger/without_cut_inheritance/Periods$T-Real$num/SDDiP-64-SBC.jld2")["sddpResults"][:solHistory]
        sddpResultsSMC = load("src/multistage_SCUC/logger/without_cut_inheritance/Periods$T-Real$num/SDDiP-64-SMC.jld2")["sddpResults"][:solHistory]

        sddpResultsSNC = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-CPT-Inf-α.jld2")["sddpResults"][:solHistory]
        
        time_LB = @df sddpResultsBC plot(
            :Time, 
            :LB,                     # 缩放到 ×10^5
            label="BC", 
            xlab = "Time (sec.)", 
            # ylab = "Lower Bound (× 1e5)",    # 纵坐标单位改成 ×10⁵
            xlim = [0, 4600], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:xcross, 2, 1.), 
            color="#3B5387", 
            yformatter=:scientific,   # <--- 科学计数法
            # yformatter=y->string(round(y, digits=1)),   # 格式化 2.0, 2.5 这种
            tickfont=font("Computer Modern"),
            legend=:outertop,  
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=4,  
            linewidth=1.5,
            linestyle=:solid
        )

        @df sddpResultsLC  plot!(:Time, :LB, marker=(:circle, 2, 1.), label="LC",  linestyle=:solid, color="#E5637B", linewidth=1.5)
        @df sddpResultsSBC plot!(:Time, :LB, marker=(:x, 2, 1.),      label="SBC", linestyle=:solid, color="#47AF79", linewidth=1.5)
        @df sddpResultsSMC plot!(:Time, :LB, marker=(:xcross, 2, 1.), label="SMC", linestyle=:solid, color=:purple, linewidth=1.5)

        @df sddpResultsFC  plot!(:Time, :LB, marker=(:vline, 2, 1.),  label="FC",  linestyle=:dash, color="#ED4043", linewidth=1.5)
        @df sddpResultsSC  plot!(:Time, :LB, marker=(:plus, 2, 1.),   label="SC",  linestyle=:dot,  color="#6A5DC4", linewidth=1.5)

        @df sddpResultsSNC plot!(:Time, :LB, marker=(:none, 2, 1.),   label="DBC", linestyle=:dashdotdot, color=:orange, linewidth=1.5)

        time_LB |> save("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/benchmarks.pdf")

    end
end

## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ##
## ----------------------------------------------------------------------- the number of MDCs -------------------------------------------------------------------------- ##
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 
for num in [5, 10]
    for T in [6, 8, 12]
        sddpResultsBC = load("src/multistage_SCUC/logger/without_cut_inheritance/Periods$T-Real$num/SDDiP-64-BC.jld2")["sddpResults"][:solHistory]
        sddpResults5 = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-CPT-5-α.jld2")["sddpResults"][:solHistory]
        sddpResults10 = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-CPT-10-α.jld2")["sddpResults"][:solHistory]
        sddpResults15 = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-CPT-15-α.jld2")["sddpResults"][:solHistory]
        sddpResultsSNC = load("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/SDDiP-64-CPT-Inf-α.jld2")["sddpResults"][:solHistory]
        
        time_LB = @df sddpResultsBC plot(
            :Time, 
            :LB,                     # 缩放到 ×10^5
            label="BC", 
            xlab = "Time (sec.)", 
            # ylab = "Lower Bound (× 1e5)",    # 纵坐标单位改成 ×10⁵
            xlim = [0, 3600], 
            titlefont = font(15,"Times New Roman"), 
            xguidefont=font(15,"Times New Roman"), 
            yguidefont=font(15,"Times New Roman"), 
            xtickfontsize=13, 
            ytickfontsize=13, 
            marker=(:xcross, 2, 1.), 
            color="#3B5387", 
            yformatter=:scientific,   # <--- 科学计数法
            # yformatter=y->string(round(y, digits=1)),   # 格式化 2.0, 2.5 这种
            tickfont=font("Computer Modern"),
            legend=:outertop,  
            legendfontsize=11, 
            legendfont=font("Times New Roman"), 
            legend_column=5,  
            linewidth=1.5,
            linestyle=:solid
        )

        @df sddpResults5  plot!(:Time, :LB, marker=(:circle, 2, 1.), label="CPT-5",  linestyle=:solid, color="#E5637B", linewidth=1.5)
        @df sddpResults10 plot!(:Time, :LB, marker=(:x, 2, 1.),      label="CPT-10", linestyle=:solid, color="#47AF79", linewidth=1.5)
        @df sddpResults15 plot!(:Time, :LB, marker=(:xcross, 2, 1.), label="CPT-15", linestyle=:solid, color=:purple, linewidth=1.5)

        @df sddpResultsSNC plot!(:Time, :LB, marker=(:none, 2, 1.),   label="DBC", linestyle=:dashdotdot, color=:orange, linewidth=1.5)

        time_LB |> save("src/multistage_SCUC/logger/cut_inheritance/Periods$T-Real$num/number_of_MDC.pdf")

    end
end

