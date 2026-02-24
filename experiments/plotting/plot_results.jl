using Pkg
using JLD2
using DataFrames
using CSV
using Statistics
using Printf
using Dates
# Use headless GR mode to avoid GKS socket conflicts in terminal/parallel runs.
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots

"""
    find_repo_root(start_dir::AbstractString)::String

# Purpose
    Resolve repository root by searching upward for `Project.toml`.

# Arguments
    1. `start_dir::AbstractString`: Start directory.

# Returns
    1. Absolute path to repository root.
"""
function find_repo_root(start_dir::AbstractString)::String
    cur = abspath(start_dir)
    while true
        if isfile(joinpath(cur, "Project.toml"))
            return cur
        end
        parent = dirname(cur)
        if parent == cur
            error("Could not find Project.toml from: " * start_dir)
        end
        cur = parent
    end
end

const REPO_ROOT = find_repo_root(@__DIR__)
Pkg.activate(REPO_ROOT)

"""
    parse_cli_args(args::Vector{String})::Dict{String, String}

# Purpose
    Parse `--key value` style command-line options.

# Arguments
    1. `args::Vector{String}`: Raw `ARGS`.

# Returns
    1. Dictionary from option key to string value.
"""
function parse_cli_args(args::Vector{String})::Dict{String, String}
    parsed = Dict{String, String}()
    idx = 1
    while idx <= length(args)
        token = args[idx]
        if startswith(token, "--")
            key = replace(token, "--" => "")
            if idx == length(args) || startswith(args[idx + 1], "--")
                parsed[key] = "true"
                idx += 1
            else
                parsed[key] = args[idx + 1]
                idx += 2
            end
        else
            idx += 1
        end
    end
    return parsed
end

"""
    parse_bool(value::String, default_value::Bool)::Bool

# Purpose
    Parse flexible boolean strings with a default fallback.

# Arguments
    1. `value::String`: Input string.
    2. `default_value::Bool`: Fallback when value is unrecognized.

# Returns
    1. Parsed boolean.
"""
function parse_bool(value::String, default_value::Bool)::Bool
    lower = lowercase(strip(value))
    if lower in ("1", "true", "yes", "y", "on")
        return true
    elseif lower in ("0", "false", "no", "n", "off")
        return false
    end
    return default_value
end

"""
    parse_axis_mode(value::String)::Symbol

# Purpose
    Parse x-axis mode.

# Arguments
    1. `value::String`: One of `time` or `iter`.

# Returns
    1. Symbol axis mode.
"""
function parse_axis_mode(value::String)::Symbol
    lower = lowercase(strip(value))
    if lower == "iter"
        return :iter
    elseif lower == "time"
        return :time
    end
    error("Unsupported x-axis mode: " * value * ". Use `time` or `iter`.")
end

"""
    parse_lb_mode(value::String)::Symbol

# Purpose
    Parse lower-bound curve mode.

# Arguments
    1. `value::String`: One of `best` or `raw`.

# Returns
    1. Symbol mode.
"""
function parse_lb_mode(value::String)::Symbol
    lower = lowercase(strip(value))
    if lower == "best"
        return :best
    elseif lower == "raw"
        return :raw
    end
    error("Unsupported lb-mode: " * value * ". Use `best` or `raw`.")
end

"""
    make_config(cli::Dict{String, String})::NamedTuple

# Purpose
    Build plotting configuration from defaults plus CLI overrides.

# Returns
    1. Plot configuration named tuple.
"""
function make_config(cli::Dict{String, String})::NamedTuple
    project = get(cli, "project", "scuc")
    if !(project in ("scuc", "generation_expansion", "sslp"))
        error("Unsupported project: " * project * ". Use `scuc`, `generation_expansion`, or `sslp`.")
    end

    dataset_filter = get(cli, "dataset-filter", ".*")
    run_filter = get(cli, "run-filter", ".*")
    exclude_filter = get(cli, "exclude-filter", "a^")
    x_axis = parse_axis_mode(get(cli, "x-axis", "time"))
    lb_mode = parse_lb_mode(get(cli, "lb-mode", "best"))
    recompute_gap = parse_bool(get(cli, "recompute-gap", "true"), true)
    width = parse(Int, get(cli, "width", "1200"))
    height = parse(Int, get(cli, "height", "700"))
    dpi = parse(Int, get(cli, "dpi", "300"))
    format = lowercase(get(cli, "format", "pdf"))
    output_root = get(
        cli,
        "output-root",
        joinpath(REPO_ROOT, "results", "figures"),
    )

    return (
        project = project,
        dataset_filter = Regex(dataset_filter),
        run_filter = Regex(run_filter),
        exclude_filter = Regex(exclude_filter),
        x_axis = x_axis,
        lb_mode = lb_mode,
        recompute_gap = recompute_gap,
        size = (width, height),
        dpi = dpi,
        format = format,
        output_root = output_root,
        legend_position = :bottomright,
        line_width = 2.5,
        title_font = 13,
        tick_font = 10,
        guide_font = 12,
    )
end

"""
    list_result_files(config::NamedTuple)::Dict{String, Vector{String}}

# Purpose
    Collect result files grouped by dataset directory.

# Returns
    1. Mapping `dataset_name => file_paths`.
"""
function list_result_files(config::NamedTuple)::Dict{String, Vector{String}}
    project_root = joinpath(REPO_ROOT, "results", config.project)
    if !isdir(project_root)
        error("Result directory does not exist: " * project_root)
    end

    grouped = Dict{String, Vector{String}}()
    for (root, _, files) in walkdir(project_root)
        local_jld = filter(name -> endswith(name, ".jld2"), files)
        isempty(local_jld) && continue

        dataset_name = basename(root)
        if !occursin(config.dataset_filter, dataset_name)
            continue
        end

        for name in local_jld
            if !occursin(config.run_filter, name)
                continue
            end
            if occursin(config.exclude_filter, name)
                continue
            end
            push!(get!(grouped, dataset_name, String[]), joinpath(root, name))
        end
    end

    for dataset_name in keys(grouped)
        sort!(grouped[dataset_name])
    end
    return grouped
end

"""
    pick_dict_value(dict_obj, keys_to_try)::Any

# Purpose
    Read dictionary entries with fallback key aliases (`String` / `Symbol`).
"""
function pick_dict_value(dict_obj, keys_to_try)::Any
    for key in keys_to_try
        if haskey(dict_obj, key)
            return dict_obj[key]
        end
    end
    return nothing
end

"""
    parse_gap_values(raw_gap)::Vector{Float64}

# Purpose
    Parse stored gap column into numeric percentages.
"""
function parse_gap_values(raw_gap)::Vector{Float64}
    if raw_gap isa AbstractVector{<:AbstractString}
        output = Float64[]
        sizehint!(output, length(raw_gap))
        for g in raw_gap
            cleaned = replace(strip(g), "%" => "")
            parsed = tryparse(Float64, cleaned)
            push!(output, isnothing(parsed) ? NaN : parsed)
        end
        return output
    elseif raw_gap isa AbstractVector
        return [ismissing(v) ? NaN : Float64(v) for v in raw_gap]
    else
        return Float64[]
    end
end

"""
    to_float_vector(values)::Vector{Float64}

# Purpose
    Robust conversion for dataframe columns that may contain numbers,
    strings, or `missing`.
"""
function to_float_vector(values)::Vector{Float64}
    output = Vector{Float64}(undef, length(values))
    for idx in eachindex(values)
        value = values[idx]
        if ismissing(value)
            output[idx] = NaN
        elseif value isa AbstractString
            parsed = tryparse(Float64, strip(value))
            output[idx] = isnothing(parsed) ? NaN : parsed
        else
            output[idx] = Float64(value)
        end
    end
    return output
end

"""
    infer_time_axis(df::DataFrame)::Vector{Float64}

# Purpose
    Resolve cumulative runtime axis from dataframe columns.
"""
function infer_time_axis(df::DataFrame)::Vector{Float64}
    if "Time" in names(df)
        return to_float_vector(df[!, "Time"])
    elseif "time" in names(df)
        values = to_float_vector(df[!, "time"])
        return cumsum(values)
    end
    return collect(1.0:nrow(df))
end

"""
    make_run_label(path::String, run_meta)::String

# Purpose
    Build concise legend label from metadata and filename.
"""
function make_run_label(path::String, run_meta)::String
    filename = basename(path)
    cut_value = pick_dict_value(run_meta, (:cut_selection, "cut_selection"))
    norm_value = pick_dict_value(run_meta, (:normalization, "normalization"))
    inherit_value = pick_dict_value(run_meta, (:inherit_disjunctive_cuts, "inherit_disjunctive_cuts"))
    disj_value = pick_dict_value(run_meta, (:disjunction_iter, "disjunction_iter"))

    cut_selection = isnothing(cut_value) ? "NA" : string(cut_value)
    normalization = isnothing(norm_value) ? "NA" : string(norm_value)
    inherit_flag = isnothing(inherit_value) ? "na" : (Bool(inherit_value) ? "inh" : "noinh")
    disj_iter = isnothing(disj_value) ? "NA" : string(disj_value)
    return string(cut_selection, " | ", normalization, " | ", inherit_flag, " | D=", disj_iter, " | ", filename[1:min(end, 28)])
end

"""
    load_run(path::String)::Union{Nothing, NamedTuple}

# Purpose
    Load one result file and normalize fields used for plotting.
"""
function load_run(path::String)::Union{Nothing, NamedTuple}
    data = load(path)
    sddp_results = pick_dict_value(data, ("sddp_results", "sddpResults", :sddp_results, :sddpResults))
    run_meta = pick_dict_value(data, ("run_meta", :run_meta))
    if isnothing(sddp_results) || isnothing(run_meta)
        return nothing
    end

    sol_history = pick_dict_value(sddp_results, (:sol_history, :solHistory, "sol_history", "solHistory"))
    if isnothing(sol_history) || !(sol_history isa DataFrame) || nrow(sol_history) == 0
        return nothing
    end

    iter_col = "Iter" in names(sol_history) ? "Iter" : "iter"
    lb_col = "LB" in names(sol_history) ? "LB" : "lb"
    ub_col = "UB" in names(sol_history) ? "UB" : "ub"
    gap_col = "gap" in names(sol_history) ? "gap" : ("Gap" in names(sol_history) ? "Gap" : nothing)

    iter = to_float_vector(sol_history[!, iter_col])
    lb = to_float_vector(sol_history[!, lb_col])
    ub = ub_col in names(sol_history) ? to_float_vector(sol_history[!, ub_col]) : fill(NaN, nrow(sol_history))
    time_axis = infer_time_axis(sol_history)
    gap_raw = isnothing(gap_col) ? fill(NaN, nrow(sol_history)) : parse_gap_values(sol_history[!, gap_col])

    best_lb = accumulate(max, lb)
    best_ub = all(isnan.(ub)) ? fill(NaN, nrow(sol_history)) : accumulate(min, ub)
    gap_recomputed = Vector{Float64}(undef, nrow(sol_history))
    for idx in eachindex(gap_recomputed)
        if isnan(best_ub[idx]) || abs(best_ub[idx]) <= 1e-9
            gap_recomputed[idx] = NaN
        else
            gap_recomputed[idx] = 100.0 * (best_ub[idx] - best_lb[idx]) / abs(best_ub[idx])
        end
    end

    label = make_run_label(path, run_meta)
    return (
        path = path,
        run_meta = run_meta,
        sol_history = sol_history,
        iter = iter,
        time = time_axis,
        lb = lb,
        ub = ub,
        best_lb = best_lb,
        best_ub = best_ub,
        gap_raw = gap_raw,
        gap_recomputed = gap_recomputed,
        label = label,
    )
end

"""
    summarize_run(run)::NamedTuple

# Purpose
    Build summary metrics for one run curve.
"""
function summarize_run(run)::NamedTuple
    best_lb_value, best_lb_idx = findmax(run.best_lb)
    best_ub_value = run.best_ub[best_lb_idx]
    final_gap = run.gap_recomputed[end]
    return (
        label = run.label,
        final_iter = Int(round(run.iter[end])),
        final_time = run.time[end],
        final_lb = run.best_lb[end],
        final_ub = run.best_ub[end],
        final_gap = final_gap,
        best_lb = best_lb_value,
        time_to_best_lb = run.time[best_lb_idx],
        iter_to_best_lb = Int(round(run.iter[best_lb_idx])),
    )
end

"""
    sanitize_filename(name::String)::String

# Purpose
    Convert labels into safe output filenames.
"""
function sanitize_filename(name::String)::String
    out = replace(name, r"[^A-Za-z0-9_\\-]+" => "_")
    out = replace(out, r"_+" => "_")
    return strip(out, '_')
end

"""
    save_plots_for_dataset(runs, dataset_name, config)

# Purpose
    Create LB and gap plots plus summary CSV for one dataset.
"""
function save_plots_for_dataset(
    runs::Vector{NamedTuple},
    dataset_name::String,
    config::NamedTuple,
)::Nothing
    isempty(runs) && return

    sort!(runs, by = r -> r.label)
    dataset_key = sanitize_filename(dataset_name)
    output_dir = joinpath(config.output_root, config.project, dataset_key)
    mkpath(output_dir)

    palette = [
        :steelblue,
        :darkorange,
        :seagreen,
        :purple,
        :goldenrod,
        :cadetblue,
        :firebrick,
        :gray40,
    ]

    x_label = config.x_axis == :time ? "Runtime (s)" : "Iteration"
    lb_title = uppercase(config.project) * " | " * dataset_name * " | Lower Bound"
    gap_title = uppercase(config.project) * " | " * dataset_name * " | Gap (%)"
    lb_plot = plot(
        title = lb_title,
        xlabel = x_label,
        ylabel = "Lower Bound",
        legend = config.legend_position,
        linewidth = config.line_width,
        size = config.size,
        dpi = config.dpi,
        grid = true,
        titlefont = font(config.title_font),
        guidefont = font(config.guide_font),
        tickfont = font(config.tick_font),
    )

    gap_plot = plot(
        title = gap_title,
        xlabel = x_label,
        ylabel = "Gap (%)",
        legend = config.legend_position,
        linewidth = config.line_width,
        size = config.size,
        dpi = config.dpi,
        grid = true,
        titlefont = font(config.title_font),
        guidefont = font(config.guide_font),
        tickfont = font(config.tick_font),
    )

    summaries = NamedTuple[]
    for (idx, run) in enumerate(runs)
        color = palette[(idx - 1) % length(palette) + 1]
        x_axis = config.x_axis == :time ? run.time : run.iter
        lb_series = config.lb_mode == :best ? run.best_lb : run.lb
        gap_series = config.recompute_gap ? run.gap_recomputed : run.gap_raw

        plot!(lb_plot, x_axis, lb_series; label = run.label, color = color)
        plot!(gap_plot, x_axis, gap_series; label = run.label, color = color)
        push!(summaries, summarize_run(run))
    end

    lb_path = joinpath(output_dir, "lb_curve." * config.format)
    gap_path = joinpath(output_dir, "gap_curve." * config.format)
    savefig(lb_plot, lb_path)
    savefig(gap_plot, gap_path)

    summary_df = DataFrame(summaries)
    csv_path = joinpath(output_dir, "curve_summary.csv")
    CSV.write(csv_path, summary_df)

    println("Saved dataset: " * dataset_name)
    println("  LB plot: " * lb_path)
    println("  Gap plot: " * gap_path)
    println("  Summary: " * csv_path)
end

"""
    run_plotting(config::NamedTuple)::Nothing

# Purpose
    Orchestrate data collection and plotting.
"""
function run_plotting(config::NamedTuple)::Nothing
    grouped = list_result_files(config)
    if isempty(grouped)
        println("No result files matched filters.")
        return
    end

    for dataset_name in sort(collect(keys(grouped)))
        runs = NamedTuple[]
        for path in grouped[dataset_name]
            loaded = load_run(path)
            if !isnothing(loaded)
                push!(runs, loaded)
            end
        end
        if isempty(runs)
            println("Skipped dataset (no loadable runs): " * dataset_name)
            continue
        end
        save_plots_for_dataset(runs, dataset_name, config)
    end
end

function print_usage()::Nothing
    println("Usage:")
    println("  julia --project=. experiments/plotting/plot_results.jl [options]")
    println("")
    println("Options:")
    println("  --project <scuc|generation_expansion|sslp>")
    println("  --dataset-filter <regex>      (default: .* )")
    println("  --run-filter <regex>          (default: .* )")
    println("  --exclude-filter <regex>      (default: a^ )")
    println("  --x-axis <time|iter>          (default: time)")
    println("  --lb-mode <best|raw>          (default: best)")
    println("  --recompute-gap <true|false>  (default: true)")
    println("  --width <int>                 (default: 1200)")
    println("  --height <int>                (default: 700)")
    println("  --dpi <int>                   (default: 300)")
    println("  --format <pdf|png>            (default: pdf)")
    println("  --output-root <path>          (default: results/figures)")
end

function main()::Nothing
    cli = parse_cli_args(ARGS)
    if haskey(cli, "help")
        print_usage()
        return
    end

    config = make_config(cli)
    println("Plot config:")
    println("  project        = " * config.project)
    println("  dataset-filter = " * string(config.dataset_filter))
    println("  run-filter     = " * string(config.run_filter))
    println("  x-axis         = " * string(config.x_axis))
    println("  lb-mode        = " * string(config.lb_mode))
    println("  recompute-gap  = " * string(config.recompute_gap))
    println("  output-root    = " * config.output_root)

    run_plotting(config)
end

main()
