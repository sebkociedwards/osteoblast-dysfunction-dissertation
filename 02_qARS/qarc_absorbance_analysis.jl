using CSV
using DataFrames
using Statistics
using CairoMakie
using HypothesisTests

# =========================
# Configuration
# =========================

const ROOT = normpath(joinpath(@__DIR__, ".."))

const INPUT_FILE = joinpath(ROOT, "raw_data", "absorbance_output", "raw_absorbance.csv")
const OUT_DIR    = joinpath(ROOT, "processed_data")
const FIG_DIR    = joinpath(ROOT, "figures")

mkpath(OUT_DIR)
mkpath(FIG_DIR)

const OUT_PDF = joinpath(FIG_DIR, "qars_delta_plot.pdf")
const OUT_PNG = joinpath(FIG_DIR, "qars_delta_plot.png")

const TREATMENT_ORDER = [
    "IS 2 mM",
    "SFN 1 µM",
    "SFN 5 µM",
    "IS 2 mM + SFN 1 µM",
    "IS 2 mM + SFN 5 µM",
]

const YLIMS = (-2.5, 1.0)

# =========================
# Theme
# =========================

function dissertation_theme!()
    set_theme!(
        Theme(
            fontsize = 12,
            fonts = (
                regular = "Times New Roman",
                bold = "Times New Roman Bold",
                italic = "Times New Roman Italic",
                bold_italic = "Times New Roman Bold Italic",
            ),
            figure_padding = 10,
            Axis = (
                xgridvisible = false,
                ygridvisible = true,
                ygridcolor = RGBAf(0, 0, 0, 0.08),
                ygridwidth = 1.0,
                spinewidth = 1.2,
                xtickwidth = 1.2,
                ytickwidth = 1.2,
                xticklabelsize = 10,
                yticklabelsize = 11,
                xlabelsize = 11,
                ylabelsize = 11,
                leftspinevisible = true,
                bottomspinevisible = true,
                rightspinevisible = false,
                topspinevisible = false,
            ),
        )
    )
end

const POINT_COLOUR = colorant"#69B38A"
const MEAN_COLOUR  = colorant"#2F6B4F"
const POINT_SIZE = 13
const MEAN_MARKER_SIZE = 18
const ERRORBAR_LINEWIDTH = 2.2
const ZERO_LINE_COLOUR = RGBAf(0, 0, 0, 0.45)

# =========================
# Helpers
# =========================

function treatment_label(condition_code::AbstractString)
    mapping = Dict(
        "is0_sfn0" => "CON",
        "is2_sfn0" => "IS 2 mM",
        "is0_sfn1" => "SFN 1 µM",
        "is0_sfn5" => "SFN 5 µM",
        "is2_sfn1" => "IS 2 mM + SFN 1 µM",
        "is2_sfn5" => "IS 2 mM + SFN 5 µM",
    )
    haskey(mapping, condition_code) || error("Unknown condition code: $condition_code")
    return mapping[condition_code]
end

function safe_log2fc(treat::Real, ctrl::Real; pseudocount::Real = 1e-6)
    return log2((treat + pseudocount) / (ctrl + pseudocount))
end

function cohens_dz(diffs::AbstractVector{<:Real})
    d = Float64.(collect(diffs))
    n = length(d)
    n < 2 && return NaN
    s = std(d)
    s == 0 && return NaN
    return mean(d) / s
end

function onesample_ttest_summary(diffs::AbstractVector{<:Real}; μ0::Real = 0.0)
    d = Float64.(collect(diffs))
    n = length(d)

    if n < 2
        return (
            t_stat = NaN,
            df = n - 1,
            p_value = NaN,
            ci_low = NaN,
            ci_high = NaN,
        )
    end

    test = OneSampleTTest(d, μ0)
    ci = confint(test)

    return (
        t_stat = test.t,
        df = test.df,
        p_value = pvalue(test),
        ci_low = ci[1],
        ci_high = ci[2],
    )
end

function shapiro_summary(diffs::AbstractVector{<:Real})
    d = Float64.(collect(diffs))
    n = length(d)

    if n < 3
        return (w_stat = NaN, p_value = NaN)
    end

    test = ShapiroWilkTest(d)
    return (w_stat = test.W, p_value = pvalue(test))
end

function bh_adjust(pvals::Vector{Float64})
    m = length(pvals)
    order = sortperm(pvals)
    ranked = pvals[order]
    adj = similar(ranked)

    for i in 1:m
        adj[i] = ranked[i] * m / i
    end

    for i in (m - 1):-1:1
        adj[i] = min(adj[i], adj[i + 1])
    end

    adj = clamp.(adj, 0.0, 1.0)

    out = similar(adj)
    out[order] = adj
    return out
end

# =========================
# Load and summarise
# =========================

raw = CSV.read(INPUT_FILE, DataFrame)
rename!(raw, Symbol.(strip.(string.(names(raw)))))

required_cols = [
    :sample_id,
    :condition,
    :sex,
    :batch,
    :is_mM,
    :sfn_uM,
    :replicate,
    :delta_absorbance_405,
]

found_cols = Set(Symbol.(names(raw)))
for c in required_cols
    c in found_cols || error("Missing required column: $c. Found: $(names(raw))")
end

raw.treatment = [treatment_label(c) for c in raw.condition]

sample_condition_summary = combine(
    groupby(raw, [:sample_id, :sex, :batch, :condition, :treatment, :is_mM, :sfn_uM]),
    :delta_absorbance_405 => mean => :mean_absorbance_405,
    :delta_absorbance_405 => std  => :sd_absorbance_405,
    :delta_absorbance_405 => length => :n_wells,
)

sort!(sample_condition_summary, [:sample_id, :condition])

# =========================
# Delta table
# =========================

function build_delta_table(sample_summary::DataFrame)
    controls = filter(:condition => ==("is0_sfn0"), sample_summary)
    treated  = filter(:condition => !=("is0_sfn0"), sample_summary)

    out = DataFrame(
        sample_id = String[],
        sex = String[],
        batch = String[],
        treatment = String[],
        condition = String[],
        control_value = Float64[],
        treatment_value = Float64[],
        log2_fc = Float64[],
    )

    for row in eachrow(treated)
        ctrl = filter(r -> r.sample_id == row.sample_id, controls)
        nrow(ctrl) == 1 || continue

        ctrl_val = Float64(ctrl[1, :mean_absorbance_405])
        treat_val = Float64(row.mean_absorbance_405)

        push!(out, (
            row.sample_id,
            row.sex,
            row.batch,
            row.treatment,
            row.condition,
            ctrl_val,
            treat_val,
            safe_log2fc(treat_val, ctrl_val),
        ))
    end

    sort!(out, [:treatment, :sample_id])
    return out
end

delta_df = build_delta_table(sample_condition_summary)

# =========================
# Statistics
# =========================

function summarise_delta_stats(df::DataFrame, metric_name::String)
    out = DataFrame(
        metric = String[],
        treatment = String[],
        n = Int[],
        mean_log2_fc = Float64[],
        sd_log2_fc = Float64[],
        sem_log2_fc = Float64[],
        ci_low_log2_fc = Float64[],
        ci_high_log2_fc = Float64[],
        fold_change = Float64[],
        cohens_dz = Float64[],
        t_statistic = Float64[],
        t_df = Float64[],
        p_value = Float64[],
        shapiro_w = Float64[],
        shapiro_p_value = Float64[],
    )

    for t in TREATMENT_ORDER
        sdf = filter(:treatment => ==(t), df)
        vals = Float64.(collect(sdf.log2_fc))
        n = length(vals)

        mean_val = mean(vals)
        sd_val = n > 1 ? std(vals) : NaN
        sem_val = n > 1 ? sd_val / sqrt(n) : NaN

        tt = onesample_ttest_summary(vals)
        sh = shapiro_summary(vals)
        fc = 2.0 ^ mean_val
        dz = cohens_dz(vals)

        push!(out, (
            metric_name,
            t,
            n,
            mean_val,
            sd_val,
            sem_val,
            tt.ci_low,
            tt.ci_high,
            fc,
            dz,
            tt.t_stat,
            tt.df,
            tt.p_value,
            sh.w_stat,
            sh.p_value,
        ))
    end

    out.bh_p_value = bh_adjust(out.p_value)
    out.ci_low_fold_change = 2.0 .^ out.ci_low_log2_fc
    out.ci_high_fold_change = 2.0 .^ out.ci_high_log2_fc
    return out
end

stats_table = summarise_delta_stats(delta_df, "qARS absorbance")

CSV.write(joinpath(OUT_DIR, "sample_condition_summary.csv"), sample_condition_summary)
CSV.write(joinpath(OUT_DIR, "delta_absorbance.csv"), delta_df)
CSV.write(joinpath(OUT_DIR, "qars_stats_table.csv"), stats_table)

println("\nqARS statistics")
show(stats_table, allrows = true, allcols = true)
println()

# =========================
# Plot helpers
# =========================

function treatment_positions(treatment_order::Vector{String})
    Dict(t => i for (i, t) in enumerate(treatment_order))
end

function summarise_by_treatment(df::DataFrame, treatment_order::Vector{String})
    g = combine(groupby(df, :treatment)) do sdf
        vals = Float64.(collect(skipmissing(sdf.log2_fc)))
        n = length(vals)
        mean_val = mean(vals)

        if n > 1
            tt = onesample_ttest_summary(vals)
            ci_low = tt.ci_low
            ci_high = tt.ci_high
        else
            ci_low = NaN
            ci_high = NaN
        end

        (; mean_log2_fc = mean_val, ci_low_log2_fc = ci_low, ci_high_log2_fc = ci_high, n = n)
    end

    order_map = Dict(t => i for (i, t) in enumerate(treatment_order))
    g.order = [get(order_map, t, typemax(Int)) for t in g.treatment]
    sort!(g, :order)
    select!(g, Not(:order))
    return g
end

function add_zero_line!(ax::Axis)
    hlines!(ax, [0.0], color = ZERO_LINE_COLOUR, linestyle = :dash, linewidth = 1.5)
end

function plot_delta!(
    ax::Axis,
    df::DataFrame;
    treatment_order::Vector{String},
    ylabel::AbstractString,
    ylimits = nothing,
    jitter_width::Float64 = 0.05,
)
    xpos = treatment_positions(treatment_order)
    summary = summarise_by_treatment(df, treatment_order)

    for t in treatment_order
        sdf = filter(:treatment => ==(t), df)
        if nrow(sdf) == 0
            continue
        end

        x0 = xpos[t]
        xvals = [x0 + randn() * jitter_width for _ in 1:nrow(sdf)]

        scatter!(
            ax,
            xvals,
            sdf.log2_fc,
            color = POINT_COLOUR,
            markersize = POINT_SIZE,
            strokewidth = 0,
        )
    end

    for row in eachrow(summary)
        x = xpos[row.treatment]
        y = row.mean_log2_fc
        lower = y - row.ci_low_log2_fc
        upper = row.ci_high_log2_fc - y

        errorbars!(
            ax,
            [x], [y], [lower], [upper],
            color = MEAN_COLOUR,
            linewidth = ERRORBAR_LINEWIDTH,
            whiskerwidth = 12,
        )

        scatter!(
            ax,
            [x], [y],
            color = MEAN_COLOUR,
            marker = :diamond,
            markersize = MEAN_MARKER_SIZE,
        )
    end

    ax.xticks = (1:length(treatment_order), treatment_order)
    ax.xticklabelrotation = π / 8
    ax.ylabel = ylabel
    add_zero_line!(ax)

    if ylimits !== nothing
        ylims!(ax, ylimits...)
    end

    return ax
end

# =========================
# Make figure
# =========================

dissertation_theme!()

fig = Figure(size = (700, 400), backgroundcolor = :white)
ax = Axis(fig[1, 1])

plot_delta!(
    ax,
    delta_df;
    treatment_order = TREATMENT_ORDER,
    ylabel = "log₂ fold change (vs control)",
    ylimits = YLIMS,
)

save(OUT_PDF, fig)
save(OUT_PNG, fig, px_per_unit = 2)

display(fig)
