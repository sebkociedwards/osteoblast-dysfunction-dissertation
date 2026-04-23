using CSV
using DataFrames
using Statistics
using CairoMakie
using HypothesisTests
using Random

# =========================
# Paths
# =========================

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW_DIR = joinpath(ROOT, "raw_data")
const PROC_DIR = joinpath(ROOT, "processed_data")
const FIG_DIR = joinpath(ROOT, "figures")

mkpath(PROC_DIR)
mkpath(FIG_DIR)

const INPUT_FILE = joinpath(RAW_DIR, "osteoblast_qpcr_raw_actb_only.csv")

const OUT_TECH = joinpath(PROC_DIR, "tech_replicate_summary.csv")
const OUT_SAMPLE = joinpath(PROC_DIR, "sample_level_dct.csv")
const OUT_DELTA = joinpath(PROC_DIR, "delta_log2_fc.csv")
const OUT_STATS = joinpath(PROC_DIR, "gene_condition_stats.csv")

const OUT_PDF = joinpath(FIG_DIR, "figure6_osteoblast_marker_qpcr.pdf")
const OUT_PNG = joinpath(FIG_DIR, "figure6_osteoblast_marker_qpcr.png")

# =========================
# Constants
# =========================

const REF_GENE = "ACTB"

const GENE_ORDER = [
    "ALPL",
    "BGLAP",
    "COL1A1",
    "OPG",
    "RANKL",
    "SPP1",
]

const CONDITION_ORDER = [
    "IS2_SFN0",
    "IS0_SFN1",
    "IS0_SFN5",
    "IS2_SFN1",
    "IS2_SFN5",
]

const CONDITION_LABELS = Dict(
    "IS2_SFN0" => "IS 2 mM",
    "IS0_SFN1" => "SFN 1 µM",
    "IS0_SFN5" => "SFN 5 µM",
    "IS2_SFN1" => "IS 2 mM + SFN 1 µM",
    "IS2_SFN5" => "IS 2 mM + SFN 5 µM",
)

const POINT_COLOUR = colorant"#69B38A"
const MEAN_COLOUR = colorant"#2F6B4F"
const GRID_COLOUR = RGBAf(0, 0, 0, 0.10)
const AXIS_COLOUR = RGBAf(0, 0, 0, 0.70)
const BG_COLOUR = :white

const RAW_MARKER_SIZE = 8
const MEAN_MARKER_SIZE = 12
const ERRORBAR_LINEWIDTH = 1.8
const REF_LINEWIDTH = 1.2

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
            figure_padding = 8,
            Axis = (
                xgridvisible = false,
                ygridvisible = true,
                ygridcolor = GRID_COLOUR,
                ygridwidth = 1.0,
                spinewidth = 1.2,
                xtickwidth = 1.2,
                ytickwidth = 1.2,
                leftspinevisible = true,
                bottomspinevisible = true,
                rightspinevisible = false,
                topspinevisible = false,
            ),
        )
    )
end

# =========================
# Helpers
# =========================

parse_bool(x) = ismissing(x) ? false : (x === true || x == 1 || lowercase(strip(string(x))) == "true")

function sem(v::AbstractVector{<:Real})
    n = length(v)
    n <= 1 && return 0.0
    return std(v) / sqrt(n)
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

function display_gene_label(g::AbstractString)
    s = uppercase(strip(String(g)))
    mapping = Dict(
        "ALPL" => "Alpl",
        "BGLAP" => "Bglap",
        "COL1A1" => "Col1a1",
        "OPG" => "Tnfrsf11b",
        "RANKL" => "Tnfsf11",
        "SPP1" => "Spp1",
    )
    return get(mapping, s, uppercase(first(lowercase(s))) * lowercase(s)[2:end])
end

italic_gene_label(g::AbstractString) = rich(display_gene_label(g), font = "Times New Roman Italic")

# =========================
# Load + clean
# =========================

raw = CSV.read(INPUT_FILE, DataFrame)
rename!(raw, Symbol.(strip.(string.(names(raw)))))

raw.sample_id = strip.(string.(raw.sample_id))
raw.sample_type = uppercase.(strip.(string.(raw.sample_type)))
raw.condition = uppercase.(strip.(string.(raw.condition)))
raw.sex = strip.(string.(raw.sex))
raw.biol_rep = strip.(string.(raw.biol_rep))
raw.gene = uppercase.(strip.(string.(raw.gene)))
raw.exclude = [parse_bool(x) for x in raw.exclude]

raw = filter(r -> r.sample_type == "SAMPLE" && !r.exclude, raw)

# =========================
# Technical replicate summary
# =========================

tech = combine(
    groupby(raw, [:sample_id, :condition, :sex, :biol_rep, :gene]),
    :ct => length => :n_tech,
    :ct => mean => :mean_cq,
    :ct => (v -> length(v) > 1 ? std(v) : 0.0) => :sd_cq,
    :ct => (v -> maximum(v) - minimum(v)) => :spread_cq,
)

sort!(tech, [:gene, :condition, :sample_id])
CSV.write(OUT_TECH, tech)

# =========================
# Sample-level ΔCt
# =========================

refs = filter(:gene => ==(REF_GENE), tech)
refs = select(refs, :sample_id, :condition, :sex, :biol_rep, :mean_cq)
rename!(refs, :mean_cq => :ref_cq)

targets = filter(r -> r.gene != REF_GENE && r.gene in GENE_ORDER, tech)
sample_dct = leftjoin(targets, refs, on = [:sample_id, :condition, :sex, :biol_rep])

sample_dct = filter(r -> !ismissing(r.ref_cq), sample_dct)
sample_dct.delta_ct = sample_dct.mean_cq .- sample_dct.ref_cq

sort!(sample_dct, [:gene, :condition, :sex, :biol_rep, :sample_id])
CSV.write(OUT_SAMPLE, sample_dct)

# =========================
# Matched replicate-level deltas
# log2 fold change = -ΔΔCt
# =========================

function build_delta_table(a::DataFrame)
    controls = filter(:condition => ==("CON"), a)
    treated = filter(r -> r.condition in CONDITION_ORDER, a)

    out = DataFrame(
        gene = String[],
        condition = String[],
        sex = String[],
        biol_rep = String[],
        control_sample_id = String[],
        treated_sample_id = String[],
        control_delta_ct = Float64[],
        treated_delta_ct = Float64[],
        delta_delta_ct = Float64[],
        log2_fc = Float64[],
        fold_change = Float64[],
    )

    for row in eachrow(treated)
        ctrl = filter(r ->
            r.gene == row.gene &&
            r.sex == row.sex &&
            r.biol_rep == row.biol_rep &&
            r.condition == "CON",
            controls
        )

        nrow(ctrl) == 1 || continue

        con_dct = Float64(ctrl[1, :delta_ct])
        trt_dct = Float64(row.delta_ct)
        ddct = trt_dct - con_dct
        log2fc = -ddct

        push!(out, (
            row.gene,
            row.condition,
            row.sex,
            row.biol_rep,
            String(ctrl[1, :sample_id]),
            String(row.sample_id),
            con_dct,
            trt_dct,
            ddct,
            log2fc,
            2.0 ^ log2fc,
        ))
    end

    sort!(out, [:gene, :condition, :sex, :biol_rep])
    return out
end

delta_df = build_delta_table(sample_dct)
CSV.write(OUT_DELTA, delta_df)

# =========================
# Stats by gene × condition
# =========================

function gene_condition_stats(df::DataFrame, gene_order::Vector{String}, condition_order::Vector{String})
    out = DataFrame(
        gene = String[],
        condition = String[],
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

    for g in gene_order
        for c in condition_order
            sdf = filter(r -> r.gene == g && r.condition == c, df)
            vals = Float64.(collect(sdf.log2_fc))
            n = length(vals)
            n == 0 && continue

            mean_val = mean(vals)
            sd_val = n > 1 ? std(vals) : NaN
            sem_val = n > 1 ? sem(vals) : NaN
            tt = onesample_ttest_summary(vals)
            sh = shapiro_summary(vals)
            dz = cohens_dz(vals)

            push!(out, (
                g,
                c,
                n,
                mean_val,
                sd_val,
                sem_val,
                tt.ci_low,
                tt.ci_high,
                2.0 ^ mean_val,
                dz,
                tt.t_stat,
                tt.df,
                tt.p_value,
                sh.w_stat,
                sh.p_value,
            ))
        end
    end

    out.bh_p_value = bh_adjust(out.p_value)
    return out
end

stats_table = gene_condition_stats(delta_df, GENE_ORDER, CONDITION_ORDER)
CSV.write(OUT_STATS, stats_table)

println("\nGene × condition stats")
show(stats_table, allrows = true, allcols = true)
println()

# =========================
# Plot helpers
# =========================

function summarise_panel(df::DataFrame)
    g = combine(groupby(df, :condition)) do sdf
        vals = Float64.(collect(skipmissing(sdf.log2_fc)))
        n = length(vals)
        mean_val = mean(vals)
        sem_val = n > 1 ? sem(vals) : 0.0
        (; mean_log2_fc = mean_val, sem_log2_fc = sem_val, n = n)
    end

    order_map = Dict(c => i for (i, c) in enumerate(CONDITION_ORDER))
    g.order = [get(order_map, r, typemax(Int)) for r in g.condition]
    sort!(g, :order)
    select!(g, Not(:order))
    return g
end

function add_integer_gridlines!(ax::Axis)
    yt = Makie.get_tickvalues(ax.yticks[], ax.scene.px_area[][2], ax.finallimits[].y)
    for y in yt
        if isfinite(y) && abs(y - round(y)) < 1e-8 && y != 0
            hlines!(ax, [y], color = GRID_COLOUR, linewidth = 1.0)
        end
    end
end

function panel_ylim(df::DataFrame)
    vals = Float64.(collect(df.log2_fc))
    isempty(vals) && return (-1, 1)
    lo = floor(minimum(vals) - 0.5)
    hi = ceil(maximum(vals) + 0.5)
    lo == hi && (hi = lo + 1)
    return (lo, hi)
end

function plot_gene_panel!(ax::Axis, df::DataFrame, gene::String)
    sdf = filter(:gene => ==(gene), df)
    summary = summarise_panel(sdf)
    xpos = Dict(c => i for (i, c) in enumerate(CONDITION_ORDER))

    Random.seed!(1)
    jitter = 0.08

    for c in CONDITION_ORDER
        cdf = filter(:condition => ==(c), sdf)
        nrow(cdf) == 0 && continue
        x0 = xpos[c]
        xs = [x0 + (rand() - 0.5) * 2 * jitter for _ in 1:nrow(cdf)]

        scatter!(
            ax,
            xs,
            cdf.log2_fc,
            markersize = RAW_MARKER_SIZE,
            color = POINT_COLOUR,
        )
    end

    for row in eachrow(summary)
        x = xpos[row.condition]
        y = row.mean_log2_fc
        e = row.sem_log2_fc

        errorbars!(
            ax,
            [x], [y], [e],
            color = MEAN_COLOUR,
            linewidth = ERRORBAR_LINEWIDTH,
            whiskerwidth = 8,
        )

        scatter!(
            ax,
            [x], [y],
            color = MEAN_COLOUR,
            marker = :diamond,
            markersize = MEAN_MARKER_SIZE,
        )
    end

    ax.title = italic_gene_label(gene)
    ax.titlesize = 11
    ax.xticks = (1:length(CONDITION_ORDER), [CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.xticklabelrotation = π / 8
    ax.xticklabelsize = 9
    ax.yticklabelsize = 10
    ax.ylabel = ""
    ax.ylabelsize = 10
    ylims!(ax, panel_ylim(sdf)...)
    hlines!(ax, [0.0], linestyle = :dash, color = AXIS_COLOUR, linewidth = REF_LINEWIDTH)
end

# =========================
# Make figure
# =========================

function make_figure(df::DataFrame)
    dissertation_theme!()

    fig = Figure(size = (700, 900), backgroundcolor = BG_COLOUR)

    axes = [
        Axis(fig[1, 1]),
        Axis(fig[1, 2]),
        Axis(fig[2, 1]),
        Axis(fig[2, 2]),
        Axis(fig[3, 1]),
        Axis(fig[3, 2]),
    ]

    for (i, (ax, gene)) in enumerate(zip(axes, GENE_ORDER))
        plot_gene_panel!(ax, df, gene)

        # Show y-axis label only on the left column; keep spacing aligned on the right.
        if i in (1, 3, 5)
            ax.ylabel = "log₂ fold change (vs control)"
        else
            ax.ylabel = "log₂ fold change (vs control)"
            ax.ylabelcolor = RGBAf(0, 0, 0, 0)
        end
    end

    return fig
end

fig = make_figure(delta_df)

save(OUT_PDF, fig)
save(OUT_PNG, fig, px_per_unit = 4)

display(fig)
