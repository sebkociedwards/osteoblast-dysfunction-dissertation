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

const INPUT_FILE = joinpath(RAW_DIR, "oxidative_stress_qpcr_data.csv")

const OUT_TECH = joinpath(PROC_DIR, "tech_replicate_summary.csv")
const OUT_SAMPLE = joinpath(PROC_DIR, "sample_level_expression.csv")
const OUT_DELTA = joinpath(PROC_DIR, "delta_log2_fc.csv")
const OUT_STATS = joinpath(PROC_DIR, "gene_stats.csv")

const OUT_PDF = joinpath(FIG_DIR, "figure5_oxidative_stress_qpcr.pdf")
const OUT_PNG = joinpath(FIG_DIR, "figure5_oxidative_stress_qpcr.png")

# =========================
# Figure settings
# =========================

const POINT_COLOUR = colorant"#69B38A"
const MEAN_COLOUR = colorant"#2F6B4F"
const GRID_COLOUR = RGBAf(0, 0, 0, 0.10)
const AXIS_COLOUR = RGBAf(0, 0, 0, 0.70)
const BG_COLOUR = :white

const RAW_MARKER_SIZE = 9
const MEAN_MARKER_SIZE = 14
const ERRORBAR_LINEWIDTH = 2.0
const REF_LINEWIDTH = 1.3

const GENE_ORDER = [
    "CAT",
    "GCLC",
    "GCLM",
    "GSR",
    "GSS",
    "HMOX1",
    "NQO1",
    "PTGS1",
    "SQSTM1",
    "SRXN1",
]

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

geommean(v::AbstractVector{<:Real}) = exp(mean(log.(v)))

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
    s = lowercase(strip(String(g)))
    isempty(s) && return s
    return uppercase(first(s)) * s[2:end]
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

# Keep only biological samples, exclude flagged wells
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
# Sample-level normalised expression
# Paper-aligned:
# control mean Cq per gene -> ΔCq = control_mean - sample_mean
# RQ = 2^ΔCq
# NF from geometric mean of ACTB/RPL4 RQ within sample
# =========================

const REF_GENES = Set(["ACTB", "RPL4"])
const CONTROL = "CON"
const CASE = "IS"

# control-group mean Cq per gene
ctrl = filter(r -> r.condition == CONTROL, tech)
ctrl_means = combine(groupby(ctrl, :gene), :mean_cq => mean => :control_mean_cq)

expr = leftjoin(tech, ctrl_means, on = :gene)
expr = filter(r -> !ismissing(r.control_mean_cq), expr)

expr.delta_cq = expr.control_mean_cq .- expr.mean_cq
expr.rq = 2.0 .^ expr.delta_cq

# sample-level NF from reference genes
refrows = filter(r -> r.gene in REF_GENES, expr)
nf = combine(
    groupby(refrows, [:sample_id, :condition, :sex, :biol_rep]),
    :rq => geommean => :norm_factor,
    :gene => length => :n_refs,
)
nf = filter(r -> r.n_refs == length(REF_GENES), nf)

targets = filter(r -> !(r.gene in REF_GENES), expr)
sample_expr = leftjoin(targets, nf, on = [:sample_id, :condition, :sex, :biol_rep])
sample_expr = filter(r -> !ismissing(r.norm_factor), sample_expr)

sample_expr.norm_expr = sample_expr.rq ./ sample_expr.norm_factor
sample_expr.log2_norm_expr = log2.(sample_expr.norm_expr)

sort!(sample_expr, [:gene, :condition, :sex, :biol_rep, :sample_id])
CSV.write(OUT_SAMPLE, sample_expr)

# =========================
# Matched replicate-level deltas
# =========================

function build_delta_table(a::DataFrame)
    controls = filter(:condition => ==(CONTROL), a)
    cases = filter(:condition => ==(CASE), a)

    out = DataFrame(
        gene = String[],
        sex = String[],
        biol_rep = String[],
        control_sample_id = String[],
        case_sample_id = String[],
        control_log2_expr = Float64[],
        case_log2_expr = Float64[],
        log2_fc = Float64[],
        fold_change = Float64[],
    )

    for row in eachrow(cases)
        ctrl = filter(r ->
            r.gene == row.gene &&
            r.sex == row.sex &&
            r.biol_rep == row.biol_rep,
            controls
        )

        nrow(ctrl) == 1 || continue

        con_val = Float64(ctrl[1, :log2_norm_expr])
        case_val = Float64(row.log2_norm_expr)
        diff = case_val - con_val

        push!(out, (
            row.gene,
            row.sex,
            row.biol_rep,
            String(ctrl[1, :sample_id]),
            String(row.sample_id),
            con_val,
            case_val,
            diff,
            2.0 ^ diff,
        ))
    end

    sort!(out, [:gene, :sex, :biol_rep])
    return out
end

delta_df = build_delta_table(sample_expr)
delta_df = filter(r -> r.gene in GENE_ORDER, delta_df)
CSV.write(OUT_DELTA, delta_df)

# =========================
# Gene-wise stats
# =========================

function gene_stats(df::DataFrame, gene_order::Vector{String})
    out = DataFrame(
        gene = String[],
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
        sdf = filter(:gene => ==(g), df)
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

    out.bh_p_value = bh_adjust(out.p_value)
    return out
end

stats_table = gene_stats(delta_df, GENE_ORDER)
CSV.write(OUT_STATS, stats_table)

println("\nGene-wise stats")
show(stats_table, allrows = true, allcols = true)
println()

# =========================
# Plot
# =========================

function summarise_by_gene(df::DataFrame, gene_order::Vector{String})
    g = combine(groupby(df, :gene)) do sdf
        vals = Float64.(collect(skipmissing(sdf.log2_fc)))
        n = length(vals)
        mean_val = mean(vals)
        sem_val = n > 1 ? sem(vals) : 0.0
        (; mean_log2_fc = mean_val, sem_log2_fc = sem_val, n = n)
    end

    order_map = Dict(g => i for (i, g) in enumerate(gene_order))
    g.order = [get(order_map, r, typemax(Int)) for r in g.gene]
    sort!(g, :order)
    select!(g, Not(:order))
    return g
end

function plot_oxidative_stress(df::DataFrame; gene_order::Vector{String})
    dissertation_theme!()

    summary = summarise_by_gene(df, gene_order)
    xpos = Dict(g => i for (i, g) in enumerate(gene_order))

    fig = Figure(size = (700, 400), backgroundcolor = BG_COLOUR)
    ax = Axis(
        fig[1, 1],
        xlabel = "",
        ylabel = "log₂ fold change (vs control)",
        xticklabelsize = 10,
        yticklabelsize = 11,
        ylabelsize = 11,
        leftspinecolor = AXIS_COLOUR,
        bottomspinecolor = AXIS_COLOUR,
        xtickcolor = AXIS_COLOUR,
        ytickcolor = AXIS_COLOUR,
        xticklabelcolor = AXIS_COLOUR,
        yticklabelcolor = AXIS_COLOUR,
        ylabelcolor = AXIS_COLOUR,
    )

    Random.seed!(1)
    jitter = 0.10

    for g in gene_order
        sdf = filter(:gene => ==(g), df)
        nrow(sdf) == 0 && continue
        x0 = xpos[g]
        xs = [x0 + (rand() - 0.5) * 2 * jitter for _ in 1:nrow(sdf)]

        scatter!(
            ax,
            xs,
            sdf.log2_fc,
            markersize = RAW_MARKER_SIZE,
            color = POINT_COLOUR,
        )
    end

    for row in eachrow(summary)
        x = xpos[row.gene]
        y = row.mean_log2_fc
        e = row.sem_log2_fc

        errorbars!(
            ax,
            [x], [y], [e],
            color = MEAN_COLOUR,
            linewidth = ERRORBAR_LINEWIDTH,
            whiskerwidth = 10,
        )

        scatter!(
            ax,
            [x], [y],
            color = MEAN_COLOUR,
            marker = :diamond,
            markersize = MEAN_MARKER_SIZE,
        )
    end

    ax.xticks = (1:length(gene_order), [italic_gene_label(g) for g in gene_order])
    ax.xticklabelrotation = π / 8
    hlines!(ax, [0.0], linestyle = :dash, color = AXIS_COLOUR, linewidth = REF_LINEWIDTH)

    return fig
end

fig = plot_oxidative_stress(delta_df; gene_order = GENE_ORDER)

save(OUT_PDF, fig)
save(OUT_PNG, fig, px_per_unit = 4)

display(fig)
