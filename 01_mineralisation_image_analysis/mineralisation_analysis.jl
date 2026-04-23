using CSV
using DataFrames
using Statistics
using CairoMakie
using HypothesisTests
using Distributions

# =========================
# Configuration
# =========================

const ROOT = normpath(joinpath(@__DIR__, ".."))

const INPUT_DIR = joinpath(ROOT, "raw_data", "imagej_outputs")
const OUT_DIR   = joinpath(ROOT, "processed_data")
const FIG_DIR   = joinpath(ROOT, "figures", "main_text")

mkpath(OUT_DIR)
mkpath(FIG_DIR)

const OUT_PDF = joinpath(FIG_DIR, "figure2_mineralisation.pdf")
const OUT_PNG = joinpath(FIG_DIR, "figure2_mineralisation.png")

const EXCLUDED_SAMPLES = Set(["female_b"])
const PX_PER_MM2 = 497.0

const TREATMENT_ORDER = [
    "IS 2 mM",
    "SFN 1 µM",
    "SFN 5 µM",
    "IS 2 mM + SFN 1 µM",
    "IS 2 mM + SFN 5 µM",
]

const AREA_YLIMS   = (-7.8, 0.5)
const NODULE_YLIMS = (-6.2, 0.5)

# =========================
# Figure theme
# =========================

function dissertation_theme!()
    set_theme!(
        Theme(
            fontsize = 14,
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
                xticklabelsize = 14,
                yticklabelsize = 14,
                xlabelsize = 16,
                ylabelsize = 16,
                leftspinevisible = true,
                bottomspinevisible = true,
                rightspinevisible = false,
                topspinevisible = false,
            ),
        )
    )
end

const FIG_W = 1950
const FIG_H = 1040

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

function parse_condition_code(code::AbstractString)
    m = match(r"^is(\d+)_sfn(\d+)_r(\d+)$", code)
    m === nothing && error("Could not parse condition code: $code")

    is_mM = parse(Int, m.captures[1])
    sfn_uM = parse(Int, m.captures[2])
    rep = parse(Int, m.captures[3])

    condition_code = "is$(is_mM)_sfn$(sfn_uM)"
    treatment = treatment_label(condition_code)

    return (condition_code, treatment, is_mM, sfn_uM, rep)
end

function infer_sample_metadata(sample_id::AbstractString)
    parts = split(sample_id, "_")
    length(parts) == 2 || error("Unexpected sample_id format: $sample_id")
    sex = parts[1]
    replicate_letter = parts[2]
    return (sex, replicate_letter)
end

# =========================
# Data loading
# =========================

function load_imagej_particle_file(path::AbstractString)
    sample_id = replace(basename(path), ".csv" => "")
    sex, replicate_letter = infer_sample_metadata(sample_id)

    wide = CSV.read(path, DataFrame)
    sample_excluded = sample_id in EXCLUDED_SAMPLES

    out = DataFrame(
        sample_id = String[],
        sex = String[],
        replicate_letter = String[],
        raw_file = String[],
        raw_column_name = String[],
        condition_code = String[],
        treatment = String[],
        is_mM = Int[],
        sfn_uM = Int[],
        within_sample_replicate = Int[],
        particle_index = Int[],
        particle_area_px = Float64[],
        included_in_analysis = Bool[],
    )

    for col in names(wide)
        colname = String(col)
        condition_code, treatment, is_mM, sfn_uM, rep = parse_condition_code(colname)

        vals = wide[!, col]
        particle_counter = 0

        for v in vals
            if ismissing(v)
                continue
            end

            s = strip(string(v))
            isempty(s) && continue

            x = tryparse(Float64, s)
            x === nothing && continue

            particle_counter += 1

            push!(out, (
                sample_id,
                sex,
                replicate_letter,
                basename(path),
                colname,
                condition_code,
                treatment,
                is_mM,
                sfn_uM,
                rep,
                particle_counter,
                x,
                !sample_excluded,
            ))
        end
    end

    return out
end

function build_particle_areas_long(input_dir::AbstractString)
    files = sort(filter(f -> endswith(f, ".csv"), readdir(input_dir; join=true)))
    isempty(files) && error("No CSV files found in $input_dir")
    dfs = [load_imagej_particle_file(f) for f in files]
    return vcat(dfs...)
end

function build_well_summary(particles::DataFrame)
    g = combine(
        groupby(
            particles,
            [:sample_id, :sex, :replicate_letter, :raw_file, :raw_column_name,
             :condition_code, :treatment, :is_mM, :sfn_uM, :within_sample_replicate,
             :included_in_analysis]
        ),
        :particle_area_px => length => :nodule_count,
        :particle_area_px => sum => :total_area_px,
        :particle_area_px => mean => :mean_particle_area_px,
    )

    g.total_area_mm2 = g.total_area_px ./ PX_PER_MM2
    g.mean_particle_area_mm2 = g.mean_particle_area_px ./ PX_PER_MM2

    return sort(g, [:sample_id, :condition_code, :within_sample_replicate])
end

# collapse within-sample wells to one value per biological replicate and condition
function build_sample_condition_summary(well_summary::DataFrame)
    g = combine(
        groupby(
            filter(:included_in_analysis => ==(true), well_summary),
            [:sample_id, :sex, :replicate_letter, :condition_code, :treatment, :is_mM, :sfn_uM]
        ),
        :nodule_count => sum => :nodule_count,
        :total_area_px => sum => :total_area_px,
        :total_area_mm2 => sum => :total_area_mm2,
        :within_sample_replicate => length => :n_wells,
    )

    return sort(g, [:sample_id, :condition_code])
end

# =========================
# Delta calculations
# =========================

function safe_log2fc(treat::Real, ctrl::Real; pseudocount::Real = 1e-6)
    return log2((treat + pseudocount) / (ctrl + pseudocount))
end

function build_delta_table(sample_summary::DataFrame, metric_col::Symbol)
    controls = filter(:condition_code => ==("is0_sfn0"), sample_summary)
    treated  = filter(:condition_code => !=("is0_sfn0"), sample_summary)

    out = DataFrame(
        sample_id = String[],
        sex = String[],
        replicate_letter = String[],
        treatment = String[],
        condition_code = String[],
        control_value = Float64[],
        treatment_value = Float64[],
        log2_fc = Float64[],
    )

    for row in eachrow(treated)
        ctrl = filter(r -> r.sample_id == row.sample_id, controls)
        nrow(ctrl) == 1 || continue

        ctrl_val = Float64(ctrl[1, metric_col])
        treat_val = Float64(row[metric_col])

        push!(out, (
            row.sample_id,
            row.sex,
            row.replicate_letter,
            row.treatment,
            row.condition_code,
            ctrl_val,
            treat_val,
            safe_log2fc(treat_val, ctrl_val),
        ))
    end

    return sort(out, [:treatment, :sample_id])
end

# =========================
# Statistics
# =========================

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

    for i in (m-1):-1:1
        adj[i] = min(adj[i], adj[i+1])
    end

    adj = clamp.(adj, 0.0, 1.0)

    out = similar(adj)
    out[order] = adj
    return out
end

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
        fc = 2.0^mean_val
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
# Run analysis
# =========================

particles = build_particle_areas_long(INPUT_DIR)
wells = build_well_summary(particles)
sample_summary = build_sample_condition_summary(wells)

delta_area = build_delta_table(sample_summary, :total_area_mm2)
delta_nodule = build_delta_table(sample_summary, :nodule_count)

area_stats = summarise_delta_stats(delta_area, "Mineralised area")
nodule_stats = summarise_delta_stats(delta_nodule, "Nodule count")
stats_table = vcat(area_stats, nodule_stats)

CSV.write(joinpath(OUT_DIR, "particle_areas_long.csv"), particles)
CSV.write(joinpath(OUT_DIR, "well_summary.csv"), wells)
CSV.write(joinpath(OUT_DIR, "sample_condition_summary.csv"), sample_summary)
CSV.write(joinpath(OUT_DIR, "delta_area.csv"), delta_area)
CSV.write(joinpath(OUT_DIR, "delta_nodule_count.csv"), delta_nodule)
CSV.write(joinpath(OUT_DIR, "figure2_stats_table.csv"), stats_table)

println("\nFigure 2 statistics")
show(stats_table, allrows=true, allcols=true)
println()

# =========================
# Make figure
# =========================

dissertation_theme!()

fig = Figure(size = (1200, 600), backgroundcolor = :white)

axA = Axis(fig[1, 1])
axB = Axis(fig[1, 2])

plot_delta!(
    axA,
    delta_area;
    treatment_order = TREATMENT_ORDER,
    ylabel = "log₂ fold change (vs control)",
    ylimits = AREA_YLIMS,
)
axA.title = "Mineralised area"
axA.titlesize = 22

plot_delta!(
    axB,
    delta_nodule;
    treatment_order = TREATMENT_ORDER,
    ylabel = "",
    ylimits = NODULE_YLIMS,
)
axB.title = "Nodule count"
axB.titlesize = 22

colgap!(fig.layout, 40)
rowgap!(fig.layout, 6)
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.5))
rowsize!(fig.layout, 1, Relative(1.0))

save(OUT_PDF, fig)
save(OUT_PNG, fig, px_per_unit = 2)

display(fig)
