using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Random
using Distributions
using Turing
using MCMCChains
using CairoMakie

# ============================================================
# Paths
# ============================================================

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW  = joinpath(ROOT, "raw_data")
const OUT  = joinpath(ROOT, "processed_data")

mkpath(OUT)

const AREA_FILE   = joinpath(RAW, "delta_area.csv")
const NODULE_FILE = joinpath(RAW, "delta_nodule_count.csv")
const QARS_FILE   = joinpath(RAW, "delta_absorbance.csv")

# ============================================================
# Settings
# ============================================================

const TREATMENT_ORDER = [
    "IS 2 mM",
    "SFN 1 µM",
    "SFN 5 µM",
    "IS 2 mM + SFN 1 µM",
    "IS 2 mM + SFN 5 µM",
]

const ENDPOINT_ORDER = [
    "Mineralised area",
    "Nodule count",
    "qARS",
]

# ============================================================
# Helpers
# ============================================================

function check_required_columns(df::DataFrame, cols::Vector{Symbol}, name::String)
    present = Set(Symbol.(names(df)))
    missing = [col for col in cols if !(col in present)]
    isempty(missing) || error("Missing columns in $name: $(missing). Present columns: $(Symbol.(names(df)))")
end

function build_index(x::Vector{String}, levels::Vector{String})
    idx = map(v -> findfirst(==(v), levels), x)
    any(isnothing, idx) && error("Found value not present in declared levels.")
    return Int.(idx)
end

function summarise_vec(x::Vector{<:Real})
    (
        mean = mean(x),
        q2_5 = quantile(x, 0.025),
        q50 = quantile(x, 0.50),
        q97_5 = quantile(x, 0.975),
        p_lt_0 = mean(x .< 0.0),
        p_lt_neg1 = mean(x .< -1.0),
    )
end

function plot_joint_summary(summary_rows::DataFrame, endpoint_levels::Vector{String}, out_pdf::String, out_png::String)
    endpoint_titles = Dict(
        "Mineralised area" => "Mineralised area",
        "Nodule count" => "Nodule count",
        "qARS" => "qARS",
    )

    treat_levels = TREATMENT_ORDER
    y = collect(length(treat_levels):-1:1)

    dark_green = colorant"#2f7d59"
    grid_gray = (:gray85, 1.0)
    zero_gray = (:gray45, 1.5)

    n_panels = length(endpoint_levels)
    fig = Figure(size = (600 * n_panels, 760), fontsize = 22, backgroundcolor = :white)

    global_low = minimum(summary_rows.ci_low)
    global_high = maximum(summary_rows.ci_high)
    xlow = min(-6.5, floor(global_low - 0.25))
    xhigh = max(0.5, ceil(global_high + 0.25))

    for (j, endpoint) in enumerate(endpoint_levels)
        sdf = filter(:endpoint => ==(endpoint), summary_rows)
        sdf = sort(sdf, [:treatment], by = x -> [findfirst(==(v), treat_levels) for v in x])

        ax = Axis(
            fig[1, j],
            title = endpoint_titles[endpoint],
            titlesize = 28,
            yticks = (y, reverse(treat_levels)),
            yticklabelsize = 18,
            xticklabelsize = 18,
            xgridcolor = grid_gray[1],
            xgridwidth = grid_gray[2],
            ygridvisible = false,
            topspinevisible = false,
            rightspinevisible = false,
            leftspinecolor = :black,
            bottomspinecolor = :black,
        )
        ax.spinewidth = 1.5
        ax.leftspinecolor = ax.bottomspinecolor = :black

        if j == 1
            ax.ylabel = "Treatment"
            ax.ylabelsize = 22
        else
            ax.yticklabelsvisible = false
            ax.yticksvisible = false
        end

        ax.xlabel = "log₂ fold change (vs control)"
        ax.xlabelsize = 22

        vlines!(ax, [0.0], color = zero_gray[1], linestyle = :dash, linewidth = zero_gray[2])

        means = Float64.(sdf.post_mean)
        lows = Float64.(sdf.ci_low)
        highs = Float64.(sdf.ci_high)

        for i in eachindex(y)
            lines!(ax, [lows[i], highs[i]], [y[i], y[i]], color = dark_green, linewidth = 3)
            scatter!(ax, [means[i]], [y[i]]; color = dark_green, marker = :diamond, markersize = 22)
        end

        xlims!(ax, xlow, xhigh)
        ylims!(ax, 0.5, length(treat_levels) + 0.5)
    end

    colgap!(fig.layout, 30)
    resize_to_layout!(fig)
    save(out_pdf, fig)
    save(out_png, fig)
    return fig
end

function plot_joint_probability_comparison(
    prob_rows_left::DataFrame,
    prob_rows_right::DataFrame,
    left_title::String,
    right_title::String,
    out_pdf::String,
    out_png::String,
)
    treat_levels = TREATMENT_ORDER
    y = collect(length(treat_levels):-1:1)
    dark_green = colorant"#2f7d59"
    grid_gray = (:gray85, 1.0)
    ref_gray = (:gray45, 1.5)

    function sorted_prob_rows(prob_rows::DataFrame)
        sort(prob_rows, [:treatment], by = x -> [findfirst(==(v), treat_levels) for v in x])
    end

    left_df = sorted_prob_rows(prob_rows_left)
    right_df = sorted_prob_rows(prob_rows_right)

    panel_specs = [
        (left_df,  :p_negative_all,   left_title,  "P(all included endpoints < 0)",  "A", 1, 1),
        (left_df,  :p_below_neg1_all, left_title,  "P(all included endpoints < -1)", "B", 1, 2),
        (right_df, :p_negative_all,   right_title, "P(all included endpoints < 0)",  "C", 2, 1),
        (right_df, :p_below_neg1_all, right_title, "P(all included endpoints < -1)", "D", 2, 2),
    ]

    fig = Figure(size = (1500, 1460), fontsize = 22, backgroundcolor = :white)

    Label(fig[0, 1:2], "Image-based endpoints only", fontsize = 28, font = :bold)
    Label(fig[2, 1:2], "Image-based endpoints + qARS", fontsize = 28, font = :bold)

    for (idx, (df_panel, prob_col, row_title, col_title, panel_label, row, col)) in enumerate(panel_specs)
        vals = Float64.(df_panel[!, prob_col])

        grid_row = row == 1 ? 1 : 3
        ax = Axis(
            fig[grid_row, col],
            title = col_title,
            titlesize = 22,
            yticks = (y, reverse(treat_levels)),
            yticklabelsize = 18,
            xticklabelsize = 18,
            xgridcolor = grid_gray[1],
            xgridwidth = grid_gray[2],
            ygridvisible = false,
            topspinevisible = false,
            rightspinevisible = false,
            leftspinecolor = :black,
            bottomspinecolor = :black,
        )
        ax.spinewidth = 1.5
        ax.leftspinecolor = ax.bottomspinecolor = :black

        if col == 1
            ax.ylabel = "Treatment"
            ax.ylabelsize = 22
        else
            ax.yticklabelsvisible = false
            ax.yticksvisible = false
        end

        ax.xlabel = "Posterior probability"
        ax.xlabelsize = 22


        vlines!(ax, [0.5], color = ref_gray[1], linestyle = :dash, linewidth = ref_gray[2])
        for i in eachindex(y)
            lines!(ax, [0.0, vals[i]], [y[i], y[i]], color = dark_green, linewidth = 3)
            scatter!(ax, [vals[i]], [y[i]]; color = dark_green, marker = :diamond, markersize = 22)
        end

        xlims!(ax, 0.0, 1.0)
        ylims!(ax, 0.5, length(treat_levels) + 0.7)
    end

    rowgap!(fig.layout, 90)
    rowsize!(fig.layout, 0, Auto(0.12))
    rowsize!(fig.layout, 2, Auto(0.12))
    rowsize!(fig.layout, 1, Auto(1))
    rowsize!(fig.layout, 3, Auto(1))
    colgap!(fig.layout, 35)
    resize_to_layout!(fig)
    save(out_pdf, fig)
    save(out_png, fig)
    return fig
end

function fit_joint_analysis(df_in::DataFrame, endpoint_levels::Vector{String}, prefix::String)
    df = filter(row -> row.endpoint in endpoint_levels, deepcopy(df_in))
    sort!(df, [:replicate, :treatment, :endpoint])
    CSV.write(joinpath(OUT, "$(prefix)_joint_long.csv"), df)

    rep_levels = sort(unique(df.replicate))
    treat_levels = TREATMENT_ORDER

    N_treat = length(treat_levels)
    N_end = length(endpoint_levels)
    N_rep = length(rep_levels)

    treat_idx = build_index(df.treatment, treat_levels)
    end_idx   = build_index(df.endpoint, endpoint_levels)
    rep_idx   = build_index(df.replicate, rep_levels)
    y         = Float64.(df.log2_fc)

    model = joint_model(y, treat_idx, end_idx, rep_idx, N_treat, N_end, N_rep)
    chain = sample(
        model,
        NUTS(0.65),
        MCMCSerial(),
        2000,
        4;
        discard_initial = 1000,
    )

    open(joinpath(OUT, "$(prefix)_joint_model_chain_summary.txt"), "w") do io
        show(io, MIME("text/plain"), chain)
    end

    chn = DataFrame(chain)

    summary_rows = DataFrame(
        treatment = String[],
        endpoint = String[],
        post_mean = Float64[],
        ci_low = Float64[],
        ci_mid = Float64[],
        ci_high = Float64[],
        p_lt_0 = Float64[],
        p_lt_neg1 = Float64[],
    )

    for t in 1:N_treat, e in 1:N_end
        col = Symbol("β[$t, $e]")
        vals = Vector{Float64}(chn[!, col])
        s = summarise_vec(vals)
        push!(summary_rows, (
            treat_levels[t],
            endpoint_levels[e],
            s.mean,
            s.q2_5,
            s.q50,
            s.q97_5,
            s.p_lt_0,
            s.p_lt_neg1,
        ))
    end
    CSV.write(joinpath(OUT, "$(prefix)_joint_model_treatment_endpoint_summary.csv"), summary_rows)

    contrast_rows = DataFrame(
        contrast = String[],
        endpoint = String[],
        post_mean = Float64[],
        ci_low = Float64[],
        ci_mid = Float64[],
        ci_high = Float64[],
        p_gt_0 = Float64[],
        p_gt_1 = Float64[],
    )

    is_idx = findfirst(==("IS 2 mM"), treat_levels)
    is_sfn1_idx = findfirst(==("IS 2 mM + SFN 1 µM"), treat_levels)
    is_sfn5_idx = findfirst(==("IS 2 mM + SFN 5 µM"), treat_levels)

    for e in 1:N_end
        β_is      = Vector{Float64}(chn[!, Symbol("β[$is_idx, $e]")])
        β_is_sfn1 = Vector{Float64}(chn[!, Symbol("β[$is_sfn1_idx, $e]")])
        β_is_sfn5 = Vector{Float64}(chn[!, Symbol("β[$is_sfn5_idx, $e]")])

        c1 = β_is_sfn1 .- β_is
        c2 = β_is_sfn5 .- β_is

        for (name, vals) in [
            ("IS+SFN 1 µM minus IS", c1),
            ("IS+SFN 5 µM minus IS", c2),
        ]
            s = summarise_vec(vals)
            push!(contrast_rows, (
                name,
                endpoint_levels[e],
                s.mean,
                s.q2_5,
                s.q50,
                s.q97_5,
                mean(vals .> 0.0),
                mean(vals .> 1.0),
            ))
        end
    end
    CSV.write(joinpath(OUT, "$(prefix)_joint_model_rescue_contrasts.csv"), contrast_rows)

    joint_rows = DataFrame(
        treatment = String[],
        p_negative_all = Float64[],
        p_below_neg1_all = Float64[],
    )

    prob_threshold = length(endpoint_levels)
    for t in 1:N_treat
        endpoint_vals = [Vector{Float64}(chn[!, Symbol("β[$t, $e]")]) for e in 1:N_end]
        neg_mask = trues(length(endpoint_vals[1]))
        neg1_mask = trues(length(endpoint_vals[1]))
        for vals in endpoint_vals
            neg_mask .&= vals .< 0.0
            neg1_mask .&= vals .< -1.0
        end
        push!(joint_rows, (
            treat_levels[t],
            mean(neg_mask),
            mean(neg1_mask),
        ))
    end
    CSV.write(joinpath(OUT, "$(prefix)_joint_model_joint_probabilities.csv"), joint_rows)

    return (df=df, chain=chain, summary_rows=summary_rows, contrast_rows=contrast_rows, joint_rows=joint_rows)
end

# ============================================================
# Read and harmonise
# ============================================================

area = CSV.read(AREA_FILE, DataFrame)
nodule = CSV.read(NODULE_FILE, DataFrame)
qars = CSV.read(QARS_FILE, DataFrame)

rename!(area,   Symbol.(strip.(String.(names(area)))))
rename!(nodule, Symbol.(strip.(String.(names(nodule)))))
rename!(qars,   Symbol.(strip.(String.(names(qars)))))

check_required_columns(area,   [:sample_id, :treatment, :log2_fc], "delta_area.csv")
check_required_columns(nodule, [:sample_id, :treatment, :log2_fc], "delta_nodule_count.csv")
check_required_columns(qars,   [:sample_id, :treatment, :log2_fc], "delta_absorbance.csv")

area_long = DataFrame(
    replicate = String.(area[!, :sample_id]),
    treatment = String.(area[!, :treatment]),
    endpoint  = fill("Mineralised area", nrow(area)),
    log2_fc   = Float64.(area[!, :log2_fc]),
)

nodule_long = DataFrame(
    replicate = String.(nodule[!, :sample_id]),
    treatment = String.(nodule[!, :treatment]),
    endpoint  = fill("Nodule count", nrow(nodule)),
    log2_fc   = Float64.(nodule[!, :log2_fc]),
)

qars_long = DataFrame(
    replicate = String.(qars[!, :sample_id]),
    treatment = String.(qars[!, :treatment]),
    endpoint  = fill("qARS", nrow(qars)),
    log2_fc   = Float64.(qars[!, :log2_fc]),
)

df = vcat(area_long, nodule_long, qars_long)
df = filter(row -> row.treatment in TREATMENT_ORDER, df)
df = filter(row -> row.endpoint in ENDPOINT_ORDER, df)
sort!(df, [:replicate, :treatment, :endpoint])

CSV.write(joinpath(OUT, "joint_mineralisation_long.csv"), df)

# ============================================================
# Model 1
# Joint hierarchical model across endpoints.
# This version is stable in Turing and keeps endpoint-specific
# treatment effects plus replicate-level partial pooling.
# ============================================================

@model function joint_model(y, treat_idx, end_idx, rep_idx, N_treat, N_end, N_rep)
    β = Matrix{Float64}(undef, N_treat, N_end)
    for t in 1:N_treat, e in 1:N_end
        β[t, e] ~ Normal(0, 2)
    end

    σ = Vector{Float64}(undef, N_end)
    τ = Vector{Float64}(undef, N_end)
    for e in 1:N_end
        σ[e] ~ truncated(Normal(0, 1), 0, Inf)
        τ[e] ~ truncated(Normal(0, 1), 0, Inf)
    end

    b = Matrix{Float64}(undef, N_rep, N_end)
    for r in 1:N_rep, e in 1:N_end
        b[r, e] ~ Normal(0, τ[e])
    end

    for i in eachindex(y)
        μ = β[treat_idx[i], end_idx[i]] + b[rep_idx[i], end_idx[i]]
        y[i] ~ Normal(μ, σ[end_idx[i]])
    end
end

# ============================================================
# Analyses
# ============================================================

Random.seed!(42)

full_result = fit_joint_analysis(df, ["Mineralised area", "Nodule count", "qARS"], "full")
image_result = fit_joint_analysis(df, ["Mineralised area", "Nodule count"], "image_only")

println(full_result.chain)
println(image_result.chain)

println("\nSaved files:")
println(joinpath(OUT, "full_joint_long.csv"))
println(joinpath(OUT, "full_joint_model_chain_summary.txt"))
println(joinpath(OUT, "full_joint_model_treatment_endpoint_summary.csv"))
println(joinpath(OUT, "full_joint_model_rescue_contrasts.csv"))
println(joinpath(OUT, "full_joint_model_joint_probabilities.csv"))
println(joinpath(OUT, "image_only_joint_long.csv"))
println(joinpath(OUT, "image_only_joint_model_chain_summary.txt"))
println(joinpath(OUT, "image_only_joint_model_treatment_endpoint_summary.csv"))
println(joinpath(OUT, "image_only_joint_model_rescue_contrasts.csv"))
println(joinpath(OUT, "image_only_joint_model_joint_probabilities.csv"))

# ============================================================
# Figures
# ============================================================

full_fig_pdf = joinpath(ROOT, "figures", "full_joint_model_forest_plot.pdf")
full_fig_png = joinpath(ROOT, "figures", "full_joint_model_forest_plot.png")
image_fig_pdf = joinpath(ROOT, "figures", "image_only_joint_model_forest_plot.pdf")
image_fig_png = joinpath(ROOT, "figures", "image_only_joint_model_forest_plot.png")
prob_compare_pdf = joinpath(ROOT, "figures", "joint_model_probability_comparison.pdf")
prob_compare_png = joinpath(ROOT, "figures", "joint_model_probability_comparison.png")

mkpath(dirname(full_fig_pdf))

plot_joint_summary(full_result.summary_rows, ["Mineralised area", "Nodule count", "qARS"], full_fig_pdf, full_fig_png)
plot_joint_summary(image_result.summary_rows, ["Mineralised area", "Nodule count"], image_fig_pdf, image_fig_png)
plot_joint_probability_comparison(
    image_result.joint_rows,
    full_result.joint_rows,
    "Image-based endpoints only",
    "Image-based endpoints + qARS",
    prob_compare_pdf,
    prob_compare_png,
)

println(full_fig_pdf)
println(full_fig_png)
println(image_fig_pdf)
println(image_fig_png)
println(prob_compare_pdf)
println(prob_compare_png)
