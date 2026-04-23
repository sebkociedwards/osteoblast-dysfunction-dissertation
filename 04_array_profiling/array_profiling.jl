using CSV
using DataFrames
using CairoMakie

# =========================
# Paths
# =========================

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RAW_DIR = joinpath(ROOT, "raw_data")
const PROC_DIR = joinpath(ROOT, "processed_data")
const FIG_DIR = joinpath(ROOT, "figures")

mkpath(PROC_DIR)
mkpath(FIG_DIR)

const CT_FILE = joinpath(RAW_DIR, "qpcr_array_ct_values.csv")
const GENE_MAP_FILE = joinpath(RAW_DIR, "qpcr_array_gene_map.csv")
const SAMPLE_META_FILE = joinpath(RAW_DIR, "qpcr_array_sample_metadata.csv")

const OUT_TABLE = joinpath(PROC_DIR, "array_main_text_gene_table.csv")
const OUT_PDF = joinpath(FIG_DIR, "figure4_array_main_text.pdf")
const OUT_PNG = joinpath(FIG_DIR, "figure4_array_main_text.png")

# =========================
# Figure settings
# =========================

const POINT_COLOUR = colorant"#69B38A"
const GRID_COLOUR = RGBAf(0, 0, 0, 0.10)
const AXIS_COLOUR = RGBAf(0, 0, 0, 0.70)
const BG_COLOUR = :white

const PAGE_WIDTH_CM = 20.99
const LEFT_MARGIN_CM = 2.54
const RIGHT_MARGIN_CM = 2.54
const TEXT_WIDTH_CM = PAGE_WIDTH_CM - LEFT_MARGIN_CM - RIGHT_MARGIN_CM
const TEXT_WIDTH_IN = TEXT_WIDTH_CM / 2.54
word_size_pt(height_in::Real) = (TEXT_WIDTH_IN * 72, height_in * 72)

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
                xgridvisible = true,
                ygridvisible = false,
                xgridcolor = GRID_COLOUR,
                xgridwidth = 1.0,
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

# Convert ordinary positive fold change to signed log2 scale.
# Values > 1 become positive and values < 1 become negative.
signed_log2_fc(x::Real) = log2(x)

function assign_pathway(gene::AbstractString)
    g = lowercase(strip(gene))

    if g in lowercase.(["Hmox1", "Nqo1", "Srxn1", "Gclc", "Txnrd1", "Gclm"])
        return "Direct Nrf2-mediated oxidative stress response"
    elseif g in lowercase.(["Gsr", "Gss", "Sod3", "Cat"])
        return "Antioxidant / indirect oxidative stress response"
    elseif g in lowercase.(["Ptgs1", "Ptgs2", "Sqstm1", "Txnip", "Nox1", "Nox4", "Ucp3"])
        return "Redox-inflammatory / stress response"
    elseif g in lowercase.(["Atr", "Ercc2", "Ercc6", "Dnm2", "Recql4"])
        return "DNA damage and stress signalling"
    elseif g in lowercase.(["Alb", "Mb", "Tpo"])
        return "Suppressed specialised / metabolic function"
    else
        return "Other"
    end
end

# Main-text subset only
# Include the full candidate set most relevant to the dissertation, including the clearly suppressed functional genes.
const MAIN_GENES = [
    "Hmox1",
    "Nqo1",
    "Srxn1",
    "Gclc",
    "Gclm",
    "Txnrd1",
    "Gsr",
    "Gss",
    "Sod3",
    "Cat",
    "Ptgs1",
    "Ptgs2",
    "Sqstm1",
    "Txnip",
    "Nox1",
    "Nox4",
    "Ucp3",
    "Atr",
    "Ercc2",
    "Ercc6",
    "Dnm2",
    "Recql4",
    "Alb",
    "Mb",
    "Tpo",
]

const GENE_ORDER = Dict(g => i for (i, g) in enumerate(MAIN_GENES))
const PATHWAY_ORDER = Dict(
    "Direct Nrf2-mediated oxidative stress response" => 1,
    "Antioxidant / indirect oxidative stress response" => 2,
    "Redox-inflammatory / stress response" => 3,
    "DNA damage and stress signalling" => 4,
    "Suppressed specialised / metabolic function" => 5,
    "Other" => 6,
)

function pick_col(df::DataFrame, candidates::Vector{String})
    names_lower = Dict(lowercase(String(n)) => String(n) for n in names(df))
    for c in candidates
        key = lowercase(c)
        if haskey(names_lower, key)
            return Symbol(names_lower[key])
        end
    end
    error("Could not find any of these columns: $(candidates). Found: $(names(df))")
end

function clean_string_col!(df::DataFrame, col::Symbol)
    df[!, col] = [ismissing(x) ? missing : strip(String(x)) for x in df[!, col]]
    return df
end

# =========================
# Load raw inputs
# =========================

ct = CSV.read(CT_FILE, DataFrame)
gene_map = CSV.read(GENE_MAP_FILE, DataFrame)
sample_meta = CSV.read(SAMPLE_META_FILE, DataFrame)

rename!(ct, Symbol.(strip.(string.(names(ct)))))
rename!(gene_map, Symbol.(strip.(string.(names(gene_map)))))
rename!(sample_meta, Symbol.(strip.(string.(names(sample_meta)))))

# Flexible column matching
ct_names_lower = Set(lowercase.(String.(names(ct))))
ct_is_premerged = all(x -> x in ct_names_lower, [
    "well_id",
    "gene_symbol",
    "well_type",
    "ct_control",
    "ct_is",
])

if ct_is_premerged
    ct_long = DataFrame(
        well = ct.well_id,
        gene_symbol = ct.gene_symbol,
        well_type = ct.well_type,
        Ct_control = ct.ct_control,
        Ct_is = ct.ct_is,
        comment_flag = fill(missing, nrow(ct)),
    )

    if :gene_description in names(ct)
        ct_long.gene_description = ct.gene_description
    end
else
    ct_well_col = pick_col(ct, ["well", "well_position", "well_id"])
    ct_ct_col = pick_col(ct, ["ct", "ct_value", "ct mean", "ct_mean"])

    gm_well_col = pick_col(gene_map, ["well", "well_position", "well_id"])
    gm_gene_col = pick_col(gene_map, ["gene_symbol", "gene", "symbol", "target"])
    gm_type_col = pick_col(gene_map, ["well_type", "type", "target_type"])

    sm_sample_col = pick_col(sample_meta, ["sample_id", "sample", "sample_name"])
    sm_condition_col = pick_col(sample_meta, ["condition", "group", "treatment"])

    # Optional flag/comment column
    flag_col = nothing
    for cand in ["comment_flag", "flag", "comments", "comment", "qiagen_flag"]
        try
            global flag_col = pick_col(gene_map, [cand])
            break
        catch
        end
    end

    clean_string_col!(ct, ct_well_col)
    clean_string_col!(gene_map, gm_well_col)
    clean_string_col!(gene_map, gm_gene_col)
    clean_string_col!(gene_map, gm_type_col)
    clean_string_col!(sample_meta, sm_sample_col)
    clean_string_col!(sample_meta, sm_condition_col)

    function ct_to_long(ct::DataFrame, well_col::Symbol, ct_col::Symbol, sample_meta::DataFrame, sample_col::Symbol)
        if sample_col in names(ct)
            return select(ct, well_col, sample_col, ct_col)
        end

        # Otherwise assume wide format: well column + one column per sample
        sample_names = Set(String.(sample_meta[!, sample_col]))
        value_cols = [n for n in names(ct) if n != well_col && String(n) in sample_names]

        if isempty(value_cols)
            error("Could not identify sample columns in ct file. Found columns: $(names(ct))")
        end

        long = stack(ct, value_cols, variable_name = :sample_id, value_name = :Ct)
        rename!(long, well_col => :well)
        return long
    end

    ct_long = ct_to_long(ct, ct_well_col, ct_ct_col, sample_meta, sm_sample_col)

    if :well ∉ names(ct_long)
        rename!(ct_long, ct_well_col => :well)
    end
    if :sample_id ∉ names(ct_long)
        rename!(ct_long, sm_sample_col => :sample_id)
    end
    if :Ct ∉ names(ct_long)
        rename!(ct_long, ct_ct_col => :Ct)
    end

    clean_string_col!(ct_long, :well)
    clean_string_col!(ct_long, :sample_id)
end

# =========================
# Merge and process
# =========================

if ct_is_premerged
    rename!(ct_long, :well => :well)
    clean_string_col!(ct_long, :well)
    clean_string_col!(ct_long, :gene_symbol)
    clean_string_col!(ct_long, :well_type)

    if :comment_flag ∉ names(ct_long)
        ct_long.comment_flag = fill(missing, nrow(ct_long))
    end

    df = ct_long
else
    gm = select(gene_map, gm_well_col, gm_gene_col, gm_type_col, filter(!isnothing, [flag_col])...)
    rename!(gm, gm_well_col => :well, gm_gene_col => :gene_symbol, gm_type_col => :well_type)
    if !isnothing(flag_col)
        rename!(gm, flag_col => :comment_flag)
    else
        gm.comment_flag = fill(missing, nrow(gm))
    end

    sm = select(sample_meta, sm_sample_col, sm_condition_col)
    rename!(sm, sm_sample_col => :sample_id, sm_condition_col => :condition)

    df = leftjoin(ct_long, gm, on = :well)
    df = leftjoin(df, sm, on = :sample_id)

    # Keep gene and housekeeping wells at this stage so normalisation can be computed.
end

# Mean Ct per gene per condition (n = 1 sample per condition here)
if ct_is_premerged
    gene_condition = select(df, :gene_symbol, :well_type, :comment_flag, :Ct_control, :Ct_is)
    rename!(gene_condition, :Ct_control => :control_ct, :Ct_is => :is_ct)
    wide = gene_condition
else
    gene_condition = combine(
        groupby(df, [:gene_symbol, :well_type, :condition, :comment_flag]),
        :Ct => mean => :mean_ct,
    )

    # Infer control and IS labels from metadata
    conditions = unique(String.(gene_condition.condition))
    control_label = findfirst(c -> occursin("con", lowercase(c)) || occursin("ctrl", lowercase(c)) || occursin("control", lowercase(c)), conditions)
    is_label = findfirst(c -> lowercase(c) == "is" || occursin("indoxyl", lowercase(c)) || occursin("is ", lowercase(c)), conditions)

    control_label === nothing && error("Could not identify control condition from: $(conditions)")
    is_label === nothing && error("Could not identify IS condition from: $(conditions)")

    control_name = conditions[control_label]
    is_name = conditions[is_label]

    wide = unstack(gene_condition, :condition, :mean_ct)

    Symbol(control_name) in names(wide) || error("Control column not found after unstack.")
    Symbol(is_name) in names(wide) || error("IS column not found after unstack.")

    wide.control_ct = wide[!, Symbol(control_name)]
    wide.is_ct = wide[!, Symbol(is_name)]
end

# Use the Qiagen-selected housekeeping genes when available.
selected_housekeepers = Set(["B2m", "Gusb"])
all_housekeepers = Set(["Actb", "Gapdh", "Hsp90ab1", "Gusb", "B2m"])

wide.is_housekeeper = [
    (!ismissing(wide.well_type[i]) && lowercase(String(wide.well_type[i])) == "housekeeping") ||
    (String(wide.gene_symbol[i]) in all_housekeepers)
    for i in 1:nrow(wide)
]

wide.use_for_normalisation = [String(g) in selected_housekeepers for g in wide.gene_symbol]
if count(wide.use_for_normalisation) == 0
    wide.use_for_normalisation = copy(wide.is_housekeeper)
end

hk_ctrl = mean(skipmissing(wide.control_ct[wide.use_for_normalisation]))
hk_is   = mean(skipmissing(wide.is_ct[wide.use_for_normalisation]))

wide.delta_ct_control = wide.control_ct .- hk_ctrl
wide.delta_ct_is = wide.is_ct .- hk_is
wide.delta_delta_ct = wide.delta_ct_is .- wide.delta_ct_control
wide.fold_change_is_vs_control = 2.0 .^ (-wide.delta_delta_ct)

# Qiagen-style fold-regulation sign convention:
# upregulated: positive fold change
# downregulated: negative reciprocal
wide.fold_regulation_is_vs_control = [
    fc >= 1 ? fc : -(1 / fc) for fc in wide.fold_change_is_vs_control
]

wide.signed_log2_fc = [signed_log2_fc(x) for x in wide.fold_change_is_vs_control]
wide.pathway = [assign_pathway(String(g)) for g in wide.gene_symbol]

# Keep only true gene wells for presentation outputs.
wide = filter(row -> !ismissing(row.well_type) && lowercase(String(row.well_type)) == "gene", wide)

# Main-text subset only
main = filter(row -> String(row.gene_symbol) in MAIN_GENES, wide)
main.gene_order = [GENE_ORDER[String(g)] for g in main.gene_symbol]
main.pathway_order = [get(PATHWAY_ORDER, String(p), 999) for p in main.pathway]
sort!(main, [:pathway_order, :gene_order])

# Save only compact presentation table
presentation_table = select(
    main,
    :gene_symbol,
    :pathway,
    :fold_regulation_is_vs_control,
    :signed_log2_fc,
    :comment_flag,
)
CSV.write(OUT_TABLE, presentation_table)

# =========================
# Plot main-text figure
# =========================

function build_y_positions(df::DataFrame)
    y = Float64[]
    labels = String[]
    pathway_breaks = Float64[]

    current_y = 1.0
    gap = 0.7

    for i in 1:nrow(df)
        push!(y, current_y)
        push!(labels, String(df.gene_symbol[i]))

        if i < nrow(df) && String(df.pathway[i]) != String(df.pathway[i + 1])
            push!(pathway_breaks, current_y + 0.5 + gap / 2)
            current_y += 1.0 + gap
        else
            current_y += 1.0
        end
    end

    return y, labels, pathway_breaks
end

function plot_main_array_figure(df::DataFrame)
    d = reverse(df)
    y, labels, pathway_breaks = build_y_positions(d)
    xmin = min(minimum(skipmissing(d.signed_log2_fc)), -0.5)
    xmax = max(maximum(skipmissing(d.signed_log2_fc)), 0.5)
    pad = 0.25

    dissertation_theme!()

    fig = Figure(
        size = (700, 600),
        backgroundcolor = BG_COLOUR,
    )

    ax = Axis(
        fig[1, 1],
        xlabel = "log₂ fold change (vs control)",
        limits = ((xmin - pad, xmax + pad), nothing),
        ylabel = "",
        yticks = (y, labels),
        xticklabelsize = 10,
        yticklabelsize = 11,
        yticklabelfont = "Times New Roman Italic",
        xlabelsize = 11,
        leftspinecolor = AXIS_COLOUR,
        bottomspinecolor = AXIS_COLOUR,
        xtickcolor = AXIS_COLOUR,
        ytickcolor = AXIS_COLOUR,
        xticklabelcolor = AXIS_COLOUR,
        yticklabelcolor = AXIS_COLOUR,
        xlabelcolor = AXIS_COLOUR,
    )

    vlines!(ax, [0], linestyle = :dash, color = AXIS_COLOUR, linewidth = 1.3)

    for i in eachindex(y)
        lines!(
            ax,
            [0, d.signed_log2_fc[i]],
            [y[i], y[i]],
            color = POINT_COLOUR,
            linewidth = 1.8,
        )
    end

    scatter!(
        ax,
        d.signed_log2_fc,
        y,
        color = POINT_COLOUR,
        markersize = 10,
    )

    flagged = findall(i -> !(ismissing(d.comment_flag[i]) || isempty(strip(String(d.comment_flag[i])))), 1:nrow(d))
    if !isempty(flagged)
        scatter!(
            ax,
            d.signed_log2_fc[flagged],
            y[flagged],
            color = :transparent,
            strokecolor = POINT_COLOUR,
            strokewidth = 1.8,
            markersize = 14,
        )
    end

    for yy in pathway_breaks
        hlines!(ax, [yy], color = GRID_COLOUR, linestyle = :dot, linewidth = 1.0)
    end

    save(OUT_PDF, fig)
    save(OUT_PNG, fig, px_per_unit = 4)
    display(fig)
end

plot_main_array_figure(main)
