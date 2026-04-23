using CSV
using DataFrames
using CairoMakie
using Statistics

# ============================================================
# 02_rnaseq_figures_vehicle_only.jl
# Figure generation for vehicle-only bulk RNA-seq analysis
# Input comparison: diff_veh vs undiff_veh
# ============================================================

# ============================================================
# Paths
# ============================================================
script_dir = @__DIR__
project_dir = normpath(joinpath(script_dir, ".."))
processed_dir = joinpath(project_dir, "processed_data")
figdir = joinpath(project_dir, "figures")
mkpath(figdir)

if !isdir(processed_dir)
    error("processed_data directory not found: $(processed_dir)")
end

pca_file = joinpath(processed_dir, "pca_vehicle_coordinates.csv")
pca_var_file = joinpath(processed_dir, "pca_vehicle_variance_explained.csv")
de_file = joinpath(processed_dir, "diff_veh_vs_undiff_veh.csv")
vst_file = joinpath(processed_dir, "vst_vehicle_matrix.csv")

for f in [pca_file, pca_var_file, de_file, vst_file]
    isfile(f) || error("Required input file not found: $(f)")
end

# ============================================================
# Theme
# ============================================================
const POINT_COLOUR = colorant"#2F6B4F"
const MEAN_COLOUR = colorant"#69B38A"
const GRID_COLOUR = RGBAf(0, 0, 0, 0.10)
const AXIS_COLOUR = RGBAf(0, 0, 0, 0.70)
const BG_COLOUR = :white
const LABEL_COLOUR = :black
const PCA_UNDIFF_COLOUR = :grey45
const PCA_DIFF_COLOUR = MEAN_COLOUR
const VOLCANO_NS_COLOUR = (:grey80, 0.7)
const VOLCANO_SIG_COLOUR = (POINT_COLOUR, 0.85)
const VOLCANO_HI_COLOUR = (MEAN_COLOUR, 0.95)

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
            Legend = (
                framevisible = false,
                labelsize = 11,
            ),
        )
    )
end

# ============================================================
# Helpers
# ============================================================
function to_float_or_nan(x)
    if ismissing(x)
        return NaN
    elseif x isa Number
        return Float64(x)
    else
        s = strip(String(x))
        if isempty(s) || lowercase(s) in ["na", "nan"]
            return NaN
        end
        y = tryparse(Float64, s)
        return isnothing(y) ? NaN : y
    end
end

function safe_neglog10(x)
    y = to_float_or_nan(x)
    return (isnan(y) || y <= 0) ? NaN : -log10(y)
end

function clean_symbol(x, fallback)
    if ismissing(x)
        return String(fallback)
    end
    s = strip(String(x))
    return (isempty(s) || s == "NA") ? String(fallback) : s
end

function coerce_numeric!(df::DataFrame, cols)
    for c in cols
        if c in names(df)
            df[!, c] = [to_float_or_nan(x) for x in df[!, c]]
        end
    end
    return df
end

# ============================================================
# Load
# ============================================================
pca = CSV.read(pca_file, DataFrame)
pca_var = CSV.read(pca_var_file, DataFrame)
de = CSV.read(de_file, DataFrame)
vst = CSV.read(vst_file, DataFrame)

coerce_numeric!(pca, [:PC1, :PC2, :PC3])
coerce_numeric!(pca_var, [:variance_explained])
coerce_numeric!(de, [:baseMean, :log2FoldChange, :lfcSE, :stat, :pvalue, :padj])

dissertation_theme!()

# ============================================================
# PCA prep
# ============================================================
pc1_var = round(100 * pca_var.variance_explained[1], digits = 1)
pc2_var = round(100 * pca_var.variance_explained[2], digits = 1)

# ============================================================
# Volcano prep
# ============================================================
de2 = copy(de)
de2.neglog10padj = [safe_neglog10(x) for x in de2.padj]
de2.log2fc_num = [to_float_or_nan(x) for x in de2.log2FoldChange]
de2.padj_num = [to_float_or_nan(x) for x in de2.padj]
de2.symbol_clean = [clean_symbol(sym, ens) for (sym, ens) in zip(de2.SYMBOL, de2.ensembl_gene_id)]

keep = .!(isnan.(de2.log2fc_num) .| isnan.(de2.neglog10padj))
de2 = de2[keep, :]


de2.sig_class = map(eachrow(de2)) do r
    if r.padj_num < 0.05 && abs(r.log2fc_num) >= 1
        "padj < 0.05 & |log2FC| ≥ 1"
    elseif r.padj_num < 0.05
        "padj < 0.05"
    else
        "not significant"
    end
end

# Genes most relevant to the dissertation narrative: osteolineage maturation,
# mineralisation, and oxidative-stress response measured in the primary osteoblast model.
dissertation_genes = [
    "Mmp13",   # strong differentiation-associated matrix remodelling marker
    "Col11a1", # strong differentiation-associated matrix gene
    "Bglap",   # directly relevant to the dissertation osteoblast-marker panel
    "Dmp1",    # late osteolineage / mineralisation-associated contextual marker
    "Spp1",    # directly relevant to the dissertation osteoblast-marker panel
    "Nqo1",    # directly relevant oxidative-stress gene from qPCR / array
    "Gclc",    # directly relevant oxidative-stress gene from qPCR / array
    "Ptgs1"    # directly relevant redox/inflammatory gene from qPCR
]

maturation_genes = ["Mmp13", "Col11a1", "Bglap", "Dmp1", "Spp1"]
redox_genes = ["Nqo1", "Gclc", "Ptgs1"]

# ============================================================
# Combined PCA + volcano figure
# ============================================================
fig = Figure(size = (1200, 450), backgroundcolor = BG_COLOUR)

ax_pca = Axis(
    fig[1, 1],
    xlabel = "PC1 ($(pc1_var)%)",
    ylabel = "PC2 ($(pc2_var)%)",
    xticklabelsize = 10,
    yticklabelsize = 11,
    xlabelsize = 11,
    ylabelsize = 11,
    leftspinecolor = AXIS_COLOUR,
    bottomspinecolor = AXIS_COLOUR,
    xtickcolor = AXIS_COLOUR,
    ytickcolor = AXIS_COLOUR,
    xticklabelcolor = AXIS_COLOUR,
    yticklabelcolor = AXIS_COLOUR,
    xlabelcolor = AXIS_COLOUR,
    ylabelcolor = AXIS_COLOUR,
)
for st in ["undiff", "diff"]
    sub = filter(:state => ==(st), pca)
    marker = st == "undiff" ? :circle : :rect
    color = st == "undiff" ? PCA_UNDIFF_COLOUR : PCA_DIFF_COLOUR
    scatter!(
        ax_pca,
        sub.PC1,
        sub.PC2,
        marker = marker,
        markersize = 20,
        color = color,
        strokecolor = AXIS_COLOUR,
        strokewidth = 1.0,
        label = st
    )
end
Legend(fig[1, 2], ax_pca)

ax_vol = Axis(
    fig[1, 3],
    xlabel = "log2 fold change (diff veh vs undiff veh)",
    ylabel = "-log10 adjusted p-value",
    xticklabelsize = 10,
    yticklabelsize = 11,
    xlabelsize = 11,
    ylabelsize = 11,
    leftspinecolor = AXIS_COLOUR,
    bottomspinecolor = AXIS_COLOUR,
    xtickcolor = AXIS_COLOUR,
    ytickcolor = AXIS_COLOUR,
    xticklabelcolor = AXIS_COLOUR,
    yticklabelcolor = AXIS_COLOUR,
    xlabelcolor = AXIS_COLOUR,
    ylabelcolor = AXIS_COLOUR,
)

for cls in ["not significant", "padj < 0.05", "padj < 0.05 & |log2FC| ≥ 1"]
    sub = filter(:sig_class => ==(cls), de2)
    nrow(sub) == 0 && continue
    color = cls == "not significant" ? VOLCANO_NS_COLOUR : cls == "padj < 0.05" ? VOLCANO_SIG_COLOUR : VOLCANO_HI_COLOUR
    scatter!(
        ax_vol,
        sub.log2fc_num,
        sub.neglog10padj,
        markersize = cls == "padj < 0.05 & |log2FC| ≥ 1" ? 9 : 6,
        color = color,
        strokewidth = 0,
        label = cls
    )
end

hlines!(ax_vol, [-log10(0.05)], linestyle = :dash, color = AXIS_COLOUR, linewidth = 1.3)
vlines!(ax_vol, [-1, 1], linestyle = :dash, color = AXIS_COLOUR, linewidth = 1.3)

label_df = filter(row -> row.symbol_clean in dissertation_genes, de2)
if nrow(label_df) > 0
    label_df.category = [g in maturation_genes ? "Maturation/mineralisation" : g in redox_genes ? "Oxidative stress/redox" : "Other" for g in label_df.symbol_clean]
    # Prefer genes with stronger evidence, but keep the selection biologically anchored
    # to the dissertation rather than labelling generic top transcriptome hits.
    sig_weight = ifelse.(label_df.padj_num .< 0.05, 10.0, 1.0)
    lfc_weight = ifelse.(abs.(label_df.log2fc_num) .>= 1, 2.0, 1.0)
    label_df.rank_score = abs.(label_df.log2fc_num) .* sig_weight .* lfc_weight
    sort!(label_df, :rank_score, rev = true)
    top_lab = label_df[1:min(12, nrow(label_df)), :]

    # Re-emphasise labelled genes with a black outline and slightly larger marker.
    scatter!(
        ax_vol,
        top_lab.log2fc_num,
        top_lab.neglog10padj,
        markersize = 12,
        color = MEAN_COLOUR,
        strokecolor = LABEL_COLOUR,
        strokewidth = 1.2,
    )

    text!(
        ax_vol,
        top_lab.log2fc_num,
        top_lab.neglog10padj,
        text = top_lab.symbol_clean,
        fontsize = 11,
        font = :bold_italic,
        color = LABEL_COLOUR,
        align = (:left, :bottom),
        offset = (5, 4)
    )
end
Legend(fig[1, 4], ax_vol)

# ============================================================
# Export compact dissertation-focused gene table
# ============================================================
selected_table = filter(row -> row.symbol_clean in dissertation_genes, de2)
if nrow(selected_table) > 0
    selected_table.category = [
        g in maturation_genes ? "Maturation/mineralisation" :
        g in redox_genes ? "Oxidative stress/redox" :
        "Other"
        for g in selected_table.symbol_clean
    ]

    selected_table.functional_relevance = [
        g == "Mmp13"  ? "Matrix remodelling / differentiation-associated" :
        g == "Col11a1" ? "Matrix-associated differentiation gene" :
        g == "Bglap"  ? "Osteoblast maturation / mineralisation marker" :
        g == "Dmp1"   ? "Late osteolineage / mineralisation-associated marker" :
        g == "Spp1"   ? "Osteoblast / mineralisation-associated marker" :
        g == "Nqo1"   ? "Oxidative stress / Nrf2-associated response" :
        g == "Gclc"   ? "Glutathione synthesis / oxidative stress response" :
        g == "Ptgs1"  ? "Redox-inflammatory context" :
        ""
        for g in selected_table.symbol_clean
    ]

    selected_table = selected_table[:, [:category, :symbol_clean, :functional_relevance, :log2fc_num, :padj_num]]
    rename!(selected_table, Dict(
        :symbol_clean => :gene,
        :log2fc_num => :log2FC_diff_vs_undiff,
        :padj_num => :padj
    ))
    # DataFrames.sort! expects `rev` as a Bool or an AbstractVector{Bool}, not a tuple.
    sort!(selected_table, [:category, :padj, :log2FC_diff_vs_undiff], rev = [false, false, true])

    CSV.write(joinpath(processed_dir, "selected_genes_for_dissertation_table.csv"), selected_table)
end

save(joinpath(figdir, "PCA_volcano_diff_veh_vs_undiff_veh.pdf"), fig)
save(joinpath(figdir, "PCA_volcano_diff_veh_vs_undiff_veh.png"), fig)

# ============================================================
# Targeted heatmap
# ============================================================
target_genes = [
    "Runx2", "Sp7", "Alpl", "Bglap", "Col1a1", "Spp1", "Dmp1", "Phex", "Fgf23",
    "Sost", "Ptprz1", "Cthrc1", "Tnc", "Mmp13", "Smpd3", "Col11a1", "Col11a2",
    "Hmox1", "Nqo1", "Gclc", "Gclm", "Gsr", "Gss", "Txn1", "Txnrd1", "Srxn1",
    "Sqstm1", "Cat", "Tfrc", "Slc2a1", "Vegfa", "Egln1", "Ptgs1"
]

pca.sample = String.(pca.sample)
pca.state = String.(pca.state)
sample_order = [s for grp in ["undiff", "diff"] for s in pca.sample[pca.state .== grp]]
sample_cols = names(vst)[2:end]
gene_ids = vst.ensembl_gene_id

annot = unique(select(de, [:ensembl_gene_id, :SYMBOL]))
rename!(annot, :SYMBOL => :symbol)

heat_df = leftjoin(DataFrame(ensembl_gene_id = gene_ids), annot, on = :ensembl_gene_id)
heat_df.symbol_clean = [clean_symbol(sym, ens) for (sym, ens) in zip(heat_df.symbol, heat_df.ensembl_gene_id)]

keep_idx = findall(x -> x in target_genes, heat_df.symbol_clean)
if !isempty(keep_idx)
    sub_heat = vst[keep_idx, [:ensembl_gene_id; Symbol.(sample_cols)]]
    sub_annot = heat_df[keep_idx, :]
    mat = Matrix(select(sub_heat, Not(:ensembl_gene_id)))

    col_idx = [findfirst(==(s), sample_cols) for s in sample_order]
    any(isnothing, col_idx) && error("At least one sample in PCA metadata was not found in the VST matrix columns.")
    col_idx = Int[i for i in col_idx]
    mat = mat[:, col_idx]

    mat_z = similar(mat, Float64)
    for i in 1:size(mat, 1)
        μ = mean(mat[i, :])
        σ = std(mat[i, :])
        mat_z[i, :] .= (σ == 0 || isnan(σ)) ? 0.0 : (mat[i, :] .- μ) ./ σ
    end

    row_order = sortperm(String.(sub_annot.symbol_clean))
    mat_z = mat_z[row_order, :]
    row_labels = sub_annot.symbol_clean[row_order]
    col_labels = sample_order

    fig = Figure(size = (1200, 900), backgroundcolor = BG_COLOUR)
    ax = Axis(
        fig[1, 1],
        xticks = (1:length(col_labels), col_labels),
        yticks = (1:length(row_labels), row_labels),
        xticklabelrotation = π / 4,
        xticklabelsize = 10,
        yticklabelsize = 11,
        xlabel = "",
        ylabel = "",
        yreversed = true,
        leftspinecolor = AXIS_COLOUR,
        bottomspinecolor = AXIS_COLOUR,
        xtickcolor = AXIS_COLOUR,
        ytickcolor = AXIS_COLOUR,
        xticklabelcolor = AXIS_COLOUR,
        yticklabelcolor = AXIS_COLOUR,
    )

    # Pass only the matrix here. CairoMakie interprets explicit x/y vectors as cell edges,
    # which must be length n+1; using center vectors caused the conversion error.
    hm = heatmap!(ax, mat_z; colormap = :vik)
    Colorbar(fig[1, 2], hm, label = "Row z-score")

    save(joinpath(figdir, "heatmap_targeted_diff_veh_vs_undiff_veh.pdf"), fig)
    save(joinpath(figdir, "heatmap_targeted_diff_veh_vs_undiff_veh.png"), fig)
end

println("Done. Figures written to: $figdir")
