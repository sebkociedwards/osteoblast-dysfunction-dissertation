# ============================================================
# 01_bulk_rnaseq_deseq2_vehicle_only.R
# Vehicle-only differentiation analysis for GSE205791 / PRJNA847597
# Comparison: diff_veh vs undiff_veh
# ============================================================

required_pkgs <- c(
  "data.table",
  "dplyr",
  "tibble",
  "stringr",
  "DESeq2",
  "apeglm",
  "org.Mm.eg.db",
  "AnnotationDbi"
)

to_install <- required_pkgs[!required_pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) {
  cran_pkgs <- setdiff(to_install, c("DESeq2", "apeglm", "org.Mm.eg.db", "AnnotationDbi"))
  if (length(cran_pkgs) > 0) install.packages(cran_pkgs)
  if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
  bioc_pkgs <- intersect(to_install, c("DESeq2", "apeglm", "org.Mm.eg.db", "AnnotationDbi"))
  if (length(bioc_pkgs) > 0) BiocManager::install(bioc_pkgs, ask = FALSE, update = FALSE)
}

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tibble)
  library(stringr)
  library(DESeq2)
  library(apeglm)
  library(org.Mm.eg.db)
  library(AnnotationDbi)
})

# ----------------------------
# Paths
# ----------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }

  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }

  # Fallback for interactive use in RStudio / source()
  return(normalizePath(getwd()))
}

script_dir <- get_script_dir()
project_dir <- normalizePath(file.path(script_dir, ".."))
raw_dir <- file.path(project_dir, "raw_data")
out_dir <- file.path(project_dir, "processed_data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

counts_file <- file.path(raw_dir, "PRJNA847597_bulkRNAseq_counts.txt")
meta_file   <- file.path(raw_dir, "PRJNA847597_metadata.txt")

if (!file.exists(counts_file)) {
  stop(
    paste0(
      "Counts file not found: ", counts_file, "\n",
      "script_dir=", script_dir, "\n",
      "project_dir=", project_dir, "\n",
      "getwd()=", getwd()
    )
  )
}

if (!file.exists(meta_file)) {
  stop(
    paste0(
      "Metadata file not found: ", meta_file, "\n",
      "script_dir=", script_dir, "\n",
      "project_dir=", project_dir, "\n",
      "getwd()=", getwd()
    )
  )
}

# ----------------------------
# Input
# ----------------------------
counts_df <- fread(counts_file, sep = "\t", header = TRUE)
meta <- fread(meta_file, sep = "\t", header = TRUE) %>% as.data.frame()

meta <- meta %>%
  dplyr::mutate(
    sample = as.character(sample),
    state = factor(state, levels = c("undiff", "diff")),
    treatment = factor(treatment),
    condition = factor(condition)
  )

counts <- counts_df %>%
  column_to_rownames("ensembl_gene_id") %>%
  as.matrix()
storage.mode(counts) <- "integer"

meta <- meta %>%
  dplyr::mutate(
    sample = as.character(sample),
    state = factor(state, levels = c("undiff", "diff")),
    treatment = factor(treatment),
    condition = factor(condition)
  )

# ----------------------------
# Restrict to vehicle only
# ----------------------------
meta_sub <- meta %>%
  dplyr::filter(treatment == "veh") %>%
  dplyr::arrange(state, replicate)

stopifnot(all(meta_sub$sample %in% colnames(counts)))
counts_sub <- counts[, meta_sub$sample, drop = FALSE]
meta_sub <- meta_sub %>% dplyr::slice(match(colnames(counts_sub), sample))
stopifnot(identical(meta_sub$sample, colnames(counts_sub)))

write.csv(meta_sub, file.path(out_dir, "sample_metadata_vehicle_only.csv"), row.names = FALSE)

# ----------------------------
# Prefilter
# ----------------------------
keep <- rowSums(counts_sub >= 10) >= 3
counts_filt <- counts_sub[keep, , drop = FALSE]

write.csv(
  data.frame(
    metric = c("genes_before_filter", "genes_after_filter"),
    value = c(nrow(counts_sub), nrow(counts_filt))
  ),
  file.path(out_dir, "filtering_summary.csv"),
  row.names = FALSE
)

# ----------------------------
# DESeq2
# ----------------------------
dds <- DESeqDataSetFromMatrix(
  countData = counts_filt,
  colData = meta_sub,
  design = ~ state
)

dds <- DESeq(dds)

write.csv(
  data.frame(sample = colnames(dds), size_factor = sizeFactors(dds)),
  file.path(out_dir, "size_factors.csv"),
  row.names = FALSE
)

# ----------------------------
# VST / PCA / distances
# ----------------------------
vsd <- vst(dds, blind = FALSE)
vst_mat <- assay(vsd)

write.csv(
  data.frame(ensembl_gene_id = rownames(vst_mat), vst_mat, check.names = FALSE),
  file.path(out_dir, "vst_vehicle_matrix.csv"),
  row.names = FALSE
)

pca <- prcomp(t(vst_mat), scale. = FALSE)
pca_percent <- (pca$sdev^2) / sum(pca$sdev^2)

pca_df <- data.frame(
  sample = rownames(pca$x),
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  PC3 = pca$x[, 3]
) %>%
  left_join(meta_sub, by = "sample")

write.csv(pca_df, file.path(out_dir, "pca_vehicle_coordinates.csv"), row.names = FALSE)
write.csv(
  data.frame(PC = paste0("PC", seq_along(pca_percent)), variance_explained = pca_percent),
  file.path(out_dir, "pca_vehicle_variance_explained.csv"),
  row.names = FALSE
)

sample_dists <- as.matrix(dist(t(vst_mat)))
write.csv(
  data.frame(sample = rownames(sample_dists), sample_dists, check.names = FALSE),
  file.path(out_dir, "sample_distance_vehicle_matrix.csv"),
  row.names = FALSE
)

# ----------------------------
# Annotation
# ----------------------------
annot <- AnnotationDbi::select(
  org.Mm.eg.db,
  keys = rownames(dds),
  keytype = "ENSEMBL",
  columns = c("SYMBOL", "GENENAME", "ENTREZID")
) %>%
  dplyr::distinct(ENSEMBL, .keep_all = TRUE) %>%
  dplyr::rename(ensembl_gene_id = ENSEMBL)

# ----------------------------
# Results helpers
# ----------------------------
make_tidy_res <- function(res_obj, contrast_name) {
  as.data.frame(res_obj) %>%
    rownames_to_column("ensembl_gene_id") %>%
    as_tibble() %>%
    dplyr::left_join(annot, by = "ensembl_gene_id") %>%
    dplyr::mutate(
      contrast = contrast_name,
      sig_padj_0_05 = !is.na(padj) & padj < 0.05,
      sig_padj_0_10 = !is.na(padj) & padj < 0.10,
      sig_lfc1_padj_0_05 = !is.na(padj) & padj < 0.05 & abs(log2FoldChange) >= 1
    ) %>%
    dplyr::arrange(padj, desc(abs(log2FoldChange)))
}

# ----------------------------
# Main contrast
# ----------------------------
res <- results(dds, contrast = c("state", "diff", "undiff"), alpha = 0.05)
res_shrunk <- lfcShrink(dds, coef = "state_diff_vs_undiff", type = "apeglm")

tidy_res <- make_tidy_res(res, "diff_veh_vs_undiff_veh")
tidy_res_shrunk <- make_tidy_res(res_shrunk, "diff_veh_vs_undiff_veh_shrunk")

write.csv(tidy_res, file.path(out_dir, "diff_veh_vs_undiff_veh.csv"), row.names = FALSE)
write.csv(tidy_res_shrunk, file.path(out_dir, "diff_veh_vs_undiff_veh_shrunk.csv"), row.names = FALSE)

write.csv(
  tibble(
    contrast = "diff_veh_vs_undiff_veh",
    n_total = nrow(tidy_res),
    n_sig_padj_0_05 = sum(tidy_res$sig_padj_0_05, na.rm = TRUE),
    n_sig_padj_0_10 = sum(tidy_res$sig_padj_0_10, na.rm = TRUE),
    n_sig_lfc1_padj_0_05 = sum(tidy_res$sig_lfc1_padj_0_05, na.rm = TRUE)
  ),
  file.path(out_dir, "deseq2_summary_counts.csv"),
  row.names = FALSE
)

# ----------------------------
# Dissertation-focused gene panels
# ----------------------------
core_maturation_genes <- c(
  "Runx2", "Sp7", "Alpl", "Col1a1", "Bglap", "Spp1", "Dmp1", "Phex", "Fgf23",
  "Sost", "Ptprz1", "Cthrc1", "Tnc", "Mmp13", "Smpd3", "Col11a1", "Col11a2"
)

stress_genes <- c(
  "Hmox1", "Nqo1", "Gclc", "Gclm", "Gsr", "Txn1", "Txnrd1", "Srxn1",
  "Sqstm1", "Prdx1", "Prdx2", "Sod1", "Sod2", "Cat", "Tfrc", "Slc2a1", "Vegfa", "Egln1"
)

target_genes <- unique(c(core_maturation_genes, stress_genes))

write.csv(
  tidy_res %>% dplyr::filter(SYMBOL %in% target_genes),
  file.path(out_dir, "target_genes_diff_veh_vs_undiff_veh.csv"),
  row.names = FALSE
)

write.csv(
  tidy_res %>% dplyr::filter(SYMBOL %in% core_maturation_genes),
  file.path(out_dir, "maturation_genes_diff_veh_vs_undiff_veh.csv"),
  row.names = FALSE
)

write.csv(
  tidy_res %>% dplyr::filter(SYMBOL %in% stress_genes),
  file.path(out_dir, "stress_genes_diff_veh_vs_undiff_veh.csv"),
  row.names = FALSE
)

# ----------------------------
# Ranked list
# ----------------------------
ranked <- tidy_res %>%
  dplyr::filter(!is.na(stat), !is.na(SYMBOL)) %>%
  dplyr::distinct(SYMBOL, .keep_all = TRUE) %>%
  dplyr::select(SYMBOL, stat) %>%
  dplyr::arrange(desc(stat))

write.table(
  ranked,
  file = file.path(out_dir, "ranked_diff_veh_vs_undiff_veh.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE
)

message("Done. Outputs written to: ", out_dir)
