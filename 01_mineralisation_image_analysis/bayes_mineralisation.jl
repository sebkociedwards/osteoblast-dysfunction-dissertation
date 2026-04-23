using CSV
using DataFrames
using Statistics
using Distributions

# =========================
# Paths
# =========================

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PROC_DIR = joinpath(ROOT, "processed_data")

const AREA_FILE = joinpath(PROC_DIR, "delta_area.csv")
const NODULE_FILE = joinpath(PROC_DIR, "delta_nodule_count.csv")

const OUT_AREA = joinpath(PROC_DIR, "bayes_area_summary.csv")
const OUT_NODULE = joinpath(PROC_DIR, "bayes_nodule_summary.csv")

# =========================
# Prior settings
# Normal-Inverse-Chi-Squared prior
# x_i ~ Normal(mu, sigma^2)
# mu | sigma^2 ~ Normal(mu0, sigma^2 / kappa0)
# sigma^2 ~ Inv-χ²(nu0, sigma0_sq)
# =========================

const MU0 = 0.0
const KAPPA0 = 0.25
const NU0 = 1.0
const SIGMA0_SQ = 4.0

const TREATMENT_ORDER = [
    "IS 2 mM",
    "SFN 1 µM",
    "SFN 5 µM",
    "IS 2 mM + SFN 1 µM",
    "IS 2 mM + SFN 5 µM",
]

function pick_col(df::DataFrame, candidates::Vector{String})
    names_lower = Dict(lowercase(String(n)) => Symbol(n) for n in names(df))
    for c in candidates
        key = lowercase(c)
        if haskey(names_lower, key)
            return names_lower[key]
        end
    end
    error("Could not find any of these columns: $(candidates). Found: $(names(df))")
end

# =========================
# Posterior update
# =========================

function posterior_params(x::Vector{Float64};
    mu0::Float64 = MU0,
    kappa0::Float64 = KAPPA0,
    nu0::Float64 = NU0,
    sigma0_sq::Float64 = SIGMA0_SQ
)
    n = length(x)
    n == 0 && error("No observations supplied.")

    xbar = mean(x)
    s2 = n > 1 ? var(x; corrected = true) : 0.0

    kappa_n = kappa0 + n
    mu_n = (kappa0 * mu0 + n * xbar) / kappa_n
    nu_n = nu0 + n

    sigma_n_sq = (
        nu0 * sigma0_sq +
        (n - 1) * s2 +
        (kappa0 * n / kappa_n) * (xbar - mu0)^2
    ) / nu_n

    return (
        n = n,
        xbar = xbar,
        s2 = s2,
        mu_n = mu_n,
        kappa_n = kappa_n,
        nu_n = nu_n,
        sigma_n_sq = sigma_n_sq,
    )
end

# Marginal posterior of mu is Student-t
function posterior_mu_dist(post)
    loc = post.mu_n
    scale = sqrt(post.sigma_n_sq / post.kappa_n)
    return LocationScale(loc, scale, TDist(post.nu_n))
end

function summarise_bayes(x::Vector{Float64})
    post = posterior_params(x)
    d = posterior_mu_dist(post)

    post_mean = mean(d)
    ci_low = quantile(d, 0.025)
    ci_high = quantile(d, 0.975)

    p_less_0 = cdf(d, 0.0)
    p_less_neg1 = cdf(d, -1.0)
    p_greater_0 = 1.0 - p_less_0

    return (
        n = post.n,
        sample_mean_log2_fc = post.xbar,
        posterior_mean_log2_fc = post_mean,
        ci_low_log2_fc = ci_low,
        ci_high_log2_fc = ci_high,
        posterior_mean_fold_change = 2.0 ^ post_mean,
        p_mu_less_0 = p_less_0,
        p_mu_less_neg1 = p_less_neg1,
        p_mu_greater_0 = p_greater_0,
        mu_n = post.mu_n,
        kappa_n = post.kappa_n,
        nu_n = post.nu_n,
        sigma_n_sq = post.sigma_n_sq,
    )
end

# =========================
# Endpoint runner
# =========================

function analyse_endpoint(file::String, endpoint_name::String)
    df = CSV.read(file, DataFrame)
    rename!(df, Symbol.(strip.(string.(names(df)))))

    treatment_col = pick_col(df, ["treatment", "condition", "group"])
    log2fc_col = pick_col(df, ["log2_fc", "log2 fold change", "delta", "log2fc"])

    out = DataFrame(
        endpoint = String[],
        treatment = String[],
        n = Int[],
        sample_mean_log2_fc = Float64[],
        posterior_mean_log2_fc = Float64[],
        ci_low_log2_fc = Float64[],
        ci_high_log2_fc = Float64[],
        posterior_mean_fold_change = Float64[],
        p_mu_less_0 = Float64[],
        p_mu_less_neg1 = Float64[],
        p_mu_greater_0 = Float64[],
    )

    for t in TREATMENT_ORDER
        sdf = filter(treatment_col => ==(t), df)
        vals = Float64.(collect(skipmissing(sdf[!, log2fc_col])))
        isempty(vals) && continue

        s = summarise_bayes(vals)

        push!(out, (
            endpoint_name,
            t,
            s.n,
            s.sample_mean_log2_fc,
            s.posterior_mean_log2_fc,
            s.ci_low_log2_fc,
            s.ci_high_log2_fc,
            s.posterior_mean_fold_change,
            s.p_mu_less_0,
            s.p_mu_less_neg1,
            s.p_mu_greater_0,
        ))
    end

    return out
end

# =========================
# Run
# =========================

area_summary = analyse_endpoint(AREA_FILE, "Mineralised area")
nodule_summary = analyse_endpoint(NODULE_FILE, "Nodule count")

CSV.write(OUT_AREA, area_summary)
CSV.write(OUT_NODULE, nodule_summary)

println("\nBayesian summary: mineralised area")
show(area_summary, allrows = true, allcols = true)
println("\n")

println("\nBayesian summary: nodule count")
show(nodule_summary, allrows = true, allcols = true)
println("\n")
