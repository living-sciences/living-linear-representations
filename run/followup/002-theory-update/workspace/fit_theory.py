#!/usr/bin/env python
"""Study 002 - theory-update.

Fit a small parametric model to how the paper's measured geometry quantities move
across the 2023-2025 era ladder, consuming ONLY followup/001-living-update/results/.
No model inference: this is a fit over 001's saved numbers.

Outputs (under ../results/):
  theory_fit.png / .pdf   - fit figure (causal off-diagonal & causal/Euclid ratio vs era)
  theory_fit.csv          - the per-model table used for the fits
  fit_results.json        - all fitted coefficients, per-era aggregates, family deltas
"""
import json, itertools
from pathlib import Path
import numpy as np

R001 = Path("/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/"
            "test-corpus/linear-representation-hypothesis/run/followup/001-living-update/results")
OUT = Path("/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/"
           "test-corpus/linear-representation-hypothesis/run/followup/002-theory-update/results")
OUT.mkdir(parents=True, exist_ok=True)

# -- model ladder (order = era ladder) --------------------------------------
MODELS = ["llama2-7b", "llama3.1-8b", "qwen2.5-7b", "qwen3-8b-base", "olmo3-7b"]
LABEL = {"llama2-7b": "Llama-2-7B", "llama3.1-8b": "Llama-3.1-8B",
         "qwen2.5-7b": "Qwen2.5-7B", "qwen3-8b-base": "Qwen3-8B-Base", "olmo3-7b": "OLMo-3-7B"}
FAMILY = {"llama2-7b": "Llama", "llama3.1-8b": "Llama", "qwen2.5-7b": "Qwen",
          "qwen3-8b-base": "Qwen", "olmo3-7b": "OLMo"}
# published/nominal total parameter counts (billions), approximate - from model cards
PARAMS_B = {"llama2-7b": 6.74, "llama3.1-8b": 8.03, "qwen2.5-7b": 7.62,
            "qwen3-8b-base": 8.19, "olmo3-7b": 7.30}

# -- load 001's saved numbers -----------------------------------------------
metrics = {m: json.load(open(R001 / f"metrics_{m}.json")) for m in MODELS}
agg = json.load(open(R001 / "aggregate.json"))
per = agg["per_model"]

rows = []
for m in MODELS:
    mm, pm = metrics[m], per[m]
    rows.append(dict(
        model=m, label=LABEL[m], family=FAMILY[m],
        year=int(mm["era"]), params_b=PARAMS_B[m],
        hidden=mm["hidden_size"], vocab=mm["vocab_size"],
        coverage_pairs=mm["coverage"]["n_singletoken_pairs_total"],
        # intersection (22 concepts common to all 5) = coverage-controlled series
        causal_inter=pm["inter_causal_mean"], euclid_inter=pm["inter_euclid_mean"],
        R_inter=pm["inter_ratio"],
        n_sep_inter=pm["inter_n_separated"], n_tot_inter=pm["inter_n_total"],
        # own viable set (paper-comparable)
        causal_own=pm["own_causal_mean"], euclid_own=pm["own_euclid_mean"],
        R_own=pm["own_ratio"], n_sep_own=pm["own_n_separated"], n_viable_own=pm["own_n_viable"],
        # all-27 (paper's raw set, no viability gate)
        causal_all27=mm["orthogonality_all27"]["causal_mean"],
        euclid_all27=mm["orthogonality_all27"]["euclid_mean"],
        R_all27=mm["orthogonality_all27"]["ratio_euclid_over_causal_mean"],
        exception=";".join(mm["subspace"]["exceptions_d_lt_1"].keys()) or "none",
    ))

# convenient arrays (primary series = intersection, coverage-controlled)
year   = np.array([r["year"] for r in rows], float)
logp   = np.log(np.array([r["params_b"] for r in rows], float))
causal = np.array([r["causal_inter"] for r in rows], float)
euclid = np.array([r["euclid_inter"] for r in rows], float)
R      = np.array([r["R_inter"] for r in rows], float)

# ---------------------------------------------------------------------------
def ols(x, y):
    """Simple linear regression y = a + b x. Returns dict with slope/intercept/R2."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    return dict(slope=float(b), intercept=float(a), r2=float(r2), pearson_r=r,
                ss_res=ss_res, ss_tot=ss_tot)

def loo(x, y):
    """Leave-one-out: refit on the other 4 points, collect slopes and held-out residuals."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    slopes, preds = [], np.full(n, np.nan)
    for i in range(n):
        mask = np.arange(n) != i
        b, a = np.polyfit(x[mask], y[mask], 1)
        slopes.append(float(b))
        preds[i] = a + b * x[i]
    slopes = np.array(slopes)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(slope_mean=float(slopes.mean()), slope_min=float(slopes.min()),
                slope_max=float(slopes.max()), slope_std=float(slopes.std(ddof=1)),
                loo_rmse=float(np.sqrt(ss_res / n)),
                loo_r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                all_slopes=[float(s) for s in slopes])

def boot_slope(x, y, B=5000, seed=0):
    """Bootstrap slope CI (crude with n=5; report alongside LOO)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2:
            continue
        b, _a = np.polyfit(x[idx], y[idx], 1)
        out.append(float(b))
    out = np.array(out)
    return dict(n_valid=int(out.size), slope_median=float(np.median(out)),
                ci2_5=float(np.percentile(out, 2.5)), ci97_5=float(np.percentile(out, 97.5)),
                frac_negative=float(np.mean(out < 0)), frac_positive=float(np.mean(out > 0)))

def fit_block(x, y, xname):
    return dict(covariate=xname, ols=ols(x, y), loo=loo(x, y), bootstrap=boot_slope(x, y))

# ---- 1. Orthogonality (causal off-diag) & ratio R vs scale/era ------------
fits = {
    "causal_inter~year":  fit_block(year, causal, "year"),
    "causal_inter~logparams": fit_block(logp, causal, "log_params"),
    "R_inter~year":       fit_block(year, R, "year"),
    "R_inter~logparams":  fit_block(logp, R, "log_params"),
    "euclid_inter~year":  fit_block(year, euclid, "year"),
}

# two-covariate fit (year + log params) - reported but flagged as overfit (n=5, p=3)
def multi_fit(y):
    X = np.column_stack([np.ones(5), year - year.mean(), logp - logp.mean()])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    n, p = 5, 3
    adj = 1 - (1 - r2) * (n - 1) / (n - p) if n - p > 0 else float("nan")
    return dict(intercept=float(coef[0]), beta_year=float(coef[1]),
                beta_logparams=float(coef[2]), r2=float(r2), adj_r2=float(adj),
                note="n=5, p=3: only 2 residual dof; adjusted-R2 and instability caveat apply")
multi = {"causal_inter": multi_fit(causal), "R_inter": multi_fit(R)}

# ---- per-era aggregates ----------------------------------------------------
eras = sorted(set(int(r["year"]) for r in rows))
def era_stats(key):
    d = {}
    for e in eras:
        vals = [r[key] for r in rows if r["year"] == e]
        d[str(e)] = dict(mean=float(np.mean(vals)), n=len(vals),
                         min=float(np.min(vals)), max=float(np.max(vals)),
                         models=[r["label"] for r in rows if r["year"] == e])
    return d
per_era = {k: era_stats(k) for k in
           ["causal_inter", "euclid_inter", "R_inter", "causal_own", "R_own",
            "n_sep_own", "n_viable_own"]}

# ---- 2. Family effect: paired within-family deltas ------------------------
def by(m, key):
    return next(r[key] for r in rows if r["model"] == m)
family_deltas = {
    "Llama-2->Llama-3.1": dict(
        d_causal=by("llama3.1-8b", "causal_inter") - by("llama2-7b", "causal_inter"),
        d_R=by("llama3.1-8b", "R_inter") - by("llama2-7b", "R_inter")),
    "Qwen2.5->Qwen3": dict(
        d_causal=by("qwen3-8b-base", "causal_inter") - by("qwen2.5-7b", "causal_inter"),
        d_R=by("qwen3-8b-base", "R_inter") - by("qwen2.5-7b", "R_inter")),
}
# across-family, same era (2024): Llama-3.1 vs Qwen2.5
cross_family = {
    "2024_Llama3.1_vs_Qwen2.5": dict(
        d_causal=by("llama3.1-8b", "causal_inter") - by("qwen2.5-7b", "causal_inter"),
        d_R=by("llama3.1-8b", "R_inter") - by("qwen2.5-7b", "R_inter")),
    "2025_Qwen3_vs_OLMo3": dict(
        d_causal=by("qwen3-8b-base", "causal_inter") - by("olmo3-7b", "causal_inter"),
        d_R=by("qwen3-8b-base", "R_inter") - by("olmo3-7b", "R_inter")),
}
mean_within_absdR = float(np.mean([abs(v["d_R"]) for v in family_deltas.values()]))
cross_2024_absdR = abs(cross_family["2024_Llama3.1_vs_Qwen2.5"]["d_R"])
stability = dict(
    mean_within_family_abs_dR=mean_within_absdR,
    cross_family_same_era_2024_abs_dR=cross_2024_absdR,
    ratio_cross_over_within=cross_2024_absdR / mean_within_absdR,
    interpretation="R changes ~{:.1f}x more across families (same era) than within a "
                   "family generation step".format(cross_2024_absdR / mean_within_absdR))

# ---- 3. Subspace survival --------------------------------------------------
subspace = {r["model"]: dict(n_sep_own=r["n_sep_own"], n_viable_own=r["n_viable_own"],
                             frac=r["n_sep_own"] / r["n_viable_own"],
                             exception=r["exception"]) for r in rows}

# ---- 4. Causal-vs-Euclidean advantage (R>1?) ------------------------------
R_all_own = np.array([r["R_own"] for r in rows])
R_all_inter = R
advantage = dict(
    R_inter_min=float(R.min()), R_inter_max=float(R.max()),
    R_own_min=float(R_all_own.min()), R_own_max=float(R_all_own.max()),
    R_gt1_all_models=bool(np.all(R_all_inter > 1) and np.all(R_all_own > 1)),
    per_era_R_inter={e: per_era["R_inter"][e]["mean"] for e in per_era["R_inter"]},
    trend_slope_R_year=fits["R_inter~year"]["ols"]["slope"],
    note="R>1 means causal strictly more orthogonal than Euclidean.")

# ---- covariate correlations (confound quantification) ---------------------
def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])
cov_corr = {
    "causal_vs_year": corr(year, causal),
    "causal_vs_logparams": corr(logp, causal),
    "causal_vs_vocab": corr([r["vocab"] for r in rows], causal),
    "causal_vs_coverage": corr([r["coverage_pairs"] for r in rows], causal),
    "R_vs_year": corr(year, R),
    "R_vs_logparams": corr(logp, R),
    "R_vs_vocab": corr([r["vocab"] for r in rows], R),
    "R_vs_coverage": corr([r["coverage_pairs"] for r in rows], R),
    # coverage/era confound: Llama-2 (2023) has far fewer single-token pairs
    "coverage_vs_year": corr(year, [r["coverage_pairs"] for r in rows]),
}

results = dict(
    provenance=dict(
        source_dir=str(R001),
        source_files=[f"metrics_{m}.json" for m in MODELS] + ["aggregate.json"],
        primary_series="22-concept intersection (inter_*), coverage-controlled across all 5 models",
        note="No model inference; all quantities read from 001's saved JSON."),
    table=rows, fits=fits, multi_covariate=multi, per_era=per_era,
    family_deltas=family_deltas, cross_family=cross_family, family_stability=stability,
    subspace=subspace, causal_advantage=advantage, covariate_correlations=cov_corr,
)
json.dump(results, open(OUT / "fit_results.json", "w"), indent=2)

# ---- CSV of the fit table --------------------------------------------------
import csv
cols = ["model", "label", "family", "year", "params_b", "hidden", "vocab", "coverage_pairs",
        "causal_inter", "euclid_inter", "R_inter", "causal_own", "euclid_own", "R_own",
        "causal_all27", "euclid_all27", "R_all27", "n_sep_own", "n_viable_own", "exception"]
with open(OUT / "theory_fit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

# ---- print a summary -------------------------------------------------------
print("=== per-era means (22-concept intersection) ===")
for e in eras:
    s = per_era["causal_inter"][str(e)]; rr = per_era["R_inter"][str(e)]
    print(f"  {e}: causal_mean={s['mean']:.4f}  R={rr['mean']:.3f}  models={s['models']}")
print("\n=== causal_inter ~ year ===")
o = fits["causal_inter~year"]
print(f"  slope={o['ols']['slope']:+.5f}/yr  R2={o['ols']['r2']:.3f}  "
      f"LOO slope[{o['loo']['slope_min']:+.5f},{o['loo']['slope_max']:+.5f}]  "
      f"boot95%[{o['bootstrap']['ci2_5']:+.5f},{o['bootstrap']['ci97_5']:+.5f}]")
print("=== R_inter ~ year ===")
o = fits["R_inter~year"]
print(f"  slope={o['ols']['slope']:+.5f}/yr  R2={o['ols']['r2']:.3f}  "
      f"LOO slope[{o['loo']['slope_min']:+.5f},{o['loo']['slope_max']:+.5f}]  "
      f"boot95%[{o['bootstrap']['ci2_5']:+.5f},{o['bootstrap']['ci97_5']:+.5f}]  "
      f"frac_neg={o['bootstrap']['frac_negative']:.2f}")
print("\n=== family stability ===")
print(f"  mean within-family |dR|={stability['mean_within_family_abs_dR']:.4f}  "
      f"cross-family(2024) |dR|={stability['cross_family_same_era_2024_abs_dR']:.4f}  "
      f"ratio={stability['ratio_cross_over_within']:.1f}x")
print("\n=== causal advantage R ===")
print(f"  R_inter range [{advantage['R_inter_min']:.3f},{advantage['R_inter_max']:.3f}]  "
      f"R>1 all models: {advantage['R_gt1_all_models']}")
print(f"  per-era R: {advantage['per_era_R_inter']}")
print("\n=== subspace ===")
for m in MODELS:
    s = subspace[m]
    print(f"  {LABEL[m]:16s} {s['n_sep_own']}/{s['n_viable_own']}  exc={s['exception']}")
print("\nwrote", OUT / "fit_results.json", OUT / "theory_fit.csv")
