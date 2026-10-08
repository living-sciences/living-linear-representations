"""Study 003: re-aggregate ALL models (5 prior 2023-2025 + 2 new 2026) and extend
the over-time figure to the 2026 point. Prior per-model results are REUSED verbatim
from study 001 (copied into this study's results/); the 2 new 2026 models are computed
this session. Intersection viable set + apples-to-apples series + over-time figure."""
import json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run/followup/003-living-update/results"

# order by release date; era labels. Last two are the NEW 2026 points.
ORDER = [
    ("llama2-7b",       "Llama-2-7B\n(2023)"),
    ("llama3.1-8b",     "Llama-3.1-8B\n(2024)"),
    ("qwen2.5-7b",      "Qwen2.5-7B\n(2024)"),
    ("qwen3-8b-base",   "Qwen3-8B\n(2025)"),
    ("olmo3-7b",        "OLMo-3-7B\n(2025)"),
    ("qwen3.5-9b-base", "Qwen3.5-9B\n(2026)"),
    ("gemma-4-12b",     "gemma-4-12B\n(2026)"),
]
NEW_2026 = {"qwen3.5-9b-base", "gemma-4-12b"}
CONCEPTS = [
    'verb=>3pSg','verb=>Ving','verb=>Ved','Ving=>3pSg','Ving=>Ved','3pSg=>Ved',
    'verb=>V + able','verb=>V + er','verb=>V + tion','verb=>V + ment',
    'adj=>un + adj','adj=>adj + ly','small=>big','thing=>color','thing=>part',
    'country=>capital','pronoun=>possessive','male=>female','lower=>upper',
    'noun=>plural','adj=>comparative','adj=>superlative','frequent=>infrequent',
    'English=>French','French=>German','French=>Spanish','German=>Spanish',
]

metrics = {k: json.load(open(f"{RES}/metrics_{k}.json")) for k, _ in ORDER}

viable_sets = [set(metrics[k]["coverage"]["viable_concept_indices"]) for k, _ in ORDER]
inter = sorted(set.intersection(*viable_sets))
inter_names = [CONCEPTS[i] for i in inter]
print(f"Intersection viable concepts across all {len(ORDER)} models (n={len(inter)}): {inter_names}")

def offdiag_stats(rows):
    C = rows / rows.norm(dim=1, keepdim=True)
    M = (C @ C.T).abs()
    n = M.shape[0]
    off = M[~torch.eye(n, dtype=bool)]
    return float(off.mean()), float(off.median())

agg = {}
for k, _ in ORDER:
    cg = torch.load(f"{RES}/concept_g_{k}.pt")[inter]
    cga = torch.load(f"{RES}/concept_gamma_{k}.pt")[inter]
    cm, cmed = offdiag_stats(cg)
    em, emed = offdiag_stats(cga)
    cohen = metrics[k]["subspace"]["cohens_d"]
    inter_sep = sum(1 for nm in inter_names if cohen.get(nm, 0) >= 1.0)
    agg[k] = {
        "era": metrics[k]["era"],
        "inter_causal_mean": cm, "inter_causal_median": cmed,
        "inter_euclid_mean": em, "inter_euclid_median": emed,
        "inter_ratio": em / cm,
        "inter_n_separated": inter_sep, "inter_n_total": len(inter),
        "all27_causal_mean": metrics[k]["orthogonality_all27"]["causal_mean"],
        "all27_euclid_mean": metrics[k]["orthogonality_all27"]["euclid_mean"],
        "all27_ratio": metrics[k]["orthogonality_all27"]["ratio_euclid_over_causal_mean"],
        "own_causal_mean": metrics[k]["orthogonality_own_viable"]["causal_mean"],
        "own_euclid_mean": metrics[k]["orthogonality_own_viable"]["euclid_mean"],
        "own_ratio": metrics[k]["orthogonality_own_viable"]["ratio_euclid_over_causal_mean"],
        "own_n_separated": metrics[k]["subspace"]["n_separated"],
        "own_n_viable": metrics[k]["subspace"]["n_viable"],
        "r_sep": metrics[k]["uncorrelatedness"]["r_separable_maleFemale_vs_engFr"],
        "r_nonsep": metrics[k]["uncorrelatedness"]["r_nonseparable_verb3pSg_vs_verbVing"],
        "is_new_2026": k in NEW_2026,
    }
    tag = " [NEW 2026]" if k in NEW_2026 else ""
    print(f"{k:16s} inter causal/euclid={cm:.4f}/{em:.4f} ratio={em/cm:.2f} "
          f"sep={inter_sep}/{len(inter)} | own sep={agg[k]['own_n_separated']}/{agg[k]['own_n_viable']}"
          f" | all27 ratio={agg[k]['all27_ratio']:.2f}{tag}")

json.dump({"n_models": len(ORDER), "intersection_indices": inter, "intersection_names": inter_names,
           "new_2026_models": sorted(NEW_2026), "per_model": agg},
          open(f"{RES}/aggregate.json", "w"), indent=2)

# ---------------- FIGURE ----------------
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "gray": "#666666", "red": "#D55E00"}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
labels = [lbl for _, lbl in ORDER]
x = np.arange(len(ORDER))
causal = [agg[k]["inter_causal_mean"] for k, _ in ORDER]
euclid = [agg[k]["inter_euclid_mean"] for k, _ in ORDER]
sep = [agg[k]["inter_n_separated"] / agg[k]["inter_n_total"] for k, _ in ORDER]
new_mask = [k in NEW_2026 for k, _ in ORDER]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8.2), sharex=True,
                               gridspec_kw={"height_ratios": [2.1, 1]})

# shade the 2026 region
first_new = next(i for i, m in enumerate(new_mask) if m)
ax1.axvspan(first_new - 0.5, len(ORDER) - 0.5, color=OKABE["orange"], alpha=0.06)
ax2.axvspan(first_new - 0.5, len(ORDER) - 0.5, color=OKABE["orange"], alpha=0.06)
ax1.text(first_new - 0.45, ax1.get_ylim()[1], "", fontsize=8)

# top: orthogonality lines
ax1.plot(x, euclid, "--o", color=OKABE["orange"], lw=2, ms=7, label="Euclidean inner product")
ax1.plot(x, causal, "-o", color=OKABE["blue"], lw=2, ms=7, label="Causal inner product")
# ring the NEW 2026 points
for xi, isnew in enumerate(new_mask):
    if isnew:
        ax1.scatter([xi, xi], [causal[xi], euclid[xi]], s=180, facecolors="none",
                    edgecolors=OKABE["red"], linewidths=1.8, zorder=5)
ax1.set_ylim(0, max(euclid) * 1.35)
ax1.set_ylabel("mean off-diagonal |inner product|\n(0 = perfectly orthogonal)")
ax1.set_title("Does the Linear Representation Hypothesis survive to 2026 models?\n"
              f"Orthogonality across 2023–2026 base models  (n={len(inter)} concepts viable in all {len(ORDER)})")
ax1.annotate("on-disk replicated Llama-2 (all 27):\n0.046 causal / 0.069 Euclid\n[concept_g.pt / concept_gamma.pt].\n"
             "Lines use the 22-concept intersection.",
             xy=(0, causal[0]), xytext=(1.2, 0.020),
             fontsize=8.3, color=OKABE["gray"], ha="left",
             arrowprops=dict(arrowstyle="->", color=OKABE["gray"], lw=1))
ax1.annotate("2026 refresh\n(this session)", xy=(len(ORDER) - 1.5, max(euclid) * 1.24),
             fontsize=9, color=OKABE["red"], ha="center", style="italic")
for xi, (c, e) in enumerate(zip(causal, euclid)):
    ax1.annotate(f"{c:.3f}", (xi, c), textcoords="offset points", xytext=(0, -15),
                 ha="center", fontsize=8, color=OKABE["blue"])
    ax1.annotate(f"{e:.3f}", (xi, e), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8, color=OKABE["orange"])
    ax1.annotate(f"×{e/c:.2f}", (xi, (c + e) / 2), textcoords="offset points", xytext=(0, 0),
                 ha="center", fontsize=8, color=OKABE["gray"], style="italic")
ax1.legend(loc="upper left", frameon=False)

# bottom: subspace separation bars
colors = [OKABE["red"] if isnew else OKABE["green"] for isnew in new_mask]
ax2.bar(x, sep, color=colors, width=0.55, alpha=0.85)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("fraction of concepts\nwith linear subspace\n(Cohen's d ≥ 1.0)")
for xi, (k, _) in enumerate(ORDER):
    ax2.annotate(f"{agg[k]['inter_n_separated']}/{agg[k]['inter_n_total']}",
                 (xi, sep[xi]), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=9)
ax2.axhline(1.0, color=OKABE["gray"], ls=":", lw=1)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RES}/over_time_orthogonality.{ext}", facecolor="white")
print("saved over_time_orthogonality.png/pdf (extended to 2026)")
