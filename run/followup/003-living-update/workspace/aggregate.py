"""Aggregate the 5 per-model metrics: intersection viable set, apples-to-apples
orthogonality + subspace series, and the required over-time figure."""
import json, torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run/followup/001-living-update/results"

# order by release date, with era labels
ORDER = [
    ("llama2-7b",     "Llama-2-7B\n(2023)"),
    ("llama3.1-8b",   "Llama-3.1-8B\n(2024)"),
    ("qwen2.5-7b",    "Qwen2.5-7B\n(2024)"),
    ("qwen3-8b-base", "Qwen3-8B\n(2025)"),
    ("olmo3-7b",      "OLMo-3-7B\n(2025)"),
]
CONCEPTS = [  # fixed order, matches run_model.CONCEPT_NAMES
    'verb=>3pSg','verb=>Ving','verb=>Ved','Ving=>3pSg','Ving=>Ved','3pSg=>Ved',
    'verb=>V + able','verb=>V + er','verb=>V + tion','verb=>V + ment',
    'adj=>un + adj','adj=>adj + ly','small=>big','thing=>color','thing=>part',
    'country=>capital','pronoun=>possessive','male=>female','lower=>upper',
    'noun=>plural','adj=>comparative','adj=>superlative','frequent=>infrequent',
    'English=>French','French=>German','French=>Spanish','German=>Spanish',
]

metrics = {k: json.load(open(f"{RES}/metrics_{k}.json")) for k, _ in ORDER}

# intersection of viable concept indices across all 5 models
viable_sets = [set(metrics[k]["coverage"]["viable_concept_indices"]) for k, _ in ORDER]
inter = sorted(set.intersection(*viable_sets))
inter_names = [CONCEPTS[i] for i in inter]
print(f"Intersection viable concepts (n={len(inter)}): {inter_names}")

def offdiag_stats(rows):
    C = rows / rows.norm(dim=1, keepdim=True)
    M = (C @ C.T).abs()
    n = M.shape[0]
    off = M[~torch.eye(n, dtype=bool)]
    return float(off.mean()), float(off.median())

# recompute orthogonality on the intersection subset per model
agg = {}
for k, _ in ORDER:
    cg = torch.load(f"{RES}/concept_g_{k}.pt")[inter]
    cga = torch.load(f"{RES}/concept_gamma_{k}.pt")[inter]
    cm, cmed = offdiag_stats(cg)
    em, emed = offdiag_stats(cga)
    # subspace separation restricted to intersection concepts
    cohen = metrics[k]["subspace"]["cohens_d"]
    inter_sep = sum(1 for nm in inter_names if cohen.get(nm, 0) >= 1.0)
    agg[k] = {
        "inter_causal_mean": cm, "inter_causal_median": cmed,
        "inter_euclid_mean": em, "inter_euclid_median": emed,
        "inter_ratio": em / cm,
        "inter_n_separated": inter_sep, "inter_n_total": len(inter),
        "own_causal_mean": metrics[k]["orthogonality_own_viable"]["causal_mean"],
        "own_euclid_mean": metrics[k]["orthogonality_own_viable"]["euclid_mean"],
        "own_ratio": metrics[k]["orthogonality_own_viable"]["ratio_euclid_over_causal_mean"],
        "own_n_separated": metrics[k]["subspace"]["n_separated"],
        "own_n_viable": metrics[k]["subspace"]["n_viable"],
    }
    print(f"{k:14s} inter causal/euclid={cm:.4f}/{em:.4f} ratio={em/cm:.2f} "
          f"sep={inter_sep}/{len(inter)} | own sep={agg[k]['own_n_separated']}/{agg[k]['own_n_viable']}")

json.dump({"intersection_indices": inter, "intersection_names": inter_names, "per_model": agg},
          open(f"{RES}/aggregate.json", "w"), indent=2)

# ---------------- FIGURE ----------------
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "gray": "#666666"}
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                               gridspec_kw={"height_ratios": [2.1, 1]})

# top: orthogonality lines
ax1.plot(x, euclid, "--o", color=OKABE["orange"], lw=2, ms=7, label="Euclidean inner product")
ax1.plot(x, causal, "-o", color=OKABE["blue"], lw=2, ms=7, label="Causal inner product")
ax1.set_ylim(0, max(euclid) * 1.35)
ax1.set_ylabel("mean off-diagonal |inner product|\n(0 = perfectly orthogonal)")
ax1.set_title("Does the Linear Representation Hypothesis survive to today's models?\n"
              f"Orthogonality across 2023–2025 base models  (n={len(inter)} concepts viable in all 5)")
# annotate the 2023 anchor with the on-disk replicated value (all-27-concept figure)
ax1.annotate("on-disk replicated Llama-2 (all 27 concepts):\n0.046 causal / 0.069 Euclid\n"
             "[concept_g.pt / concept_gamma.pt].\nLines use the 22-concept intersection.",
             xy=(0, causal[0]), xytext=(1.35, 0.024),
             fontsize=8.5, color=OKABE["gray"], ha="left",
             arrowprops=dict(arrowstyle="->", color=OKABE["gray"], lw=1))
for xi, (c, e) in enumerate(zip(causal, euclid)):
    ax1.annotate(f"{c:.3f}", (xi, c), textcoords="offset points", xytext=(0, -15),
                 ha="center", fontsize=8, color=OKABE["blue"])
    ax1.annotate(f"{e:.3f}", (xi, e), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8, color=OKABE["orange"])
    ax1.annotate(f"×{e/c:.2f}", (xi, (c + e) / 2), textcoords="offset points", xytext=(0, 0),
                 ha="center", fontsize=8, color=OKABE["gray"], style="italic")
ax1.legend(loc="upper right", frameon=False)

# bottom: subspace separation bars
bars = ax2.bar(x, sep, color=OKABE["green"], width=0.55, alpha=0.85)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("fraction of concepts\nwith linear subspace\n(Cohen's d ≥ 1.0)")
for xi, (k, _) in enumerate(ORDER):
    ax2.annotate(f"{agg[k]['inter_n_separated']}/{agg[k]['inter_n_total']}",
                 (xi, sep[xi]), textcoords="offset points", xytext=(0, 3),
                 ha="center", fontsize=9)
ax2.axhline(1.0, color=OKABE["gray"], ls=":", lw=1)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RES}/over_time_orthogonality.{ext}", facecolor="white")
print("saved over_time_orthogonality.png/pdf")
