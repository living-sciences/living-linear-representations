"""Second figure: is the causal-orthogonality advantage (Euclid/causal ratio) a
function of MODEL FAMILY, not release year or scale? Plots the all-27-concept ratio
per model, colored by family, with per-family mean bands. The 2026 Qwen point tests
whether Qwen stays in its family band. All numbers from this session's aggregate.json."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run/followup/003-living-update/results"
agg = json.load(open(f"{RES}/aggregate.json"))["per_model"]

# (key, display, family, release_year, params_B)
ROWS = [
    ("llama2-7b",       "Llama-2-7B",   "LLaMA", 2023, 6.74),
    ("llama3.1-8b",     "Llama-3.1-8B", "LLaMA", 2024, 8.03),
    ("qwen2.5-7b",      "Qwen2.5-7B",   "Qwen",  2024, 7.62),
    ("qwen3-8b-base",   "Qwen3-8B",     "Qwen",  2025, 8.19),
    ("olmo3-7b",        "OLMo-3-7B",    "OLMo",  2025, 7.30),
    ("qwen3.5-9b-base", "Qwen3.5-9B",   "Qwen",  2026, 9.0),
    ("gemma-4-12b",     "gemma-4-12B",  "Gemma", 2026, 12.0),
]
FAM_COLOR = {"LLaMA": "#0072B2", "Qwen": "#D55E00", "OLMo": "#009E73", "Gemma": "#CC79A7"}
FAM_MARK  = {"LLaMA": "o", "Qwen": "s", "OLMo": "^", "Gemma": "D"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
fig, ax = plt.subplots(figsize=(9.5, 5.2))

# jitter x slightly by year so 2024/2025/2026 pairs don't overlap
year_counts = {}
xs, ys, fams, labels_new = [], [], [], []
for key, disp, fam, yr, pB in ROWS:
    ratio = agg[key]["all27_ratio"]
    n = year_counts.get(yr, 0); year_counts[yr] = n + 1
    x = yr + (0.10 * n if n else 0.0)
    xs.append(x); ys.append(ratio); fams.append(fam); labels_new.append(agg[key]["is_new_2026"])

# per-family mean band
for fam in ["LLaMA", "Qwen", "OLMo", "Gemma"]:
    fys = [y for y, f in zip(ys, fams) if f == fam]
    m = np.mean(fys)
    ax.axhline(m, color=FAM_COLOR[fam], lw=1, ls="--", alpha=0.45)
    ax.text(2026.55, m, f"{fam} mean {m:.2f}", color=FAM_COLOR[fam], fontsize=8.5, va="center")

# points
seen = set()
for key, disp, fam, yr, pB in ROWS:
    i = [r[0] for r in ROWS].index(key)
    x, y = xs[i], ys[i]
    lbl = fam if fam not in seen else None; seen.add(fam)
    ax.scatter(x, y, s=170, color=FAM_COLOR[fam], marker=FAM_MARK[fam],
               edgecolors="black" if labels_new[i] else "none",
               linewidths=1.8 if labels_new[i] else 0, zorder=4, label=lbl)
    dy = 0.015 if not labels_new[i] else -0.028
    ax.annotate(f"{disp}\n{y:.2f}", (x, y), textcoords="offset points",
                xytext=(0, 12 if dy > 0 else -22), ha="center", fontsize=7.8,
                color=FAM_COLOR[fam])

ax.axhspan(1.5, 2.5, color="#666666", alpha=0.07)
ax.text(2023.0, 2.44, "paper's stated 1.5–2.5× band", fontsize=8.5, color="#666666", style="italic")
ax.set_ylim(1.0, 2.6)
ax.set_xlim(2022.6, 2027.2)
ax.set_xticks([2023, 2024, 2025, 2026])
ax.set_xlabel("model release year")
ax.set_ylabel("causal orthogonality advantage\n(Euclidean / causal mean off-diag, all 27 concepts)")
ax.set_title("Family, not scale or year: the causal advantage clusters by model family\n"
             "(black-ringed = 2026 refresh; the new Qwen3.5 lands in the Qwen band)")
ax.axhline(1.0, color="black", lw=0.8)
ax.legend(loc="upper right", frameon=False, title="family")
fig.tight_layout()
fig.savefig(f"{RES}/fig_family_not_scale.png", facecolor="white")
print("saved fig_family_not_scale.png")
for key, disp, fam, yr, pB in ROWS:
    print(f"  {disp:14s} {fam:6s} {yr}  ratio={agg[key]['all27_ratio']:.3f}")
