#!/usr/bin/env python
"""Fit figure for Study 002: causal off-diagonal & causal/Euclid ratio vs era ladder,
with the 5 measured points, per-era means, OLS trend and a leave-one-out band."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path("/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/"
           "test-corpus/linear-representation-hypothesis/run/followup/002-theory-update/results")
d = json.load(open(OUT / "fit_results.json"))
rows = d["table"]

FAM_COLOR = {"Llama": "#3B6FB6", "Qwen": "#D1873B", "OLMo": "#4E9A6B"}
FAM_MARK = {"Llama": "o", "Qwen": "s", "OLMo": "^"}

year = np.array([r["year"] for r in rows], float)
# small horizontal jitter so models sharing a year don't overlap
jit = {}
for e in sorted(set(year)):
    idx = [i for i, y in enumerate(year) if y == e]
    for k, i in enumerate(idx):
        jit[i] = (k - (len(idx) - 1) / 2) * 0.10
xj = np.array([year[i] + jit[i] for i in range(len(rows))])

def loo_band(x, y, xs):
    """5 leave-one-out fit lines; return (ols_line, lo, hi) over xs."""
    b, a = np.polyfit(x, y, 1)
    lines = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        bb, aa = np.polyfit(x[m], y[m], 1)
        lines.append(aa + bb * xs)
    lines = np.array(lines)
    return a + b * xs, lines.min(0), lines.max(0), b

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
xs = np.linspace(2022.7, 2025.3, 50)

panels = [
    ("causal_inter", "Causal off-diagonal mean  |ip|\n(lower = more orthogonal)",
     "A.  Orthogonality vs era — essentially flat", None),
    ("R_inter", "Ratio  R = Euclid / causal\n(higher = bigger causal advantage)",
     "B.  Causal advantage vs era — declines, but a family split", 1.0),
]
for ax, (key, ylab, title, hline) in zip(axes, panels):
    y = np.array([r[key] for r in rows], float)
    ols_line, lo, hi, slope = loo_band(year, y, xs)
    ax.fill_between(xs, lo, hi, color="#999999", alpha=0.18, lw=0, label="leave-one-out band")
    ax.plot(xs, ols_line, color="#555555", lw=1.8, label=f"OLS trend (slope={slope:+.4f}/yr)")
    # per-era mean markers (black x)
    for e in sorted(set(year)):
        vals = [r[key] for r in rows if r["year"] == e]
        ax.plot(e, np.mean(vals), marker="x", color="black", ms=10, mew=2.2, zorder=5)
    # measured points
    for i, r in enumerate(rows):
        ax.scatter(xj[i], y[i], s=90, color=FAM_COLOR[r["family"]],
                   marker=FAM_MARK[r["family"]], edgecolor="white", lw=1.0, zorder=6)
    if hline is not None:
        ax.axhline(hline, color="#C0392B", ls=":", lw=1.3, zorder=1)
        ax.text(2025.25, hline + 0.006, "R = 1\n(no advantage)", color="#C0392B",
                fontsize=8.5, ha="right", va="bottom")
    ax.set_xticks([2023, 2024, 2025])
    ax.set_xlabel("Release era (year)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11.5, loc="left")
    ax.margins(x=0.04)

# legends
fam_handles = [Line2D([0], [0], marker=FAM_MARK[f], color="w", markerfacecolor=FAM_COLOR[f],
                      markeredgecolor="white", markersize=10, label=f) for f in ["Llama", "Qwen", "OLMo"]]
fit_handles = [Line2D([0], [0], color="#555555", lw=1.8, label="OLS trend"),
               Line2D([0], [0], marker="x", color="black", lw=0, markersize=10, mew=2.2, label="per-era mean"),
               plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.25, label="leave-one-out band")]
axes[0].legend(handles=fam_handles, loc="upper left", frameon=False, fontsize=9, title="model family")
axes[1].legend(handles=fit_handles, loc="upper right", frameon=False, fontsize=9)

fig.suptitle("Study 002 — Causal-inner-product geometry across the 2023–2025 base-model ladder "
             "(5 models, 22-concept intersection)", fontsize=12.5, y=1.02, x=0.02, ha="left")
fig.text(0.02, -0.04,
         "5 data points ⇒ descriptive fit, not a scaling law. Causal off-diagonal is flat across eras "
         "(slope +0.001/yr, R²=0.06); the ratio R declines with year (R²=0.38) but the LOO band and family\n"
         "markers show the decline is a Llama→Qwen/OLMo family split (R changes ~4.8× more across families "
         "than within a family step), not a scale effect (R vs log-params corr ≈ −0.09). R > 1 on all 5.",
         fontsize=8.3, ha="left", va="top", color="#333333")
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
fig.savefig(OUT / "theory_fit.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "theory_fit.pdf", bbox_inches="tight")
print("wrote", OUT / "theory_fit.png")
