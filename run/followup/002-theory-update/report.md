# Study 002 — theory-update: a written-down updated model of causal-inner-product geometry vs scale / family / era

**Paper:** Park, Choe, Veitch, *"The Linear Representation Hypothesis and the Geometry of Large
Language Models"* (arXiv 2311.03658, ICML 2024).
**Run date:** 2026-09-09. **Compute:** none — this study is a parametric fit over the numbers
Study 001 already saved to disk. No model inference was launched.
**Consumes:** `followup/001-living-update/results/` (`metrics_<model>.json` ×5, `aggregate.json`).

## Question

Study 001 re-ran the paper's core geometry diagnostics on a 5-model era ladder
(Llama-2-7B '23, Llama-3.1-8B '24, Qwen2.5-7B '24, Qwen3-8B-Base '25, OLMo-3-7B '25) and found the
Linear Representation Hypothesis survives. This follow-up asks the *quantitative* successor question:
**how do the measured geometry quantities move across scale, family and era, and what is the updated,
per-era statement of the causal-inner-product claim for 2025?** Concretely (4 parts): (1) does the
causal inner product get *more* or *less* orthogonalizing as models advance? (2) is the geometry more
stable *within* a family than *across* families? (3) is the paper's "26/27 concepts linear" still
representative, and is the exception set stable? (4) does the causal advantage `R = Euclid/causal > 1`
persist across all eras, and does it grow or shrink (Appendix D.2 predicted Euclidean "somewhat works"
on LLaMA-2 but fails on newer/other models)? With only **5 data points** the mandate is an honest,
uncertainty-bearing descriptive fit — **not** a scaling law.

## Approach

Everything here is read from 001's saved JSON; no re-computation of any geometry. For each model I
took `ortho_causal_mean`, `ortho_euclid_mean`, the gap ratio `R`, `n_separated / n_viable`, the
per-concept Cohen's d, the uncorrelatedness r, and covariates (era year, params, hidden dim d, vocab,
family, single-token coverage). The **primary fit series is the 22-concept intersection** (the
concepts viable in all 5 models: `inter_*` in `aggregate.json`) because it removes the tokenizer-coverage
confound; I also report each model's **own-viable** set (paper-comparable) and the **all-27** set.

Fits (`workspace/fit_theory.py`): for the causal off-diagonal and for `R` I ran single-covariate OLS
against era-year and against log(params), plus a per-era mean (2023/2024/2025); uncertainty is a
**leave-one-out** (LOO) band (refit on the other 4, 5 slopes + held-out residuals) and a 5000-draw
bootstrap slope CI. A 2-covariate `year + log(params)` fit is reported only to show it overfits
(n=5, p=3). Family effects are paired within-family deltas (Llama-2→Llama-3.1, Qwen2.5→Qwen3) versus
same-era cross-family contrasts, with OLMo-3 as the fully-open reference. Figure: `workspace/plot_theory.py`.

## Results

### Parametric summary table (per era; own-viable set unless noted)

| Era | Models | Causal off-diag mean | Euclid mean | **R = Euclid/causal** | n_separated / n_viable | Exception |
|---|---|---|---|---|---|---|
| **2023** | Llama-2-7B | 0.052 | 0.074 | **1.42** | 21 / 22 | thing⇒part (d=0.85) |
| **2024** | Llama-3.1-8B, Qwen2.5-7B | 0.059 (0.058–0.060) | 0.080 | **1.35** (1.23–1.48) | 26 / 26 | none |
| **2025** | Qwen3-8B-Base, OLMo-3-7B | 0.055 (0.053–0.057) | 0.070 | **1.27** (1.24–1.30) | 26 / 26 | none |

(Intersection series gives the same picture: causal 0.052 → 0.060 → 0.056; R 1.42 → 1.35 → 1.24.)
Every number traces to `followup/001-living-update/results/`; the 2023 row reproduces the on-disk
replication anchor (all-27 causal/Euclid = 0.046 / 0.069, ratio 1.52) exactly per 001's cross-check.

### 1. Orthogonality vs scale / era — **flat, no trend**

The causal off-diagonal (the paper's orthogonality quantity; lower = more orthogonal) stays in a tight
band **~0.052–0.062** across all 5 models. Fitting `causal ~ year`: slope **+0.0010 / yr**, **R² = 0.055**,
and the LOO slope range **[−0.0042, +0.0028]** *changes sign* (bootstrap 95% CI **[−0.0063, +0.0085]**,
straddling 0). Against log(params) the OLS R² looks higher (0.52) but **LOO R² = −0.05** and the
bootstrap CI still spans 0 — i.e. not generalizable. **Answer: the causal inner product is neither
measurably more nor less orthogonalizing as models scale/advance; it is stable.** The concepts stay
approximately orthogonal (off-diagonal ≈ 0.05) at every generation.

### 2. Family effect — **geometry is a family property, not an era property**

Within-family generation steps barely move the geometry; across-family steps move it a lot:

| Contrast | Δ causal off-diag | Δ R |
|---|---|---|
| Llama-2 → Llama-3.1 (within, '23→'24) | +0.0065 | +0.071 |
| Qwen2.5 → Qwen3 (within, '24→'25) | −0.0040 | +0.041 |
| **Llama-3.1 vs Qwen2.5 (across, same '24 era)** | −0.0030 | **+0.271** |
| Qwen3 vs OLMo-3 (across, same '25 era) | +0.0033 | +0.035 |

Mean within-family |ΔR| = **0.056**; the same-era Llama-vs-Qwen |ΔR| = **0.271** — **R varies ≈ 4.8×
more across families than within a family generation step.** The split is clean: the causal advantage
`R` is **high for the LLaMA lineage (1.42, 1.48)** and **low for Qwen/OLMo (1.22–1.30)**. OLMo-3 (the
fully-open 2025 reference) lands with Qwen (R = 1.24), **not** with any "2025 era" value — confirming
the axis is family/tokenizer, not calendar. Correlations: `R vs year = −0.62` but `R vs log(params) =
−0.09` (essentially none), so what looks like an era decline of R (§4) is really the changing family
mix (2023 = Llama-only → 2025 = Qwen/OLMo-only).

### 3. Subspace survival — **26/27 still representative; exception set is Llama-2-specific**

Fraction of viable concepts with a linear subspace (Cohen's d ≥ 1): **21/22 for Llama-2, 26/26 for each
of the four 2024–2025 models** (mean ≈ **99%** of viable concepts, vs the paper's 26/27 ≈ 96%). The
paper's claim is therefore **still representative in 2025 and, if anything, strengthens.** The exception
set is **not stable across models** — it is Llama-2-only: `thing⇒part` has d = 0.85 (< 1) on Llama-2
but rises above threshold on every newer model (d = 1.17 Llama-3.1, 1.35 Qwen2.5, 1.23 Qwen3, 1.36
OLMo-3). No other concept ever drops below d = 1. So the paper's documented exception is a
Llama-2 artifact that later models resolve, not a persistent property of the concept.

### 4. Causal-vs-Euclidean advantage — **R > 1 everywhere; shrinks by family, contradicting D.2's direction**

`R > 1` on **all 5 models and all three concept sets** (intersection range **1.22–1.49**; own-viable
1.22–1.48; all-27 1.25–1.52) — the causal inner product is strictly more orthogonal than the Euclidean
one in every era. The per-era mean **declines** (1.42 → 1.35 → 1.24; `R ~ year` slope −0.091, R² = 0.38,
LOO slopes all negative [−0.11, −0.08], bootstrap 91% negative but 95% CI [−0.25, +0.03] includes 0).
**But** §2 shows this decline is the family mix, not scale. Read against **Appendix D.2** — which
predicted Euclidean "somewhat works" on LLaMA-2 but fails on newer/other models — the data run the
**opposite** direction: the causal advantage is **largest** on the LLaMA lineage (R ≈ 1.4–1.5, i.e.
Euclidean is *furthest* from causal there) and **smallest** on the newer non-LLaMA Qwen/OLMo models
(R ≈ 1.2–1.3, Euclidean nearly as orthogonal), and the absolute Euclidean off-diagonal is if anything
*lower* on OLMo-3 (0.065) than Llama-2 (0.074). So **Euclidean does not fail on newer models; it gets
relatively closer to causal** — D.2's directional hypothesis is not supported by this ladder.

### The updated claim (quantified per era)

> **Across 2023–2025 base models (7–8B), the causal inner product keeps causally-separable concepts
> approximately orthogonal (mean off-diagonal |ip| ≈ 0.052–0.060, essentially flat across generations)
> and remains R = 1.22–1.48× more orthogonal than the Euclidean inner product on every model; subspace
> linearity holds for ≈ 99% of viable single-token concepts (26/26 on the four 2024–2025 models, 21/22
> on Llama-2), with the only exception, thing⇒part, confined to Llama-2. The size of the causal
> advantage is governed by model *family/tokenizer* — high for the LLaMA lineage (R ≈ 1.4–1.5), lower
> for Qwen/OLMo (R ≈ 1.2–1.3) — not by scale or calendar year (R vs log-params corr ≈ −0.09); the
> apparent era-decline of R is the changing family mix. Contra the paper's Appendix-D.2 conjecture,
> the Euclidean inner product does not degrade on newer models — it draws closer to the causal one.**

## Deviations & limitations

1. **5 points ⇒ descriptive fit, not a scaling law.** Reported honestly: **every** fit has a negative
   leave-one-out R² and a bootstrap slope CI that includes 0. No trend here is statistically
   established; the robust statements are qualitative (causal off-diagonal flat; R > 1 everywhere;
   family split with |ΔR|_cross ≈ 4.8× |ΔR|_within).
2. **Era is confounded with family and with tokenizer coverage.** 2023 = Llama-only, 2025 =
   Qwen/OLMo-only, so "year" cannot be separated from "family" with 5 points. Coverage (single-token
   pairs) correlates with year at **+0.80** (Llama-2's 32k vocab yields 641 pairs vs ~1150 for the
   others) and causal off-diagonal correlates with vocab at **+0.89** — the intersection series is used
   precisely to neutralize this, but the confound is real and is why I do not read the causal~year or
   causal~logparams slopes as scale effects.
3. **Scope carried forward from 001.** Binary single-token concepts only; geometry computed in float32
   on the unembedding; the 22-concept intersection drops 5 morphological concepts that Llama-2's
   tokenizer cannot realize as single tokens. Params are published/nominal totals (approximate).
4. **No model/era dropped.** All 5 models 001 produced are used; nothing was imputed or extrapolated
   beyond the 5 measured points.

## Artifacts (under `results/`)

- `theory_fit.png` / `.pdf` — the fit figure (causal off-diagonal & R vs era, points + per-era means +
  OLS trend + LOO band; family markers).
- `theory_fit.csv` — the per-model table feeding the fits (both concept sets + covariates).
- `fit_results.json` — all fitted coefficients, LOO/bootstrap bands, per-era aggregates, family deltas,
  covariate correlations.
- `workspace/fit_theory.py`, `workspace/plot_theory.py` — this session's code.
