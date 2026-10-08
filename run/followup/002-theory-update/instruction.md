# Study 002 — theory-update: a written-down updated model of causal-inner-product geometry vs scale/family/era

OUTLINE (finalize after 001 runs). This study consumes ONLY `followup/001-living-update/results/` (the per-model
`metrics_<model>.json`, the over-time figure data) — it launches no new model inference. It fits a small parametric
model to how the paper's measured geometry quantities move across the 5-model era ladder and states, in words, an
updated version of the linear-representation / causal-inner-product claim for 2025.

Run mechanism: `python -m veritas.cli.main followup <run_dir> --instruction-file 002-theory-update.outline.md
--name theory-update`, reusing 001's `results/`. Deliverables mirror 001: `report.md`, `followup_summary.json`,
`result_card.json` (schema `sai.followup.result_card/v1`), figures under `results/`.

## Inputs from 001 (per model: Llama-2-7B'23, Llama-3.1-8B'24, Qwen2.5-7B'24, Qwen3-8B-Base'25, OLMo-3-7B'25)
- `ortho_causal_mean`, `ortho_euclid_mean` (+ medians); the gap ratio `R = euclid_mean / causal_mean`.
- `n_separated / n_viable` and per-concept Cohen's d (subspace separation); the exception concept(s) per model.
- Uncorrelatedness r (separable vs non-separable pair).
- Covariates: release year, params, hidden dim d, vocab size, family (Llama/Qwen/OLMo), tokenizer coverage.

## What to fit (small, honest — 5 data points; report uncertainty, avoid overfitting)
1. **Orthogonality vs scale/era.** Fit `ortho_causal_mean ~ a + b·year + c·log(params)` (and separately the gap
   ratio `R`). With only 5 points, prefer single-covariate fits + a per-era mean (2023 / 2024 / 2025) rather than a
   many-term regression; report slope sign, R², and a bootstrap or leave-one-out band. Question answered: does the
   causal inner product get *more* or *less* orthogonalizing as models scale and advance?
2. **Family effect.** Compare within-family generation steps (Llama-2→Llama-3.1, Qwen2.5→Qwen3) as paired deltas;
   is the geometry more stable within a family than across families? Treat OLMo-3 as the fully-open reference point.
3. **Subspace survival.** Model `n_separated / n_viable` across the ladder; is the paper's 26/27 still ~representative
   in 2025, and is the exception set stable (still thing⇒part) or model-dependent?
4. **Causal-vs-Euclidean advantage.** State whether `R > 1` (causal strictly better) persists across all eras, and
   whether it grows or shrinks — the paper's Appendix D.2 hypothesis was that Euclidean "somewhat works" on LLaMA-2
   but fails on newer/other models; test that trend directly.

## The updated model / claim (the written deliverable)
- A one-paragraph **updated statement** of the linear-representation-hypothesis finding, quantified per era, e.g.:
  "Across 2023–2025 base models (7–8B), the causal inner product keeps causally-separable concepts approximately
  orthogonal (mean off-diagonal |ip| in [__, __]) and remains R = __–__× more orthogonal than the Euclidean inner
  product; subspace linearity holds for __/__ concepts on average, with the exception set {…}." Fill blanks from 001.
- A small **parametric summary table** (per era: causal mean, R, n_separated/n_viable) and one **fit figure**
  (`results/theory_fit.png`): the fitted trend line(s) over the era axis with the 5 measured points and their band.
- Explicit **scope/limits** (carry forward from the landscape notes): binary single-token concepts only; tokenizer
  coverage differs by family (a confound, quantified from 001); 5 points ⇒ descriptive fit, not a scaling law.

## Deliverables
- `report.md`: the fits (with slope/R²/uncertainty), the era summary table, the updated written claim, limits.
- `results/theory_fit.png` (+ caption/alt) and any supporting CSV of the fit.
- `followup_summary.json`: fitted coefficients, per-era aggregates, the updated-claim string.
- `result_card.json` (`sai.followup.result_card/v1`): headline = the updated quantified claim with its key number;
  metrics each paired with the 001 baseline value + provenance (`followup/001-living-update/results/metrics_*.json`).

## Anti-goals
- **Foreground only; never background or "wait".** No new GPU inference — this is a fit over 001's saved numbers.
- Every fitted value derives from 001's `results/`; do not invent points or extrapolate beyond the 5 models. With 5
  data points, state uncertainty honestly and do not claim a scaling law. Disclose any era/model dropped by 001.