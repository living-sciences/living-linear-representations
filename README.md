# The Linear Representation Hypothesis and the Geometry of Large Language Models (living paper)

**Paper:** The Linear Representation Hypothesis and the Geometry of Large Language Models
**Authors:** Kiho Park, Yo Joong Choe, Victor Veitch
**Venue:** Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR volume 235
**Identifier:** arXiv:2311.03658 (https://arxiv.org/abs/2311.03658)
**Original code:** github.com/KihoPark/linear_rep_geometry

The paper formalizes the "linear representation hypothesis" using counterfactuals, and
defines a non-Euclidean *causal inner product* that renders causally separable concepts
approximately orthogonal. It shows, on LLaMA-2, that high-level binary concepts have linear
(subspace) representations and that this inner product unifies linear probing and model steering.

## What a living paper is

A living paper is a reproduction of a published result that is kept alive over time: we re-run
the original study, then add small extension studies that re-test the paper's central claim on
new models and from new angles, so the finding can be checked again as the field moves on.

## What this repo contains

- **Replication** (`run/replication/`): a from-scratch re-run of the paper's five notebooks on
  LLaMA-2-7B (subspace, heatmap/orthogonality, measurement, intervention, sanity check), using
  the authors' package as the `codebase/`. All five diagnostics reproduced the paper's picture.

- **Study 001 - living-update** (`run/followup/001-living-update/`): re-ran the core diagnostics
  on a ladder of five base models from 2023 to 2025. Headline: the hypothesis survives to 2025.
  The causal inner product stays more orthogonal than the Euclidean one on every model, and
  essentially all binary single-token concepts stay linear, though the advantage weakens outside
  the LLaMA family (about 1.5x for LLaMA, about 1.25x for Qwen and OLMo).

- **Study 002 - theory-update** (`run/followup/002-theory-update/`): fit a small descriptive model
  (no new inference) to Study 001's numbers across the era ladder. Headline: the causal advantage
  is set by model family (LLaMA high, Qwen and OLMo lower), not by scale or release year; with only
  five data points every fit is descriptive, not a scaling law.

- **Study 003 - living-update (2026 refresh)** (`run/followup/003-living-update/`): extended the
  ladder to 2026 models (Qwen3.5-9B-Base and gemma-4-12B), reusing all prior results. Headline:
  the hypothesis survives to 2026; the causal inner product stays more orthogonal than Euclidean on
  both 2026 models, all binary single-token concepts stay linear, and the advantage still clusters
  by family rather than scale or year.

Each study directory holds its `instruction.md` (the task), `workspace/` (the code and scripts it
ran), `results/` (figures, json, csv, and small result tensors), `report.md`, `result_card.json`,
and `followup_summary.json`.

## Living page

The living page for this paper is at: https://livingscience.ai/safety/living-linear-representations

## License and attribution of the authors' package

Files under each `codebase/` directory are derived from the authors' original replication package
(github.com/KihoPark/linear_rep_geometry) and remain under the authors' original license and terms.
Please cite the paper above when you use them.

## Large files

Some large artifacts (full unembedding matrices, covariance roots, and model weights) are too big
for GitHub and are not included here. Every omitted file, with its size and how to regenerate or
obtain it, is listed in [LARGE_FILES_OMITTED.md](LARGE_FILES_OMITTED.md).
