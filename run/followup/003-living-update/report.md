# Study 003 — living-update (2026 model-refresh tick): does the Linear Representation Hypothesis survive to 2026 models?

**Paper:** Park, Choe, Veitch, *"The Linear Representation Hypothesis and the Geometry of Large
Language Models"* (arXiv 2311.03658, ICML 2024).
**Run date:** 2026-09-11. **Hardware:** 1× NVIDIA H200 NVL (143 GB). **New GPU time this session:** ≈ 1 min
(2 new models × ~12 s metrics + a 2-model C8 intervention pass; the 5 prior models were **reused, not recomputed**).

## Question

The prior living-update (study 001, 2026-09-09) established that the paper's central finding survives
on a 5-model ladder from 2023→2025: under a **causal inner product** (whitening the unembedding space
by `Cov(γ)^-1/2`) causally-separable concepts become approximately **orthogonal**, improving on the
plain Euclidean inner product, and (almost) every binary single-token concept has a **linear subspace
representation**. This 2026 tick **extends that ladder to the newest 2026 models** — primarily
**Qwen3.5-9B-Base (Mar 2026)**, plus **gemma-4-12B (base)** — reusing all prior results, and asks:
does the finding still hold on the 2026 flagships, and does the study-001 observation that the *size*
of the causal advantage tracks **model family, not scale or release year** still hold once a 2026 point
is added?

## Approach

**Reuse (nothing recomputed).** I copied study 001's `workspace/` scripts and its entire `results/`
(all five 2023–2025 `metrics_*.json`, `concept_g/gamma_*.pt`, `intervention_C8.json`) into this study's
`results/`, and treated every model 001 already covered as **DONE**. Their numbers below are read
verbatim from those on-disk artifacts; only the **two new 2026 models** were computed this session.
The math is identical to 001/replication: `g = γ @ Cov(γ)^-1/2`; concept direction = unit-normalized
mean of target−base unembedding differences; orthogonality = `|Ĉ Ĉᵀ|` off-diagonal on unit-normalized
concept rows; leave-one-out (LOO) counterfactual projections vs 100 000 random vocab-pair projections
(seed 100); vocabulary-wide C11 uncorrelatedness scatter. Data (`word_pairs/`, `paired_contexts/`) is
reused verbatim. `run_model_2026.py` imports the exact helper functions from 001's `run_model.py`.

**The only new code — the 2026 loader.** The 2026 flagships are **multimodal-arch** and cannot be
wrapped by nnsight or `AutoModelForCausalLM`. Per the verified 2026-09-11 recipe I load them with
`AutoModelForImageTextToText` (bf16, `device_map="auto"`) and pull the unembedding via
`m.get_output_embeddings().weight`. **LRH's core needs only the unembedding + input embeddings — no
nnsight, no forward passes** — so the geometry pipeline transfers unchanged; I only swapped the loader
and the unembedding accessor. As in 001, `Cov(γ)` and `eigh` are computed in **float32** (CUDA has no
bf16 eigh kernel).

**Smoke test before the full run (required).** On Qwen3.5-9B-Base I verified, *before* any metric:
load OK (`Qwen3_5ForConditionalGeneration`); unembedding `(248320, 4096)`, **untied** from the input
embedding; a text forward pass returns 33 hidden states (= n_layers+1, 32 layers) and logits
`(1,3,248320)`; and a residual-stream forward-hook on `dec.layers[16]` reads a `(1,3,4096)` activation.
One recipe correction was needed and is logged under Deviations: the text decoder is nested at
**`m.model.language_model`**, not `m.language_model` as the recipe stated. This did not affect the LRH
core (which uses `get_output_embeddings()`), only the hook/forward path used by the optional C8 check.

**Primary series = the 22 concepts viable in ALL 7 models** (apples-to-apples; unchanged from 001,
because Llama-2's smaller tokenizer remains the binding constraint). **Secondary series = each model's
own viable set** and the all-27-concept anchor. I also ran the optional **C8 intervention** on both new
models to match 001's treatment of the earlier five.

## Results

**Headline: the Linear Representation Hypothesis survives to the 2026 models.** On both Qwen3.5-9B-Base
and gemma-4-12B the causal inner product is strictly more orthogonal than the Euclidean one (ratio > 1)
with a low absolute causal off-diagonal (~0.063–0.069), **all** viable binary single-token concepts show
linear subspace structure (26/26 own; 22/22 on the intersection), the C11 uncorrelatedness sanity check
holds, and the male⇒female steering still flips king→queen. See `results/over_time_orthogonality.png`
(the ladder extended to 2026) and `results/fig_family_not_scale.png`.

### Orthogonality anchor — mean off-diagonal |inner product| (all 27 concepts, causal / Euclidean, ratio)

Original column is read from the replication artifacts on disk; the 2023–2025 columns are reused from
study 001's on-disk `metrics_*.json`; the **2026** columns are computed this session.

| Metric (all 27 concepts) | Original Llama-2-7B (artifact) | Llama-2 (2023) | Llama-3.1 (2024) | Qwen2.5 (2024) | Qwen3-8B (2025) | OLMo-3 (2025) | **Qwen3.5-9B (2026)** | **gemma-4-12B (2026)** |
|---|---|---|---|---|---|---|---|---|
| Causal mean | **0.046** (`concept_g.pt`, `verify/C5.json`) | 0.046 | 0.055 | 0.057 | 0.054 | 0.050 | **0.063** | **0.064** |
| Euclidean mean | **0.069** (`concept_gamma.pt`) | 0.070 | 0.082 | 0.071 | 0.071 | 0.062 | **0.081** | **0.088** |
| Causal median | **0.012** (`concept_g.pt`) | 0.012 | 0.014 | 0.018 | 0.015 | 0.014 | **0.019** | **0.018** |
| Euclidean median | **0.029** (`concept_gamma.pt`) | 0.029 | 0.040 | 0.034 | 0.026 | 0.023 | **0.035** | **0.034** |
| **Ratio Euclid/causal** | **1.52** | 1.52 | 1.50 | 1.25 | 1.31 | 1.25 | **1.27** | **1.38** |

On the **22-concept intersection** (apples-to-apples across all 7 models) the ratios are: Llama-2 1.42,
Llama-3.1 1.49, Qwen2.5 1.22, Qwen3-8B 1.26, OLMo-3 1.22, **Qwen3.5-9B 1.23**, **gemma-4-12B 1.29**
(causal/Euclid means for the two new models: 0.067/0.083 and 0.069/0.089). Both new models keep the
causal line clearly below the Euclidean one.

**Reading.** The causal metric stays clearly below the Euclidean one on both 2026 models; concepts stay
approximately orthogonal under the causal geometry (~0.063–0.069). So the **qualitative claim survives on
2026 models**. The quantitative "1.5–2.5×" band is again **not** reached by the newest models (Qwen3.5
1.27, gemma-4 1.38, all-27) — consistent with the study-001 pattern that the LLaMA lineage sits at ~1.5×
while later families sit lower.

### Family, not scale or year (the study-001 observation, tested with a 2026 point)

| Family | Members (year, all-27 ratio) | Family mean ratio |
|---|---|---|
| LLaMA | Llama-2-7B (2023, 1.52), Llama-3.1-8B (2024, 1.50) | **1.51** |
| Qwen | Qwen2.5-7B (2024, 1.25), Qwen3-8B (2025, 1.31), **Qwen3.5-9B (2026, 1.27)** | **1.28** |
| OLMo | OLMo-3-7B (2025, 1.25) | **1.25** |
| Gemma | **gemma-4-12B (2026, 1.38)** | **1.38** |

**The family-not-scale finding holds with the 2026 point.** The new **Qwen3.5-9B (2026, 9B params)**
lands at 1.27 — squarely inside the Qwen band (1.25–1.31) set by Qwen2.5-7B and Qwen3-8B — even though it
is a year newer and larger than both. The magnitude of the causal-orthogonality advantage tracks **model
family**, not release year or parameter count: across three Qwen generations spanning 2024→2026 and
7.6B→9B it stays ~1.25–1.31, while the two LLaMA models sit at ~1.5×. gemma-4-12B adds a *new* family
point at 1.38, between the LLaMA and Qwen bands — a single data point, so its "family mean" is just
itself. See `results/fig_family_not_scale.png`.

### Subspace separation — the 26/27 analog (Cohen's d ≥ 1.0)

| Quantity | Original Llama-2 | Llama-2 | Llama-3.1 | Qwen2.5 | Qwen3-8B | OLMo-3 | **Qwen3.5-9B** | **gemma-4-12B** |
|---|---|---|---|---|---|---|---|---|
| n_separated / n_viable (own set) | 26/27 (`verify/C1.json`, `C2.json`)\* | 21/22 | 26/26 | 26/26 | 26/26 | 26/26 | **26/26** | **26/26** |
| n_separated / n_viable (22-intersection) | — | 21/22 | 22/22 | 22/22 | 22/22 | 22/22 | **22/22** | **22/22** |
| Exception concept(s) | thing⇒part | thing⇒part (d=0.85) | none | none | none | none | **none** | **none** |

\* The replication's 26/27 counts all 27 concepts with no ≥10-pair viability gate (see 001). Both 2026
models separate **all** their viable concepts, including thing⇒part (Qwen3.5 d=1.23, gemma-4 d≈comparable) —
the subspace claim survives on the 2026 models.

### Uncorrelatedness (C11 analog) — Pearson r over the vocabulary

| Pair | Llama-2 | Llama-3.1 | Qwen2.5 | Qwen3-8B | OLMo-3 | **Qwen3.5-9B** | **gemma-4-12B** |
|---|---|---|---|---|---|---|---|
| separable (male⇒female vs English⇒French) — expect ≈0 | 0.022 | −0.002 | 0.030 | −0.010 | 0.007 | **0.010** | **−0.020** |
| non-separable (verb⇒3pSg vs verb⇒Ving) — expect larger | 0.365 | 0.379 | 0.378 | 0.368 | 0.364 | **0.392** | **0.418** |

The C11 sanity check **holds identically on the 2026 models**: the causally-separable pair is essentially
uncorrelated (|r| ≤ 0.02) while the non-separable pair is clearly correlated (r ≈ 0.39–0.42).

### Coverage (single-token pairs; larger 2026 vocabularies → more coverage)

| Model | vocab | hidden | unembed tied? | total single-token pairs | viable concepts (≥10 pairs) |
|---|---|---|---|---|---|
| Qwen3.5-9B-Base (2026) | 248,320 | 4096 | no | 1460 | 26/27 |
| gemma-4-12B (2026) | 262,144 | 3840 | **yes** | 1610 | 26/27 |

Both 2026 models have larger BPE vocabularies than any prior model and correspondingly more single-token
pairs (1460 / 1610 vs 641–1159 before). Each loses exactly **one** concept for lack of ≥10 single-token
pairs — `pronoun⇒possessive` (Qwen3.5: 6 pairs; gemma-4: 8 pairs). The 22-concept apples-to-apples
intersection is unchanged from study 001 (Llama-2's smaller 32 K tokenizer remains the binding
constraint, dropping 5 morphological concepts). gemma-4 has **tied** input/output embeddings (LRH uses
the unembedding = the tied weight); Qwen3.5 has a real separate unembedding.

### Optional C8 intervention — "Long live the" + α·(male⇒female)

Both 2026 models still **flip top-1 king→queen and push "king" out of the top-5** (full tables in
`results/intervention_C8.json`): top-1 = `king` at α=0; a `Queen`/`queen` variant becomes top-1 by α=0.1.
For Qwen3.5, `king` has left the top-5 by α=0.3 (→ queen/Queen/princess/goddess/girl); for gemma-4,
`king` sits at rank 4 at α=0.3 (queen/Queen/princess/king/King) and is gone by α=0.4
(queen/Queen/princess/woman/Princess). The causal male⇒female steering direction still works on the
2026 models.

## Verdict per model (2026 additions)

| Model | Orthogonality (causal < Euclid) | 1.5–2.5× band? | Subspace linear? | C11 | C8 |
|---|---|---|---|---|---|
| **Qwen3.5-9B-Base (2026)** | ✅ (ratio 1.23–1.27) | ⚠️ below (~1.27×, in Qwen band) | ✅ 26/26 | ✅ | ✅ |
| **gemma-4-12B (2026)** | ✅ (ratio 1.29–1.38) | ⚠️ below (~1.38×) | ✅ 26/26 | ✅ | ✅ |

**Bottom line: the Linear Representation Hypothesis survives to the 2026 models.** On Qwen3.5-9B-Base and
gemma-4-12B the causal inner product is more orthogonal than the Euclidean one, all viable binary
single-token concepts show linear subspace structure, C11 holds, and the male⇒female intervention still
steers king→queen. The one quantitative caveat carried over from 001 persists: the *magnitude* of the
causal advantage is ~1.5× only for the LLaMA lineage and ~1.25–1.38× for later families — and the
**family-not-scale** reading is reinforced by the 2026 Qwen point landing inside the Qwen band despite
being newer and larger.

## Deviations & limitations

1. **Recipe attribute path corrected.** The 2026 loading recipe said the text decoder is `m.language_model`;
   for both `Qwen3_5ForConditionalGeneration` and `Gemma4UnifiedForConditionalGeneration` it is actually
   nested at **`m.model.language_model`** (probed and confirmed). This affects only the optional C8
   forward/hook path; the LRH core uses `m.get_output_embeddings()` and was unaffected. No model was
   descoped — both 2026 models loaded, forwarded, and hooked successfully.
2. **gemma-4-12B added as the second 2026 point.** It was fully downloaded (~23 GB) and hookable, so it is
   included per the instruction's "add IF present and hookable." I used the **base** model (`gemma-4-12B`,
   not `-it`), matching this paper's base-model convention. Its embeddings are **tied**; LRH's causal
   geometry uses the unembedding (= the tied weight), which is the correct object.
3. **"1.5–2.5×" band not met by the 2026 models.** Reported honestly: Qwen3.5 1.27, gemma-4 1.38 (all-27).
   The causal metric still improves orthogonality (ratio > 1, low absolute off-diagonal) — the qualitative
   claim holds — but the magnitude is below the paper's band, consistent with the family pattern.
4. **22-concept intersection unchanged.** The apples-to-apples series still uses 22 concepts because
   Llama-2's tokenizer (not the 2026 models) is the binding constraint. Both 2026 models are individually
   26/27-viable; the only concept they drop (`pronoun⇒possessive`) is already outside the intersection.
   No value was imputed or fabricated; every dropped concept is disclosed above.
5. **Prior five models reused, not recomputed** (per instruction). Their numbers are read from study 001's
   on-disk `metrics_*.json` / `concept_*_*.pt`; re-aggregating them here reproduces 001's table exactly
   (e.g. Llama-2 all-27 causal/Euclid 0.046/0.070, ratio 1.52). The 2023 anchor is read from the
   replication artifacts (`concept_g.pt`/`concept_gamma.pt`, `verify/C5.json`).
6. **C6 measurement probe not run** (as in 001) — optional figure-only diagnostic; the required metrics +
   C8 already answer the survival question. Every number here is from this session's own execution
   (2026 models) or read from disk (2023–2025 baselines).

## Artifacts

- `results/over_time_orthogonality.png` / `.pdf` — the required over-time figure, **extended to 2026**
  (7 models, 2026 points ringed and shaded).
- `results/fig_family_not_scale.png` — the causal advantage vs release year, colored by family, showing
  the 2026 Qwen point lands in the Qwen band.
- `results/metrics_qwen3.5-9b-base.json`, `results/metrics_gemma-4-12b.json` — the two new models'
  per-model metrics (this session).
- `results/metrics_{llama2-7b,llama3.1-8b,qwen2.5-7b,qwen3-8b-base,olmo3-7b}.json` — reused from study 001.
- `results/aggregate.json` — 7-model intersection set + intersection/own/all-27 series.
- `results/intervention_C8.json` — C8 rank tables for all 7 models (5 reused + 2 new).
- `results/concept_g_*.pt`, `concept_gamma_*.pt` — 27 concept directions per model (5 reused + 2 new).
- `workspace/run_model_2026.py`, `aggregate_2026.py`, `run_intervention_2026.py`, `fig_family.py`,
  `smoke_test.py`, `probe.py` — this session's code.
