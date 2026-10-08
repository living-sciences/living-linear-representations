# Study 001 — living-update: does the Linear Representation Hypothesis survive to today's models?

You are a study agent. You have never read the paper; everything you need is here. This study re-runs the main
diagnostics of *"The Linear Representation Hypothesis and the Geometry of Large Language Models"* (Park, Choe,
Veitch; arXiv 2311.03658, ICML 2024) on a ladder of newer model generations and produces an **over-time figure**:
did the paper's central finding — that a *causal inner product* renders causally-separable concepts approximately
orthogonal, and that concepts have linear (subspace) representations — hold up in 2024 and 2025 open models?

Run mechanism (already handled by the launcher): this file was passed to
`python -m veritas.cli.main followup <run_dir> --instruction-file 001-living-update.instruction.md --name living-update`,
creating `followup/001-living-update/` inside the completed run dir. You work inside that directory and **reuse the
replicated codebase** at `<run_dir>/replication/codebase/` (copy it in; never modify the original run dir).

`<run_dir>` = `/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run`

---

## THE QUESTION

For each of 5 model generations (2023 → 2025), compute the paper's core quantities on that model's unembedding
matrix and answer: **is the causal inner product still ~1.5–2.5× more orthogonal than the Euclidean one, and do
~all binary single-token concepts still show linear subspace structure?** Plot each quantity across the model
ladder ("did the finding survive to today?").

## CLAIMS BEING UPDATED (IDs, paper values, replicated values, artifact paths)

Read the replicated values from disk (they are your original/2023 baselines; cite the file). All statuses in
`<run_dir>/verify/verdicts.json` are `match`.

- **C3 / C9 (HEADLINE — orthogonality of the causal inner product).** Paper: under the causal inner product most
  of the 27 causally-separable concepts are nearly orthogonal (block-diagonal off-diagonals ≈ 0); the Euclidean
  inner product "somewhat works" but the causal one clearly improves on it. **Replicated LLaMA-2-7B (compute from
  `<run_dir>/replication/codebase/matrices/concept_g.pt` and `concept_gamma.pt`; unit-normalize rows, take
  |C @ Cᵀ| off-diagonal):** mean off-diag = **0.046** causal vs **0.069** Euclidean; median **0.012** vs **0.029**.
  (This same number is quoted in `<run_dir>/verify/C5.json`.) Figure: `figures/three_heatmaps.pdf`.
- **C1 / C2 (subspace separation).** Paper: for (almost) every concept, counterfactual-pair differences project
  onto the concept direction far more strongly than random-pair differences (right-skewed histogram); **thing⇒part**
  is the sole exception. Replicated: confirmed for 26/27; thing⇒part overlaps random (`<run_dir>/verify/C1.json`,
  `C2.json`; figure `figures/appendix_right-skewed_LOO_g.pdf`).
- **C11 (sanity-check uncorrelatedness).** Paper: over the vocabulary, λ_Wᵀγ and λ_Zᵀγ are uncorrelated for a
  causally-separable pair (male⇒female vs English⇒French) and correlated for a non-separable pair (verb⇒3pSg vs
  verb⇒Ving). Replicated: confirmed (`<run_dir>/verify/C11.json`; `figures/sanity_check.png`).
- **C5, C10 (specific heatmap cells, use as exact numeric baselines).** From `<run_dir>/verify/C5.json`: row 19
  lower⇒upper vs language pairs |ip| = 0.086 (Eng⇒Fr), 0.091 (Fr⇒De), 0.054 (De⇒Es), 0.0036 (Fr⇒Es). From
  `C10.json`: English⇒French causal 0.234/0.260/0.0039 vs French⇒German/French⇒Spanish/German⇒Spanish, Euclidean
  0.121/0.145/0.080.
- **C8 (intervention rank table — OPTIONAL diagnostic).** Paper Table 1 / replicated: "Long live the" + α·(male⇒
  female) flips top-1 king→queen by α=0.2 and drops "king" out of top-5 by α=0.3. Both tables in `verify/C8.json`.

**Claims explicitly NOT updated (and why):** C4 (block = semantic grouping) — qualitative labelling of C3, folded
into the block-structure narrative, not a separate number. C6 (measurement probe) and C7 (intervention arrows) —
figure-only diagnostics that need forward passes; run them only if GPU budget remains (see Optional). **C12** (extra
Gemma-2B model) — was out-of-scope in the replication (no code path); superseded by this study, which adds newer
models directly. Do not attempt to reproduce the paper's Gemma-2B appendix figure.

## MODEL / DATA TABLE (verified 2026-09-09; load OFFLINE from the local snapshot paths — zero download)

Set `HF_HOME=/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`. Load each model by its **explicit local snapshot directory** below (not by bare HF id —
some slots live in a different cache). All are **base** models with `tie_word_embeddings=False` (a real separate
`lm_head.weight`). Never load an instruct/chat variant.

| # | Era | HF id (for the report) | params | bf16 GB | gated | Local snapshot path to load |
|---|---|---|---|---|---|---|
| 1 | 2023 (original) | meta-llama/Llama-2-7b-hf | 6.74B | 13.5 | yes (cached, no token needed) | `/net/projects2/chai-lab/shared_models/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9` |
| 2 | 2024 | meta-llama/Llama-3.1-8B | 8.03B | 16.1 | yes (cached, no token needed) | `/net/projects2/chai-lab/shared_models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` |
| 3 | 2024 | Qwen/Qwen2.5-7B | 7.62B | 15.2 | no | `/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796` |
| 4 | 2025 | Qwen/Qwen3-8B-Base | 8.19B | 16.4 | no | `/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| 5 | 2025 | allenai/Olmo-3-1025-7B | 7.30B | 14.6 | no | `/net/projects2/chai-lab/shared_models/hub/models--allenai--Olmo-3-1025-7B/snapshots/0a15c289b17ad4f27aabc863276ec525d5312c01` |

If a snapshot path has moved, resolve it with `ls .../models--<org>--<name>/snapshots/`. If OLMo-3 fails to load
(transformers too old for `model_type=olmo3`), first try upgrading transformers, else fall back to a 4-model ladder
(drop row 5) and say so — **do not** silently substitute a different model.

**Data (reused verbatim, copied from the replicated codebase — no new datasets):**
`word_pairs/` (27 `[a - b].txt`, 1998 tab-separated single-token counterfactual pairs; concept order fixed in
`store_matrices.py`) and `paired_contexts/` (`en-fr,fr-de,fr-es,es-de.jsonl` Wikipedia context pairs). Both live at
`<run_dir>/replication/codebase/`. Copy them into the study workspace.

## EXACT REUSE OF THE REPLICATED CODEBASE

1. `cp -r <run_dir>/replication/codebase <followup_dir>/codebase` and work on the copy. Key files: `store_matrices.py`
   (builds gamma, `g = gamma @ Cov(gamma)^-1/2`, and the 27 concept directions in gamma- and g-space), and
   `linear_rep_geometry.py` (model load + all diagnostic helpers: `get_counterfactual_pairs`, `concept_direction`,
   `inner_product_loo`, `draw_heatmaps`, `get_embeddings`, `show_intervention`, `show_rank`, `sanity_check`).
2. **What to change to swap the model (the only real code change):**
   - In `linear_rep_geometry.py`, replace `LlamaTokenizer`/`LlamaForCausalLM` with `AutoTokenizer`/
     `AutoModelForCausalLM`, and parametrize `MODEL_PATH` (loop over the 5 snapshot paths). Load in **bfloat16**
     (`dtype=torch.bfloat16`, `device_map={"":0}`).
   - In `store_matrices.py`, compute `Cov(gamma)` and `torch.linalg.eigh` in **float32** (cast the d×d covariance
     to float32, or move it to CPU/float64) — CUDA has no half/bf16 eigh kernel. Keep gamma itself in the model dtype.
   - **Fix the single-token assumption (critical for Qwen/OLMo, which do not prepend BOS).** In
     `get_counterfactual_pairs` (and anywhere using `tokenizer.encode(word)[1]` / the `len==2` filter, incl.
     `get_logit` and the sanity-check `ind_1/ind_2` lists), switch to a tokenizer-agnostic single-token lookup:
     ```python
     def single_token_id(tok, word):
         for w in (word, " " + word):
             ids = tok(w, add_special_tokens=False)["input_ids"]
             if len(ids) == 1:
                 return ids[0]
         return None   # skip pairs where either side is not single-token
     ```
     Keep a counterfactual pair only if BOTH sides return a non-None id. This preserves the paper's single-token
     design across tokenizers. Record, per model, how many pairs survive per concept.
3. **Do not change** the math: `g = gamma @ Cov(gamma)^-1/2`, concept direction = mean of (target−base) unembedding
   differences, unit-normalized; heatmap = |Ĉ @ Ĉᵀ| on unit-normalized concept rows; LOO projections and the
   uncorrelatedness scatter are exactly as in `linear_rep_geometry.py`.

## PER-ERA (PER-MODEL) COMPUTATION — the required numeric outputs

For each of the 5 models, after building `concept_g` / `concept_gamma` (adapted `store_matrices.py`), compute and
save to a per-model JSON (`results/metrics_<model>.json`):

1. **Orthogonality anchor (HEADLINE):** on unit-normalized concept rows, `C = |Ĉ @ Ĉᵀ|`; report mean and median of
   the off-diagonal for both **causal** (`concept_g`) and **Euclidean** (`concept_gamma`); also `euclid_mean/causal_mean`
   ratio. Cross-check: for model #1 this must reproduce 0.046 / 0.069 (mean) — if it does not, stop and debug before
   trusting the other models.
2. **Subspace separation (the 26/27 analog):** for each concept, compute leave-one-out counterfactual-pair
   projections (reuse `inner_product_loo`, causal/g-space) and projections of ~100K random single-token word-pair
   differences (seed 100) onto the concept direction. Per concept report **Cohen's d** = (mean_counterfactual −
   mean_random)/pooled_sd, and mark "separated" if d ≥ 1.0 (a clear right-shift). Report count `n_separated /
   n_viable` and list any exception concepts (the model's analog of thing⇒part).
3. **Uncorrelatedness (C11 analog):** Pearson r of (λ_Wᵀγ, λ_Zᵀγ) over the vocabulary for the separable pair
   (male⇒female, English⇒French) and the non-separable pair (verb⇒3pSg, verb⇒Ving). Expect |r| small for the first,
   larger for the second.
4. **Coverage:** `n_singletoken_pairs` total and per concept, and `n_viable` concepts (≥10 usable pairs). Different
   tokenizers → different coverage; this is expected and must be reported, not hidden.

Primary series = metrics computed over the **intersection** of concepts viable in ALL 5 models (apples-to-apples).
Secondary series = each model's own full viable set. Report both.

**OPTIONAL (only if GPU-hours remain, ≤4 total):** run the measurement probe (C6, `3_measurement.ipynb` logic:
does the on-target language direction separate the two language classes while an off-target does not?) and the
intervention rank table (C8, `show_rank("Long live the", α∈{0,.1,.2,.3,.4}, concept=male⇒female)`: does king→queen
still flip and king leave top-5?) per model, as qualitative "still holds / weaker / broken" notes.

## THE REQUIRED OVER-TIME FIGURE

`results/over_time_orthogonality.(png|pdf)` — the headline "did it survive?" figure:
- **X axis:** the 5 models ordered by release date, grouped by era: `Llama-2-7B (2023)` | `Llama-3.1-8B (2024)`,
  `Qwen2.5-7B (2024)` | `Qwen3-8B-Base (2025)`, `OLMo-3-7B (2025)`. Label each tick with model + year.
- **Y axis:** mean off-diagonal |inner product| (0 at bottom = perfectly orthogonal). **Two lines:** causal (solid)
  and Euclidean (dashed). Annotate the 2023 point with the paper/replicated value (0.046 causal / 0.069 Euclidean).
  The finding "survives" if causal stays low AND stays clearly below Euclidean across the ladder.
- Overlay or second panel: bar of `n_separated / n_viable` (the 26/27 analog) per model, so both the orthogonality
  and the subspace claim are shown over time.
Every plotted number must come from this session's own `results/metrics_<model>.json`. Caption must state the metric,
the concept set (intersection vs full), and that the 2023 point matches the on-disk replication artifact.

## ENVIRONMENT CONSTRAINTS

- **1 GPU** (A40 48GB or A100 80GB); every model ≤8.2B bf16 (≤16.4 GB) fits with room. Budget **≤4 GPU-hours total**
  (required metrics ≈1.5 h; full/optional ≈3 h). Do not exceed 8.
- Build your own venv with `uv` under the followup dir (`uv venv && uv pip install ...`). Use the replication's
  known-good stack: `torch==2.13.0 transformers==5.16.1 accelerate numpy seaborn matplotlib tqdm sentencepiece`.
  transformers 5.16.1 supports qwen3 and olmo3.
- **Storage: everything under `/net/projects2/chai-lab-models/haokunliu/`.** NEVER write under `/home`.
  `HF_HOME=/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`. Zero downloads — all 5 models are cached; load from the snapshot paths above.
- Compute the eigendecomposition in float32 (see reuse notes). No Stata/MATLAB — Python only.

## REQUIRED DELIVERABLES (leave these in `followup/001-living-update/`)

- **`report.md`** — narrative + a comparison table with one row per claim/metric and columns:
  `[original LLaMA-2-7B value, artifact file it came from, then the new value for each of the 5 models]`. Include:
  the orthogonality anchor (causal & Euclidean mean/median), the `n_separated/n_viable` subspace count and each
  model's exception concept(s), the uncorrelatedness r's, and coverage. State plainly whether the finding survived,
  weakened, or broke per model, and disclose every data hole (concepts dropped for lack of single-token pairs).
- **`followup_summary.json`** — machine-readable summary: per-model metrics, the intersection concept set, the
  headline verdict, and pointers to figures.
- **`result_card.json`** — schema `sai.followup.result_card/v1`: a one-sentence `headline` containing the key number
  (e.g. "The causal inner product stays N× more orthogonal than Euclidean across 2023–2025 models"); `status`; 1–5
  `metrics`, EACH paired with its `baseline` value and `baseline_provenance` (e.g. `concept_g.pt` /
  `verify/C5.json`); ≤2 tables; 0–3 figures under `results/` each with `caption` + `alt`; `notes`.
- **`results/`** — `over_time_orthogonality.(png|pdf)` (required), `metrics_<model>.json` for all 5 models, and any
  optional measurement/intervention outputs. Each figure has a caption and alt text.

## ANTI-GOALS (READ — these are hard rules)

- **NEVER background a long job and NEVER end your turn to "wait" for one. Run everything in the FOREGROUND** — if
  the session ends, the process is killed and the work is lost. Do not `nohup`/`&`/`sbatch`-detach the compute.
- **No token or turn budgets in this spec** — do not stop early because of an imagined budget; finish the 5 models.
- **Every reported number must come from THIS session's own execution.** Do not copy a new-model number from anywhere;
  compute it. **Every original/2023-side value must be read from the replication artifacts on disk** (`concept_g.pt`,
  `concept_gamma.pt`, `verify/*.json`) — and model #1 recomputed here must match them (0.046/0.069) or you debug first.
- **Disclose every data hole and any hand-coded value.** If a concept is dropped for a model because its pairs are not
  single-token under that tokenizer, say so in `report.md` and exclude it from the intersection series — never
  fabricate or impute a value. Do not hardcode results into the figure.
- Do not modify anything under `<run_dir>` outside `followup/001-living-update/`. Work on the copied `codebase/`.