# Large files omitted from this repo

GitHub rejects files over 100 MB and discourages files over 50 MB. The files below were part of the
run but are left out of the repository because they are large matrices or model weights. Everything
here is regenerable from the kept code plus the base model weights. The small result tensors that are
genuine deliverables (for example `concept_g.pt`, `concept_gamma.pt`, and the per-model
`concept_g_<model>.pt` / `concept_gamma_<model>.pt` under each `results/`) are kept in the repo.

All three omitted matrices below are produced by `store_matrices.py` in the same `codebase/`
directory, which loads the base model and writes the unembedding matrix, its causal-inner-product
transform, and the covariance square root into `matrices/`. Regenerating them requires the base model
weights (not included; download from HuggingFace, for example `meta-llama/Llama-2-7b-hf` for the
replication) and a GPU.

| Repo-relative path | Size | What it is / how to regenerate |
| --- | --- | --- |
| run/replication/codebase/matrices/gamma.pt | 500.0 MiB | Stacked unembedding matrix gamma (vocab x hidden, about [32000, 4096]) for LLaMA-2-7B. Regenerate with `run/replication/codebase/store_matrices.py`. |
| run/replication/codebase/matrices/g.pt | 500.0 MiB | Unembedding transformed into the causal-inner-product basis g (same shape). Regenerate with `run/replication/codebase/store_matrices.py`. |
| run/replication/codebase/matrices/sqrt_Cov_gamma.pt | 64.0 MiB | Square root of the covariance of gamma (about [4096, 4096]), used to define the causal inner product. Regenerate with `run/replication/codebase/store_matrices.py`. |
| run/followup/001-living-update/workspace/codebase/matrices/gamma.pt | 500.0 MiB | Same as the replication gamma.pt; this is the study's working copy of the codebase. Regenerate with that copy's `store_matrices.py`. |
| run/followup/001-living-update/workspace/codebase/matrices/g.pt | 500.0 MiB | Same as the replication g.pt (study working copy). Regenerate with that copy's `store_matrices.py`. |
| run/followup/001-living-update/workspace/codebase/matrices/sqrt_Cov_gamma.pt | 64.0 MiB | Same as the replication sqrt_Cov_gamma.pt (study working copy). Regenerate with that copy's `store_matrices.py`. |
| run/followup/003-living-update/workspace/codebase/matrices/gamma.pt | 500.0 MiB | Same as the replication gamma.pt (study working copy). Regenerate with that copy's `store_matrices.py`. |
| run/followup/003-living-update/workspace/codebase/matrices/g.pt | 500.0 MiB | Same as the replication g.pt (study working copy). Regenerate with that copy's `store_matrices.py`. |
| run/followup/003-living-update/workspace/codebase/matrices/sqrt_Cov_gamma.pt | 64.0 MiB | Same as the replication sqrt_Cov_gamma.pt (study working copy). Regenerate with that copy's `store_matrices.py`. |

## Model weights

No model weights are checked into this repo. The replication used `meta-llama/Llama-2-7b-hf`; the
living-update studies additionally used the base checkpoints of Llama-3.1-8B, Qwen2.5-7B,
Qwen3-8B-Base, OLMo-3-7B, Qwen3.5-9B-Base, and gemma-4-12B. Each model's own unembedding matrix is
read with `get_output_embeddings()` after loading the base model from HuggingFace. Download the
corresponding checkpoints from HuggingFace to reproduce the per-model results.
