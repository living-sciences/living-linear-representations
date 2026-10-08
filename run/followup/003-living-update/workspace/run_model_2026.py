"""Follow-up study 003 (2026 model-refresh): compute the paper's core geometry
quantities for a 2026 MULTIMODAL-arch model's UNEMBEDDING matrix.

Identical math to study 001's run_model.py (g = gamma @ Cov(gamma)^-1/2, concept
dir = unit-normalized mean of target-base diffs, |C C^T| off-diagonal, LOO
projections, uncorrelatedness r) — REUSED verbatim by importing from run_model.

Only the model LOAD changes: 2026 flagships are multimodal-arch and nnsight/
AutoModelForCausalLM cannot wrap them, so we load with AutoModelForImageTextToText
and pull the unembedding via m.get_output_embeddings().weight (verified 2026-09-11).

Usage: python run_model_2026.py <model_key>
"""
import os, sys, json, time
import torch
import numpy as np
from transformers import AutoModelForImageTextToText, AutoTokenizer

# reuse the EXACT math + concept order from study 001
from run_model import (
    FILENAMES, CONCEPT_NAMES, I_MALE_FEMALE, I_ENG_FR, I_VERB_3PSG, I_VERB_VING,
    single_token_id, get_counterfactual_pairs, concept_direction,
    inner_product_loo, offdiag_stats, cohens_d,
)

RESULTS = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run/followup/003-living-update/results"

MODELS = {
    "qwen3.5-9b-base": ("Qwen/Qwen3.5-9B-Base", "2026",
        "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9"),
    "gemma-4-12b": ("google/gemma-4-12B", "2026",
        "/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache/hub/models--google--gemma-4-12B/snapshots/023679ed352de9bb66cc873c9009ce3482585c08"),
}


def main():
    key = sys.argv[1]
    hf_id, era, path = MODELS[key]
    t0 = time.time()
    print(f"=== {key} ({hf_id}, {era}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    # 2026 multimodal-arch: load with AutoModelForImageTextToText (NOT nnsight / AutoModelForCausalLM)
    model = AutoModelForImageTextToText.from_pretrained(
        path, low_cpu_mem_usage=True, device_map="auto", dtype=torch.bfloat16).eval()
    dev = torch.device("cuda:0")

    # unembedding matrix via get_output_embeddings (the causal-inner-product geometry);
    # cast to float32 for the geometry (matches study 001 / replication fp32 pipeline)
    gamma = model.get_output_embeddings().weight.detach().to(dev).float()
    W, d = gamma.shape
    tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    print(f"  gamma (unembedding) shape = {tuple(gamma.shape)}; tied_with_input_emb={tied}", flush=True)

    gamma_bar = gamma.mean(dim=0)
    centered = gamma - gamma_bar
    Cov = centered.T @ centered / W               # float32
    evals, evecs = torch.linalg.eigh(Cov)          # float32 eigh
    inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
    g = gamma @ inv_sqrt

    # concept directions + coverage
    concept_gamma = torch.zeros(len(FILENAMES), d, device=dev)
    concept_g = torch.zeros(len(FILENAMES), d, device=dev)
    per_concept_pairs, base_inds, target_inds = {}, [], []
    for i, cstr in enumerate(FILENAMES):
        bi, ti = get_counterfactual_pairs(tok, cstr)
        base_inds.append(bi); target_inds.append(ti)
        per_concept_pairs[CONCEPT_NAMES[i]] = len(bi)
        if len(bi) >= 1:
            dg, _ = concept_direction(bi, ti, gamma); concept_gamma[i] = dg
            dgg, _ = concept_direction(bi, ti, g);     concept_g[i] = dgg
    n_total_pairs = sum(per_concept_pairs.values())
    viable = [i for i in range(len(FILENAMES)) if per_concept_pairs[CONCEPT_NAMES[i]] >= 10]
    print(f"  total single-token pairs = {n_total_pairs}; viable concepts = {len(viable)}/27", flush=True)

    torch.save(concept_g.cpu(), f"{RESULTS}/concept_g_{key}.pt")
    torch.save(concept_gamma.cpu(), f"{RESULTS}/concept_gamma_{key}.pt")

    # --- 1. orthogonality anchor (headline) ---
    def ortho_for(idxs):
        cm, cmed = offdiag_stats(concept_g[idxs])
        em, emed = offdiag_stats(concept_gamma[idxs])
        return {"causal_mean": cm, "causal_median": cmed,
                "euclid_mean": em, "euclid_median": emed,
                "ratio_euclid_over_causal_mean": em / cm}
    ortho_all = ortho_for(list(range(len(FILENAMES))))
    ortho_viable = ortho_for(viable)

    # --- 2. subspace separation (Cohen's d) ---
    torch.manual_seed(100)
    num_sample = 100000
    idx1 = torch.multinomial(torch.ones(W, device=dev), num_sample, replacement=True)
    idx2 = torch.multinomial(torch.ones(W, device=dev), num_sample, replacement=True)
    random_pairs_g = g[idx1] - g[idx2]
    cohen = {}
    for i in viable:
        cf = inner_product_loo(base_inds[i], target_inds[i], g)
        rnd = random_pairs_g @ concept_g[i]
        cohen[CONCEPT_NAMES[i]] = cohens_d(cf, rnd)
    separated = [c for c, dd in cohen.items() if dd >= 1.0]
    exceptions = [c for c, dd in cohen.items() if dd < 1.0]

    # --- 3. uncorrelatedness (C11 analog) ---
    def pearson(a_i, b_i):
        if per_concept_pairs[CONCEPT_NAMES[a_i]] < 10 or per_concept_pairs[CONCEPT_NAMES[b_i]] < 10:
            return None
        x = (g @ concept_g[a_i]).cpu().numpy()
        y = (g @ concept_g[b_i]).cpu().numpy()
        return float(np.corrcoef(x, y)[0, 1])
    r_sep = pearson(I_MALE_FEMALE, I_ENG_FR)
    r_nonsep = pearson(I_VERB_3PSG, I_VERB_VING)

    out = {
        "model_key": key, "hf_id": hf_id, "era": era, "snapshot_path": path,
        "loader": "AutoModelForImageTextToText", "unembedding_tied_with_input": bool(tied),
        "vocab_size": int(W), "hidden_size": int(d),
        "coverage": {
            "n_singletoken_pairs_total": int(n_total_pairs),
            "per_concept_pairs": per_concept_pairs,
            "n_viable_concepts": len(viable),
            "viable_concept_indices": viable,
            "viable_concept_names": [CONCEPT_NAMES[i] for i in viable],
        },
        "orthogonality_all27": ortho_all,
        "orthogonality_own_viable": ortho_viable,
        "subspace": {
            "cohens_d": cohen,
            "n_separated": len(separated),
            "n_viable": len(viable),
            "exceptions_d_lt_1": {c: cohen[c] for c in exceptions},
        },
        "uncorrelatedness": {
            "r_separable_maleFemale_vs_engFr": r_sep,
            "r_nonseparable_verb3pSg_vs_verbVing": r_nonsep,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(f"{RESULTS}/metrics_{key}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ortho causal/euclid mean = {ortho_all['causal_mean']:.4f}/{ortho_all['euclid_mean']:.4f} "
          f"(ratio {ortho_all['ratio_euclid_over_causal_mean']:.2f})", flush=True)
    print(f"  separated {len(separated)}/{len(viable)}; exceptions={exceptions}", flush=True)
    print(f"  r_sep={r_sep}, r_nonsep={r_nonsep}", flush=True)
    print(f"  DONE {key} in {out['runtime_sec']}s -> metrics_{key}.json", flush=True)


if __name__ == "__main__":
    main()
