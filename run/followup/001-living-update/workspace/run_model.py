"""
Follow-up study 001 (living-update): compute the paper's core geometry quantities
for ONE model's unembedding matrix. Reuses the replicated methodology
(store_matrices.py + linear_rep_geometry.py) but:
  - loads any HF causal LM (AutoTokenizer/AutoModelForCausalLM), bf16
  - uses a tokenizer-agnostic single-token lookup (works for Qwen/OLMo w/o BOS)
  - computes Cov(gamma) / eigh in float32
The math (g = gamma @ Cov(gamma)^-1/2, concept dir = unit-normalized mean of
target-base diffs, |C C^T| off-diagonal, LOO projections, uncorrelatedness r)
is unchanged from linear_rep_geometry.py.

Usage: python run_model.py <model_key>
"""
import os, sys, json, time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

RESULTS = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run/followup/001-living-update/results"
WORD_DIR = "codebase/word_pairs"

MODELS = {
    "llama2-7b":    ("meta-llama/Llama-2-7b-hf", "2023", "/net/projects2/chai-lab/shared_models/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"),
    "llama3.1-8b":  ("meta-llama/Llama-3.1-8B",  "2024", "/net/projects2/chai-lab/shared_models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"),
    "qwen2.5-7b":   ("Qwen/Qwen2.5-7B",          "2024", "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"),
    "qwen3-8b-base":("Qwen/Qwen3-8B-Base",       "2025", "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3-8B-Base/snapshots/49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
    "olmo3-7b":     ("allenai/Olmo-3-1025-7B",   "2025", "/net/projects2/chai-lab/shared_models/hub/models--allenai--Olmo-3-1025-7B/snapshots/0a15c289b17ad4f27aabc863276ec525d5312c01"),
}

# concept order fixed exactly as in store_matrices.py
FILENAMES = [
    'verb - 3pSg','verb - Ving','verb - Ved','Ving - 3pSg','Ving - Ved','3pSg - Ved',
    'verb - V + able','verb - V + er','verb - V + tion','verb - V + ment',
    'adj - un + adj','adj - adj + ly','small - big','thing - color','thing - part',
    'country - capital','pronoun - possessive','male - female','lower - upper',
    'noun - plural','adj - comparative','adj - superlative','frequent - infrequent',
    'English - French','French - German','French - Spanish','German - Spanish',
]
CONCEPT_NAMES = [f"{c.split(' - ')[0]}=>{c.split(' - ')[1]}" for c in FILENAMES]
# indices for the uncorrelatedness (C11) analog
I_MALE_FEMALE, I_ENG_FR = 17, 23
I_VERB_3PSG, I_VERB_VING = 0, 1


def single_token_id(tok, word):
    """tokenizer-agnostic single-token lookup (paper's single-token design)."""
    for w in (word, " " + word):
        ids = tok(w, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    return None


def get_counterfactual_pairs(tok, concept_str):
    fn = os.path.join(WORD_DIR, f"[{concept_str}].txt")
    with open(fn) as f:
        pairs = [ln.strip().split('\t') for ln in f if ln.strip()]
    base_ind, target_ind = [], []
    for a, b in pairs:
        ia, ib = single_token_id(tok, a), single_token_id(tok, b)
        if ia is not None and ib is not None and ia != ib:
            base_ind.append(ia); target_ind.append(ib)
    return base_ind, target_ind


def concept_direction(base_ind, target_ind, data):
    diff = data[target_ind] - data[base_ind]
    mean = diff.mean(dim=0)
    return mean / mean.norm(), diff


def inner_product_loo(base_ind, target_ind, data):
    diff = data[target_ind] - data[base_ind]
    prods = []
    for i in range(diff.shape[0]):
        mask = torch.ones(diff.shape[0], dtype=bool); mask[i] = False
        m = diff[mask].mean(dim=0)
        prods.append((m / m.norm()) @ diff[i])
    return torch.stack(prods)


def offdiag_stats(rows):
    """rows: (k,d) tensor of concept directions. returns mean,median of |C C^T| off-diag."""
    C = rows / rows.norm(dim=1, keepdim=True)
    M = (C @ C.T).abs()
    n = M.shape[0]
    off = M[~torch.eye(n, dtype=bool)]
    return float(off.mean()), float(off.median())


def cohens_d(cf, rnd):
    n1, n2 = len(cf), len(rnd)
    s1, s2 = cf.std(unbiased=True), rnd.std(unbiased=True)
    pooled = torch.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return float((cf.mean() - rnd.mean()) / pooled)


def main():
    key = sys.argv[1]
    hf_id, era, path = MODELS[key]
    t0 = time.time()
    print(f"=== {key} ({hf_id}, {era}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, low_cpu_mem_usage=True, device_map={"": 0}, dtype=torch.bfloat16)
    dev = torch.device("cuda:0")

    # unembedding matrix; cast to float32 for the geometry (matches replication's fp32 pipeline)
    gamma = model.lm_head.weight.detach().to(dev).float()
    W, d = gamma.shape
    print(f"  gamma shape = {tuple(gamma.shape)}", flush=True)

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
