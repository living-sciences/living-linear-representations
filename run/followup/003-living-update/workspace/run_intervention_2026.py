"""OPTIONAL C8 diagnostic for the 2026 models (same as study 001's run_intervention.py
but with the AutoModelForImageTextToText loader + get_output_embeddings unembedding).
show_rank("Long live the", alpha in {0,.1,.2,.3,.4}, concept = male=>female):
does adding alpha*(male=>female) flip top-1 king->queen and push king out of top-5?
Merges the 2 new models into the existing intervention_C8.json (prior 5 preserved)."""
import json, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from run_model import get_counterfactual_pairs, concept_direction, FILENAMES, CONCEPT_NAMES
from run_model_2026 import MODELS

RES = "/net/projects2/chai-lab-models/haokunliu/alignment-batch/alignment_papers/test-corpus/linear-representation-hypothesis/run/followup/003-living-update/results"
I_MALE_FEMALE = 17
PROMPT = "Long live the"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4]

out_all = json.load(open(f"{RES}/intervention_C8.json"))  # prior 5 models
for key, (hf_id, era, path) in MODELS.items():
    print(f"=== {key} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(
        path, low_cpu_mem_usage=True, device_map="auto", dtype=torch.bfloat16).eval()
    dev = torch.device("cuda:0")
    gamma = model.get_output_embeddings().weight.detach().to(dev).float()
    W, d = gamma.shape
    centered = gamma - gamma.mean(0)
    Cov = centered.T @ centered / W
    ev, evec = torch.linalg.eigh(Cov)
    inv_sqrt = evec @ torch.diag(1 / torch.sqrt(ev)) @ evec.T
    sqrt_cov = evec @ torch.diag(torch.sqrt(ev)) @ evec.T
    g = gamma @ inv_sqrt

    bi, ti = get_counterfactual_pairs(tok, FILENAMES[I_MALE_FEMALE])
    cdir, _ = concept_direction(bi, ti, g)

    ids = tok(PROMPT, return_tensors="pt").to(dev)
    with torch.no_grad():
        hs = model(**ids, output_hidden_states=True).hidden_states[-1][0, -1, :].float()
    l = hs @ sqrt_cov

    table = {}
    for a in ALPHAS:
        val = g @ (l + a * cdir)
        top = torch.topk(val, 5).indices.tolist()
        table[f"alpha={a}"] = [tok.decode(t).strip() for t in top]
    def isq(w): return w.lower() == "queen"
    def isk(w): return w.lower() == "king"
    flipped = any(isq(table[f"alpha={a}"][0]) for a in ALPHAS if a > 0)
    king_out = any(not any(isk(w) for w in table[f"alpha={a}"]) for a in ALPHAS if a >= 0.3)
    out_all[key] = {"hf_id": hf_id, "era": era, "prompt": PROMPT,
                    "concept": CONCEPT_NAMES[I_MALE_FEMALE], "n_pairs": len(bi),
                    "table": table, "queen_becomes_top1": flipped, "king_leaves_top5_by_0.3+": king_out}
    print(json.dumps(table, indent=2), flush=True)
    print(f"  queen->top1={flipped}, king out top5 (a>=0.3)={king_out}", flush=True)
    del model, gamma, g, centered, Cov
    torch.cuda.empty_cache()

json.dump(out_all, open(f"{RES}/intervention_C8.json", "w"), indent=2)
print("saved intervention_C8.json (now includes 2026 models)", flush=True)
