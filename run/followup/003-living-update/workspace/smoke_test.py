"""Smoke test for 2026 multimodal-arch models (Qwen3.5-9B-Base, gemma-4-12B).
Verifies: load via AutoModelForImageTextToText, unembedding via get_output_embeddings,
one text forward pass, one residual-stream activation read, and single-token lookup."""
import sys, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

PATHS = {
    "qwen3.5-9b-base": "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9",
    "gemma-4-12b":     "/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache/hub/models--google--gemma-4-12B/snapshots/023679ed352de9bb66cc873c9009ce3482585c08",
}

key = sys.argv[1]
path = PATHS[key]
print(f"=== smoke test {key} ===", flush=True)
tok = AutoTokenizer.from_pretrained(path)
m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
print("loaded. model class:", type(m).__name__, flush=True)

# text decoder (2026 Qwen3.5/gemma4 nest it under m.model.language_model, not m.language_model)
dec = m.model.language_model
n_layers = len(dec.layers)
print("language_model class:", type(dec).__name__, "n_layers:", n_layers, flush=True)

# unembedding + input embeddings
unemb = m.get_output_embeddings().weight
inemb = m.get_input_embeddings().weight
print("unembedding shape:", tuple(unemb.shape), "dtype:", unemb.dtype, flush=True)
print("input emb shape:", tuple(inemb.shape), flush=True)
print("tied (same storage):", unemb.data_ptr() == inemb.data_ptr(), flush=True)

# single-token lookup sanity
for w in ["king", " king", "queen", " queen"]:
    ids = tok(w, add_special_tokens=False)["input_ids"]
    print(f"  tok({w!r}) -> {ids} (len {len(ids)})", flush=True)

# one forward pass with hidden states
ids = tok("Long live the", return_tensors="pt").to(m.device)
with torch.no_grad():
    out = m(**ids, output_hidden_states=True)
hs = out.hidden_states
print("n hidden_states:", len(hs), "(expect n_layers+1 =", n_layers + 1, ")", flush=True)
print("last hidden state shape:", tuple(hs[-1].shape), flush=True)
print("logits shape:", tuple(out.logits.shape), flush=True)

# residual-stream forward-hook read on a layer
acts = {}
h = dec.layers[n_layers // 2].register_forward_hook(
    lambda mod, inp, o: acts.__setitem__("mid", (o[0] if isinstance(o, tuple) else o).shape))
with torch.no_grad():
    m(**ids)
h.remove()
print("hook read layer", n_layers // 2, "output shape:", acts["mid"], flush=True)
print("SMOKE OK", flush=True)
