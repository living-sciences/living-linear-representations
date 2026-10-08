import sys, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
PATHS = {
    "qwen3.5-9b-base": "/net/projects2/chai-lab/shared_models/hub/models--Qwen--Qwen3.5-9B-Base/snapshots/2d021f1887f1fe402bf2c53ed69d7f0fc4709ec9",
    "gemma-4-12b":     "/net/projects2/chai-lab-models/haokunliu/alignment-batch/hf-cache/hub/models--google--gemma-4-12B/snapshots/023679ed352de9bb66cc873c9009ce3482585c08",
}
key = sys.argv[1]
m = AutoModelForImageTextToText.from_pretrained(PATHS[key], dtype=torch.bfloat16, device_map="auto").eval()
print("top-level children:", [n for n, _ in m.named_children()])
for n, c in m.named_children():
    print(f" {n}: {type(c).__name__}; children={[cn for cn,_ in c.named_children()]}")
# find modules that have .layers
print("--- modules with a .layers attr ---")
for n, mod in m.named_modules():
    if hasattr(mod, "layers") and isinstance(getattr(mod, "layers"), torch.nn.ModuleList):
        print(f"  {n or '<root>'}: {type(mod).__name__}, n_layers={len(mod.layers)}, layer0={type(mod.layers[0]).__name__}")
oe = m.get_output_embeddings()
print("get_output_embeddings:", type(oe).__name__, "weight", tuple(oe.weight.shape))
ie = m.get_input_embeddings()
print("get_input_embeddings:", type(ie).__name__, "weight", tuple(ie.weight.shape))
