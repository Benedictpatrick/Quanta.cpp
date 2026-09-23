"""Which parts of the GPTQ-4bit model cause the quality loss? Start from all-4bit (gptq_sym.pt) and
restore one (layer, attention|mlp) group to full precision at a time; report the perplexity recovered."""
import math, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "models/qwen2.5-0.5b-instruct"
torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32).eval()
evald = tok(open("research/eval.txt", encoding="utf-8").read(), return_tensors="pt").input_ids[:, :2048]
g = torch.load("research/gptq_sym.pt")
quant = g["quant"] if "quant" in g else g
orig, deq = {}, {}
for n, p in model.named_parameters():
    if n in quant:
        qint, d, _ = quant[n]
        orig[n] = p.detach().clone()
        deq[n] = ((qint.float() - 8).reshape(p.shape[0], -1, 32) * d.float()[:, :, None]).reshape(p.shape)
params = dict(model.named_parameters())

def ppl():
    with torch.no_grad():
        nll, cnt = 0.0, 0
        for s in range(0, evald.shape[1] - 1, 1024):
            c = evald[:, s:s + 1024]
            nll += model(c, labels=c).loss.item() * (c.shape[1] - 1); cnt += c.shape[1] - 1
    return math.exp(nll / cnt)

def set_all(src):
    with torch.no_grad():
        for n in quant: params[n].copy_(src[n])

set_all(orig); base = ppl()
set_all(deq); q4 = ppl()
print(f"fp32 {base:.4f} | all-Q4 GPTQ {q4:.4f} ({(q4/base-1)*100:+.2f}%)", flush=True)
rows = []
for li in range(len(model.model.layers)):
    for grp, keys in (("attn", ["q_proj", "k_proj", "v_proj", "o_proj"]), ("mlp", ["gate_proj", "up_proj", "down_proj"])):
        names = [f"model.layers.{li}.{'self_attn' if grp == 'attn' else 'mlp'}.{k}.weight" for k in keys]
        with torch.no_grad():
            for n in names: params[n].copy_(orig[n])
        p = ppl()
        with torch.no_grad():
            for n in names: params[n].copy_(deq[n])
        share = (q4 - p) / (q4 - base) * 100
        rows.append((share, li, grp))
        print(f"  layer {li:2d} {grp:4s}: restoring recovers {share:5.1f}% of the loss", flush=True)
rows.sort(reverse=True)
print("top:", ", ".join(f"L{li}.{g}={s:.1f}%" for s, li, g in rows[:10]))
