"""Compare 4-bit weight formats on Qwen2.5-0.5B by perplexity (fake-quant in PyTorch).

Layers are quantized; the tied embedding/head stays Q8_0 in every config (as in export.py).
Calibration text (for importance weights) and evaluation text are disjoint.
"""
import glob, math, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
MD = "models/qwen2.5-0.5b-instruct"
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32).eval()
LIB = sys.base_prefix + "/Lib/"

def read(p, n):
    try: return open(p, encoding="utf-8").read()[:n]
    except Exception: return ""
files = sorted(glob.glob(LIB + "*.py"))
calib_text = "".join(read(p, 3000) for p in files[0:80:2])
eval_text = read(sys.base_prefix + "/LICENSE.txt", 8000) + "".join(read(p, 2500) for p in files[1:80:2])
calib = tok(calib_text, return_tensors="pt").input_ids[:, :4096]
evald = tok(eval_text, return_tensors="pt").input_ids[:, :4096]

linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n}
orig = {n: m.weight.detach().clone() for n, m in linears.items()}

# importance = mean squared activation per input column (the "imatrix")
imp = {n: torch.zeros(m.in_features) for n, m in linears.items()}
def collect(n):
    def hook(mod, inp):
        imp[n].add_(inp[0].reshape(-1, inp[0].shape[-1]).pow(2).sum(0))  # returns None: input unchanged
    return hook
hooks = [m.register_forward_pre_hook(collect(n)) for n, m in linears.items()]
with torch.no_grad():
    for s in range(0, calib.shape[1], 1024): model(calib[:, s:s + 1024])
for h in hooks: h.remove()

def ppl(ids=evald):
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, ids.shape[1] - 1, 1024):
            c = ids[:, s:s + 1024]
            nll += model(c, labels=c).loss.item() * (c.shape[1] - 1); cnt += c.shape[1] - 1
    return math.exp(nll / cnt)

def f16(x): return x.to(torch.float16).to(torch.float32)

# ---- formats: each takes W [rows, cols], importance w [cols]; returns dequantized W
def q8_0(W, w):
    b = W.reshape(-1, 32); d = f16(b.abs().amax(-1, keepdim=True) / 127)
    return (torch.clamp(torch.round(b / d.clamp_min(1e-12)), -127, 127) * d).reshape(W.shape)

def q4_0_naive(W, w):
    b = W.reshape(-1, 32)
    mx = b.gather(1, b.abs().argmax(-1, keepdim=True)); d = f16(mx / -8)
    return (torch.clamp(torch.round(b / d.where(d != 0, torch.ones_like(d))), -8, 7) * d).reshape(W.shape)

def search_sym(W, w, bits=4, grid=21):
    """Symmetric per-32 scale chosen to minimize importance-weighted squared error."""
    b = W.reshape(-1, 32); wt = w.repeat(W.shape[0]).reshape(-1, 32) + 1e-8 * w.mean()
    qmax = 2 ** (bits - 1)
    mx = b.gather(1, b.abs().argmax(-1, keepdim=True))
    best_err = torch.full((b.shape[0], 1), float("inf")); best = torch.zeros_like(b)
    for f in torch.linspace(0.7, 1.15, grid):
        for sgn in (-1, 1):
            d = f16(mx / (sgn * qmax) * f); d = d.where(d != 0, torch.full_like(d, 1e-8))
            q = torch.clamp(torch.round(b / d), -qmax, qmax - 1) * d
            err = (wt * (b - q) ** 2).sum(-1, keepdim=True)
            better = err < best_err
            best_err = torch.where(better, err, best_err); best = torch.where(better, q, best)
    return best.reshape(W.shape)

def search_asym(W, w, bits=4, grid=15):
    """Q4_1-style: scale + min per 32, grid search over range shrink, importance-weighted."""
    b = W.reshape(-1, 32); wt = w.repeat(W.shape[0]).reshape(-1, 32) + 1e-8 * w.mean()
    lo, hi = b.min(-1, keepdim=True).values, b.max(-1, keepdim=True).values
    L = 2 ** bits - 1
    best_err = torch.full((b.shape[0], 1), float("inf")); best = torch.zeros_like(b)
    for s_lo in torch.linspace(0, 0.2, 5):
        for s_hi in torch.linspace(0, 0.2, 5):
            l = lo + (hi - lo) * s_lo; h = hi - (hi - lo) * s_hi
            d = f16((h - l) / L).clamp_min(1e-8); m = f16(l)
            q = torch.clamp(torch.round((b - m) / d), 0, L) * d + m
            err = (wt * (b - q) ** 2).sum(-1, keepdim=True)
            better = err < best_err
            best_err = torch.where(better, err, best_err); best = torch.where(better, q, best)
    return best.reshape(W.shape)

def apply(fmt_for):  # fmt_for(name) -> function or None (keep fp32)
    with torch.no_grad():
        for n, m in linears.items():
            f = fmt_for(n)
            m.weight.copy_(orig[n] if f is None else f(orig[n], imp[n]))

def run(label, fmt_for, bpw):
    t = time.time(); apply(fmt_for); p = ppl()
    print(f"{label:48s} {bpw:5.2f} bpw  ppl {p:7.3f}  ({(p / base - 1) * 100:+6.2f}%)  [{time.time() - t:.0f}s]", flush=True)
    return p

base = ppl()
print(f"{'fp32 baseline':48s}        ppl {base:7.3f}")
run("Q8_0", lambda n: q8_0, 8.5)
run("Q4_0 naive (current export)", lambda n: q4_0_naive, 4.5)
run("Q4_0 + weighted scale search", lambda n: search_sym, 4.5)
run("Q4_1 + weighted range search", lambda n: search_asym, 5.0)

# per-tensor-type sensitivity with the best 4-bit symmetric format, others at Q8
kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
for k in kinds:
    run(f"only {k} at Q4 (search), rest Q8", lambda n, k=k: search_sym if n.endswith(k) else q8_0, 0)
