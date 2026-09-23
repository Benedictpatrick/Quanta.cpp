"""GPTQ 4-bit (group 32) for Qwen2.5-0.5B: quantize layer by layer, report perplexity, save the result.

  python -u research/gptq.py --format sym    # Q4_0 layout: int4 in [-8,7] * d          (4.5 bits/weight)
  python -u research/gptq.py --format asym   # Q4_1 layout: int4 in [0,15] * d + m      (5.0 bits/weight)

GPTQ picks integers column by column and pushes each column's rounding error onto the columns not yet
quantized (weighted by the inverse Hessian of the layer inputs), so the *layer output* error stays small.
--hadamard rotates (a) each attention head's values with a 64x64 Hadamard, folded into v_proj/o_proj
offline (free at runtime), and (b) the down_proj input with a block-diagonal 256x256 Hadamard, which the
engine applies online (a fast Walsh-Hadamard transform per token). Rotations spread outliers so 4-bit
rounding hurts less; the math is otherwise identical.
The storage format is unchanged, so the engine's kernels don't care. Saves research/gptq_<format>.pt with
{tensor name: (q uint8 [rows, cols], d fp16 [rows, cols/32], m fp16 or None)}.
"""
import argparse, glob, math, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "models/qwen2.5-0.5b-instruct"
GS = 32


def read(p, n):
    try:
        return open(p, encoding="utf-8", errors="ignore").read()[:n]
    except Exception:
        return ""


def texts(tok):
    lib = sys.base_prefix + "/Lib/"
    files = sorted(glob.glob(lib + "*.py"))
    docs = sorted(glob.glob(lib + "site-packages/*.dist-info/METADATA"))
    calib = "".join(read(p, 3000) for p in files[0:80:2]) + "".join(read(p, 3000) for p in docs[:150])
    evalt = open("research/eval.txt", encoding="utf-8").read()
    c = tok(calib, return_tensors="pt").input_ids[0]
    n = (min(len(c), 16384) // 1024) * 1024
    return c[:n].reshape(-1, 1024), tok(evalt, return_tensors="pt").input_ids[:, :4096]


def f16(x):
    return x.to(torch.float16).to(torch.float32)


def group_params(w, h, fmt):
    """Best scale (and min) for each row of a [rows, 32] group, minimizing h-weighted squared error."""
    best_err = torch.full((w.shape[0], 1), float("inf"))
    if fmt == "sym":
        mx = w.gather(1, w.abs().argmax(-1, keepdim=True))
        best_d = torch.ones_like(mx)
        for f in torch.linspace(0.7, 1.15, 19):
            for sgn in (-1, 1):
                d = f16(mx / (sgn * 8) * f)
                d = d.where(d != 0, torch.full_like(d, 1e-8))
                err = ((w - torch.clamp(torch.round(w / d), -8, 7) * d) ** 2 * h).sum(-1, keepdim=True)
                better = err < best_err
                best_err, best_d = torch.where(better, err, best_err), torch.where(better, d, best_d)
        return best_d, torch.zeros_like(best_d)
    lo, hi = w.min(-1, keepdim=True).values, w.max(-1, keepdim=True).values
    bd, bm = torch.ones_like(lo), torch.zeros_like(lo)
    for a in torch.linspace(0, 0.25, 6):
        for b in torch.linspace(0, 0.25, 6):
            l, u = lo + (hi - lo) * a, hi - (hi - lo) * b
            d, m = f16((u - l) / 15).clamp_min(1e-8), f16(l)
            err = ((w - (torch.clamp(torch.round((w - m) / d), 0, 15) * d + m)) ** 2 * h).sum(-1, keepdim=True)
            better = err < best_err
            best_err, bd, bm = torch.where(better, err, best_err), torch.where(better, d, bd), torch.where(better, m, bm)
    return bd, bm


def gptq(W, H, fmt, block=128, damp=0.01):
    W = W.clone().float()
    rows, cols = W.shape
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    hdiag = torch.diag(H)
    lo_q, hi_q = (-8, 7) if fmt == "sym" else (0, 15)

    Q = torch.zeros_like(W)
    qint = torch.zeros(rows, cols, dtype=torch.uint8)
    D = torch.zeros(rows, cols // GS)
    M = torch.zeros(rows, cols // GS)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols)
        W1, E1, Hi = W[:, i1:i2].clone(), torch.zeros(rows, i2 - i1), Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            c = i1 + i
            if c % GS == 0:  # group starts: fix its params from the error-updated weights
                d, m = group_params(W1[:, i:i + GS], hdiag[c:c + GS], fmt)
                D[:, c // GS], M[:, c // GS] = d[:, 0], m[:, 0]
            d, m = D[:, c // GS], M[:, c // GS]
            w = W1[:, i]
            qi = torch.clamp(torch.round((w - m) / d), lo_q, hi_q)
            q = qi * d + m
            Q[:, c] = q
            qint[:, c] = (qi - lo_q).to(torch.uint8)  # stored as 0..15 (sym: q+8, like Q4_0)
            err = (w - q) / Hi[i, i]
            W1[:, i:] -= err[:, None] * Hi[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q, qint, D.to(torch.float16), (M.to(torch.float16) if fmt == "asym" else None)


def hadamard(n):
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)  # symmetric and orthonormal: H @ H = I


def block_rotate(x, H):  # x [..., n] with n a multiple of H.size -> x' = blockdiag(H) x (per row)
    b = H.shape[0]
    return (x.reshape(*x.shape[:-1], -1, b) @ H).reshape(x.shape)


def apply_hadamard(model):
    """Rotate in place. Returns biases that changed (to be exported instead of the originals)."""
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    H64, H256 = hadamard(hd), hadamard(256)
    changed = {}
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            at, mlp = layer.self_attn, layer.mlp
            # values: v' = H v per kv head  -> rows of W_v and b_v rotated
            Wv = at.v_proj.weight.reshape(-1, hd, cfg.hidden_size)            # [kv_heads, hd, dim]
            at.v_proj.weight.copy_((H64 @ Wv).reshape(-1, cfg.hidden_size))
            bv = at.v_proj.bias.reshape(-1, hd)
            at.v_proj.bias.copy_((bv @ H64).reshape(-1))                        # H symmetric
            changed[f"model.layers.{li}.self_attn.v_proj.bias"] = at.v_proj.bias.detach().clone()
            # attention output per head is rotated the same way -> W_o' = W_o blockdiag(H64)
            at.o_proj.weight.copy_(block_rotate(at.o_proj.weight, H64))
            # down_proj: W x = (W B)(B x) with B = blockdiag(H256); engine rotates x online
            mlp.down_proj.weight.copy_(block_rotate(mlp.down_proj.weight, H256))
            mlp.down_proj.register_forward_pre_hook(lambda mod, inp: (block_rotate(inp[0], H256),))
    return changed


def ppl(model, ids):
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, ids.shape[1] - 1, 1024):
            c = ids[:, s:s + 1024]
            nll += model(c, labels=c).loss.item() * (c.shape[1] - 1)
            cnt += c.shape[1] - 1
    return math.exp(nll / cnt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["sym", "asym"], default="sym")
    ap.add_argument("--hadamard", action="store_true")
    ap.add_argument("--mlp-only", action="store_true", help="GPTQ only the MLP; attention stays for Q8 export")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(8)

    tok = AutoTokenizer.from_pretrained(MD)
    model = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    calib, evald = texts(tok)
    base = ppl(model, evald)
    changed = {}
    if args.hadamard:
        changed = apply_hadamard(model)
        print(f"hadamard applied: fp32 ppl {ppl(model, evald):.4f} (should equal baseline)", flush=True)
    print(f"calib {calib.numel()} tokens | eval {evald.shape[1]} tokens | fp32 ppl {base:.4f}", flush=True)

    t0 = time.time()
    saved = {}
    with torch.no_grad():
        hs = model.model.embed_tokens(calib)
        pos = torch.arange(calib.shape[1])[None].expand(calib.shape[0], -1)
        pe = model.model.rotary_emb(hs, pos)
        for li, layer in enumerate(model.model.layers):
            lin = {n: m for n, m in layer.named_modules()
                   if isinstance(m, torch.nn.Linear) and (not args.mlp_only or n.startswith("mlp."))}
            Hs = {n: torch.zeros(m.in_features, m.in_features) for n, m in lin.items()}

            def mk(n):
                def hook(mod, inp):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    Hs[n].addmm_(x.T, x, alpha=2.0 / calib.numel())
                return hook

            hooks = [m.register_forward_pre_hook(mk(n)) for n, m in lin.items()]
            for b in range(calib.shape[0]):
                layer(hs[b:b + 1], position_ids=pos[b:b + 1], position_embeddings=tuple(t[b:b + 1] for t in pe))
            for h in hooks:
                h.remove()
            for n, m in lin.items():
                Q, qint, D, M = gptq(m.weight, Hs[n], args.format)
                m.weight.copy_(Q)
                saved[f"model.layers.{li}.{n}.weight"] = (qint, D, M)
            # outputs of the quantized layer feed the next one
            hs = torch.cat([layer(hs[b:b + 1], position_ids=pos[b:b + 1],
                                  position_embeddings=tuple(t[b:b + 1] for t in pe))[0] for b in range(calib.shape[0])])
            print(f"  layer {li + 1:2d}/{len(model.model.layers)}  {time.time() - t0:5.0f}s", flush=True)

    p = ppl(model, evald)
    bpw = 4.5 if args.format == "sym" else 5.0
    tag = args.format + ("_had" if args.hadamard else "") + ("_mlp" if args.mlp_only else "")
    print(f"GPTQ {tag} ({bpw} bpw): ppl {p:.4f} vs fp32 {base:.4f} ({(p / base - 1) * 100:+.2f}%)", flush=True)
    torch.save({"quant": saved, "float_overrides": changed, "hadamard": args.hadamard}, f"research/gptq_{tag}.pt")


if __name__ == "__main__":
    main()
