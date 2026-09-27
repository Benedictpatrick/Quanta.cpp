"""Funnel test: can whole groups of vocabulary rows be ruled out exactly, before the 4-bit screen?

Level 1  clusters: k-means on the 8-bit head rows; a cluster's best possible logit is bounded by
         centroid . x + min(R * |x|, sum_b R_b * |x_b|)   (R = cluster radius, R_b = per-32-block radius).
Level 2  4-bit screen: coarse score + sum_b err_b * |x_b| (the current Exact Shortlist Head bound).
Level 3  exact 8-bit rows.
For each real hidden state we report the share of rows that survive level 1 (their cluster's bound reaches the
true k-th best logit: the minimum any exact method must touch with these clusters) and the share that also
survive level 2. Both are "oracle threshold" numbers: a real run finds the threshold on the way, so it touches a
little more.

  python -u research/funnel.py [--clusters 1024 4096]
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from nested_accept import MD, PROMPTS, QK, q8_0  # noqa: E402

CACHE = Path("research/funnel_hidden.pt")


def hidden_states(n_gen):
    if CACHE.exists():
        return torch.load(CACHE)
    tok = AutoTokenizer.from_pretrained(MD)
    m = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32).eval()
    xs = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      return_tensors="pt")
        seq = m.generate(ids, max_new_tokens=n_gen, do_sample=False, pad_token_id=tok.eos_token_id)
        h = m.model(seq).last_hidden_state[0]  # after the final RMSNorm: exactly what the head multiplies
        xs.append(h[ids.shape[1] - 1 : seq.shape[1] - 1])
    x = torch.cat(xs)
    torch.save(x, CACHE)
    return x


def q4_rtn(w):
    b = w.reshape(-1, QK)
    idx = b.abs().argmax(dim=1)
    mx = b[torch.arange(len(b)), idx]
    d = mx / -8.0
    inv = torch.where(d != 0, 1.0 / torch.where(d != 0, d, torch.ones_like(d)), torch.zeros_like(d))
    q = torch.clamp(torch.round(b * inv[:, None] + 8.0), 0, 15)
    return ((q - 8) * d.half().float()[:, None]).reshape(w.shape)


def kmeans(w, c, iters=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = w[torch.randperm(len(w), generator=g)[:c]].clone()
    for it in range(iters):
        assign = torch.empty(len(w), dtype=torch.long)
        cn = (cent * cent).sum(1)
        for i in range(0, len(w), 16384):
            blk = w[i : i + 16384]
            assign[i : i + 16384] = (cn[None, :] - 2 * blk @ cent.T).argmin(1)
        new = torch.zeros_like(cent).index_add_(0, assign, w)
        cnt = torch.bincount(assign, minlength=c).float()
        empty = cnt == 0
        cent = torch.where(empty[:, None], cent, new / cnt.clamp(min=1)[:, None])
    return cent, assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--n-gen", type=int, default=128)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    x = hidden_states(args.n_gen)  # [N, d]
    m = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32)
    w = m.model.embed_tokens.weight.detach().clone()  # tied head [V, d]
    del m
    q, dq = q8_0(w)
    w8 = (q * dq[:, None]).reshape(w.shape)
    V, d = w8.shape
    nb = d // QK
    N = len(x)
    print(f"{N} hidden states, head {V} x {d}")

    xb = x.reshape(N, nb, QK).norm(dim=2)  # [N, nb] block norms of x
    xn = x.norm(dim=1)                     # [N]
    exact = x @ w8.T                       # [N, V]
    kth = {k: exact.topk(k, dim=1).values[:, -1] for k in (1, 20)}

    # Level 2 alone (current method), oracle threshold.
    w4 = q4_rtn(w8)
    err = (w8 - w4).reshape(V, nb, QK).norm(dim=2)  # [V, nb]
    up4 = x @ w4.T + xb @ err.T                      # [N, V]
    for k in (1, 20):
        s2 = (up4 >= kth[k][:, None]).float().mean().item()
        print(f"level 2 only (4-bit screen), top-{k}: rows needing exact {s2:.2%}")

    for c in args.clusters:
        t = time.time()
        cent, assign = kmeans(w8, c)
        diff = w8 - cent[assign]
        R = torch.zeros(c).index_reduce_(0, assign, diff.norm(dim=1), "amax")
        Rb = torch.zeros(c, nb).index_reduce_(0, assign, diff.reshape(V, nb, QK).norm(dim=2), "amax")
        ub = x @ cent.T + torch.minimum(R[None, :] * xn[:, None], xb @ Rb.T)  # [N, c]
        sizes = torch.bincount(assign, minlength=c).float()
        for k in (1, 20):
            live = ub >= kth[k][:, None]                     # [N, c] clusters that must be opened
            s1 = (live.float() @ sizes / V)                  # share of rows in opened clusters
            rows_live = live[:, assign]                      # [N, V]
            s12 = ((up4 >= kth[k][:, None]) & rows_live).float().mean(1)
            print(f"clusters {c:5d} top-{k:2d}: open clusters {live.float().mean():6.2%}  "
                  f"rows screened (4-bit) {s1.mean():6.2%} (p90 {s1.quantile(.9):6.2%})  "
                  f"rows exact {s12.mean():6.2%}   [{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
