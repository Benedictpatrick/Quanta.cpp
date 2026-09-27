"""Deep study of the Exact Shortlist Head: which screen reads the fewest bytes per token, exactly?

Every variant bounds each row's exact 8-bit logit from above, finds a threshold tau from a few exactly scored
"seed" rows (the k+8 best screen scores; tau = k-th best exact among them), and rescores exactly every row whose
bound reaches tau. The answer is identical to the full 8-bit head by construction. We report bytes read per token
as a share of the full 8-bit head (Q8_0: 34 bytes per 32 weights).

Variants
  q4copy    current engine: separate Q4 copy (+ per-32-block f16 error norms), exact rows re-read from Q8.
  hi4       no extra copy: the Q8 ints stored as two nibble planes; screen = high plane (q>>4 at mid-point),
            exact rows read only the low plane. Bound 8*d*|x_b|_1 (no storage) or stored L2 error norms.
  hi2>hi4   progressive planes: screen with the top 2 bits, survivors read bits 3-4 and are re-bounded,
            survivors of that read the low nibble.
  svdR      rank-R projection P (from the head's SVD): screen = int8 (P^T w).(P^T x) per row, bound
            = |dq_row|*|P^T x| + |w_perp_row|*|x_perp|; exact rows read from Q8.

  python -u research/head_deep.py      (uses research/funnel_hidden.pt from research/funnel.py)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nested_accept import MD, QK, q8_0  # noqa: E402
from funnel import hidden_states, q4_rtn  # noqa: E402

KS = (1, 20)
SEEDS = 8


def run(name, x, exact, screen, bound, bytes_fn):
    """screen, bound: [N, V]. bytes_fn(survivors [N] ints, level-2 survivors or None) -> bytes share [N]."""
    ub = screen + bound
    out = []
    for k in KS:
        seeds = screen.topk(k + SEEDS, dim=1).indices
        tau = exact.gather(1, seeds).topk(k, dim=1).values[:, -1]
        live = ub >= tau[:, None]
        assert (live.gather(1, exact.topk(k, dim=1).indices)).all(), "bound violated"
        share = bytes_fn(live)
        out.append(f"top-{k:2d}: rows exact {live.float().mean():6.2%}  bytes/token {share.mean():6.1%} of 8-bit head")
    print(f"{name:26s} " + "   ".join(out), flush=True)


def main():
    torch.set_grad_enabled(False)
    x = hidden_states(128)
    from transformers import AutoModelForCausalLM
    w = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32).model.embed_tokens.weight.detach()
    q, dq = q8_0(w)  # q: [V*nb, 32] ints, dq: [V*nb]
    V, D = w.shape
    nb = D // QK
    w8 = (q * dq[:, None]).reshape(V, D)
    exact = x @ w8.T
    N = len(x)
    xb1 = x.reshape(N, nb, QK).abs().sum(2)   # block L1 norms of x
    xb2 = x.reshape(N, nb, QK).norm(dim=2)    # block L2 norms
    full = D * 34 / 32                        # bytes per row, Q8_0
    print(f"{N} tokens, head {V} x {D}; bytes shares are per token, relative to {V * full / 1e6:.0f} MB")

    # ---- q4copy (current engine)
    w4 = q4_rtn(w8)
    err = (w8 - w4).reshape(V, nb, QK).norm(dim=2)
    run("q4copy (+85 MB)", x, exact, x @ w4.T, xb2 @ err.T,
        lambda live: (V * (D * 18 / 32 + nb * 2) + live.sum(1) * full) / (V * full))

    # ---- hi4 nibble planes (no extra copy)
    h4 = torch.floor(q / 16)
    c4 = ((16 * h4 + 8) * dq[:, None]).reshape(V, D)
    d_row = dq.reshape(V, nb)
    plane4 = D * 0.5 + nb * 2  # high nibble + the block scale
    lowrd = D * 0.5            # the low nibble completes the row
    run("hi4, analytic bound", x, exact, x @ c4.T, 8 * xb1 @ d_row.T,
        lambda live: (V * plane4 + live.sum(1) * lowrd) / (V * full))
    err4 = (w8 - c4).reshape(V, nb, QK).norm(dim=2)
    run("hi4, stored L2 norms", x, exact, x @ c4.T, xb2 @ err4.T,
        lambda live: (V * (plane4 + nb * 2) + live.sum(1) * lowrd) / (V * full))

    # ---- progressive hi2 -> hi4 -> full
    h2 = torch.floor(q / 64)
    c2 = ((64 * h2 + 32) * dq[:, None]).reshape(V, D)
    s2, b2 = x @ c2.T, 32 * xb1 @ d_row.T
    s4, b4 = x @ c4.T, 8 * xb1 @ d_row.T
    for k in KS:
        seeds = s2.topk(k + SEEDS, dim=1).indices
        tau = exact.gather(1, seeds).topk(k, dim=1).values[:, -1]
        live2 = (s2 + b2) >= tau[:, None]
        live4 = live2 & ((s4 + b4) >= tau[:, None])
        share = (V * (D * 0.25 + nb * 2) + live2.sum(1) * D * 0.25 + live4.sum(1) * lowrd) / (V * full)
        print(f"{'hi2>hi4 (no extra copy)':26s} top-{k:2d}: level-1 survivors {live2.float().mean():6.2%}  "
              f"rows exact {live4.float().mean():6.2%}  bytes/token {share.mean():6.1%}", flush=True)

    # ---- low-rank screen with a two-sided residual bound
    U, S, Vh = torch.linalg.svd(w8, full_matrices=False)
    for R in (64, 128, 256):
        P = Vh[:R].T                          # [D, R] orthonormal
        wr = w8 @ P                           # [V, R] coefficients
        amax = wr.abs().amax(1, keepdim=True)
        sc = amax / 127
        wq = torch.round(wr / sc) * sc        # int8 per row
        dqn = (wr - wq).norm(dim=1)           # quantization error of the coefficients
        wperp = (w8 - wr @ P.T).norm(dim=1)   # part of the row outside the subspace
        xr = x @ P
        xperp = (x - xr @ P.T).norm(dim=1)
        screen = xr @ wq.T
        bound = dqn[None, :] * xr.norm(dim=1)[:, None] + wperp[None, :] * xperp[:, None]
        run(f"svd{R} (+{V * (R + 6) / 1e6:.0f} MB)", x, exact, screen, bound,
            lambda live, R=R: (V * (R + 6) + live.sum(1) * full) / (V * full))
        print(f"    |x_perp|/|x| mean {(xperp / x.norm(dim=1)).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
