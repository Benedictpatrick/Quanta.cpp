"""Step 1a of the nested-weights idea: how often does the "top-half" draft model agree with the 8-bit model?

The 8-bit model (target) is Qwen2.5-0.5B-Instruct with every linear weight in Q8_0 (as the engine's q8_0 file).
Draft models reuse the same int8 numbers, so they cost no extra memory:
  hi4      top 4 bits of each int8 (q >> 4) at the block midpoint: w ~ d * (16 * (q >> 4) + 8)
  q4_rtn   an independent 4-bit round-to-nearest copy (NOT free; shown as a reference for 4-bit quality)
For greedy decoding, a draft token is accepted iff it equals the target's greedy token given the same prefix,
so teacher-forcing the draft on the target's own output gives the exact accepted lengths of speculative decoding.

  python -u research/nested_accept.py [--n-gen 128]
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MD = "models/qwen2.5-0.5b-instruct"
QK = 32

PROMPTS = [
    "Write a short story about a lighthouse keeper.",
    "Explain how a refrigerator works to a 10-year-old.",
    "What are the main causes of the French Revolution?",
    "Write a Python function that checks if a string is a palindrome, with comments.",
    "Give me a 5-day travel plan for Tokyo.",
    "Summarize the theory of evolution in one paragraph.",
    "Write a polite email asking my manager for a day off next Friday.",
    "What is the difference between TCP and UDP?",
    "Solve step by step: a train travels 180 km in 2.5 hours. What is its average speed?",
    "Write a haiku about monsoon rain.",
    "List ten healthy breakfast ideas.",
    "Explain what a neural network is in simple words.",
    "Translate to French: The weather is nice today and I want to go for a walk.",
    "Write a SQL query that finds the top 3 customers by total order value.",
    "Why is the sky blue?",
    "Give arguments for and against school uniforms.",
    "Write a limerick about a cat who loves coffee.",
    "How do vaccines train the immune system?",
    "Explain recursion with an example in JavaScript.",
    "What should I consider when buying a used car?",
]


def q8_0(w):
    """Returns (q int8-valued float [rows, cols], d float [rows, cols/32]) exactly like scripts/export.py."""
    b = w.reshape(-1, QK)
    d = b.abs().amax(dim=1) / 127.0
    inv = torch.where(d > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)), torch.zeros_like(d))
    q = torch.clamp(torch.round(b * inv[:, None]), -127, 127)
    d16 = d.half().float()
    return q, d16


def target_w(w):
    q, d = q8_0(w)
    return (q * d[:, None]).reshape(w.shape)


def hi4_w(w):
    q, d = q8_0(w)
    h = torch.floor(q / 16)  # arithmetic shift q >> 4: [-8, 7]
    return ((16 * h + 8) * d[:, None]).reshape(w.shape)


def hi_bits_w(bits):
    shift = 8 - bits
    def fn(w):
        q, d = q8_0(w)
        h = torch.floor(q / 2**shift)  # q >> shift
        return ((2**shift * h + 2**(shift - 1)) * d[:, None]).reshape(w.shape)
    return fn


def q4_rtn_w(w):
    b = w.reshape(-1, QK)
    idx = b.abs().argmax(dim=1)
    mx = b[torch.arange(len(b)), idx]
    d = mx / -8.0
    inv = torch.where(d != 0, 1.0 / torch.where(d != 0, d, torch.ones_like(d)), torch.zeros_like(d))
    q = torch.clamp(torch.round(b * inv[:, None] + 8.0), 0, 15)
    return ((q - 8) * d.half().float()[:, None]).reshape(w.shape)


def build(base_sd, fn):
    m = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32, tie_word_embeddings=False)
    sd = {}
    for k, v in base_sd.items():
        if v.dim() == 2 and "embed_tokens" not in k:  # every linear weight + lm_head; embedding lookup stays exact
            sd[k] = fn(v)
        else:
            sd[k] = v
    m.load_state_dict(sd)
    return m.eval()


def spec_tokens_per_step(match, k):
    """Greedy speculative decoding with draft length k: each verify pass yields accepted + 1 tokens."""
    i, steps, n = 0, 0, len(match)
    while i < n:
        a = 0
        while a < k and i + a < n and match[i + a]:
            a += 1
        i += a + 1
        steps += 1
    return n / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=128)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MD)
    base = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32, tie_word_embeddings=False)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    base_sd["lm_head.weight"] = base_sd["model.embed_tokens.weight"].clone()  # Qwen2.5-0.5B ties them
    del base

    t0 = time.time()
    target = build(base_sd, target_w)
    drafts = {"hi4 (free)": build(base_sd, hi4_w), "hi3 (free)": build(base_sd, hi_bits_w(3)),
              "hi2 (free)": build(base_sd, hi_bits_w(2)), "q4_rtn (extra copy)": build(base_sd, q4_rtn_w)}
    print(f"models built in {time.time() - t0:.0f}s")

    matches = {name: [] for name in drafts}
    for pi, p in enumerate(PROMPTS):
        ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                      return_tensors="pt")
        out = target.generate(ids, max_new_tokens=args.n_gen, do_sample=False, pad_token_id=tok.eos_token_id)
        seq = out[:, : ids.shape[1] + args.n_gen]
        gen = seq[0, ids.shape[1]:]
        # Target's own argmax at each generated position (== gen, sanity) and the drafts' argmax.
        for name, d in drafts.items():
            logits = d(seq).logits[0, ids.shape[1] - 1 : seq.shape[1] - 1]
            matches[name].append((logits.argmax(-1) == gen).tolist())
        print(f"prompt {pi + 1}/{len(PROMPTS)}: {len(gen)} tokens", flush=True)

    print("\ngreedy agreement with the 8-bit model and speculative tokens per verify pass (k = draft length)")
    for name, ms in matches.items():
        flat = [x for m in ms for x in m]
        alpha = sum(flat) / len(flat)
        tps = {k: sum(spec_tokens_per_step(m, k) * len(m) for m in ms) / len(flat) for k in (1, 2, 3, 4, 6, 8)}
        print(f"{name:22s} agree {alpha:6.1%}   " + "  ".join(f"k={k}: {v:.2f}" for k, v in tps.items()))


if __name__ == "__main__":
    main()
