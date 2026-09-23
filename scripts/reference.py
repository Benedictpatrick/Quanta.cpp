"""Dump Hugging Face reference outputs that the C++ engine is tested against.

Writes tests/data/ref.json (prompts, input ids, greedy continuation ids) and
tests/data/ref_logits_<i>.npy (float32 logits for every prompt position).
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "<|im_start|>user\nWrite a haiku about phones.<|im_end|>\n<|im_start|>assistant\n",
]
N_GEN = 64


def main():
    model_dir = "models/qwen2.5-0.5b-instruct"
    out = Path("tests/data")
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    cases = []
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0].numpy().astype(np.float32)
            # Pure greedy: override generation_config.json (Qwen ships repetition_penalty=1.1, top_k, ...).
            # Explicit mask: otherwise HF infers one from pad_token_id (= <|im_end|>) and hides
            # the <|im_end|> tokens inside chat prompts.
            gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=N_GEN, do_sample=False, repetition_penalty=1.0,
                                 temperature=None, top_p=None, top_k=None, eos_token_id=None,
                                 pad_token_id=tok.eos_token_id)
        np.save(out / f"ref_logits_{i}.npy", logits)
        greedy = gen[0, ids.shape[1]:].tolist()
        cases.append({"prompt": p, "input_ids": ids[0].tolist(), "greedy_ids": greedy})
        print(f"[{i}] {len(ids[0])} tokens -> {tok.decode(greedy)!r}")

    (out / "ref.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
