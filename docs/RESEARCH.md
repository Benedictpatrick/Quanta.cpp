# Quanta Engine — Research & Strategy (Sep 2026)

## 1. Where the field is today
| Fact | Source |
|---|---|
| Phone decode is **memory-bandwidth bound**: phones have 50–90 GB/s vs 2–3 TB/s on datacenter GPUs. | [State of the Union 2026](https://v-chandra.github.io/on-device-llms/) |
| Flagship phones run 3–8B 4-bit models at **20–50 tok/s**; tiny models (≤0.5B) reach 100+ tok/s. | [Octomil guide](https://docs.octomil.com/blog/on-device-llm-inference-2025-2026/) |
| **NPU wins prefill** (>1,400 tok/s), **CPU wins decode** (~70 tok/s). Framework choice alone can cause 15× gaps. | [Is Your NPU Ready for LLMs?](https://arxiv.org/html/2607.05475v1) |
| **Thermal throttling** dominates sustained speed: −41% (iPhone 16 Pro), −69% after 30 min (SD 8 Gen 3). A thermal-aware pipeline kept 77% of peak vs 31%. | [Edge LLM under sustained load](https://arxiv.org/pdf/2603.23640), [DEV thermal article](https://dev.to/software_mvp-factory/thermal-throttling-and-sustained-on-device-llm-inference-on-android-4nh5) |
| **LUT kernels** (T-MAC) replace multiplies with table lookups: 4–5× over llama.cpp for 1–2 bit models on CPU. | [T-MAC](https://arxiv.org/abs/2407.00088), [Vec-LUT](https://arxiv.org/pdf/2512.06443) |
| **Speculative decoding** gives 2–3× (EAGLE-3); mobile-specific versions exist (Lever, 1.5× over plain spec-decode). | [Lever](https://arxiv.org/html/2605.16786) |
| **Prompt-lookup (n-gram) drafting** needs no draft model and no extra RAM: up to 2.4–2.8× on copy-heavy tasks. | [prompt-lookup-decoding](https://github.com/apoorvumang/prompt-lookup-decoding) |
| **Activation sparsity** (TEAL): 40–50% sparsity, 1.5–1.8× decode on **7B+** models. | [TEAL](https://arxiv.org/abs/2408.14690) |
| Leaders: llama.cpp (CPU, GGUF), ExecuTorch 1.0 (Meta), Qualcomm GENIE (NPU), MNN, MLC. | above |

## 2. Our own measurements (Qwen2.5-0.5B, perplexity on 4k tokens)
| Experiment | Result | Takeaway |
|---|---|---|
| Quanta f32 vs Hugging Face | max logit diff 3.6e-4, greedy 64/64 | Engine is correct |
| Plain Q4_0 | top-1 agreement 60–92% | Too lossy for a tiny model → need smarter 4-bit |
| Uniform activation sparsity 25% / 40% | ppl +2.7% / +15.4% | Uniform, uncalibrated sparsity hurts this tiny model fast |
| Sparsity on `down_proj` input only, 50% / 60% | ppl +2.4% / +6.4% | Usable, but saves only ~17% of weight reads → nice bonus, not a headline |

*Preliminary:* 4k tokens of Python source/license text, uniform thresholds with no per-layer
calibration (TEAL calibrates), applied at every position. Not directly comparable to TEAL's 7B numbers.

**Conclusion:** the research's biggest wins are on big models. For small phone models the most promising
levers are *sustained* speed (thermal), memory-free speculation, and skipping prefill on reopened chats.

## 3. Quanta's bet
**The techniques below all exist separately** (in papers, llama.cpp, MNN and others). Quanta's bet is to
combine them in one small engine built for small phone models, and to **prove sustained, real-use gains
with published head-to-head numbers**. The novelty is the combination and the evidence, not any single trick.

### N1. Quanta Governor — thermal-aware adaptive decoding
The engine reads the phone's thermal state and battery each second and re-tunes itself live: thread
count, big/little core placement, speculation depth, sparsity level. Prior art: an adaptive pipeline in
the sustained-load study reached 77% of peak after 30 min; MNN-AECS does adaptive core selection.
Target: **≥75% of peak speed after 30 min**, while also tuning speculation and sparsity (not only threads).
*Pitch: "Quanta doesn't slow down when your phone heats up."*

### N2. Free Speculation — drafting with zero extra memory
Stack three free draft sources, verified in one batched forward pass:
1. **Prompt lookup** — copy from the conversation (great for code, summaries, edits).
2. **Personal n-gram memory** — an on-device store of the user's own frequent phrases that persists
   across sessions (prior art: llama.cpp's saved dynamic n-gram cache, REST's datastore drafting).
3. **Self-drafting via layer skip** — draft with the model's first layers, verify with all layers.
Target: **≥1.5× decode on code/editing tasks**. Gains on open-ended chat are expected to be smaller;
we report them without a target.

### N3. Instant Resume — zero-prefill conversations
Save the KV cache of the system prompt and each chat to flash, then mmap it back when the app reopens.
Known technique (llama.cpp has prompt-cache/state files); rarely shipped in mobile chat apps.
Reopening a long chat costs ~0 ms instead of re-reading every past token.
*Pitch: "Open a 10,000-word chat instantly."*

### N4. Smart 4-bit — quality-aware mixed precision
Measure each layer's sensitivity, keep fragile layers at 8-bit and robust ones at 4-bit (or lower with
LUT kernels). Target: 4-bit size with **≤2% perplexity loss** on a 0.5B model where plain Q4_0 fails.

### N5. Sparse down-projection (bonus)
50% activation sparsity on `down_proj` only (measured +2.4% ppl), using a column-major layout so skipped
activations skip whole weight columns → ~17% less weight traffic per token.

### N6. Phase split (later)
Prefill on NPU/GPU, decode on CPU — the published best practice, rarely done well.

## 3b. Budget & mid-range phones ("Quanta runs everywhere")
**Why cheap phones are slow:** decode speed ≈ memory bandwidth ÷ bytes read per token. Budget phones have
LPDDR4X and small/older cores (A53/A55/A76/A78), several times less bandwidth than flagships, and adding little cores
to decode *hurts* ([COTS mobile study](https://arxiv.org/html/2410.03613v2)). So on cheap phones the whole game is
**reading fewer bytes per token** and not wasting the weak CPU on dequantization.

### N7. Exact Shortlist Head (our finding — prior-art check still needed)
Qwen2.5-0.5B's output head (152K vocab × 896) is **136M of 494M params (28%)**, and it is read in full every
token. With 4-bit layers and an 8-bit head (the usual setup), the head is **~42% of all bytes per token**.

Idea: score the whole vocabulary with a cheap **4-bit copy** of the head, then recompute exact 8-bit logits only for
tokens that *could* still win. A per-32-weight-block error norm gives a hard bound
`|exact − coarse| ≤ Σ_blocks ‖err_block‖·‖h_block‖`, so the result is **provably identical** to the full head
(exact greedy / exact top-k), not an approximation.

Measured on Qwen2.5-0.5B (scratch experiments, ~2k prose tokens + 1k generated chat tokens):
| Method | Top-1 found | Head bytes vs Q8 |
|---|---|---|
| Low-rank shortlist (rank 192, K=256) | 93–96% ✗ | 22% |
| Frequent-vocab only (ids < 32K) | 93–96% ✗ | 22% |
| Coarse 2-bit + rerank top-64 | 97.4–99.8% ✗ | 29% |
| Coarse 4-bit + rerank top-16 | 99.95–100% | 53% |
| **Coarse 4-bit + block error bound (exact)** | **100% guaranteed** | **~59% + re-checks** |

C++ engine measurements (2026-09-24, `quanta checkhead`, 725 prose tokens, cut-off = k-th best exact score of the
best k+8 coarse tokens): greedy re-checks mean 0.47% of rows (median 251, p99 5,351); top-k 20 (chat default)
mean 6.3% (median 7,485, p99 45,135 ≈ 30%). 0 bound violations, output bit-identical to the full head.
Speed on a phone: **not measured yet**.

Related prior work is approximate or draft-only: [VocabTrim](https://arxiv.org/html/2506.22694v1),
[SlimSpec](https://arxiv.org/abs/2605.10453), [FR-Spec](https://arxiv.org/pdf/2502.14856),
[vocab trimming](https://blog.squeezebits.com/vocabulary-trimming-methods), [learning to screen](https://arxiv.org/pdf/1810.12406).
Bound-based pruning is classic in max-inner-product search; applying it for an exact quantized LM head was not
found in our search. Caveats: the tail (p99 ≈ 7.5K–16K re-checks) needs a cap/fallback; storage grows by the 4-bit
copy (~77 MB) since the 8-bit head is still needed for embeddings and re-checks.

### Bytes per token and speed ceilings (Qwen2.5-0.5B, theoretical upper bounds)
Assumed *effective* bandwidth (rough estimates, to be measured): budget ~12 GB/s, mid ~25 GB/s, flagship ~60 GB/s.
| Config | MB/token | Budget | Mid | Flagship |
|---|---|---|---|---|
| Q4 layers + Q8 head (typical today) | 346 | 35 | 72 | 173 |
| Q4 layers + Exact Shortlist Head | 284 | 42 | 88 | 211 |
| 2-bit layers + Exact Shortlist Head | 195 | 62 | 128 | 308 |
Real speed lands well below the ceiling; the ratios are the point (**up to ~1.8× less memory traffic** than typical).

### Other low-end requirements
- **2-bit layers** need quantization-aware training to keep quality ([ParetoQ](https://arxiv.org/pdf/2502.02631):
  2-bit 1B beats 4-bit 600M). **LUT kernels** ([T-MAC](https://arxiv.org/abs/2407.00088)) avoid multiplies and shine on
  weak cores (11 tok/s 3B on a Raspberry Pi 5). Once layers are 2-bit, the head dominates even more → N7 matters more.
- **Every CPU gets a kernel:** plain NEON for A53 (no dot-product instructions), `sdot` for A55/A76+, `i8mm` where
  present — chosen at runtime.
- **Quanta Calibrate:** on first launch, measure real bandwidth and cores, then pick model variant, thread count
  (big cores only for decode) and kernels automatically.
- **Fits in 4 GB phones:** mmap'd weights, 8-bit KV cache, no f32 copies.

## 4. Roadmap
| Step | Delivers | Gate |
|---|---|---|
| M2 | Tokenizer, sampler, chat CLI | Token ids identical to HF |
| M3 | Smart 4-bit (N4) + multithreading | ≤2% ppl loss at ~4.5 bits/weight |
| M4 | ARM NEON kernels (A53/A55/A76+ paths), runs on phone | **Match or beat llama.cpp** decode on the same phone |
| M4b | Exact Shortlist Head (N7) in C++ | Identical output to full head; measured head-time drop on a real phone |
| M5 | Free Speculation (N2) | ≥1.5× decode on code/edit prompts; chat reported |
| M6 | Instant Resume (N3) + Android app | Reopen long chat < 100 ms |
| M7 | Quanta Governor (N1) | ≥75% of peak speed after 30 min |
| M8 | Sparse down-proj (N5), NPU prefill (N6) | measured gains |

## 5. How we prove "top tier"
Publish a benchmark that measures what users actually feel, head to head with llama.cpp and ExecuTorch
on the same phones — at least one **budget** (A53/A55-class), one **mid-range** and one **flagship** device:
- **Sustained tok/s** at 1, 10 and 30 minutes (not just peak)
- **Energy per token** (mJ/token) from Android battery stats
- **Time to first token** for new chats and reopened chats
- **Quality**: perplexity and exact-match versus the full-precision model

Claims go in the marketing only after these numbers exist.
