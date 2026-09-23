# Quanta — progress log

## Done
- **M1** f32 engine: logits match Hugging Face (max diff 3.6e-4), greedy 64/64.
- **M2** tokenizer (67/67 cases identical to HF), sampler, `quanta chat` / `run` / `ppl`.
- **M3 (partial)**
  - Thread pool: f32 ~4 → ~10 tok/s on PC (8 threads).
  - Integer path (int8 activations): Q8_0 ~20 tok/s on PC, perplexity unchanged (6.4090 vs 6.4134 f32).
  - 16-bit KV cache (default; `--kv-f32` for exact tests).
  - ARM NEON kernels (`sdot` + A53 fallback) with runtime dispatch; Android arm64 build works
    (`build-android/quanta`), **not yet run on a phone**. First phone command: `quanta selftest`.
  - Format v2 with Hadamard flag; exporter accepts GPTQ results (`--gptq research/gptq_*.pt`).

## 4-bit quality results (Qwen2.5-0.5B, perplexity vs fp32)
| Method | bits/weight | loss |
|---|---|---|
| Q4_0 naive | 4.5 | +13% |
| + weighted scale search | 4.5 | +9.2% |
| + GPTQ | 4.5 | +5.5% |
| + GPTQ + Hadamard rotation | 4.5 | +5.5% (no gain on this model) |
| GPTQ asym (Q4_1 style) + Hadamard | 5.0 | +4.5% |
| Q8_0 | 8.5 | ~0% |
| **q4mix**: attention Q8 + MLP GPTQ Q4 | ~4.8 | **+3.1%** |

Engine check (`quanta ppl -n 1024`, PC, 8 threads):
| File | size | ppl | loss | speed |
|---|---|---|---|---|
| f32 | 1980 MB | 6.3935 | — | 11.0 tok/s |
| q8_0 | 529 MB | 6.3739 | ~0% | 24.5 tok/s |
| q4mix | 372 MB | 6.5939 | +3.1% | 22.7 tok/s |
| q4g | 350 MB | 6.6984 | +4.8% | 18.8 tok/s |
**Decision:** default "fast" file = `q4mix` (only 6% bigger than all-4-bit, much better quality);
"quality" file = `q8_0`.
Loss is spread across many layers; attention (12% of weights) ≈ 40% of the loss.

## Batched prefill (2026-09-24)
`forward_batch` runs the prompt in chunks of 32 tokens: each weight matrix is read once per chunk.
`quanta checkbatch` proves logits are **bit-identical** to token-by-token (f32, q8_0, q4mix; incl. a 2nd turn at pos > 0).
PC (ref kernel, compute-bound): prefill 27 → 40 tok/s (q8_0), 24 → 34 tok/s (q4mix). Phone gain should be larger
(bandwidth-bound). `--no-batch` = old path.
**Multi-token kernels (same day):** `matmul_q` (ref + NEON base + NEON sdot) loads/unpacks each weight block once
per 4 tokens, same summation order as matvec → bit-identical (checked by `selftest` and `checkbatch`).
PC single thread: 2.9× (Q8) / 3.3× (Q4) vs looped matvec. PC prompt speed now **91 tok/s (q8_0), 86 tok/s (q4mix)**
vs 30 / 27 token-by-token (~3×). NEON versions compile but are **not run yet** (no ARM device) — `selftest` on
the phone checks them.

## Exact Shortlist Head, N7 (2026-09-24) — working in C++
Export: `python scripts/export.py --dtype q8_0 --gptq research/gptq_sym_mlp.pt --shortlist --out models/qwen2.5-0.5b-q4mix-sl.qnt`
(adds a 4-bit copy of the head + per-32-block error norms; file 372 → 457 MB).
Per token: score all 151,936 tokens with the 4-bit copy, compute the best k+8 exactly, then re-check only tokens
whose upper bound (4-bit score + Σ err_b·|x_b|) can still reach the k-th best. Repetition-penalized tokens are
always exact. `quanta checkhead` over 725 tokens: **0 bound violations, 0 mismatches**; sampled output identical
to `--full-head`.
| Sampling | 8-bit head rows re-checked (mean / median / p99) |
|---|---|
| greedy (k=1) | 0.47% / 251 rows / 5,351 rows |
| top-k 20 (chat default) | 6.3% / 7,485 / 45,135 |
Head bytes per token: ~85 MB (4-bit copy + norms) + re-checks vs 145 MB → q4mix decode reads ~316 MB instead of
372 MB (**~15% less memory traffic — an estimate, not a measured speedup**). On PC it is *slower*
(head 9.7 ms vs 7.3 ms; decode 26.1 vs 27.7 tok/s) because the x86 fallback 4-bit kernel is compute-bound.
Keep plain `q4mix` as the default until `checkhead` on a phone shows the shortlist beating the full head.
Also verified identical to `--full-head`: 3-turn chat, greedy with repetition penalty, 3 seeds × 400 tokens.
Ideas: tighter bound (it is ~30× the real error), 8-bit log-scale norms.

## Android app (2026-09-24)
`android/` (Java, no dependencies; Gradle wrapper 8.14, AGP 8.11.1, NDK 28.2): chat screen + benchmark mode.
Build: `cd android && ./gradlew assembleRelease` → copy to `dist/quanta-0.1.apk` (458 MB; model = q4mix-sl
as an uncompressed asset, copied to app storage on first launch; arm64 + x86_64 libs).
Shared engine code for CLI + app: `engine/src/session.*` (Session, Chat) and `engine/src/checks.*`
(kernel_selftest, check_batch, check_head, run_benchmark). CLI: `quanta bench -m file [-t N]`.
Testing without a phone: Firebase Test Lab Game Loop (project `quanta-bench`, gcloud) — see docs/TESTLAB.md.
Raw reports go to `results/`.

**First ARM run (virtual ARM phone, Test Lab MediumPhone.arm API 34, 4 cores w/ sdot — server hardware,
not a real phone):** app installs and runs; all exactness checks PASS (batch, shortlist head, identical output);
NEON kernels bit-exact matmul-vs-matvec. Only "failure" was a too-strict float tolerance in the kernel check
(1.1e-4 vs 1e-4 per-row; fixed to error relative to output scale). Speeds (not phone-representative):
prefill 117 tok/s; decode 84 tok/s @2 threads; @4 threads full head 99 → **shortlist 126 tok/s (1.28×)**.
Kernel note: plain-NEON (no sdot) Q4 matmul is 0.94× of looped matvec → needs tuning for A53-class phones.

## First real-phone results (2026-09-24, Test Lab, app q4mix-sl, raw reports in results/)
All 4 phones: **ALL PASS** (kernels, batch exactness, shortlist bound + exactness, full-vs-shortlist identical).
| Phone (chip) | sdot | decode tok/s, 4 thr (full / shortlist) | best decode | prefill 500 tok (batched / 1-by-1) |
|---|---|---|---|---|
| Galaxy S24 (SD 8 Gen 3) | yes | 78.6 / 71.3 (0.91×) | 78.6 @4 | 98 / 81 |
| Galaxy A54 (Exynos 1380) | yes | 28.4 / 32.2 (1.13×) | 32.2 @4 SL | 103 / 27 |
| Galaxy A15 (Helio G99) | yes | 15.0 / 15.5 (1.03×) | 23.2 @2 SL | 36 / 15 |
| Galaxy A03s (Helio P35, A53 only, 3 GB) | no | 8.75 / 8.73 | 8.75 @4 | 12 / 5.5 |
Findings:
- 8 threads is always bad (little cores stall the rest: S24 27, A54 5.5 tok/s). Best thread count differs per phone
  (A15: 2 threads 23 tok/s vs 4 threads 15) → **auto-calibrate threads on first launch**.
- Shortlist head helps where memory is the limit (A54 +13%, A03s @2thr +5%), hurts on the S24 (−9%, compute-bound)
  → turn it on per device from a quick calibration.
- Q4 kernels are compute-bound: q4 neon_dot reaches ~40% of q8's GB/s on every phone → faster Q4 unpacking helps all.
- S24 prefill barely beats token-by-token (98 vs 81): something besides the matmuls dominates (attention / sync?).
- A03s (no sdot) sampled text diverges after ~20 words from the sdot phones: different float summation order in
  the plain-NEON kernel changes sampling — expected, not a bug (each phone is self-consistent).

## Next
0. **From phone results:** (a) calibrate threads + shortlist on/off per device, (b) faster Q4 kernels,
   (c) find the S24 prefill bottleneck, (d) same-phone comparison vs llama.cpp.
1. ~~Mixed-precision comparison~~ done (2026-09-24): q4mix = fast, q8_0 = quality.
2. If ≤2% loss at ~4.5 bits is still needed: quantization-aware training (also needed for 2-bit, see RESEARCH.md).
3. **Connect an Android phone** (USB debugging on) → `adb push build-android/quanta` + models → run
   `selftest`, `checkbatch`, `checkhead` (exactness + head time), `ppl`, then `run` speed for q8_0 / q4mix / q4mix-sl
   with and without `--no-batch` / `--full-head`.
4. ~~Batched prefill~~, ~~Exact Shortlist Head (N7)~~ done; multi-token matmul kernels done (phone-verify) → then the Android app.
