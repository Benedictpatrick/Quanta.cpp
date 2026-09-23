# Quanta

A transformer inference engine for Android phones, written from scratch in C++17 with no dependencies.
It is not based on llama.cpp or any other runtime.

The first supported model is **Qwen2.5-0.5B-Instruct**. It runs as a command-line tool on PC, as an `adb shell`
binary, or inside an Android chat app.

## Highlights

- **Correct first.** Logits match Hugging Face to within 3.6e-4 (f32), greedy output is identical over 64 tokens,
  and tokenization matches the Hugging Face tokenizer on all 67 test cases.
- **Quantized weights.** The weights are memory-mapped straight from a custom `.qnt` file, so there is no load-time copy.
  - Q8_0: about 0% quality loss, 529 MB.
  - **q4mix**: attention in Q8, MLP in 4-bit GPTQ. About 4.8 bits per weight, +3.1% perplexity, 372 MB.
- **ARM NEON kernels.** An `sdot` kernel for ARMv8.2+ cores and a plain-NEON fallback for Cortex-A53, chosen at runtime.
- **Batched prefill.** The prompt is processed in chunks of 32 tokens, with multi-token matmul kernels. The logits are
  **bit-identical** to running one token at a time.
- **Exact Shortlist Head.** The 151,936-word output layer is first scored with a 4-bit copy, and only the tokens
  that could still win are recomputed exactly. The output is provably the same as the full head, while reading
  about 15% fewer bytes per token. It helps on memory-bound phones.
- **Built-in checks.** The `selftest`, `checkbatch`, `checkhead` and `bench` commands check exactness and speed on
  the device itself.

## Results on real phones

Run through Firebase Test Lab. Model: q4mix with the shortlist head. Raw reports are in [`results/`](results/).

| Phone (chip) | Decode, tok/s | Prompt (prefill), tok/s | Checks |
|---|---|---|---|
| Galaxy S24 (Snapdragon 8 Gen 3) | 78.6 | 98 | all pass |
| Galaxy A54 (Exynos 1380) | 32.2 | 103 | all pass |
| Galaxy A15 (Helio G99) | 23.2 | 36 | all pass |
| Galaxy A03s (Helio P35, Cortex-A53 only, 3 GB) | 8.8 | 12 | all pass |

The details, including thread counts and full head versus shortlist, are in [`docs/PROGRESS.md`](docs/PROGRESS.md).

## Build

Requirements: CMake 3.20 or newer, a C++17 compiler (clang or gcc) and Ninja.

```sh
# PC
cmake -B build -G Ninja && cmake --build build

# Android command-line binary (arm64)
cmake -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26
cmake --build build-android
```

## Get a model

The model files are not stored in the repo. Create them from the Hugging Face weights with Python 3.11,
`torch`, `transformers`, `safetensors` and `numpy`:

```sh
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/qwen2.5-0.5b-instruct

# Quality file: Q8_0
python scripts/export.py --dtype q8_0 --out models/qwen2.5-0.5b-q8_0.qnt

# Fast file: q4mix (GPTQ on the MLP, attention stays Q8), plus the shortlist head
python -u research/gptq.py --format sym --mlp-only    # writes research/gptq_sym_mlp.pt
python scripts/export.py --dtype q8_0 --gptq research/gptq_sym_mlp.pt --shortlist \
  --out models/qwen2.5-0.5b-q4mix-sl.qnt
```

## Use

```sh
quanta chat  -m models/qwen2.5-0.5b-q4mix-sl.qnt
quanta run   -m models/qwen2.5-0.5b-q8_0.qnt -p "The capital of France is" -n 64 --greedy
quanta ppl   -m models/qwen2.5-0.5b-q8_0.qnt -f text.txt -n 1024
quanta bench -m models/qwen2.5-0.5b-q4mix-sl.qnt     # full report: checks plus speed
quanta selftest                                       # kernel checks, no model needed
```

Common flags:
- `-t THREADS`
- `--ctx 4096`
- `--kv-f32`: exact 32-bit KV cache (the default is 16-bit)
- `--no-batch`
- `--full-head`

Sampling flags:
- `--temp 0.7`
- `--top-k 20`
- `--top-p 0.8`
- `--repeat-penalty 1.1`
- `--seed N`
- `--greedy`

## Android app

[`android/`](android/) is a small Java chat app with a benchmark button and no library dependencies.
It needs the Android SDK and NDK 28.2. The build packs `models/qwen2.5-0.5b-q4mix-sl.qnt` into the APK:

```sh
cd android && ./gradlew assembleRelease
```

To test on real phones without owning them, see [`docs/TESTLAB.md`](docs/TESTLAB.md) (Firebase Test Lab).

## Tests

```sh
python scripts/reference.py        # dump Hugging Face reference logits into tests/data/
python tests/test_parity.py        # engine logits vs Hugging Face
python tests/test_tokenizer.py     # tokenizer vs Hugging Face
```

## Layout

| Path | What it holds |
|---|---|
| `engine/src/` | model, kernels (`ops_*`), tokenizer, sampler, thread pool, session, checks |
| `tools/cli/` | the `quanta` command-line tool |
| `scripts/` | model export and reference dumps |
| `research/` | quantization experiments (GPTQ, sensitivity) |
| `android/` | Android app and JNI bridge |
| `bench/llamabench/` | llama.cpp harness for same-protocol comparisons |
| `docs/` | progress log, research notes, Test Lab guide |

## Roadmap

- Pick the thread count and shortlist on/off per device automatically.
- Faster 4-bit kernels (the Q4 path is compute-bound today).
- Faster prefill on flagship phones.
- A same-phone comparison with llama.cpp.

## License

MIT, see [LICENSE](LICENSE). The Qwen2.5 model weights have their own license (Apache 2.0).
