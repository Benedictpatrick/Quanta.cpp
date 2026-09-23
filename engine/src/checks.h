// Self-checks and benchmarks, shared by the CLI (`quanta selftest / checkbatch / checkhead / bench`) and the
// Android app's benchmark mode (Firebase Test Lab). Every check prints PASS/FAIL lines through `log`.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "model.h"
#include "tokenizer.h"

namespace quanta {

using Log = std::function<void(const std::string&)>;

// CPU features and which kernels were picked.
void log_cpu_info(const Log& log);

// Every integer kernel vs the scalar reference (matvec), and multi-token kernels vs matvec (bit-exact). Times each.
bool kernel_selftest(const Log& log);

// Batched prefill must give bit-identical logits to token-by-token decode (incl. a second turn at pos > 0).
// quick = fewer cases (for phones on a time limit).
bool check_batch(Model& model, const std::vector<int>& ids, const Log& log, bool quick = false);

// Shortlist head vs full head over ids: bound holds everywhere; top-k and `must` logits are bit-identical.
bool check_head(Model& model, const Tokenizer& tok, const std::vector<int>& ids, const Log& log);

struct BenchOptions {
    double max_seconds = 150;          // stop starting new sections after this (Test Lab has a time limit)
    std::vector<int> thread_counts;    // extra decode-speed runs with these thread counts (0 = default)
};
// Full on-device report: CPU info, kernel checks, exactness checks, prefill/decode speed (full vs shortlist head).
bool run_benchmark(const std::string& model_path, const BenchOptions& opt, const Log& log);

// A few hundred tokens of plain English used by the checks and benchmark.
const char* sample_text();

// ---- Engine comparison (Quanta vs llama.cpp vs LiteRT), one protocol for every engine:
// prompt = compare_prompt() (~500 tokens, each engine counts with its own tokenizer), prefill it in one go,
// then decode kCompareGen tokens greedily ignoring end-of-text; one untimed warm-up, then kCompareReps timed
// repetitions per thread count. Output lines: "RESULT engine=... threads=N prefill_tok=... prefill_tps=...
// decode_tok=... decode_tps=..." so all engines' reports parse the same way.
constexpr int kCompareGen = 64;
constexpr int kCompareReps = 2;
std::string compare_prompt();
bool run_compare(const std::string& model_path, const std::string& label, const std::vector<int>& threads,
                 const Log& log);

}  // namespace quanta
