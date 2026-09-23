// llama.cpp side of the engine comparison. Same protocol as quanta::run_compare (engine/src/checks.h):
// prefill the whole prompt in one llama_decode (n_batch = n_ubatch = 512, like llama-bench's pp512),
// then decode kGen tokens greedily ignoring end-of-text; 1 untimed warm-up + kReps timed repetitions.
#include "llama_harness.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <vector>

#include "llama.h"

namespace {

constexpr int kGen = 64;
constexpr int kReps = 2;

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void logf(const LlamaLog& log, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log(buf);
}

int argmax(const float* v, int n) { return int(std::max_element(v, v + n) - v); }

}  // namespace

bool llama_harness_run(const std::string& model_path, const std::string& label, const std::string& prompt,
                       const std::vector<int>& threads, const LlamaLog& log) {
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);  // quiet
    logf(log, "llama.cpp system info: %s", llama_print_system_info());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        logf(log, "llama.cpp: failed to load %s", model_path.c_str());
        return false;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> ids(prompt.size() + 16);
    const int n = llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), ids.data(), int(ids.size()),
                                 /*add_special=*/true, /*parse_special=*/false);
    if (n <= 0) {
        log("llama.cpp: tokenize failed");
        return false;
    }
    ids.resize(size_t(n));

    bool ok = true;
    for (int nt : threads) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 1024;
        cp.n_batch = 512;
        cp.n_ubatch = 512;
        cp.n_threads = nt;
        cp.n_threads_batch = nt;
        cp.no_perf = true;
        llama_context* ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            log("llama.cpp: context failed");
            ok = false;
            break;
        }
        double pre_tps = 0, dec_tps = 0;
        for (int rep = -1; rep < kReps; ++rep) {  // rep -1 = warm-up (untimed)
            llama_memory_clear(llama_get_memory(ctx), true);
            const int n_prompt = rep < 0 ? 32 : int(ids.size());
            const int n_gen = rep < 0 ? 8 : kGen;
            auto t = Clock::now();
            for (int i = 0; i < n_prompt; i += 512) {  // prompts here are <= 512 tokens: one call
                llama_batch b = llama_batch_get_one(ids.data() + i, std::min(512, n_prompt - i));
                if (llama_decode(ctx, b) != 0) {
                    log("llama.cpp: decode failed (prompt)");
                    ok = false;
                }
            }
            const double t_pre = since(t);
            llama_token tok = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            t = Clock::now();
            for (int g = 0; g < n_gen; ++g) {
                llama_batch b = llama_batch_get_one(&tok, 1);
                if (llama_decode(ctx, b) != 0) {
                    log("llama.cpp: decode failed (gen)");
                    ok = false;
                    break;
                }
                tok = argmax(llama_get_logits_ith(ctx, -1), n_vocab);
            }
            const double t_dec = since(t);
            if (rep < 0) continue;
            pre_tps += n_prompt / t_pre / kReps;
            dec_tps += n_gen / t_dec / kReps;
        }
        logf(log,
             "RESULT engine=llama.cpp model=%s threads=%d prefill_tok=%zu prefill_tps=%.2f decode_tok=%d "
             "decode_tps=%.2f",
             label.c_str(), nt, ids.size(), pre_tps, kGen, dec_tps);
        llama_free(ctx);
    }
    llama_model_free(model);
    return ok;
}
