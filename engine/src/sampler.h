// Token sampling: repetition penalty, temperature, top-k, top-p, min-p. temperature <= 0 means greedy.
#pragma once

#include <cstdint>
#include <vector>

namespace quanta {

struct SamplerParams {
    // Defaults follow Qwen2.5-Instruct's generation_config.json.
    float temperature = 0.7f;
    int top_k = 20;             // 0 = disabled
    float top_p = 0.8f;         // 1 = disabled
    float min_p = 0.0f;         // 0 = disabled
    float repeat_penalty = 1.1f;  // 1 = disabled; HF semantics (divide positive logits, multiply negative)
    int repeat_window = 0;      // how many recent tokens the penalty looks at (0 = all, like HF)
    uint64_t seed = 0;          // 0 = random
};

class Sampler {
public:
    explicit Sampler(const SamplerParams& p);

    // `logits` is modified in place. Only ids < n_valid can be sampled (the model's output
    // may have padding rows beyond the real vocabulary).
    int sample(float* logits, int n_valid, const std::vector<int>& history);

    const SamplerParams& params() const { return p_; }

private:
    uint64_t next_u64();
    float next_uniform() { return (next_u64() >> 40) * (1.0f / 16777216.0f); }

    SamplerParams p_;
    uint64_t state_;
    std::vector<std::pair<float, int>> cand_;
};

}  // namespace quanta
