#include "sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>

namespace quanta {

Sampler::Sampler(const SamplerParams& p) : p_(p) {
    state_ = p.seed ? p.seed : uint64_t(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    state_ |= 1;
}

uint64_t Sampler::next_u64() {  // xorshift64*
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1DULL;
}

int Sampler::sample(float* logits, int n_valid, const std::vector<int>& history) {
    if (p_.repeat_penalty != 1.0f && !history.empty()) {
        const size_t from = (p_.repeat_window > 0 && history.size() > size_t(p_.repeat_window))
                                ? history.size() - p_.repeat_window
                                : 0;
        std::unordered_set<int> seen(history.begin() + from, history.end());
        for (int id : seen) {
            if (id < 0 || id >= n_valid) continue;
            float& l = logits[id];
            l = l > 0 ? l / p_.repeat_penalty : l * p_.repeat_penalty;
        }
    }

    if (p_.temperature <= 0.0f) {
        return int(std::max_element(logits, logits + n_valid) - logits);
    }

    cand_.clear();
    cand_.reserve(n_valid);
    for (int i = 0; i < n_valid; ++i) cand_.emplace_back(logits[i], i);
    auto by_logit = [](const auto& a, const auto& b) { return a.first > b.first; };

    size_t k = cand_.size();
    if (p_.top_k > 0 && size_t(p_.top_k) < k) {
        k = size_t(p_.top_k);
        std::nth_element(cand_.begin(), cand_.begin() + k, cand_.end(), by_logit);
        cand_.resize(k);
    }
    std::sort(cand_.begin(), cand_.end(), by_logit);

    // Softmax with temperature over the survivors.
    const float mx = cand_[0].first;
    float sum = 0.f;
    for (auto& c : cand_) {
        c.first = std::exp((c.first - mx) / p_.temperature);
        sum += c.first;
    }
    for (auto& c : cand_) c.first /= sum;

    size_t keep = cand_.size();
    if (p_.min_p > 0.f) {
        const float floor = cand_[0].first * p_.min_p;
        while (keep > 1 && cand_[keep - 1].first < floor) --keep;
    }
    if (p_.top_p < 1.f) {
        float cum = 0.f;
        for (size_t i = 0; i < keep; ++i) {
            cum += cand_[i].first;
            if (cum >= p_.top_p) {
                keep = i + 1;
                break;
            }
        }
    }

    float total = 0.f;
    for (size_t i = 0; i < keep; ++i) total += cand_[i].first;
    float r = next_uniform() * total;
    for (size_t i = 0; i < keep; ++i) {
        r -= cand_[i].first;
        if (r <= 0.f) return cand_[i].second;
    }
    return cand_[keep - 1].second;
}

}  // namespace quanta
