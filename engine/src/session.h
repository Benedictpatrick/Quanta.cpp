// Conversation state on top of Model: feeds prompts into the KV cache and streams generations.
// Shared by the CLI and the Android app.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

namespace quanta {

class Session {
public:
    Session(Model& m, const Tokenizer& t) : model_(m), tok_(t) {}

    int pos() const { return int(history_.size()); }
    void reset() { history_.clear(); }
    void set_batch_prefill(bool on) { batch_prefill_ = on; }
    // Use the Exact Shortlist Head (if the file has one) for tokens sampled with these params.
    void set_sampler(const SamplerParams* sp) { sp_ = sp; }

    // Feeds tokens into the KV cache. Returns false if they don't fit in the context.
    bool feed(const std::vector<int>& ids, double* seconds);

    // Samples up to max_tokens, calling on_token for each one. Stops at a stop token (which is NOT fed
    // into the cache) or when on_token returns false. Returns the number of sampled tokens.
    int generate(Sampler& sampler, int max_tokens, const std::vector<int>& stops,
                 const std::function<bool(int)>& on_token, double* seconds);

    // Average share of the 8-bit output head read per token so far (1.0 without the shortlist head).
    double head_fraction() const;

private:
    void count_head();
    // Which logits the sampler can actually use: its top-k, plus the tokens its repetition penalty lowers.
    const HeadQuery* head_query();

    Model& model_;
    const Tokenizer& tok_;
    std::vector<int> history_;
    const float* logits_ = nullptr;
    std::vector<float> scratch_;
    bool batch_prefill_ = true;
    const SamplerParams* sp_ = nullptr;
    HeadQuery hq_;
    std::vector<int> must_;
    long head_rows_ = 0, head_calls_ = 0;
};

// Multi-turn chat in Qwen's ChatML format, keeping the KV cache across turns.
class Chat {
public:
    Chat(Model& m, const Tokenizer& t, std::string system, const SamplerParams& sp, bool batch_prefill = true,
         bool shortlist_head = true);

    struct Stats {
        int prompt_tokens = 0, reply_tokens = 0;
        double prefill_seconds = 0, decode_seconds = 0;
        int ctx_used = 0;
    };
    // Generates the assistant's reply; on_piece gets decoded text bytes as they come (return false to stop).
    // Returns false if the message doesn't fit in the context even after clearing the conversation.
    bool reply(const std::string& user, int max_tokens, const std::function<bool(const std::string&)>& on_piece,
               Stats* stats = nullptr);
    void reset();

private:
    const Tokenizer& tok_;
    std::string system_;
    SamplerParams sp_;
    Sampler sampler_;
    Session session_;
    std::vector<int> stops_;
    bool open_turn_ = false;  // the last reply's <|im_end|> is not in the cache yet
};

}  // namespace quanta
