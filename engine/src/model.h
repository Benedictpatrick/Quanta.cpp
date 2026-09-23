// Qwen2-style decoder-only transformer: RMSNorm, GQA attention with RoPE + QKV bias,
// SwiGLU MLP, tied embeddings. Processes one token per forward() call (decode path).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model_file.h"
#include "ops.h"
#include "threadpool.h"

namespace quanta {

struct LayerWeights {
    const Tensor *attn_norm, *wq, *wk, *wv, *wo, *bq, *bk, *bv;
    const Tensor *ffn_norm, *w_gate, *w_up, *w_down;
};

// Which logits the caller needs from forward(). Default (nullptr / top_k <= 0): all of them.
// With top_k > 0 and a file that has the Exact Shortlist Head, only these are computed exactly:
// the top_k largest logits among ids [0, n_valid) and every id in `must`. All others come back as
// -infinity. The exact values are bit-identical to the full head, so top-k sampling is unchanged.
struct HeadQuery {
    int top_k = 0;
    int n_valid = 0;            // ids >= n_valid are padding and ignored (0 = whole vocab)
    const int* must = nullptr;  // e.g. tokens a repetition penalty will lower (they may fall out of the top-k)
    int n_must = 0;
};

class Model {
public:
    // ctx_len <= 0 uses the max sequence length stored in the file; n_threads <= 0 picks a default.
    // kv_f16 stores the attention KV cache in 16-bit floats (half the memory; tiny accuracy cost).
    bool load(const std::string& path, int ctx_len, std::string* err, int n_threads = 0, bool kv_f16 = true);

    // Runs one token at position `pos` (0-based) and returns logits[vocab_size].
    // Positions must be fed in order; the KV cache holds entries [0, pos].
    const float* forward(int token, int pos, const HeadQuery* hq = nullptr);

    // Prefill: runs n tokens at positions pos0 .. pos0+n-1 in one pass and returns the logits of the
    // last one. Each weight matrix is read from memory once per chunk of kBatch tokens instead of once
    // per token, which is what makes prompt processing fast on bandwidth-limited phones.
    const float* forward_batch(const int* tokens, int n, int pos0, const HeadQuery* hq = nullptr);

    // Recomputes the output head for the last forward() token (for tests / comparing head modes).
    const float* head(const HeadQuery* hq);
    bool has_shortlist_head() const { return head_q4_ != nullptr; }
    int last_head_rows() const { return last_head_rows_; }  // 8-bit head rows read by the last head()
    const float* head_coarse() const { return coarse_.data(); }  // last shortlist pass: 4-bit scores
    const float* head_upper() const { return upper_.data(); }    // ... and their upper bounds
    static constexpr int kBatch = 32;

    const HParams& hparams() const { return file_.hparams(); }
    const ModelFile& file() const { return file_; }
    int ctx_len() const { return ctx_len_; }
    int n_threads() const { return pool_ ? pool_->size() : 1; }
    size_t kv_cache_bytes() const;

private:
    // out = W x. For quantized W, xq must hold x quantized with quantize_q8 (shared across
    // the matrices that read the same input, e.g. q/k/v).
    void linear(float* out, const Tensor& W, const float* x, const QuantVec& xq);
    void prepare_input(const float* x, int n, QuantVec& xq);  // quantizes only if some consumer needs it
    // Batched linear: x is [n][W.cols], out is [n][W.rows]; xq[t] is token t's quantized input.
    void linear_batch(float* out, const Tensor& W, const float* x, const QuantVec* xq, int n);
    // Chunk of at most kBatch tokens.
    void forward_chunk(const int* tokens, int n, int pos0);
    void shortlist_head(const HeadQuery& hq);  // logits_ from x_ / xq_ via the 4-bit copy + error bound

    const Tensor* head_q4_ = nullptr;   // 4-bit copy of the 8-bit head
    const Tensor* head_err_ = nullptr;  // [vocab][dim/32] f16 upper bound of ||w8_block - w4_block||
    std::vector<float> coarse_, upper_, xnorm_;
    std::vector<uint8_t> must_mark_;
    std::vector<std::vector<std::pair<float, int>>> heap_;  // per-thread best coarse scores (seeds)
    std::vector<std::vector<int>> cand_;    // per-thread candidate rows
    std::vector<int> rows_;
    int last_head_rows_ = 0;

    std::unique_ptr<ThreadPool> pool_;
    bool any_quantized_ = false;
    QuantVec xq_, hq_;

    ModelFile file_;
    std::vector<LayerWeights> layers_;
    const Tensor* embed_ = nullptr;
    const Tensor* final_norm_ = nullptr;
    const Tensor* lm_head_ = nullptr;
    int ctx_len_ = 0;

    // Activations (single token).
    std::vector<float> x_, xb_, xb2_, q_, k_, v_, att_, hb_, hb2_, logits_;
    // Activations for a prefill chunk: [kBatch][width].
    std::vector<float> bx_, bxb_, bxb2_, bq_, bk_, bv_, bhb_, bhb2_;
    std::vector<QuantVec> bxq_, bhq_;
    // KV cache: [layer][pos][kv_dim], either f32 or f16 (only one pair is allocated).
    bool kv_f16_ = true;
    std::vector<float> k_cache_, v_cache_;
    std::vector<_Float16> k_cache16_, v_cache16_;
};

}  // namespace quanta
