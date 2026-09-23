#include "model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace quanta {

bool Model::load(const std::string& path, int ctx_len, std::string* err, int n_threads, bool kv_f16) {
    if (!file_.open(path, err)) return false;
    const HParams& hp = file_.hparams();
    try {
        embed_ = &file_.get("model.embed_tokens.weight");
        final_norm_ = &file_.get("model.norm.weight");
        lm_head_ = hp.tied_embeddings ? embed_ : &file_.get("lm_head.weight");
        layers_.resize(hp.n_layers);
        for (int l = 0; l < hp.n_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l) + ".";
            LayerWeights& L = layers_[l];
            L.attn_norm = &file_.get(p + "input_layernorm.weight");
            L.wq = &file_.get(p + "self_attn.q_proj.weight");
            L.wk = &file_.get(p + "self_attn.k_proj.weight");
            L.wv = &file_.get(p + "self_attn.v_proj.weight");
            L.wo = &file_.get(p + "self_attn.o_proj.weight");
            L.bq = hp.qkv_bias ? &file_.get(p + "self_attn.q_proj.bias") : nullptr;
            L.bk = hp.qkv_bias ? &file_.get(p + "self_attn.k_proj.bias") : nullptr;
            L.bv = hp.qkv_bias ? &file_.get(p + "self_attn.v_proj.bias") : nullptr;
            L.ffn_norm = &file_.get(p + "post_attention_layernorm.weight");
            L.w_gate = &file_.get(p + "mlp.gate_proj.weight");
            L.w_up = &file_.get(p + "mlp.up_proj.weight");
            L.w_down = &file_.get(p + "mlp.down_proj.weight");
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return false;
    }

    if ((hp.flags & FLAG_HADAMARD_DOWN) && hp.ffn_dim % 256 != 0) {
        if (err) *err = "hadamard flag needs ffn_dim % 256 == 0";
        return false;
    }
    if (hp.n_heads * hp.head_dim != hp.dim || hp.n_heads % hp.n_kv_heads != 0) {
        if (err) *err = "unsupported head configuration";
        return false;
    }
    head_q4_ = file_.find("quanta.head_q4");
    head_err_ = file_.find("quanta.head_err");
    if (head_q4_ || head_err_) {
        const bool ok = head_q4_ && head_err_ && head_q4_->dtype == DType::Q4_0 && head_err_->dtype == DType::F16 &&
                        lm_head_->dtype == DType::Q8_0 && head_q4_->rows == lm_head_->rows &&
                        head_q4_->cols == lm_head_->cols && head_err_->rows == lm_head_->rows &&
                        head_err_->cols == lm_head_->cols / QK;
        if (!ok) {
            if (err) *err = "bad shortlist head tensors";
            return false;
        }
    }
    pool_ = std::make_unique<ThreadPool>(n_threads);
    any_quantized_ = is_quantized(lm_head_->dtype);
    for (const LayerWeights& L : layers_)
        for (const Tensor* t : {L.wq, L.wk, L.wv, L.wo, L.w_gate, L.w_up, L.w_down})
            any_quantized_ |= is_quantized(t->dtype);

    ctx_len_ = (ctx_len > 0 && ctx_len < hp.max_seq) ? ctx_len : hp.max_seq;
    const int q_dim = hp.n_heads * hp.head_dim;
    const int kv_dim = hp.n_kv_heads * hp.head_dim;
    x_.assign(hp.dim, 0.f);
    xb_.assign(hp.dim, 0.f);
    xb2_.assign(hp.dim, 0.f);
    q_.assign(q_dim, 0.f);
    k_.assign(kv_dim, 0.f);
    v_.assign(kv_dim, 0.f);
    att_.assign(size_t(hp.n_heads) * ctx_len_, 0.f);
    hb_.assign(hp.ffn_dim, 0.f);
    hb2_.assign(hp.ffn_dim, 0.f);
    logits_.assign(hp.vocab_size, 0.f);
    bx_.assign(size_t(kBatch) * hp.dim, 0.f);
    bxb_.assign(size_t(kBatch) * hp.dim, 0.f);
    bxb2_.assign(size_t(kBatch) * hp.dim, 0.f);
    bq_.assign(size_t(kBatch) * q_dim, 0.f);
    bk_.assign(size_t(kBatch) * kv_dim, 0.f);
    bv_.assign(size_t(kBatch) * kv_dim, 0.f);
    bhb_.assign(size_t(kBatch) * hp.ffn_dim, 0.f);
    bhb2_.assign(size_t(kBatch) * hp.ffn_dim, 0.f);
    bxq_.resize(kBatch);
    bhq_.resize(kBatch);
    if (head_q4_) {
        coarse_.assign(hp.vocab_size, 0.f);
        upper_.assign(hp.vocab_size, 0.f);
        must_mark_.assign(hp.vocab_size, 0);
        xnorm_.assign(hp.dim / QK, 0.f);
        heap_.resize(pool_->size());
        cand_.resize(pool_->size());
    }
    kv_f16_ = kv_f16;
    const size_t kv_elems = size_t(hp.n_layers) * ctx_len_ * kv_dim;
    if (kv_f16_) {
        k_cache16_.assign(kv_elems, _Float16(0));
        v_cache16_.assign(kv_elems, _Float16(0));
    } else {
        k_cache_.assign(kv_elems, 0.f);
        v_cache_.assign(kv_elems, 0.f);
    }
    return true;
}

size_t Model::kv_cache_bytes() const {
    return k_cache_.size() * 4 + v_cache_.size() * 4 + k_cache16_.size() * 2 + v_cache16_.size() * 2;
}

namespace {

// Attention for query heads [h0, h1) at position `pos` over cached keys/values of type T (float or _Float16).
template <class T>
void attend(int h0, int h1, const float* q, const T* kc, const T* vc, float* att_all, float* out_all, int pos,
            int ctx_len, int kv_dim, int hd, int group, float scale) {
    for (int h = h0; h < h1; ++h) {
        const float* qh = q + h * hd;
        const int kvh = h / group;
        float* att = att_all + size_t(h) * ctx_len;
        for (int t = 0; t <= pos; ++t) {
            const T* kt = kc + size_t(t) * kv_dim + kvh * hd;
            float s = 0.f;
            for (int i = 0; i < hd; ++i) s += qh[i] * float(kt[i]);
            att[t] = s * scale;
        }
        softmax(att, pos + 1);
        float* out = out_all + h * hd;
        std::fill(out, out + hd, 0.f);
        for (int t = 0; t <= pos; ++t) {
            const T* vt = vc + size_t(t) * kv_dim + kvh * hd;
            const float a = att[t];
            for (int i = 0; i < hd; ++i) out[i] += a * float(vt[i]);
        }
    }
}

template <class T>
void store(T* dst, const float* src, int n) {
    for (int i = 0; i < n; ++i) dst[i] = T(src[i]);
}

}  // namespace

void Model::prepare_input(const float* x, int n, QuantVec& xq) {
    if (any_quantized_) quantize_q8(x, n, xq);
}

void Model::linear(float* out, const Tensor& W, const float* x, const QuantVec& xq) {
    // Grain of 16 rows keeps chunks cache-friendly and threads' outputs on separate cache lines.
    if (is_quantized(W.dtype)) {
        pool_->parallel_for(W.rows, 16, [&](int r0, int r1) { matvec_q(out, W, xq, r0, r1); });
    } else {
        pool_->parallel_for(W.rows, 16, [&](int r0, int r1) { matvec_ref(out, W, x, r0, r1); });
    }
}

static void add_bias(float* v, const Tensor* b) {
    if (!b) return;
    const float* bb = b->f32();
    for (int i = 0; i < b->cols; ++i) v[i] += bb[i];
}

const float* Model::forward(int token, int pos, const HeadQuery* hq) {
    const HParams& hp = file_.hparams();
    if (pos < 0 || pos >= ctx_len_) throw std::out_of_range("position exceeds context length");
    const int dim = hp.dim, hd = hp.head_dim;
    const int kv_dim = hp.n_kv_heads * hd;
    const int group = hp.n_heads / hp.n_kv_heads;  // query heads per kv head
    const float att_scale = 1.0f / std::sqrt(float(hd));

    get_row(x_.data(), *embed_, token);

    for (int l = 0; l < hp.n_layers; ++l) {
        const LayerWeights& L = layers_[l];

        // ---- attention
        rmsnorm(xb_.data(), x_.data(), L.attn_norm->f32(), dim, hp.norm_eps);
        prepare_input(xb_.data(), dim, xq_);
        linear(q_.data(), *L.wq, xb_.data(), xq_);
        linear(k_.data(), *L.wk, xb_.data(), xq_);
        linear(v_.data(), *L.wv, xb_.data(), xq_);
        add_bias(q_.data(), L.bq);
        add_bias(k_.data(), L.bk);
        add_bias(v_.data(), L.bv);
        rope(q_.data(), hp.n_heads, hd, pos, hp.rope_theta);
        rope(k_.data(), hp.n_kv_heads, hd, pos, hp.rope_theta);

        const size_t layer_off = size_t(l) * ctx_len_ * kv_dim, pos_off = size_t(pos) * kv_dim;
        // xb_ receives the attention output (q_dim == dim).
        if (kv_f16_) {
            _Float16* kc = k_cache16_.data() + layer_off;
            _Float16* vc = v_cache16_.data() + layer_off;
            store(kc + pos_off, k_.data(), kv_dim);
            store(vc + pos_off, v_.data(), kv_dim);
            pool_->parallel_for(hp.n_heads, 1, [&](int h0, int h1) {
                attend(h0, h1, q_.data(), kc, vc, att_.data(), xb_.data(), pos, ctx_len_, kv_dim, hd, group, att_scale);
            });
        } else {
            float* kc = k_cache_.data() + layer_off;
            float* vc = v_cache_.data() + layer_off;
            store(kc + pos_off, k_.data(), kv_dim);
            store(vc + pos_off, v_.data(), kv_dim);
            pool_->parallel_for(hp.n_heads, 1, [&](int h0, int h1) {
                attend(h0, h1, q_.data(), kc, vc, att_.data(), xb_.data(), pos, ctx_len_, kv_dim, hd, group, att_scale);
            });
        }
        prepare_input(xb_.data(), dim, xq_);
        linear(xb2_.data(), *L.wo, xb_.data(), xq_);
        for (int i = 0; i < dim; ++i) x_[i] += xb2_[i];

        // ---- MLP
        rmsnorm(xb_.data(), x_.data(), L.ffn_norm->f32(), dim, hp.norm_eps);
        prepare_input(xb_.data(), dim, xq_);
        linear(hb_.data(), *L.w_gate, xb_.data(), xq_);
        linear(hb2_.data(), *L.w_up, xb_.data(), xq_);
        swiglu(hb_.data(), hb_.data(), hb2_.data(), hp.ffn_dim);
        if (hp.flags & FLAG_HADAMARD_DOWN) fwht_blocks(hb_.data(), hp.ffn_dim, 256);
        prepare_input(hb_.data(), hp.ffn_dim, hq_);
        linear(xb2_.data(), *L.w_down, hb_.data(), hq_);
        for (int i = 0; i < dim; ++i) x_[i] += xb2_[i];
    }

    rmsnorm(x_.data(), x_.data(), final_norm_->f32(), dim, hp.norm_eps);
    prepare_input(x_.data(), dim, xq_);
    return head(hq);
}

const float* Model::head(const HeadQuery* hq) {
    if (hq && hq->top_k > 0 && head_q4_) {
        shortlist_head(*hq);
    } else {
        linear(logits_.data(), *lm_head_, x_.data(), xq_);
        last_head_rows_ = lm_head_->rows;
    }
    return logits_.data();
}

void Model::shortlist_head(const HeadQuery& hq) {
    // exact - coarse = (w8 - w4) . x = sum over 32-blocks b of (w8_b - w4_b) . x_b, so by Cauchy-Schwarz
    // |exact - coarse| <= sum_b err_b * |x_b|. x is the int8-quantized activation both heads really see.
    // A token whose upper bound is below the k-th largest lower bound can't reach the top-k: skip it.
    const int V = lm_head_->rows, nb = lm_head_->cols / QK;
    const int nv = (hq.n_valid > 0 && hq.n_valid < V) ? hq.n_valid : V;
    const int k = hq.top_k;
    for (int b = 0; b < nb; ++b) {
        float ss = 0.f;
        for (int i = 0; i < QK; ++i) ss += float(xq_.q[b * QK + i]) * float(xq_.q[b * QK + i]);
        xnorm_[b] = xq_.d[b] * std::sqrt(ss);
    }
    for (int i = 0; i < hq.n_must; ++i)
        if (hq.must[i] >= 0 && hq.must[i] < nv) must_mark_[hq.must[i]] = 1;

    const _Float16* err = reinterpret_cast<const _Float16*>(head_err_->data);  // hardware f16 on ARM
    const int n_seed = k + 8;  // best coarse scores get exact values first; they set the cut-off
    using Scored = std::pair<float, int>;
    auto min_heap = [](const Scored& a, const Scored& b) { return a.first > b.first; };
    pool_->run([&](int t, int nt) {
        const int chunks = (V + 15) / 16;
        const int r0 = std::min(V, int(int64_t(chunks) * t / nt) * 16);
        const int r1 = std::min(V, int(int64_t(chunks) * (t + 1) / nt) * 16);
        std::vector<Scored>& heap = heap_[t];
        heap.clear();
        if (r0 >= r1) return;
        matvec_q(coarse_.data(), *head_q4_, xq_, r0, r1);
        for (int r = r0; r < r1; ++r) {
            const _Float16* e = err + size_t(r) * nb;
            float bound = 0.f;
            for (int b = 0; b < nb; ++b) bound += float(e[b]) * xnorm_[b];
            bound = bound * 1.001f + 1e-5f;  // covers float rounding in both dot products
            upper_[r] = coarse_[r] + bound;
            logits_[r] = -INFINITY;
            if (r >= nv || must_mark_[r]) continue;
            if (int(heap.size()) < n_seed) {
                heap.emplace_back(coarse_[r], r);
                std::push_heap(heap.begin(), heap.end(), min_heap);
            } else if (coarse_[r] > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(), min_heap);
                heap.back() = {coarse_[r], r};
                std::push_heap(heap.begin(), heap.end(), min_heap);
            }
        }
    });

    // Same kernel on the same row -> bit-identical to the full head.
    auto exact_rows = [&](const int* rows, int n) {
        pool_->parallel_for(n, 8, [&](int i0, int i1) {
            for (int i = i0; i < i1; ++i) matvec_q(logits_.data(), *lm_head_, xq_, rows[i], rows[i] + 1);
        });
    };

    // Seeds: the n_seed best coarse scores overall, computed exactly. Their k-th best exact logit is a
    // real lower bound on the k-th best logit, so a token whose upper bound is below it can't be in the top-k.
    std::vector<Scored>& seeds = heap_[0];
    for (size_t t = 1; t < heap_.size(); ++t) seeds.insert(seeds.end(), heap_[t].begin(), heap_[t].end());
    auto by_score = [](const Scored& a, const Scored& b) { return a.first > b.first; };
    if (int(seeds.size()) > n_seed) {
        std::nth_element(seeds.begin(), seeds.begin() + n_seed, seeds.end(), by_score);
        seeds.resize(n_seed);
    }
    rows_.clear();
    for (const Scored& sd : seeds) {
        rows_.push_back(sd.second);
        must_mark_[sd.second] = 2;  // "already exact": skipped by the candidate scan below
    }
    exact_rows(rows_.data(), int(rows_.size()));
    float tau = -INFINITY;
    if (int(seeds.size()) >= k) {
        for (Scored& sd : seeds) sd.first = logits_[sd.second];
        std::nth_element(seeds.begin(), seeds.begin() + (k - 1), seeds.end(), by_score);
        tau = seeds[k - 1].first;
    }

    pool_->run([&](int t, int nt) {
        std::vector<int>& c = cand_[t];
        c.clear();
        const int r0 = int(int64_t(nv) * t / nt), r1 = int(int64_t(nv) * (t + 1) / nt);
        for (int r = r0; r < r1; ++r)
            if (!must_mark_[r] && upper_[r] >= tau) c.push_back(r);
    });
    const size_t n_seeded = rows_.size();
    for (const auto& c : cand_) rows_.insert(rows_.end(), c.begin(), c.end());
    for (int i = 0; i < hq.n_must; ++i) {
        const int id = hq.must[i];
        if (id >= 0 && id < nv && must_mark_[id] == 1) {
            rows_.push_back(id);
            must_mark_[id] = 0;  // also drops repeated ids
        }
    }
    for (size_t i = 0; i < n_seeded; ++i) must_mark_[rows_[i]] = 0;
    exact_rows(rows_.data() + n_seeded, int(rows_.size() - n_seeded));
    last_head_rows_ = int(rows_.size());
}

void Model::linear_batch(float* out, const Tensor& W, const float* x, const QuantVec* xq, int n) {
    // Each thread owns a band of rows and runs every token through it, so the band's weights are
    // fetched from RAM once and then re-read from cache n-1 times.
    const int rows = W.rows, cols = W.cols;
    if (is_quantized(W.dtype)) {
        pool_->parallel_for(rows, 16, [&](int r0, int r1) { matmul_q(out, rows, W, xq, n, r0, r1); });
    } else {
        pool_->parallel_for(rows, 16, [&](int r0, int r1) {
            for (int t = 0; t < n; ++t) matvec_ref(out + size_t(t) * rows, W, x + size_t(t) * cols, r0, r1);
        });
    }
}

const float* Model::forward_batch(const int* tokens, int n, int pos0, const HeadQuery* hq) {
    const HParams& hp = file_.hparams();
    if (n <= 0) throw std::invalid_argument("forward_batch needs at least one token");
    if (pos0 < 0 || pos0 + n > ctx_len_) throw std::out_of_range("position exceeds context length");
    for (int i = 0; i < n; i += kBatch) forward_chunk(tokens + i, std::min(kBatch, n - i), pos0 + i);

    // Logits only for the last token: the LM head is the biggest matrix and prefill needs just one row.
    const int last = (n - 1) % kBatch;
    std::copy_n(bx_.data() + size_t(last) * hp.dim, hp.dim, x_.data());
    rmsnorm(x_.data(), x_.data(), final_norm_->f32(), hp.dim, hp.norm_eps);
    prepare_input(x_.data(), hp.dim, xq_);
    return head(hq);
}

void Model::forward_chunk(const int* tokens, int n, int pos0) {
    const HParams& hp = file_.hparams();
    const int dim = hp.dim, hd = hp.head_dim, ffn = hp.ffn_dim;
    const int q_dim = hp.n_heads * hd, kv_dim = hp.n_kv_heads * hd;
    const int group = hp.n_heads / hp.n_kv_heads;
    const float att_scale = 1.0f / std::sqrt(float(hd));
    auto per_token = [&](auto&& fn) { pool_->parallel_for(n, 1, [&](int t0, int t1) { for (int t = t0; t < t1; ++t) fn(t); }); };
    auto row = [](std::vector<float>& v, int t, int width) { return v.data() + size_t(t) * width; };

    per_token([&](int t) { get_row(row(bx_, t, dim), *embed_, tokens[t]); });

    for (int l = 0; l < hp.n_layers; ++l) {
        const LayerWeights& L = layers_[l];

        // ---- attention
        per_token([&](int t) {
            rmsnorm(row(bxb_, t, dim), row(bx_, t, dim), L.attn_norm->f32(), dim, hp.norm_eps);
            prepare_input(row(bxb_, t, dim), dim, bxq_[t]);
        });
        linear_batch(bq_.data(), *L.wq, bxb_.data(), bxq_.data(), n);
        linear_batch(bk_.data(), *L.wk, bxb_.data(), bxq_.data(), n);
        linear_batch(bv_.data(), *L.wv, bxb_.data(), bxq_.data(), n);

        const size_t layer_off = size_t(l) * ctx_len_ * kv_dim;
        auto run_attention = [&](auto* kc, auto* vc) {
            // Store every token's K/V first: token t attends to itself and all earlier tokens of the chunk.
            per_token([&](int t) {
                float *q = row(bq_, t, q_dim), *k = row(bk_, t, kv_dim), *v = row(bv_, t, kv_dim);
                add_bias(q, L.bq);
                add_bias(k, L.bk);
                add_bias(v, L.bv);
                rope(q, hp.n_heads, hd, pos0 + t, hp.rope_theta);
                rope(k, hp.n_kv_heads, hd, pos0 + t, hp.rope_theta);
                const size_t off = size_t(pos0 + t) * kv_dim;
                store(kc + off, k, kv_dim);
                store(vc + off, v, kv_dim);
            });
            pool_->parallel_for(hp.n_heads, 1, [&](int h0, int h1) {
                for (int t = 0; t < n; ++t)
                    attend(h0, h1, row(bq_, t, q_dim), kc, vc, att_.data(), row(bxb_, t, dim), pos0 + t, ctx_len_,
                           kv_dim, hd, group, att_scale);
            });
        };
        if (kv_f16_) run_attention(k_cache16_.data() + layer_off, v_cache16_.data() + layer_off);
        else run_attention(k_cache_.data() + layer_off, v_cache_.data() + layer_off);

        per_token([&](int t) { prepare_input(row(bxb_, t, dim), dim, bxq_[t]); });
        linear_batch(bxb2_.data(), *L.wo, bxb_.data(), bxq_.data(), n);

        // ---- MLP
        per_token([&](int t) {
            float *x = row(bx_, t, dim), *xb = row(bxb_, t, dim), *o = row(bxb2_, t, dim);
            for (int i = 0; i < dim; ++i) x[i] += o[i];
            rmsnorm(xb, x, L.ffn_norm->f32(), dim, hp.norm_eps);
            prepare_input(xb, dim, bxq_[t]);
        });
        linear_batch(bhb_.data(), *L.w_gate, bxb_.data(), bxq_.data(), n);
        linear_batch(bhb2_.data(), *L.w_up, bxb_.data(), bxq_.data(), n);
        per_token([&](int t) {
            float* h = row(bhb_, t, ffn);
            swiglu(h, h, row(bhb2_, t, ffn), ffn);
            if (hp.flags & FLAG_HADAMARD_DOWN) fwht_blocks(h, ffn, 256);
            prepare_input(h, ffn, bhq_[t]);
        });
        linear_batch(bxb2_.data(), *L.w_down, bhb_.data(), bhq_.data(), n);
        per_token([&](int t) {
            float *x = row(bx_, t, dim), *o = row(bxb2_, t, dim);
            for (int i = 0; i < dim; ++i) x[i] += o[i];
        });
    }
}

}  // namespace quanta
