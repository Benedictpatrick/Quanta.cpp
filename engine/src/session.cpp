#include "session.h"

#include <algorithm>
#include <chrono>

namespace quanta {

namespace {
using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
}  // namespace

bool Session::feed(const std::vector<int>& ids, double* seconds) {
    if (pos() + int(ids.size()) >= model_.ctx_len()) return false;
    auto t0 = Clock::now();
    if (ids.empty()) return true;
    // history_ is updated before each forward so the head query's penalized tokens include the new ones.
    if (batch_prefill_) {
        const int pos0 = pos();
        history_.insert(history_.end(), ids.begin(), ids.end());
        logits_ = model_.forward_batch(ids.data(), int(ids.size()), pos0, head_query());
    } else {
        for (int id : ids) {
            history_.push_back(id);
            logits_ = model_.forward(id, pos() - 1, head_query());
        }
    }
    count_head();
    if (seconds) *seconds = seconds_since(t0);
    return true;
}

int Session::generate(Sampler& sampler, int max_tokens, const std::vector<int>& stops,
                      const std::function<bool(int)>& on_token, double* seconds) {
    const int n_valid = std::min(tok_.n_tokens(), model_.hparams().vocab_size);
    auto t0 = Clock::now();
    int n = 0;
    scratch_.assign(logits_, logits_ + model_.hparams().vocab_size);
    while (n < max_tokens && pos() < model_.ctx_len() - 1) {
        const int id = sampler.sample(scratch_.data(), n_valid, history_);
        ++n;
        if (std::find(stops.begin(), stops.end(), id) != stops.end()) break;
        const bool more = on_token(id);
        history_.push_back(id);
        logits_ = model_.forward(id, pos() - 1, head_query());
        count_head();
        scratch_.assign(logits_, logits_ + model_.hparams().vocab_size);
        if (!more) break;
    }
    if (seconds) *seconds = seconds_since(t0);
    return n;
}

double Session::head_fraction() const {
    return head_calls_ ? head_rows_ / (double(head_calls_) * model_.hparams().vocab_size) : 1.0;
}

void Session::count_head() {
    head_rows_ += model_.last_head_rows();
    ++head_calls_;
}

const HeadQuery* Session::head_query() {
    if (!sp_ || !model_.has_shortlist_head()) return nullptr;
    const int k = sp_->temperature <= 0.f ? 1 : sp_->top_k;
    if (k <= 0) return nullptr;
    must_.clear();
    if (sp_->repeat_penalty != 1.f) {
        const size_t from = (sp_->repeat_window > 0 && history_.size() > size_t(sp_->repeat_window))
                                ? history_.size() - sp_->repeat_window
                                : 0;
        must_.assign(history_.begin() + from, history_.end());
    }
    hq_.top_k = k;
    hq_.n_valid = std::min(tok_.n_tokens(), model_.hparams().vocab_size);
    hq_.must = must_.data();
    hq_.n_must = int(must_.size());
    return &hq_;
}

Chat::Chat(Model& m, const Tokenizer& t, std::string system, const SamplerParams& sp, bool batch_prefill,
           bool shortlist_head)
    : tok_(t), system_(std::move(system)), sp_(sp), sampler_(sp), session_(m, t) {
    session_.set_batch_prefill(batch_prefill);
    if (shortlist_head) session_.set_sampler(&sp_);
    stops_ = {tok_.token_id("<|im_end|>"), tok_.token_id("<|endoftext|>")};
}

void Chat::reset() {
    session_.reset();
    open_turn_ = false;
}

bool Chat::reply(const std::string& user, int max_tokens, const std::function<bool(const std::string&)>& on_piece,
                 Stats* stats) {
    std::string turn;
    if (session_.pos() == 0) turn = "<|im_start|>system\n" + system_ + "<|im_end|>\n";
    if (open_turn_) turn += "<|im_end|>\n";
    turn += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = tok_.encode(turn);
    Stats st;
    if (!session_.feed(ids, &st.prefill_seconds)) {  // context full: start over with just this message
        reset();
        ids = tok_.encode("<|im_start|>system\n" + system_ + "<|im_end|>\n<|im_start|>user\n" + user +
                          "<|im_end|>\n<|im_start|>assistant\n");
        if (!session_.feed(ids, &st.prefill_seconds)) {
            reset();
            return false;
        }
    }
    st.prompt_tokens = int(ids.size());
    st.reply_tokens = session_.generate(
        sampler_, max_tokens, stops_, [&](int id) { return on_piece(tok_.decode(id)); }, &st.decode_seconds);
    open_turn_ = true;
    st.ctx_used = session_.pos();
    if (stats) *stats = st;
    return true;
}

}  // namespace quanta
