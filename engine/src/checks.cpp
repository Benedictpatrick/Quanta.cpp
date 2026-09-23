#include "checks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <thread>

#include "ops.h"
#include "session.h"
#include "threadpool.h"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace quanta {

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void logf(const Log& log, const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    log(buf);
}

}  // namespace

const char* sample_text() {
    return "The lighthouse stood at the edge of the island for more than a hundred years. Every night its keeper "
           "climbed the narrow stairs, cleaned the great lens, and lit the lamp so that ships could find their way "
           "through the rocks. In winter the storms were so strong that the whole tower seemed to shake, and the "
           "keeper wrote in his logbook by candlelight while the wind screamed outside. He recorded the weather, the "
           "ships that passed, and small things he noticed: a seal resting on the beach, the first flowers of spring, "
           "a letter from his sister in the city. When electric lamps arrived, engineers came to install a machine "
           "that could run the light without anyone watching it. The keeper showed them every corner of the tower, "
           "explained how the old clockwork turned the lens, and asked many questions about the new equipment. On "
           "his last evening he climbed the stairs one more time, not to light the lamp, but to watch the sun go "
           "down over the water. Years later, visitors to the island museum can still read his logbook. The "
           "handwriting is neat and patient, and the entries show how a quiet job can become a way of paying "
           "attention to the world. Students often ask why he stayed so long. The answer, written on the final "
           "page, is simple: someone had to keep the light, and he liked knowing that strangers were safe because "
           "of it.";
}

void log_cpu_info(const Log& log) {
    logf(log, "cpu: %u hardware threads, default threads %d", std::thread::hardware_concurrency(),
         ThreadPool::default_threads());
#if defined(__aarch64__) && defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    logf(log, "cpu features: asimddp(sdot)=%d fphp=%d asimdhp=%d i8mm=%d sve=%d", int((hw >> 20) & 1),
         int((hw >> 9) & 1), int((hw >> 10) & 1), int((hw2 >> 13) & 1), int((hw >> 22) & 1));
#endif
    logf(log, "kernel: %s", matvec_q_kernel_name());
}

bool kernel_selftest(const Log& log) {
    uint64_t rng = 0x9E3779B97F4A7C15ULL;
    auto next = [&]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };
    bool ok = true;
    for (DType dt : {DType::Q8_0, DType::Q4_0}) {
        const int rows = 4864, cols = 896;  // one Qwen2.5-0.5B MLP matrix
        const size_t qbytes = dt == DType::Q8_0 ? size_t(rows) * cols : size_t(rows) * cols / 2;
        std::vector<uint8_t> buf(qbytes + size_t(rows) * cols / QK * 2);
        for (size_t i = 0; i < qbytes; ++i) buf[i] = uint8_t(next());
        uint16_t* scales = reinterpret_cast<uint16_t*>(buf.data() + qbytes);
        for (int i = 0; i < rows * cols / QK; ++i) scales[i] = uint16_t(0x2000 + next() % 0x800);  // ~0.001-0.004
        Tensor W;
        W.dtype = dt;
        W.rows = rows;
        W.cols = cols;
        W.data = buf.data();

        std::vector<float> x(cols);
        for (float& v : x) v = float(int(next() % 2001) - 1000) / 1000.f;
        QuantVec xq;
        quantize_q8(x.data(), cols, xq);

        std::vector<float> want(rows), got(rows);
        matvec_q_ref(want.data(), W, xq, 0, rows);
        for (const auto& [name, fn] : available_matvec_q_kernels()) {
            fn(got.data(), W, xq, 0, rows);
            // Error relative to the output scale: SIMD kernels sum in a different order (and may fuse
            // multiply-adds), so a row whose result happens to be near 0 can't have a tiny per-row relative error.
            double max_abs = 0, scale = 0;
            for (int r = 0; r < rows; ++r) {
                max_abs = std::max(max_abs, std::fabs(double(got[r]) - want[r]));
                scale = std::max(scale, std::fabs(double(want[r])));
            }
            const double max_rel = max_abs / scale;
            const int iters = 50;
            auto t = Clock::now();
            for (int i = 0; i < iters; ++i) fn(got.data(), W, xq, 0, rows);
            const double sec = seconds_since(t) / iters;
            const bool pass = max_rel < 1e-4;  // real kernel bugs give errors near 1
            ok &= pass;
            logf(log, "%-5s %-9s matvec max err %.2e (of output scale) %s  %7.1f us  %6.2f GB/s (1 thread)", dtype_name(dt), name,
                 max_rel, pass ? "PASS" : "FAIL", sec * 1e6, double(buf.size()) / sec / 1e9);
        }

        // Prefill kernels: n tokens at once must equal the same family's matvec per token, bit for bit.
        const int n = 32;
        std::vector<QuantVec> xs(n);
        for (auto& v : xs) {
            for (float& f : x) f = float(int(next() % 2001) - 1000) / 1000.f;
            quantize_q8(x.data(), cols, v);
        }
        const auto mvs = available_matvec_q_kernels();
        const auto mms = available_matmul_q_kernels();
        std::vector<float> want_mm(size_t(n) * rows), got_mm(size_t(n) * rows);
        for (size_t k = 0; k < mms.size(); ++k) {
            for (int t = 0; t < n; ++t) mvs[k].second(want_mm.data() + size_t(t) * rows, W, xs[t], 0, rows);
            bool exact = true;
            for (int n_try : {n, 7}) {  // also a ragged token count (tile of 4 + remainder)
                std::fill(got_mm.begin(), got_mm.end(), 0.f);
                mms[k].second(got_mm.data(), rows, W, xs.data(), n_try, 0, rows);
                exact &= std::equal(got_mm.begin(), got_mm.begin() + size_t(n_try) * rows, want_mm.begin());
            }
            ok &= exact;
            const int iters = 5;
            auto t = Clock::now();
            for (int i = 0; i < iters; ++i) mms[k].second(got_mm.data(), rows, W, xs.data(), n, 0, rows);
            const double mm_sec = seconds_since(t) / iters;
            t = Clock::now();
            for (int i = 0; i < iters; ++i)
                for (int tk = 0; tk < n; ++tk) mvs[k].second(got_mm.data() + size_t(tk) * rows, W, xs[tk], 0, rows);
            const double mv_sec = seconds_since(t) / iters;
            logf(log, "%-5s %-9s matmul x%d: bit-exact %s  %6.2f ms vs %6.2f ms looped matvec (%.2fx, 1 thread)",
                 dtype_name(dt), mms[k].first, n, exact ? "PASS" : "FAIL", mm_sec * 1e3, mv_sec * 1e3,
                 mv_sec / mm_sec);
        }
    }
    return ok;
}

bool check_batch(Model& model, const std::vector<int>& ids, const Log& log, bool quick) {
    const int V = model.hparams().vocab_size;
    bool ok = true;
    auto check = [&](int pos0, int n) {
        if (pos0 + n > int(ids.size()) || pos0 + n > model.ctx_len()) return;
        std::vector<float> want;
        for (int p = 0; p < pos0 + n; ++p) {
            const float* lg = model.forward(ids[p], p);
            if (p == pos0 + n - 1) want.assign(lg, lg + V);
        }
        // Rebuild the cache the batched way: first [0, pos0) (as an earlier chat turn), then [pos0, pos0+n).
        if (pos0 > 0) model.forward_batch(ids.data(), pos0, 0);
        const float* got = model.forward_batch(ids.data() + pos0, n, pos0);
        float maxd = 0.f;
        for (int i = 0; i < V; ++i) maxd = std::max(maxd, std::fabs(got[i] - want[i]));
        logf(log, "batch pos0=%4d n=%4d  max |diff| %.3g  %s", pos0, n, maxd, maxd == 0.f ? "PASS" : "FAIL");
        ok &= maxd == 0.f;
    };
    if (quick) {
        check(0, 1);
        check(0, 33);  // one full chunk of 32 + 1
        check(40, 45);
        return ok;
    }
    for (int n : {1, 31, 32, 33, 100}) check(0, n);
    check(40, 45);
    if (ids.size() > 100) check(0, int(ids.size()));
    return ok;
}

bool check_head(Model& model, const Tokenizer& tok, const std::vector<int>& ids, const Log& log) {
    if (!model.has_shortlist_head()) {
        log("this file has no shortlist head (export with --shortlist)");
        return false;
    }
    const int V = model.hparams().vocab_size, nv = std::min(tok.n_tokens(), V);
    const int n = std::min(int(ids.size()), model.ctx_len());
    std::vector<float> full;
    long bad = 0, violations = 0;
    std::vector<int> order(nv);
    for (int k : {1, 20}) {
        double t_full = 0, t_sl = 0, rows = 0, sum_bound = 0, sum_diff = 0;
        std::vector<int> row_counts;
        for (int p = 0; p < n; ++p) {
            model.forward(ids[p], p);
            auto t = Clock::now();
            const float* lg = model.head(nullptr);  // time the full head alone
            t_full += seconds_since(t);
            full.assign(lg, lg + V);
            // Every penalized token (last 64 ids) must also come back exact.
            std::vector<int> must(ids.begin() + std::max(0, p - 63), ids.begin() + p + 1);
            HeadQuery hq;
            hq.top_k = k;
            hq.n_valid = nv;
            hq.must = must.data();
            hq.n_must = int(must.size());
            t = Clock::now();
            const float* sl = model.head(&hq);
            t_sl += seconds_since(t);
            row_counts.push_back(model.last_head_rows());
            rows += model.last_head_rows();
            // The bound must hold for every token.
            for (int v = 0; v < V; ++v) {
                const float bnd = model.head_upper()[v] - model.head_coarse()[v];
                const float dif = std::fabs(full[v] - model.head_coarse()[v]);
                if (dif > bnd) ++violations;
                sum_bound += bnd;
                sum_diff += dif;
            }
            // Exact rows must be bit-identical; the true top-k (and must ids) must be among them.
            for (int v = 0; v < nv; ++v) order[v] = v;
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int x, int y) { return full[x] > full[y]; });
            for (int v = 0; v < nv; ++v)
                if (sl[v] != -INFINITY && sl[v] != full[v]) ++bad;
            for (int i = 0; i < k; ++i)
                if (sl[order[i]] != full[order[i]]) ++bad;
            for (int id : must)
                if (id < nv && sl[id] != full[id]) ++bad;
        }
        std::sort(row_counts.begin(), row_counts.end());
        logf(log,
             "head top_k=%2d over %d tokens: exact rows mean %.0f median %d p99 %d of %d (%.2f%%) | "
             "head time full %.2f ms, shortlist %.2f ms",
             k, n, rows / n, row_counts[n / 2], row_counts[std::min(n - 1, n * 99 / 100)], V, 100.0 * rows / n / V,
             1e3 * t_full / n, 1e3 * t_sl / n);
        logf(log, "  mean bound %.3f vs mean actual |exact - coarse| %.3f", sum_bound / n / V, sum_diff / n / V);
    }
    logf(log, "head bound violations: %ld   mismatches: %ld   %s", violations, bad,
         violations == 0 && bad == 0 ? "PASS" : "FAIL");
    return violations == 0 && bad == 0;
}

std::string compare_prompt() {
    return std::string(sample_text()) + "\n\n" + sample_text() + "\n\nWrite a new chapter of this story:";
}

bool run_compare(const std::string& model_path, const std::string& label, const std::vector<int>& threads,
                 const Log& log) {
    const std::string prompt = compare_prompt();
    for (int nt : threads) {
        Model m;
        std::string err;
        if (!m.load(model_path, 2048, &err, nt)) {
            logf(log, "load failed: %s", err.c_str());
            return false;
        }
        Tokenizer tok(m.file().tokenizer());
        const std::vector<int> ids = tok.encode(prompt);
        // With a shortlist head, report both head modes (greedy = top-1 query, bit-identical output).
        for (int sl = 0; sl < (m.has_shortlist_head() ? 2 : 1); ++sl) {
            SamplerParams greedy;
            greedy.temperature = 0.f;
            greedy.repeat_penalty = 1.f;
            double pre_tps = 0, dec_tps = 0;
            for (int rep = -1; rep < kCompareReps; ++rep) {  // rep -1 = warm-up (untimed)
                Session s(m, tok);
                if (sl) s.set_sampler(&greedy);
                Sampler sampler(greedy);
                double t_pre = 0, t_dec = 0;
                const std::vector<int> feed_ids = rep < 0 ? std::vector<int>(ids.begin(), ids.begin() + 32) : ids;
                s.feed(feed_ids, &t_pre);
                const int n = s.generate(sampler, rep < 0 ? 8 : kCompareGen, {}, [](int) { return true; }, &t_dec);
                if (rep < 0) continue;
                pre_tps += feed_ids.size() / t_pre / kCompareReps;
                dec_tps += n / t_dec / kCompareReps;
            }
            logf(log, "RESULT engine=quanta model=%s%s threads=%d prefill_tok=%zu prefill_tps=%.2f decode_tok=%d "
                      "decode_tps=%.2f",
                 label.c_str(), sl ? "+shortlist" : "", m.n_threads(), ids.size(), pre_tps, kCompareGen, dec_tps);
        }
    }
    return true;
}

bool run_benchmark(const std::string& model_path, const BenchOptions& opt, const Log& log) {
    const auto t_start = Clock::now();
    auto time_left = [&](const char* section) {
        if (seconds_since(t_start) < opt.max_seconds) return true;
        logf(log, "(skipped %s: time budget of %.0f s used up)", section, opt.max_seconds);
        return false;
    };
    bool ok = true;
    log("== quanta benchmark");
    log_cpu_info(log);

    log("== kernels");
    ok &= kernel_selftest(log);

    log("== load");
    Model model;
    std::string err;
    auto t = Clock::now();
    if (!model.load(model_path, 2048, &err)) {
        logf(log, "load failed: %s", err.c_str());
        return false;
    }
    Tokenizer tok(model.file().tokenizer());
    logf(log, "loaded %s in %.2f s, %d threads, shortlist head: %s", model_path.c_str(), seconds_since(t),
         model.n_threads(), model.has_shortlist_head() ? "yes" : "no");

    std::string text = sample_text();
    const std::vector<int> ids = tok.encode(text);  // ~250 tokens
    std::vector<int> long_ids;                      // ~500 tokens for prefill speed
    while (long_ids.size() < 500) long_ids.insert(long_ids.end(), ids.begin(), ids.end());
    long_ids.resize(500);

    const std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                               "Write a short story about a lighthouse keeper.<|im_end|>\n<|im_start|>assistant\n";
    const std::vector<int> pids = tok.encode(prompt);

    // Chat-style sampling (Qwen defaults, fixed seed): the full head and the shortlist head must produce the
    // same text; only speed may differ. Runs are ordered A-B-B-A so phone heat-up doesn't favour either one.
    auto decode_speed = [&](Model& m) {
        const bool has_sl = m.has_shortlist_head();
        const std::vector<int> order = has_sl ? std::vector<int>{0, 1, 1, 0} : std::vector<int>{0};
        double tok_s[2] = {0, 0}, head[2] = {1, 1};
        int runs[2] = {0, 0};
        std::vector<int> outs[2];
        for (int sl : order) {
            SamplerParams sp;  // Qwen chat defaults
            sp.seed = 1234;
            Sampler sampler(sp);
            Session s(m, tok);
            if (sl) s.set_sampler(&sp);
            double t_pre = 0, t_dec = 0;
            s.feed(pids, &t_pre);
            std::vector<int> out;
            const int n = s.generate(sampler, 64, {}, [&](int id) { out.push_back(id); return true; }, &t_dec);
            tok_s[sl] += n / t_dec;
            head[sl] = s.head_fraction();
            ++runs[sl];
            if (outs[sl].empty()) outs[sl] = out;
            else ok &= outs[sl] == out;
        }
        logf(log, "decode full head:      %6.2f tok/s", tok_s[0] / runs[0]);
        if (has_sl) {
            logf(log, "decode shortlist head: %6.2f tok/s  (8-bit head read %.1f%%)  -> %.2fx", tok_s[1] / runs[1],
                 100 * head[1], (tok_s[1] / runs[1]) / (tok_s[0] / runs[0]));
            const bool same = outs[0] == outs[1];
            ok &= same;
            logf(log, "decode output identical (full vs shortlist): %s", same ? "PASS" : "FAIL");
        }
        return outs[0];
    };

    if (time_left("prefill speed")) {
        log("== prefill speed");
        Session s(model, tok);
        double sec = 0;
        s.feed(long_ids, &sec);
        logf(log, "prefill batched        %3zu tok: %6.1f tok/s", long_ids.size(), long_ids.size() / sec);
        s.reset();
        s.set_batch_prefill(false);
        const std::vector<int> short_ids(long_ids.begin(), long_ids.begin() + 64);
        s.feed(short_ids, &sec);
        logf(log, "prefill token-by-token  %2zu tok: %6.1f tok/s", short_ids.size(), short_ids.size() / sec);
    }
    if (time_left("decode speed")) {
        logf(log, "== decode speed, %d threads (default)", model.n_threads());
        std::string sample;
        for (int id : decode_speed(model)) sample += tok.decode(id);
        logf(log, "sample output: %.300s", sample.c_str());
    }
    if (time_left("batch check")) {
        log("== exactness: batched prefill");
        ok &= check_batch(model, std::vector<int>(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 100)), log,
                          true);
    }
    if (model.has_shortlist_head() && time_left("head check")) {
        log("== exactness: shortlist head");
        ok &= check_head(model, tok, std::vector<int>(ids.begin(), ids.begin() + std::min<size_t>(ids.size(), 24)), log);
    }
    for (int nt : opt.thread_counts) {
        if (nt <= 0 || nt == model.n_threads()) continue;
        if (!time_left("thread sweep")) break;
        Model m;
        if (!m.load(model_path, 2048, &err, nt)) {
            logf(log, "load failed: %s", err.c_str());
            return false;
        }
        logf(log, "== decode speed, %d threads", m.n_threads());
        decode_speed(m);
    }
    logf(log, "== done in %.1f s: %s", seconds_since(t_start), ok ? "ALL PASS" : "SOME CHECKS FAILED");
    return ok;
}

}  // namespace quanta
