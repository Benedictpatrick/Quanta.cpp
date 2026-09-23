// quanta — command-line driver for the Quanta engine.
//
//   quanta chat     -m model.qnt                          interactive chat (ChatML)
//   quanta run      -m model.qnt -p "text" [-n 128]       complete a prompt, streaming
//   quanta tokenize -m model.qnt -f text.txt              print token ids
//   quanta logits   -m model.qnt --ids 1,2,3 -o out.bin   dump float32 logits for every position
//   quanta gen      -m model.qnt --ids 1,2,3 -n 64        greedy-generate token ids
//   quanta ppl      -m model.qnt -f text.txt -n 1024      perplexity
//
// Sampling flags (run/chat): --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0 --repeat-penalty 1.1 --seed N --greedy
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "model.h"
#include "ops.h"
#include "sampler.h"
#include "tokenizer.h"
#include "checks.h"
#include "session.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

using Clock = std::chrono::steady_clock;
using quanta::Session;

namespace {

const char* kDefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

struct Args {
    std::string cmd, model, out, ids, prompt, file, system = kDefaultSystem;
    int n_gen = 256;
    int ctx = 4096;
    int threads = 0;
    bool kv_f16 = true;
    bool batch = true;
    bool full_head = false;
    quanta::SamplerParams sp;
};

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  quanta chat     -m model.qnt [--system TEXT] [sampling flags]\n"
                 "  quanta run      -m model.qnt (-p TEXT | -f FILE) [-n 256] [sampling flags]\n"
                 "  quanta tokenize -m model.qnt (-p TEXT | -f FILE)\n"
                 "  quanta logits   -m model.qnt --ids 1,2,3 -o out.bin\n"
                 "  quanta gen      -m model.qnt --ids 1,2,3 [-n 64]\n"
                 "  quanta ppl      -m model.qnt -f FILE [-n 1024]         perplexity of the first n tokens\n"
                 "  quanta checkbatch -m model.qnt (-p TEXT | -f FILE)     batched prefill must equal token-by-token\n"
                 "  quanta checkhead  -m model.qnt (-p TEXT | -f FILE)     shortlist head must equal the full head\n"
                 "  quanta compare   -m model.qnt [-t N]                   engine-comparison protocol (threads 2,4)\n"
                 "  quanta prompt                                          print the comparison prompt\n"
                 "  quanta bench     -m model.qnt                          full on-device report (checks + speed)\n"
                 "  quanta selftest                                        check + time kernels (no model needed)\n"
                 "common: --ctx 4096  -t THREADS  --kv-f32 (exact 32-bit KV cache; default is 16-bit)\n"
                 "        --no-batch (prefill token by token)  --full-head (skip the shortlist head)\n"
                 "sampling: --temp 0.7 --top-k 20 --top-p 0.8 --min-p 0 --repeat-penalty 1.1 --seed N --greedy\n");
    std::exit(2);
}

Args parse(int argc, char** argv) {
    if (argc < 2) usage();
    Args a;
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (k == "-m") a.model = next();
        else if (k == "-o") a.out = next();
        else if (k == "--ids") a.ids = next();
        else if (k == "-p") a.prompt = next();
        else if (k == "-f") a.file = next();
        else if (k == "--system") a.system = next();
        else if (k == "-n") a.n_gen = std::atoi(next().c_str());
        else if (k == "--ctx") a.ctx = std::atoi(next().c_str());
        else if (k == "-t") a.threads = std::atoi(next().c_str());
        else if (k == "--kv-f32") a.kv_f16 = false;
        else if (k == "--no-batch") a.batch = false;
        else if (k == "--full-head") a.full_head = true;
        else if (k == "--temp") a.sp.temperature = float(std::atof(next().c_str()));
        else if (k == "--top-k") a.sp.top_k = std::atoi(next().c_str());
        else if (k == "--top-p") a.sp.top_p = float(std::atof(next().c_str()));
        else if (k == "--min-p") a.sp.min_p = float(std::atof(next().c_str()));
        else if (k == "--repeat-penalty") a.sp.repeat_penalty = float(std::atof(next().c_str()));
        else if (k == "--seed") a.sp.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (k == "--greedy") {
            a.sp.temperature = 0.f;
            a.sp.repeat_penalty = 1.f;
        } else usage();
    }
    if (a.model.empty()) usage();
    return a;
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> ids;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) ids.push_back(std::atoi(tok.c_str()));
    return ids;
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Reads one line of UTF-8 from stdin (uses the wide console API on Windows so non-ASCII input works).
bool read_line(std::string& line) {
#ifdef _WIN32
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(in, &mode)) {
        std::wstring w;
        wchar_t buf[1024];
        DWORD n = 0;
        while (true) {
            if (!ReadConsoleW(in, buf, 1024, &n, nullptr) || n == 0) return false;
            w.append(buf, n);
            if (!w.empty() && w.back() == L'\n') break;
        }
        while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
        int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
        line.assign(size_t(len), '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), line.data(), len, nullptr, nullptr);
        return true;
    }
#endif
    if (!std::getline(std::cin, line)) return false;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return true;
}

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

void print_piece(const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    const quanta::Log print = [](const std::string& line) { std::printf("%s\n", line.c_str()); };
    if (argc >= 2 && std::strcmp(argv[1], "prompt") == 0) {  // the comparison prompt, for other engines' harnesses
        std::fputs(quanta::compare_prompt().c_str(), stdout);
        return 0;
    }
    if (argc >= 2 && std::strcmp(argv[1], "selftest") == 0) {
        quanta::log_cpu_info(print);
        return quanta::kernel_selftest(print) ? 0 : 1;
    }
    Args a = parse(argc, argv);
    if (a.cmd == "compare") {  // engine-comparison protocol (see checks.h); prompt text: `quanta prompt`
        const std::vector<int> threads = a.threads > 0 ? std::vector<int>{a.threads} : std::vector<int>{2, 4};
        return quanta::run_compare(a.model, a.model, threads, print) ? 0 : 1;
    }
    if (a.cmd == "bench") {
        quanta::BenchOptions opt;
        opt.max_seconds = 1e9;
        if (a.threads > 0) opt.thread_counts = {a.threads};
        return quanta::run_benchmark(a.model, opt, print) ? 0 : 1;
    }

    quanta::Model model;
    std::string err;
    auto t0 = Clock::now();
    if (!model.load(a.model, a.ctx, &err, a.threads, a.kv_f16)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const quanta::HParams& hp = model.hparams();
    quanta::Tokenizer tok(model.file().tokenizer());
    std::fprintf(stderr, "quanta: loaded %s in %.2fs (dim=%d layers=%d heads=%d/%d vocab=%d ctx=%d threads=%d)\n",
                 a.model.c_str(), seconds_since(t0), hp.dim, hp.n_layers, hp.n_heads, hp.n_kv_heads, hp.vocab_size,
                 model.ctx_len(), model.n_threads());

    std::string text = a.file.empty() ? a.prompt : read_file(a.file);
    const std::vector<int> stops = {tok.token_id("<|im_end|>"), tok.token_id("<|endoftext|>")};

    if (a.cmd == "tokenize") {
        const std::vector<int> ids = tok.encode(text);
        for (size_t i = 0; i < ids.size(); ++i) std::printf(i ? ",%d" : "%d", ids[i]);
        std::printf("\n");
        return 0;
    }

    if (a.cmd == "logits" || a.cmd == "gen") {
        const std::vector<int> ids = a.ids.empty() ? tok.encode(text) : parse_ids(a.ids);
        if (ids.empty()) {
            std::fprintf(stderr, "no input\n");
            return 1;
        }
        if (a.cmd == "logits") {
            FILE* f = std::fopen(a.out.c_str(), "wb");
            if (!f) {
                std::fprintf(stderr, "cannot write %s\n", a.out.c_str());
                return 1;
            }
            for (size_t p = 0; p < ids.size(); ++p)
                std::fwrite(model.forward(ids[p], int(p)), sizeof(float), hp.vocab_size, f);
            std::fclose(f);
            return 0;
        }
        Session s(model, tok);
        s.set_batch_prefill(a.batch);
        if (!a.full_head) s.set_sampler(&a.sp);
        quanta::SamplerParams greedy;
        greedy.temperature = 0.f;
        greedy.repeat_penalty = 1.f;
        quanta::Sampler sampler(greedy);
        double t_prefill = 0, t_decode = 0;
        if (!s.feed(ids, &t_prefill)) {
            std::fprintf(stderr, "prompt too long for context\n");
            return 1;
        }
        std::vector<int> out;
        s.generate(sampler, a.n_gen, {}, [&](int id) { out.push_back(id); return true; }, &t_decode);
        for (size_t i = 0; i < out.size(); ++i) std::printf(i ? ",%d" : "%d", out[i]);
        std::printf("\n");
        std::fprintf(stderr, "prefill: %zu tok %.2f tok/s | decode: %zu tok %.2f tok/s\n", ids.size(),
                     ids.size() / t_prefill, out.size(), out.size() / t_decode);
        return 0;
    }

    if (a.cmd == "checkbatch") return quanta::check_batch(model, tok.encode(text), print) ? 0 : 1;
    if (a.cmd == "checkhead") return quanta::check_head(model, tok, tok.encode(text), print) ? 0 : 1;

    if (a.cmd == "ppl") {
        // Perplexity of the first -n tokens of the text (one context window, scored from token 1 on).
        std::vector<int> ids = tok.encode(text);
        if (int(ids.size()) > a.n_gen) ids.resize(a.n_gen);
        if (int(ids.size()) >= model.ctx_len()) ids.resize(model.ctx_len() - 1);
        const int V = std::min(tok.n_tokens(), hp.vocab_size);
        double nll = 0;
        t0 = Clock::now();
        for (size_t p = 0; p + 1 < ids.size(); ++p) {
            const float* lg = model.forward(ids[p], int(p));
            float mx = lg[0];
            for (int i = 1; i < V; ++i) mx = std::max(mx, lg[i]);
            double sum = 0;
            for (int i = 0; i < V; ++i) sum += std::exp(double(lg[i] - mx));
            nll += (std::log(sum) + mx) - lg[ids[p + 1]];
            if ((p + 1) % 256 == 0)
                std::fprintf(stderr, "  %zu/%zu  ppl so far %.4f\n", p + 1, ids.size() - 1, std::exp(nll / double(p + 1)));
        }
        const double n = double(ids.size() - 1);
        std::printf("ppl %.4f over %d tokens (%.1f tok/s)\n", std::exp(nll / n), int(n), n / seconds_since(t0));
        return 0;
    }

    quanta::Sampler sampler(a.sp);

    if (a.cmd == "run") {
        if (text.empty()) usage();
        Session s(model, tok);
        s.set_batch_prefill(a.batch);
        if (!a.full_head) s.set_sampler(&a.sp);
        double t_prefill = 0, t_decode = 0;
        const std::vector<int> ids = tok.encode(text);
        if (!s.feed(ids, &t_prefill)) {
            std::fprintf(stderr, "prompt too long for context\n");
            return 1;
        }
        print_piece(text);
        const int n = s.generate(
            sampler, a.n_gen, stops, [&](int id) { print_piece(tok.decode(id)); return true; }, &t_decode);
        std::fprintf(stderr, "\n[prefill %zu tok @ %.1f tok/s | decode %d tok @ %.1f tok/s | 8-bit head read %.1f%%]\n",
                     ids.size(), ids.size() / t_prefill, n, n / t_decode, 100 * s.head_fraction());
        return 0;
    }

    if (a.cmd == "chat") {
        quanta::Chat chat(model, tok, a.system, a.sp, a.batch, !a.full_head);
        std::fprintf(stderr, "Quanta chat. Commands: /reset, /exit\n");
        std::string line;
        while (true) {
            std::fprintf(stderr, "\n> ");
            if (!read_line(line) || line == "/exit") break;
            if (line.empty()) continue;
            if (line == "/reset") {
                chat.reset();
                std::fprintf(stderr, "(conversation cleared)\n");
                continue;
            }
            quanta::Chat::Stats st;
            if (!chat.reply(line, a.n_gen, [](const std::string& piece) { print_piece(piece); return true; }, &st)) {
                std::fprintf(stderr, "(message too long)\n");
                continue;
            }
            std::fprintf(stderr, "\n[prefill %d tok @ %.1f tok/s | %d tok @ %.1f tok/s | ctx %d/%d]\n",
                         st.prompt_tokens, st.prompt_tokens / st.prefill_seconds, st.reply_tokens,
                         st.reply_tokens / st.decode_seconds, st.ctx_used, model.ctx_len());
        }
        return 0;
    }

    usage();
}
