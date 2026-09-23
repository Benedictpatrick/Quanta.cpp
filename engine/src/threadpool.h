// Persistent worker pool tuned for LLM decode: one token issues a few hundred small parallel jobs,
// so workers spin briefly between jobs (cheap wake-up) and only sleep after a longer idle period.
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace quanta {

class ThreadPool {
public:
    // n_threads includes the calling thread; <= 0 picks a default for this machine.
    explicit ThreadPool(int n_threads = 0);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int size() const { return n_threads_; }

    // Calls fn(thread_index, n_threads) on every thread (caller included) and waits for all.
    void run(const std::function<void(int, int)>& fn);

    // Splits [0, n) into contiguous chunks (multiples of `grain`) and calls fn(begin, end) in parallel.
    template <class F>
    void parallel_for(int n, int grain, F&& fn) {
        if (n_threads_ == 1 || n <= grain) {
            fn(0, n);
            return;
        }
        run([&](int t, int nt) {
            const int chunks = (n + grain - 1) / grain;
            const int c0 = int(int64_t(chunks) * t / nt), c1 = int(int64_t(chunks) * (t + 1) / nt);
            const int b = c0 * grain, e = std::min(n, c1 * grain);
            if (b < e) fn(b, e);
        });
    }

    static int default_threads();

private:
    void worker(int index);

    int n_threads_;
    std::vector<std::thread> threads_;
    const std::function<void(int, int)>* job_ = nullptr;
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    std::atomic<int> sleepers_{0};
    std::mutex mu_;
    std::condition_variable cv_;
};

}  // namespace quanta
