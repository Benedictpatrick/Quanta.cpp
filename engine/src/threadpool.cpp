#include "threadpool.h"

#include <algorithm>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define QUANTA_PAUSE() _mm_pause()
#elif defined(__aarch64__)
#define QUANTA_PAUSE() __asm__ __volatile__("yield")
#else
#define QUANTA_PAUSE() ((void)0)
#endif

namespace quanta {

namespace {
constexpr auto kSpinBeforeSleep = std::chrono::milliseconds(2);
}

int ThreadPool::default_threads() {
    // Decode is memory-bound: beyond the physical/big cores extra threads mostly add contention.
    const int hw = int(std::thread::hardware_concurrency());
    return std::max(1, std::min(8, hw > 1 ? hw / 2 : 1));
}

ThreadPool::ThreadPool(int n_threads) : n_threads_(n_threads > 0 ? n_threads : default_threads()) {
    for (int i = 1; i < n_threads_; ++i) threads_.emplace_back(&ThreadPool::worker, this, i);
}

ThreadPool::~ThreadPool() {
    stop_.store(true);
    generation_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(mu_);
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
}

void ThreadPool::run(const std::function<void(int, int)>& fn) {
    if (n_threads_ == 1) {
        fn(0, 1);
        return;
    }
    job_ = &fn;
    pending_.store(n_threads_ - 1, std::memory_order_relaxed);
    generation_.fetch_add(1);  // seq_cst: pairs with the sleepers_ check below (Dekker-style)
    if (sleepers_.load() > 0) {
        std::lock_guard<std::mutex> lk(mu_);
        cv_.notify_all();
    }
    fn(0, n_threads_);
    while (pending_.load(std::memory_order_acquire) > 0) QUANTA_PAUSE();
}

void ThreadPool::worker(int index) {
    uint64_t seen = 0;
    while (true) {
        // Spin for a short while waiting for the next job, then fall back to sleeping.
        auto spin_start = std::chrono::steady_clock::now();
        uint64_t gen;
        int spins = 0;
        while ((gen = generation_.load(std::memory_order_acquire)) == seen) {
            QUANTA_PAUSE();
            if (++spins == 1024) {
                spins = 0;
                if (std::chrono::steady_clock::now() - spin_start > kSpinBeforeSleep) {
                    std::unique_lock<std::mutex> lk(mu_);
                    sleepers_.fetch_add(1);
                    cv_.wait(lk, [&] { return generation_.load() != seen; });
                    sleepers_.fetch_sub(1);
                    spin_start = std::chrono::steady_clock::now();
                }
            }
        }
        seen = gen;
        if (stop_.load()) return;
        (*job_)(index, n_threads_);
        pending_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

}  // namespace quanta
