// Picks the fastest integer matvec kernel the CPU supports, once, at startup.
// Override with the environment variable QUANTA_KERNEL=ref|neon|neon_dot (for testing).
#include <cstdlib>
#include <cstring>

#include "ops.h"

#if defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif

namespace quanta {

#if defined(__aarch64__)
void matvec_q_neon_base(float* out, const Tensor& W, const QuantVec& x, int r0, int r1);
void matvec_q_neon_dot(float* out, const Tensor& W, const QuantVec& x, int r0, int r1);
void matmul_q_neon_base(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1);
void matmul_q_neon_dot(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1);
#endif

namespace {

struct Kernel {  // matvec (decode) and matmul (prefill) from the same family, so both give identical sums
    MatvecQFn fn;
    MatmulQFn mm;
    const char* name;
};

Kernel detect() {
    const char* force = std::getenv("QUANTA_KERNEL");
#if defined(__aarch64__)
    const bool has_dot = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    if (force && std::strcmp(force, "ref") == 0) return {matvec_q_ref, matmul_q_ref, "ref"};
    if (force && std::strcmp(force, "neon") == 0) return {matvec_q_neon_base, matmul_q_neon_base, "neon"};
    if (has_dot) return {matvec_q_neon_dot, matmul_q_neon_dot, "neon_dot"};
    return {matvec_q_neon_base, matmul_q_neon_base, "neon"};
#else
    (void)force;
    return {matvec_q_ref, matmul_q_ref, "ref"};
#endif
}

const Kernel& kernel() {
    static const Kernel k = detect();
    return k;
}

}  // namespace

void matvec_q(float* out, const Tensor& W, const QuantVec& x, int r0, int r1) { kernel().fn(out, W, x, r0, r1); }

void matmul_q(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1) {
    kernel().mm(out, ldo, W, x, n, r0, r1);
}

const char* matvec_q_kernel_name() { return kernel().name; }

std::vector<std::pair<const char*, MatvecQFn>> available_matvec_q_kernels() {
    std::vector<std::pair<const char*, MatvecQFn>> ks = {{"ref", matvec_q_ref}};
#if defined(__aarch64__)
    ks.push_back({"neon", matvec_q_neon_base});
    if (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ks.push_back({"neon_dot", matvec_q_neon_dot});
#endif
    return ks;
}

std::vector<std::pair<const char*, MatmulQFn>> available_matmul_q_kernels() {
    std::vector<std::pair<const char*, MatmulQFn>> ks = {{"ref", matmul_q_ref}};
#if defined(__aarch64__)
    ks.push_back({"neon", matmul_q_neon_base});
    if (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ks.push_back({"neon_dot", matmul_q_neon_dot});
#endif
    return ks;
}

}  // namespace quanta
