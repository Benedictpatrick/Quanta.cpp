// Core numeric ops used by the transformer forward pass.
// ops_ref.cpp holds plain scalar implementations: they are the ground truth that every
// optimized kernel (NEON, AVX2) is tested against.
#pragma once

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "model_file.h"

namespace quanta {

inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: normalize
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                --exp;
            }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// out[r] = dot(W[r, :], x) for r in [r0, r1). x has W.cols elements. Float math for every dtype.
void matvec_ref(float* out, const Tensor& W, const float* x, int r0, int r1);

// Activation vector quantized to int8 in blocks of 32 (one float scale per block), which is what
// the integer kernels consume: dot(Wq, xq) = sum_b dW_b * dx_b * sum_i qW_i * qx_i.
struct QuantVec {
    std::vector<int8_t> q;
    std::vector<float> d;
    int n = 0;
};
void quantize_q8(const float* x, int n, QuantVec& out);

inline bool is_quantized(DType t) { return t == DType::Q8_0 || t == DType::Q4_0; }

// Integer-dot matvec for quantized weights (Q8_0 / Q4_0) against a quantized activation.
using MatvecQFn = void (*)(float* out, const Tensor& W, const QuantVec& x, int r0, int r1);
void matvec_q_ref(float* out, const Tensor& W, const QuantVec& x, int r0, int r1);

// Fastest kernel for this CPU (chosen once at startup; see ops_dispatch.cpp).
void matvec_q(float* out, const Tensor& W, const QuantVec& x, int r0, int r1);
const char* matvec_q_kernel_name();
std::vector<std::pair<const char*, MatvecQFn>> available_matvec_q_kernels();  // for self-tests

// Prefill: the same product for n tokens at once, out[t * ldo + r] for t in [0, n), r in [r0, r1).
// Each weight block is loaded/unpacked once and used for several tokens. Results are bit-identical to
// calling the matching matvec kernel per token (same per-(row, token) summation order).
using MatmulQFn = void (*)(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1);
void matmul_q_ref(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1);
void matmul_q(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1);
std::vector<std::pair<const char*, MatmulQFn>> available_matmul_q_kernels();  // same order as matvec list

// Copies (dequantizing if needed) row `row` of W into out.
void get_row(float* out, const Tensor& W, int row);

void rmsnorm(float* out, const float* x, const float* w, int n, float eps);

// Rotary embedding, "rotate_half" (NeoX) convention used by Qwen/Llama HF checkpoints:
// pairs are (i, i + head_dim/2) within each head.
void rope(float* v, int n_heads, int head_dim, int pos, float theta);

void softmax(float* x, int n);

// In-place orthonormal Walsh-Hadamard transform on each consecutive block of `block` (power of 2) values.
void fwht_blocks(float* x, int n, int block);

// out[i] = silu(gate[i]) * up[i]
void swiglu(float* out, const float* gate, const float* up, int n);

}  // namespace quanta
