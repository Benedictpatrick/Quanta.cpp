// ARM NEON integer matvec kernels. Included twice by ops_neon_dot.cpp / ops_neon_base.cpp:
// with QUANTA_NEON_DOT = 1 it uses the ARMv8.2 `sdot` instruction (Cortex-A55/A76 and newer);
// otherwise widening multiplies that work on every ARMv8 core (e.g. Cortex-A53).
// Must produce the same result as matvec_q_ref (integer sums are exact; only float rounding differs).
#include <arm_neon.h>

#include <algorithm>

#include "ops.h"

namespace quanta {

namespace {

// Sum of 16 int8 x int8 products into an int32x4 accumulator.
inline int32x4_t dot16(int32x4_t acc, int8x16_t a, int8x16_t b) {
#if QUANTA_NEON_DOT
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vpadalq_s16(vpadalq_s16(acc, lo), hi);
#endif
}

inline float half(uint16_t h) {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

}  // namespace

void QUANTA_NEON_FN(float* out, const Tensor& W, const QuantVec& x, int r0, int r1) {
    const int n = W.cols, nb = n / QK;
    const uint16_t* dw = W.scales();
    const int8_t* qx = x.q.data();
    const float* dx = x.d.data();

    if (W.dtype == DType::Q8_0) {
        const int8_t* q = reinterpret_cast<const int8_t*>(W.data);
        for (int r = r0; r < r1; ++r) {
            const int8_t* qr = q + size_t(r) * n;
            const uint16_t* dr = dw + size_t(r) * nb;
            float32x4_t accf = vdupq_n_f32(0.f);
            for (int b = 0; b < nb; ++b) {
                int32x4_t s = vdupq_n_s32(0);
                s = dot16(s, vld1q_s8(qr + b * QK), vld1q_s8(qx + b * QK));
                s = dot16(s, vld1q_s8(qr + b * QK + 16), vld1q_s8(qx + b * QK + 16));
                accf = vmlaq_n_f32(accf, vcvtq_f32_s32(s), half(dr[b]) * dx[b]);
            }
            out[r] = vaddvq_f32(accf);
        }
        return;
    }

    // Q4_0: 16 bytes per block; low nibbles are elements 0..15, high nibbles 16..31; value = q - 8.
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    for (int r = r0; r < r1; ++r) {
        const uint8_t* qr = W.data + size_t(r) * n / 2;
        const uint16_t* dr = dw + size_t(r) * nb;
        float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
        int b = 0;
        for (; b + 1 < nb; b += 2) {  // two blocks per iteration for instruction-level parallelism
            const uint8x16_t v0 = vld1q_u8(qr + b * 16), v1 = vld1q_u8(qr + b * 16 + 16);
            const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, mask)), eight);
            const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), eight);
            const int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, mask)), eight);
            const int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), eight);
            const int8_t* x0 = qx + b * QK;
            int32x4_t s0 = dot16(dot16(vdupq_n_s32(0), l0, vld1q_s8(x0)), h0, vld1q_s8(x0 + 16));
            int32x4_t s1 = dot16(dot16(vdupq_n_s32(0), l1, vld1q_s8(x0 + 32)), h1, vld1q_s8(x0 + 48));
            acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(s0), half(dr[b]) * dx[b]);
            acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(s1), half(dr[b + 1]) * dx[b + 1]);
        }
        for (; b < nb; ++b) {
            const uint8x16_t v = vld1q_u8(qr + b * 16);
            const int8x16_t l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
            const int8x16_t h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
            int32x4_t s = dot16(dot16(vdupq_n_s32(0), l, vld1q_s8(qx + b * QK)), h, vld1q_s8(qx + b * QK + 16));
            acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(s), half(dr[b]) * dx[b]);
        }
        out[r] = vaddvq_f32(vaddq_f32(acc0, acc1));
    }
}

// Multi-token version (prefill). Tiles of 1 row x 4 tokens: each weight block is loaded (and for Q4,
// unpacked) once per tile instead of once per token. Per (row, token) it does exactly the matvec steps
// above in the same order, so results are bit-identical to QUANTA_NEON_FN.
void QUANTA_NEON_MM_FN(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1) {
    const int cols = W.cols, nb = cols / QK;
    const uint16_t* dw = W.scales();
    constexpr int T = 4;
    for (int t0 = 0; t0 < n; t0 += T) {
        const int nt = std::min(T, n - t0);
        const int8_t* qx[T];
        const float* dx[T];
        for (int t = 0; t < T; ++t) {  // pad a short tile by repeating the last token (results discarded)
            const QuantVec& xv = x[t0 + std::min(t, nt - 1)];
            qx[t] = xv.q.data();
            dx[t] = xv.d.data();
        }
        if (W.dtype == DType::Q8_0) {
            const int8_t* q = reinterpret_cast<const int8_t*>(W.data);
            for (int r = r0; r < r1; ++r) {
                const int8_t* qr = q + size_t(r) * cols;
                const uint16_t* dr = dw + size_t(r) * nb;
                float32x4_t acc[T];
                for (int t = 0; t < T; ++t) acc[t] = vdupq_n_f32(0.f);
                for (int b = 0; b < nb; ++b) {
                    const int8x16_t w0 = vld1q_s8(qr + b * QK), w1 = vld1q_s8(qr + b * QK + 16);
                    const float d = half(dr[b]);
                    for (int t = 0; t < T; ++t) {
                        int32x4_t s = vdupq_n_s32(0);
                        s = dot16(s, w0, vld1q_s8(qx[t] + b * QK));
                        s = dot16(s, w1, vld1q_s8(qx[t] + b * QK + 16));
                        acc[t] = vmlaq_n_f32(acc[t], vcvtq_f32_s32(s), d * dx[t][b]);
                    }
                }
                for (int t = 0; t < nt; ++t) out[size_t(t0 + t) * ldo + r] = vaddvq_f32(acc[t]);
            }
            continue;
        }
        const uint8x16_t mask = vdupq_n_u8(0x0F);
        const int8x16_t eight = vdupq_n_s8(8);
        for (int r = r0; r < r1; ++r) {
            const uint8_t* qr = W.data + size_t(r) * cols / 2;
            const uint16_t* dr = dw + size_t(r) * nb;
            float32x4_t acc0[T], acc1[T];
            for (int t = 0; t < T; ++t) acc0[t] = acc1[t] = vdupq_n_f32(0.f);
            int b = 0;
            for (; b + 1 < nb; b += 2) {  // even blocks -> acc0, odd -> acc1, as in the matvec kernel
                const uint8x16_t v0 = vld1q_u8(qr + b * 16), v1 = vld1q_u8(qr + b * 16 + 16);
                const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, mask)), eight);
                const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), eight);
                const int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, mask)), eight);
                const int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), eight);
                const float d0 = half(dr[b]), d1 = half(dr[b + 1]);
                for (int t = 0; t < T; ++t) {
                    const int8_t* x0 = qx[t] + b * QK;
                    const int32x4_t s0 = dot16(dot16(vdupq_n_s32(0), l0, vld1q_s8(x0)), h0, vld1q_s8(x0 + 16));
                    const int32x4_t s1 = dot16(dot16(vdupq_n_s32(0), l1, vld1q_s8(x0 + 32)), h1, vld1q_s8(x0 + 48));
                    acc0[t] = vmlaq_n_f32(acc0[t], vcvtq_f32_s32(s0), d0 * dx[t][b]);
                    acc1[t] = vmlaq_n_f32(acc1[t], vcvtq_f32_s32(s1), d1 * dx[t][b + 1]);
                }
            }
            for (; b < nb; ++b) {
                const uint8x16_t v = vld1q_u8(qr + b * 16);
                const int8x16_t l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
                const int8x16_t h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
                const float d = half(dr[b]);
                for (int t = 0; t < T; ++t) {
                    const int32x4_t s = dot16(dot16(vdupq_n_s32(0), l, vld1q_s8(qx[t] + b * QK)), h,
                                              vld1q_s8(qx[t] + b * QK + 16));
                    acc0[t] = vmlaq_n_f32(acc0[t], vcvtq_f32_s32(s), d * dx[t][b]);
                }
            }
            for (int t = 0; t < nt; ++t) out[size_t(t0 + t) * ldo + r] = vaddvq_f32(vaddq_f32(acc0[t], acc1[t]));
        }
    }
}

}  // namespace quanta
