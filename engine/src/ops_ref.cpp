#include <algorithm>
#include <cmath>

#include "ops.h"

namespace quanta {

void matvec_ref(float* out, const Tensor& W, const float* x, int r0, int r1) {
    const int n = W.cols;
    switch (W.dtype) {
        case DType::F32:
            for (int r = r0; r < r1; ++r) {
                const float* w = W.f32() + size_t(r) * n;
                float acc = 0.f;
                for (int i = 0; i < n; ++i) acc += w[i] * x[i];
                out[r] = acc;
            }
            break;
        case DType::F16: {
            const uint16_t* base = reinterpret_cast<const uint16_t*>(W.data);
            for (int r = r0; r < r1; ++r) {
                const uint16_t* w = base + size_t(r) * n;
                float acc = 0.f;
                for (int i = 0; i < n; ++i) acc += fp16_to_f32(w[i]) * x[i];
                out[r] = acc;
            }
            break;
        }
        case DType::Q8_0: {
            const int8_t* q = reinterpret_cast<const int8_t*>(W.data);
            const uint16_t* d = W.scales();
            const int nb = n / QK;
            for (int r = r0; r < r1; ++r) {
                float acc = 0.f;
                for (int b = 0; b < nb; ++b) {
                    const int8_t* qb = q + (size_t(r) * n + size_t(b) * QK);
                    const float* xb = x + b * QK;
                    float s = 0.f;
                    for (int i = 0; i < QK; ++i) s += qb[i] * xb[i];
                    acc += s * fp16_to_f32(d[size_t(r) * nb + b]);
                }
                out[r] = acc;
            }
            break;
        }
        case DType::Q4_0: {
            const uint8_t* q = W.data;
            const uint16_t* d = W.scales();
            const int nb = n / QK;
            for (int r = r0; r < r1; ++r) {
                float acc = 0.f;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* qb = q + (size_t(r) * n + size_t(b) * QK) / 2;
                    const float* xb = x + b * QK;
                    float s = 0.f;
                    for (int j = 0; j < QK / 2; ++j) {
                        s += (int(qb[j] & 0xF) - 8) * xb[j];
                        s += (int(qb[j] >> 4) - 8) * xb[j + QK / 2];
                    }
                    acc += s * fp16_to_f32(d[size_t(r) * nb + b]);
                }
                out[r] = acc;
            }
            break;
        }
    }
}

void quantize_q8(const float* x, int n, QuantVec& out) {
    const int nb = n / QK;
    out.n = n;
    out.q.resize(n);
    out.d.resize(nb);
    for (int b = 0; b < nb; ++b) {
        const float* xb = x + b * QK;
        float amax = 0.f;
        for (int i = 0; i < QK; ++i) amax = std::max(amax, std::fabs(xb[i]));
        const float d = amax / 127.f;
        const float inv = d > 0 ? 1.f / d : 0.f;
        for (int i = 0; i < QK; ++i) out.q[b * QK + i] = int8_t(std::lround(xb[i] * inv));
        out.d[b] = d;
    }
}

void matvec_q_ref(float* out, const Tensor& W, const QuantVec& x, int r0, int r1) {
    const int n = W.cols, nb = n / QK;
    const uint16_t* dw = W.scales();
    const int8_t* qx = x.q.data();
    if (W.dtype == DType::Q8_0) {
        const int8_t* q = reinterpret_cast<const int8_t*>(W.data);
        for (int r = r0; r < r1; ++r) {
            const int8_t* qr = q + size_t(r) * n;
            const uint16_t* dr = dw + size_t(r) * nb;
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) {
                int32_t s = 0;
                for (int i = 0; i < QK; ++i) s += int32_t(qr[b * QK + i]) * qx[b * QK + i];
                acc += float(s) * fp16_to_f32(dr[b]) * x.d[b];
            }
            out[r] = acc;
        }
    } else {  // Q4_0
        for (int r = r0; r < r1; ++r) {
            const uint8_t* qr = W.data + size_t(r) * n / 2;
            const uint16_t* dr = dw + size_t(r) * nb;
            float acc = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* qb = qr + b * QK / 2;
                const int8_t* xb = qx + b * QK;
                int32_t s = 0;
                for (int j = 0; j < QK / 2; ++j) {
                    s += (int32_t(qb[j] & 0xF) - 8) * xb[j];
                    s += (int32_t(qb[j] >> 4) - 8) * xb[j + QK / 2];
                }
                acc += float(s) * fp16_to_f32(dr[b]) * x.d[b];
            }
            out[r] = acc;
        }
    }
}

void matmul_q_ref(float* out, int ldo, const Tensor& W, const QuantVec* x, int n, int r0, int r1) {
    const int cols = W.cols, nb = cols / QK;
    const uint16_t* dw = W.scales();
    constexpr int T = 4;  // tokens per tile
    int8_t wb[QK];
    for (int r = r0; r < r1; ++r) {
        const uint16_t* dr = dw + size_t(r) * nb;
        for (int t0 = 0; t0 < n; t0 += T) {
            const int nt = std::min(T, n - t0);
            float acc[T] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb; ++b) {
                if (W.dtype == DType::Q8_0) {
                    std::memcpy(wb, reinterpret_cast<const int8_t*>(W.data) + size_t(r) * cols + size_t(b) * QK, QK);
                } else {  // Q4_0 unpacked once for all tokens of the tile
                    const uint8_t* qb = W.data + (size_t(r) * cols + size_t(b) * QK) / 2;
                    for (int j = 0; j < QK / 2; ++j) {
                        wb[j] = int8_t(int(qb[j] & 0xF) - 8);
                        wb[j + QK / 2] = int8_t(int(qb[j] >> 4) - 8);
                    }
                }
                const float d = fp16_to_f32(dr[b]);
                for (int t = 0; t < nt; ++t) {
                    const int8_t* xb = x[t0 + t].q.data() + b * QK;
                    int32_t s = 0;
                    if (W.dtype == DType::Q8_0) {
                        for (int i = 0; i < QK; ++i) s += int32_t(wb[i]) * xb[i];
                    } else {  // same pairing order as matvec_q_ref
                        for (int j = 0; j < QK / 2; ++j) {
                            s += int32_t(wb[j]) * xb[j];
                            s += int32_t(wb[j + QK / 2]) * xb[j + QK / 2];
                        }
                    }
                    acc[t] += float(s) * d * x[t0 + t].d[b];
                }
            }
            for (int t = 0; t < nt; ++t) out[size_t(t0 + t) * ldo + r] = acc[t];
        }
    }
}

void get_row(float* out, const Tensor& W, int row) {
    const int n = W.cols;
    switch (W.dtype) {
        case DType::F32:
            std::memcpy(out, W.f32() + size_t(row) * n, sizeof(float) * n);
            break;
        case DType::F16: {
            const uint16_t* w = reinterpret_cast<const uint16_t*>(W.data) + size_t(row) * n;
            for (int i = 0; i < n; ++i) out[i] = fp16_to_f32(w[i]);
            break;
        }
        case DType::Q8_0: {
            const int8_t* q = reinterpret_cast<const int8_t*>(W.data) + size_t(row) * n;
            const uint16_t* d = W.scales() + size_t(row) * (n / QK);
            for (int i = 0; i < n; ++i) out[i] = q[i] * fp16_to_f32(d[i / QK]);
            break;
        }
        case DType::Q4_0: {
            const uint8_t* q = W.data + size_t(row) * n / 2;
            const uint16_t* d = W.scales() + size_t(row) * (n / QK);
            for (int b = 0; b < n / QK; ++b) {
                const float s = fp16_to_f32(d[b]);
                for (int j = 0; j < QK / 2; ++j) {
                    out[b * QK + j] = (int(q[b * QK / 2 + j] & 0xF) - 8) * s;
                    out[b * QK + j + QK / 2] = (int(q[b * QK / 2 + j] >> 4) - 8) * s;
                }
            }
            break;
        }
    }
}

void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
    float ss = 0.f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

void rope(float* v, int n_heads, int head_dim, int pos, float theta) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        const float freq = std::pow(theta, -2.0f * i / head_dim);
        const float angle = pos * freq;
        const float c = std::cos(angle), s = std::sin(angle);
        for (int h = 0; h < n_heads; ++h) {
            float* p = v + h * head_dim;
            const float a = p[i], b = p[i + half];
            p[i] = a * c - b * s;
            p[i + half] = b * c + a * s;
        }
    }
}

void softmax(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; ++i) mx = x[i] > mx ? x[i] : mx;
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

void fwht_blocks(float* x, int n, int block) {
    const float norm = 1.0f / std::sqrt(float(block));
    for (int b0 = 0; b0 < n; b0 += block) {
        float* v = x + b0;
        for (int h = 1; h < block; h *= 2)
            for (int i = 0; i < block; i += 2 * h)
                for (int j = i; j < i + h; ++j) {
                    const float a = v[j], c = v[j + h];
                    v[j] = a + c;
                    v[j + h] = a - c;
                }
        for (int i = 0; i < block; ++i) v[i] *= norm;
    }
}

void swiglu(float* out, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; ++i) {
        const float g = gate[i];
        out[i] = g / (1.0f + std::exp(-g)) * up[i];
    }
}

}  // namespace quanta
