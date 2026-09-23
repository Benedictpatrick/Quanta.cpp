// Reader for the .qnt model format (see scripts/export.py for the layout).
// The file is memory-mapped; tensors point straight into the mapping, so loading is
// near-instant and weights are paged in by the OS on demand.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quanta {

enum class DType : uint32_t { F32 = 0, F16 = 1, Q8_0 = 2, Q4_0 = 3 };
constexpr int QK = 32;  // quantization block size

// down_proj expects its input rotated by a block-diagonal 256x256 Walsh-Hadamard transform
// (weights were rotated offline; see research/gptq.py --hadamard).
constexpr uint32_t FLAG_HADAMARD_DOWN = 1;

const char* dtype_name(DType t);

// 2-D view of a weight matrix (1-D tensors have rows == 1).
// For Q8_0 / Q4_0 the quants come first, followed by one fp16 scale per 32 weights.
struct Tensor {
    DType dtype = DType::F32;
    int rows = 0;
    int cols = 0;
    const uint8_t* data = nullptr;

    const float* f32() const { return reinterpret_cast<const float*>(data); }
    size_t quant_bytes() const;                    // size of the quant section
    const uint16_t* scales() const {               // fp16 block scales (quantized types)
        return reinterpret_cast<const uint16_t*>(data + quant_bytes());
    }
};

struct HParams {
    int vocab_size, dim, n_layers, n_heads, n_kv_heads, head_dim, ffn_dim, max_seq;
    float rope_theta, norm_eps;
    bool tied_embeddings, qkv_bias;
    uint32_t flags;  // FLAG_* below (file version 2+)
};

struct TokenizerData {
    std::vector<std::string> tokens;                          // raw bytes per token id
    std::vector<std::array<uint32_t, 3>> merges;              // (left, right, result) in rank order
    std::vector<std::pair<uint32_t, std::string>> specials;   // added/special tokens
    std::vector<std::pair<uint32_t, uint32_t>> letters, numbers, spaces;  // codepoint ranges
};

class ModelFile {
public:
    ModelFile() = default;
    ~ModelFile();
    ModelFile(const ModelFile&) = delete;
    ModelFile& operator=(const ModelFile&) = delete;

    bool open(const std::string& path, std::string* err);

    const HParams& hparams() const { return hp_; }
    const TokenizerData& tokenizer() const { return tok_; }
    const Tensor* find(const std::string& name) const;
    const Tensor& get(const std::string& name) const;  // throws if missing
    size_t size_bytes() const { return size_; }

private:
    void unmap();

    HParams hp_{};
    TokenizerData tok_;
    std::unordered_map<std::string, Tensor> tensors_;
    const uint8_t* base_ = nullptr;
    size_t size_ = 0;
    void* file_handle_ = nullptr;     // Windows: HANDLE
    void* mapping_handle_ = nullptr;  // Windows: mapping HANDLE
};

}  // namespace quanta
