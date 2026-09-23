// Byte-level BPE tokenizer compatible with Qwen2 / GPT-4-style tokenizers.
// Input text is expected to be NFC-normalized already (on Android: java.text.Normalizer).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model_file.h"

namespace quanta {

class Tokenizer {
public:
    explicit Tokenizer(const TokenizerData& data);

    // Added/special tokens written literally in the text (e.g. "<|im_start|>") are recognized.
    std::vector<int> encode(const std::string& text) const;
    std::string decode(int id) const;  // raw bytes; may be a partial UTF-8 sequence
    std::string decode(const std::vector<int>& ids) const;

    int n_tokens() const { return int(tokens_.size()); }  // real tokens (model vocab may be padded)
    int token_id(const std::string& literal) const;       // special token id, or -1

private:
    enum CharClass : uint8_t { OTHER = 0, LETTER = 1, NUMBER = 2, SPACE = 4 };
    uint8_t classify(uint32_t cp) const;
    void pretokenize(const std::string& text, std::vector<std::pair<size_t, size_t>>& out) const;
    void bpe(const std::string& piece, std::vector<int>& out) const;

    std::vector<std::string> tokens_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;  // (left<<32|right) -> (rank, result)
    std::vector<std::pair<std::string, int>> specials_;          // sorted longest first
    std::unordered_map<std::string, int> special_ids_;
    int byte_token_[256];
    std::vector<std::pair<uint32_t, uint32_t>> letters_, numbers_, spaces_;
};

}  // namespace quanta
