#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace quanta {

namespace {

// Decodes one UTF-8 code point starting at s[i]; returns its byte length (invalid bytes -> length 1).
int utf8_next(const std::string& s, size_t i, uint32_t* cp) {
    const unsigned char c = s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > s.size()) {
        *cp = 0xFFFD;
        return 1;
    }
    uint32_t v = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = s[i + k];
        if ((cc >> 6) != 0x2) {
            *cp = 0xFFFD;
            return 1;
        }
        v = (v << 6) | (cc & 0x3F);
    }
    *cp = v;
    return len;
}

bool in_ranges(const std::vector<std::pair<uint32_t, uint32_t>>& r, uint32_t cp) {
    auto it = std::upper_bound(r.begin(), r.end(), cp, [](uint32_t v, const auto& p) { return v < p.first; });
    return it != r.begin() && cp <= std::prev(it)->second;
}

uint64_t pair_key(int a, int b) { return (uint64_t(uint32_t(a)) << 32) | uint32_t(b); }

}  // namespace

Tokenizer::Tokenizer(const TokenizerData& data)
    : tokens_(data.tokens), letters_(data.letters), numbers_(data.numbers), spaces_(data.spaces) {
    merges_.reserve(data.merges.size() * 2);
    for (size_t rank = 0; rank < data.merges.size(); ++rank) {
        const auto& m = data.merges[rank];
        merges_.emplace(pair_key(int(m[0]), int(m[1])), std::make_pair(int(rank), int(m[2])));
    }
    int first_special = int(tokens_.size());
    for (const auto& [id, text] : data.specials) {
        specials_.emplace_back(text, int(id));
        special_ids_[text] = int(id);
        first_special = std::min(first_special, int(id));
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    std::fill(std::begin(byte_token_), std::end(byte_token_), -1);
    for (int id = 0; id < first_special; ++id)
        if (tokens_[id].size() == 1) byte_token_[static_cast<unsigned char>(tokens_[id][0])] = id;
    for (int b = 0; b < 256; ++b)
        if (byte_token_[b] < 0) throw std::runtime_error("tokenizer is missing a byte token");
}

int Tokenizer::token_id(const std::string& literal) const {
    auto it = special_ids_.find(literal);
    return it == special_ids_.end() ? -1 : it->second;
}

uint8_t Tokenizer::classify(uint32_t cp) const {
    if (cp < 0x80) {  // ASCII fast path
        if ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') return LETTER;
        if (cp >= '0' && cp <= '9') return NUMBER;
        if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return SPACE;
        return OTHER;
    }
    if (in_ranges(letters_, cp)) return LETTER;
    if (in_ranges(numbers_, cp)) return NUMBER;
    if (in_ranges(spaces_, cp)) return SPACE;
    return OTHER;
}

// Hand-written equivalent of the Qwen2 split regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]*
//   | \s*[\r\n]+ | \s+(?!\S) | \s+
// Emits [begin, end) byte ranges of `text`.
void Tokenizer::pretokenize(const std::string& text, std::vector<std::pair<size_t, size_t>>& out) const {
    std::vector<uint32_t> cp;
    std::vector<uint8_t> cls;
    std::vector<size_t> off;
    for (size_t i = 0; i < text.size();) {
        uint32_t c;
        int len = utf8_next(text, i, &c);
        cp.push_back(c);
        cls.push_back(classify(c));
        off.push_back(i);
        i += len;
    }
    off.push_back(text.size());
    const size_t n = cp.size();

    auto L = [&](size_t k) { return k < n && cls[k] == LETTER; };
    auto S = [&](size_t k) { return k < n && cls[k] == SPACE; };
    auto crlf = [&](size_t k) { return k < n && (cp[k] == '\r' || cp[k] == '\n'); };
    auto punct = [&](size_t k) { return k < n && cls[k] == OTHER; };  // [^\s\p{L}\p{N}]
    auto lower = [&](size_t k) { return k < n && cp[k] < 0x80 ? (cp[k] | 0x20) : 0u; };

    size_t i = 0;
    while (i < n) {
        size_t j = 0;
        // 1. contractions
        if (cp[i] == '\'') {
            const uint32_t a = lower(i + 1), b = lower(i + 2);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
        }
        // 2. optional non-letter/number/newline prefix, then letters
        if (!j) {
            size_t k = SIZE_MAX;
            if (L(i)) k = i;
            else if (!crlf(i) && cls[i] != NUMBER && L(i + 1)) k = i + 1;
            if (k != SIZE_MAX) {
                while (L(k)) ++k;
                j = k;
            }
        }
        // 3. a single number
        if (!j && cls[i] == NUMBER) j = i + 1;
        // 4. optional space, punctuation run, trailing newlines
        if (!j) {
            size_t k = SIZE_MAX;
            if (cp[i] == ' ' && punct(i + 1)) k = i + 1;
            else if (punct(i)) k = i;
            if (k != SIZE_MAX) {
                while (punct(k)) ++k;
                while (crlf(k)) ++k;
                j = k;
            }
        }
        // 5-7. whitespace
        if (!j && S(i)) {
            size_t e = i;
            size_t last_nl = SIZE_MAX;
            while (S(e)) {
                if (crlf(e)) last_nl = e;
                ++e;
            }
            if (last_nl != SIZE_MAX) j = last_nl + 1;            // \s*[\r\n]+
            else if (e == n) j = e;                               // \s+(?!\S) at end of text
            else if (e - i >= 2) j = e - 1;                       // \s+(?!\S): leave one space for the next word
            else j = e;                                           // \s+
        }
        if (!j) j = i + 1;  // unreachable in practice; guarantees progress
        out.emplace_back(off[i], off[j]);
        i = j;
    }
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    std::vector<int> ids;
    ids.reserve(piece.size());
    for (unsigned char c : piece) ids.push_back(byte_token_[c]);

    while (ids.size() > 1) {
        int best_rank = INT_MAX, best_pos = -1, best_id = -1;
        for (size_t p = 0; p + 1 < ids.size(); ++p) {
            auto it = merges_.find(pair_key(ids[p], ids[p + 1]));
            if (it != merges_.end() && it->second.first < best_rank) {
                best_rank = it->second.first;
                best_pos = int(p);
                best_id = it->second.second;
            }
        }
        if (best_pos < 0) break;
        ids[best_pos] = best_id;
        ids.erase(ids.begin() + best_pos + 1);
    }
    out.insert(out.end(), ids.begin(), ids.end());
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    std::vector<std::pair<size_t, size_t>> pieces;
    auto encode_plain = [&](size_t b, size_t e) {
        if (b >= e) return;
        const std::string chunk = text.substr(b, e - b);
        pieces.clear();
        pretokenize(chunk, pieces);
        for (const auto& [pb, pe] : pieces) bpe(chunk.substr(pb, pe - pb), out);
    };

    size_t start = 0;
    for (size_t i = 0; i < text.size();) {
        const std::pair<std::string, int>* hit = nullptr;
        for (const auto& sp : specials_) {
            if (text.compare(i, sp.first.size(), sp.first) == 0) {
                hit = &sp;
                break;
            }
        }
        if (hit) {
            encode_plain(start, i);
            out.push_back(hit->second);
            i += hit->first.size();
            start = i;
        } else {
            ++i;
        }
    }
    encode_plain(start, text.size());
    return out;
}

std::string Tokenizer::decode(int id) const {
    if (id < 0 || id >= int(tokens_.size())) return {};
    return tokens_[id];
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string s;
    for (int id : ids) s += decode(id);
    return s;
}

}  // namespace quanta
