#include "model_file.h"

#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace quanta {

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32: return "f32";
        case DType::F16: return "f16";
        case DType::Q8_0: return "q8_0";
        case DType::Q4_0: return "q4_0";
    }
    return "?";
}

size_t Tensor::quant_bytes() const {
    const size_t n = size_t(rows) * cols;
    switch (dtype) {
        case DType::F32: return n * 4;
        case DType::F16: return n * 2;
        case DType::Q8_0: return n;
        case DType::Q4_0: return n / 2;
    }
    return 0;
}

namespace {

// Bounds-checked sequential reader over the mapped bytes.
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;

    void need(size_t n) {
        if (size_t(end - p) < n) throw std::runtime_error("model file truncated");
    }
    uint32_t u32() {
        need(4);
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    uint64_t u64() {
        need(8);
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    float f32() {
        uint32_t u = u32();
        float f;
        std::memcpy(&f, &u, 4);
        return f;
    }
    std::string str() {
        uint32_t n = u32();
        need(n);
        std::string s(reinterpret_cast<const char*>(p), n);
        p += n;
        return s;
    }
    std::vector<std::pair<uint32_t, uint32_t>> ranges() {
        std::vector<std::pair<uint32_t, uint32_t>> r(u32());
        for (auto& x : r) {
            x.first = u32();
            x.second = u32();
        }
        return r;
    }
};

}  // namespace

ModelFile::~ModelFile() { unmap(); }

void ModelFile::unmap() {
#ifdef _WIN32
    if (base_) UnmapViewOfFile(base_);
    if (mapping_handle_) CloseHandle(mapping_handle_);
    if (file_handle_) CloseHandle(file_handle_);
#else
    if (base_) munmap(const_cast<uint8_t*>(base_), size_);
#endif
    base_ = nullptr;
    mapping_handle_ = file_handle_ = nullptr;
    size_ = 0;
}

bool ModelFile::open(const std::string& path, std::string* err) {
    unmap();
#ifdef _WIN32
    HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) {
        CloseHandle(f);
        if (err) *err = "CreateFileMapping failed";
        return false;
    }
    file_handle_ = f;
    mapping_handle_ = m;
    base_ = static_cast<const uint8_t*>(MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0));
    size_ = size_t(sz.QuadPart);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    struct stat st;
    fstat(fd, &st);
    size_ = size_t(st.st_size);
    void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        size_ = 0;
        if (err) *err = "mmap failed";
        return false;
    }
    base_ = static_cast<const uint8_t*>(p);
#endif
    if (!base_) {
        if (err) *err = "mapping failed";
        unmap();
        return false;
    }

    try {
        Cursor c{base_, base_ + size_};
        c.need(4);
        if (std::memcmp(c.p, "QNT1", 4) != 0) throw std::runtime_error("bad magic (not a .qnt file)");
        c.p += 4;
        const uint32_t version = c.u32();
        if (version < 1 || version > 2) throw std::runtime_error("unsupported version " + std::to_string(version));

        hp_.vocab_size = int(c.u32());
        hp_.dim = int(c.u32());
        hp_.n_layers = int(c.u32());
        hp_.n_heads = int(c.u32());
        hp_.n_kv_heads = int(c.u32());
        hp_.head_dim = int(c.u32());
        hp_.ffn_dim = int(c.u32());
        hp_.max_seq = int(c.u32());
        hp_.rope_theta = c.f32();
        hp_.norm_eps = c.f32();
        hp_.tied_embeddings = c.u32() != 0;
        hp_.qkv_bias = c.u32() != 0;
        hp_.flags = version >= 2 ? c.u32() : 0;

        // Tokenizer section is length-prefixed; parse it with its own cursor.
        uint32_t tok_len = c.u32();
        c.need(tok_len);
        Cursor t{c.p, c.p + tok_len};
        c.p += tok_len;
        tok_.tokens.resize(t.u32());
        for (auto& s : tok_.tokens) s = t.str();
        tok_.merges.resize(t.u32());
        for (auto& m : tok_.merges) m = {t.u32(), t.u32(), t.u32()};
        tok_.specials.resize(t.u32());
        for (auto& s : tok_.specials) {
            s.first = t.u32();
            s.second = t.str();
        }
        tok_.letters = t.ranges();
        tok_.numbers = t.ranges();
        tok_.spaces = t.ranges();

        uint32_t n_tensors = c.u32();
        for (uint32_t i = 0; i < n_tensors; ++i) {
            std::string name = c.str();
            Tensor tt;
            tt.dtype = DType(c.u32());
            uint32_t ndim = c.u32();
            if (ndim == 1) {
                tt.rows = 1;
                tt.cols = int(c.u32());
            } else if (ndim == 2) {
                tt.rows = int(c.u32());
                tt.cols = int(c.u32());
            } else {
                throw std::runtime_error("unsupported ndim for " + name);
            }
            uint64_t off = c.u64();
            if (off > size_) throw std::runtime_error("bad offset for " + name);
            tt.data = base_ + off;
            size_t total = tt.quant_bytes();
            if (tt.dtype == DType::Q8_0 || tt.dtype == DType::Q4_0) total += size_t(tt.rows) * tt.cols / QK * 2;
            if (off + total > size_) throw std::runtime_error("tensor out of bounds: " + name);
            tensors_.emplace(std::move(name), tt);
        }
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        unmap();
        return false;
    }
    return true;
}

const Tensor* ModelFile::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& ModelFile::get(const std::string& name) const {
    if (auto* t = find(name)) return *t;
    throw std::runtime_error("missing tensor: " + name);
}

}  // namespace quanta
