"""Export a Hugging Face Qwen2 checkpoint to the Quanta .qnt format.

Layout (all little-endian):
  header      : magic 'QNT1', version, hparams, flags (v2+)
  tokenizer   : vocab (raw bytes per token), merges (left, right, result), special tokens,
                unicode category ranges (letters / numbers / whitespace) for the pre-tokenizer
  tensor table: name, dtype, shape, absolute data offset
  tensor data : each tensor 64-byte aligned so the engine can mmap and use it in place

Quantized layouts (per tensor, rows x cols, cols % 32 == 0):
  Q8_0: int8 quants [rows*cols]            then fp16 scales [rows*cols/32]
  Q4_0: packed nibbles [rows*cols/2]       then fp16 scales [rows*cols/32]
        block of 32 -> 16 bytes, byte j = (q[j] & 0xF) | (q[j+16] << 4), q in [0,15], value = (q-8)*d
"""
import argparse
import json
import struct
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

MAGIC = b"QNT1"
VERSION = 2
FLAG_HADAMARD_DOWN = 1  # engine applies a block-256 Walsh-Hadamard transform to the down_proj input
ALIGN = 64

DT_F32, DT_F16, DT_Q8_0, DT_Q4_0 = 0, 1, 2, 3
DTYPES = {"f32": DT_F32, "f16": DT_F16, "q8_0": DT_Q8_0, "q4_0": DT_Q4_0}
QK = 32


def bytes_to_unicode():
    """GPT-2 byte-level mapping: byte -> printable unicode char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_ranges(pred):
    """Compact [start, end] codepoint ranges where pred(ch) holds."""
    ranges, start = [], None
    for cp in range(0x110000):
        ok = pred(chr(cp)) if not (0xD800 <= cp <= 0xDFFF) else False
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            ranges.append((start, cp - 1))
            start = None
    if start is not None:
        ranges.append((start, 0x10FFFF))
    return ranges


def is_letter(ch):
    return unicodedata.category(ch).startswith("L")


def is_number(ch):
    return unicodedata.category(ch).startswith("N")


# Unicode White_Space property, which is what \s means in the HF (Rust regex) pre-tokenizer.
# Python's str.isspace() differs (it also accepts U+001C..U+001F).
WHITE_SPACE = {*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
               0x2028, 0x2029, 0x202F, 0x205F, 0x3000}


def is_space(ch):
    return ord(ch) in WHITE_SPACE


# ---------------------------------------------------------------- quantization

def quantize_q8_0(w: np.ndarray):
    blocks = w.reshape(-1, QK)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0)
    q = np.clip(np.round(blocks * inv[:, None]), -127, 127).astype(np.int8)
    return q.tobytes() + d.astype(np.float16).tobytes()


def quantize_q4_0(w: np.ndarray):
    blocks = w.reshape(-1, QK)
    idx = np.abs(blocks).argmax(axis=1)
    mx = blocks[np.arange(len(blocks)), idx]  # signed value with largest magnitude
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0)
    q = np.clip(np.round(blocks * inv[:, None] + 8.0), 0, 15).astype(np.uint8)
    packed = (q[:, :16] & 0xF) | (q[:, 16:] << 4)
    return packed.astype(np.uint8).tobytes() + d.astype(np.float16).tobytes()


def pack_q4_0(qint: np.ndarray, d: np.ndarray) -> bytes:
    """Precomputed Q4_0 (e.g. from research/gptq.py): qint uint8 [rows, cols] in 0..15, d fp16 [rows, cols/32]."""
    q = qint.reshape(-1, QK).astype(np.uint8)
    packed = (q[:, :16] & 0xF) | (q[:, 16:] << 4)
    return packed.astype(np.uint8).tobytes() + d.astype(np.float16).reshape(-1).tobytes()


def dequant_q8_0(w: np.ndarray) -> np.ndarray:
    """What the engine's 8-bit head actually computes with (float32 [rows, cols])."""
    blocks = w.reshape(-1, QK).astype(np.float32)
    d = np.abs(blocks).max(axis=1) / 127.0  # same steps as quantize_q8_0: round with f32 d, store f16 d
    q = np.clip(np.round(blocks * np.where(d > 0, 1.0 / np.where(d > 0, d, 1), 0)[:, None]), -127, 127)
    return (q * d.astype(np.float16).astype(np.float32)[:, None]).reshape(w.shape)


def shortlist_head(w8: np.ndarray, chunk=8192):
    """Exact Shortlist Head (N7): a 4-bit copy of the 8-bit head plus, per 32-block, an upper bound on
    ||w8_block - w4_block||. The engine scores all tokens with the 4-bit copy, bounds the error with
    sum_b err_b * ||x_b||, and recomputes exact 8-bit logits only for tokens that could still be in the top-k."""
    rows, cols = w8.shape
    packed, scales, errs = [], [], []
    for r0 in range(0, rows, chunk):
        blocks = w8[r0:r0 + chunk].reshape(-1, QK).astype(np.float32)
        mx = blocks[np.arange(len(blocks)), np.abs(blocks).argmax(axis=1)]
        best_e = np.full(len(blocks), np.inf)
        best_d = np.zeros(len(blocks), np.float32)
        best_q = np.zeros(blocks.shape, np.uint8)
        for f in np.linspace(0.8, 1.1, 13):  # small scale search: minimize the block error the bound pays for
            for sgn in (-8.0, 8.0):
                d = (mx / sgn * f).astype(np.float16).astype(np.float32)
                inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0)
                q = np.clip(np.round(blocks * inv[:, None] + 8.0), 0, 15)
                e = (((q - 8.0) * d[:, None] - blocks) ** 2).sum(axis=1)
                better = e < best_e
                best_e[better], best_d[better], best_q[better] = e[better], d[better], q[better]
        diff = (best_q.astype(np.float64) - 8.0) * best_d[:, None].astype(np.float64) - blocks.astype(np.float64)
        err = np.sqrt((diff ** 2).sum(axis=1))
        e16 = err.astype(np.float16)
        e16 = np.where(e16.astype(np.float64) < err, np.nextafter(e16, np.float16(np.inf)), e16)  # round up
        packed.append(((best_q[:, :16] & 0xF) | (best_q[:, 16:] << 4)).astype(np.uint8))
        scales.append(best_d.astype(np.float16))
        errs.append(e16)
    head_q4 = np.concatenate(packed).tobytes() + np.concatenate(scales).tobytes()
    head_err = np.concatenate(errs).astype(np.float16).tobytes()
    return head_q4, head_err


def encode_tensor(w: np.ndarray, dtype: int) -> bytes:
    if dtype == DT_F32:
        return w.astype(np.float32).tobytes()
    if dtype == DT_F16:
        return w.astype(np.float16).tobytes()
    if dtype == DT_Q8_0:
        return quantize_q8_0(w.astype(np.float32))
    if dtype == DT_Q4_0:
        return quantize_q4_0(w.astype(np.float32))
    raise ValueError(dtype)


# ---------------------------------------------------------------- writer helpers

def u32(x):
    return struct.pack("<I", x)


def f32(x):
    return struct.pack("<f", x)


def blob(b: bytes):
    return u32(len(b)) + b


def build_tokenizer_section(model_dir: Path) -> bytes:
    tok = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    merges = tok["model"]["merges"]
    byte_dec = {v: k for k, v in bytes_to_unicode().items()}

    added = {t["content"]: t["id"] for t in tok["added_tokens"]}
    n_tokens = max(max(vocab.values()), max(added.values()) if added else 0) + 1

    token_bytes = [b""] * n_tokens
    for s, i in vocab.items():
        token_bytes[i] = bytes(byte_dec[c] for c in s)
    for s, i in added.items():
        token_bytes[i] = s.encode("utf-8")

    out = [u32(n_tokens)]
    out += [blob(b) for b in token_bytes]

    out.append(u32(len(merges)))
    for m in merges:
        a, b = m.split(" ") if isinstance(m, str) else m
        out.append(u32(vocab[a]) + u32(vocab[b]) + u32(vocab[a + b]))

    out.append(u32(len(added)))
    for s, i in sorted(added.items(), key=lambda kv: kv[1]):
        out.append(u32(i) + blob(s.encode("utf-8")))

    for pred in (is_letter, is_number, is_space):
        rs = unicode_ranges(pred)
        out.append(u32(len(rs)))
        out += [u32(a) + u32(b) for a, b in rs]
    return b"".join(out)


def tensor_dtype(name: str, wanted: int) -> int:
    """Which storage type each tensor gets."""
    if name.endswith(".bias") or "norm" in name:
        return DT_F32  # tiny, precision-sensitive
    if name == "model.embed_tokens.weight" and wanted == DT_Q4_0:
        return DT_Q8_0  # tied embedding doubles as lm_head; keep it at 8-bit
    return wanted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/qwen2.5-0.5b-instruct")
    ap.add_argument("--dtype", default="f32", choices=DTYPES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--shortlist", action="store_true", help="add the Exact Shortlist Head tensors (N7)")
    ap.add_argument("--gptq", default=None, help="research/gptq_sym.pt: use GPTQ integers for the layers (Q4_0)")
    args = ap.parse_args()
    gptq, overrides, flags = {}, {}, 0
    if args.gptq:
        g = torch.load(args.gptq)
        gptq, overrides = g["quant"], g["float_overrides"]
        if g["hadamard"]:
            flags |= FLAG_HADAMARD_DOWN
        if any(m is not None for _, _, m in gptq.values()):
            sys.exit("only symmetric (Q4_0) GPTQ results can be exported")

    model_dir = Path(args.model)
    cfg = json.loads((model_dir / "config.json").read_text())
    out_path = Path(args.out or f"models/qwen2.5-0.5b-{args.dtype}.qnt")
    wanted = DTYPES[args.dtype]

    state = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        state.update(load_file(str(f)))
    state = {k: v.to(torch.float32).numpy() for k, v in state.items()}
    for k, v in overrides.items():  # e.g. rotated v_proj biases from research/gptq.py --hadamard
        state[k] = v.float().numpy()

    tied = int(cfg.get("tie_word_embeddings", False) or "lm_head.weight" not in state)
    if tied:
        state.pop("lm_head.weight", None)

    dim = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    hparams = b"".join([
        u32(cfg["vocab_size"]), u32(dim), u32(cfg["num_hidden_layers"]), u32(n_heads),
        u32(cfg["num_key_value_heads"]), u32(dim // n_heads), u32(cfg["intermediate_size"]),
        u32(args.max_seq), f32(cfg.get("rope_theta", 10000.0)), f32(cfg["rms_norm_eps"]),
        u32(tied), u32(1),  # has_qkv_bias
        u32(flags),
    ])
    tok_section = build_tokenizer_section(model_dir)

    names = sorted(state)
    encoded, entries = {}, []
    for n in names:
        w = state[n]
        dt = tensor_dtype(n, wanted)
        if dt in (DT_Q8_0, DT_Q4_0) and (w.ndim != 2 or w.shape[1] % QK):
            dt = DT_F16
        if n in gptq:
            qint, d, _ = gptq[n]
            dt = DT_Q4_0
            encoded[n] = (dt, w.shape, pack_q4_0(qint.numpy(), d.numpy()))
        else:
            encoded[n] = (dt, w.shape, encode_tensor(w, dt))
        print(f"  {n:55s} {str(w.shape):16s} -> {list(DTYPES)[dt]}", file=sys.stderr)

    if args.shortlist:
        head = "model.embed_tokens.weight" if tied else "lm_head.weight"
        if encoded[head][0] != DT_Q8_0:
            sys.exit("--shortlist needs an 8-bit (q8_0) head")
        print("  building shortlist head ...", file=sys.stderr)
        rows, cols = state[head].shape
        q4, err = shortlist_head(dequant_q8_0(state[head]))
        encoded["quanta.head_q4"] = (DT_Q4_0, (rows, cols), q4)
        encoded["quanta.head_err"] = (DT_F16, (rows, cols // QK), err)
        names = sorted(encoded)

    # First pass: size of the table so we know where data starts.
    def table_bytes(offsets):
        t = [u32(len(names))]
        for n in names:
            dt, shape, _ = encoded[n]
            t.append(blob(n.encode()) + u32(dt) + u32(len(shape)) + b"".join(u32(s) for s in shape)
                     + struct.pack("<Q", offsets.get(n, 0)))
        return b"".join(t)

    head = MAGIC + u32(VERSION) + hparams + blob(tok_section)
    table_len = len(table_bytes({}))
    pos = len(head) + table_len
    offsets = {}
    for n in names:
        pos = (pos + ALIGN - 1) // ALIGN * ALIGN
        offsets[n] = pos
        pos += len(encoded[n][2])

    with open(out_path, "wb") as f:
        f.write(head)
        f.write(table_bytes(offsets))
        for n in names:
            f.write(b"\0" * (offsets[n] - f.tell()))
            f.write(encoded[n][2])
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
