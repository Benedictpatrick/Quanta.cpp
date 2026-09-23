"""Check the C++ tokenizer produces exactly the same token ids as Hugging Face.

  python tests/test_tokenizer.py [--model models/qwen2.5-0.5b-f32.qnt] [--fuzz 40]

Inputs must be NFC-normalized (the engine leaves normalization to the app layer),
so every case is NFC-normalized before comparison.
"""
import argparse
import random
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
QUANTA = ROOT / "build" / ("quanta.exe" if sys.platform == "win32" else "quanta")

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure you'll love it. They've gone; we'd go. DON'T SHOUT'S 'quoted' it's",
    "Numbers: 12345, 3.14159, 1,000,000 and 2026-09-23.",
    "def f(x):\n    return x * 2  # double\n\n\n\tindented\ttabs\n",
    "   leading spaces and trailing spaces   ",
    "multiple   inner    spaces\n\n\nand\r\nwindows\r\nnewlines \n \n",
    "emoji 😀🚀👍🏽 family 👨‍👩‍👧 flags 🇮🇳🇺🇸",
    "中文测试：你好，世界！这是一个句子。",
    "हिन्दी में एक वाक्य है।",
    "مرحبا بالعالم",
    "Café naïve résumé coöperate",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "not<|im_end|>special<|endoftext|>tokens<tool_call>x</tool_call>",
    "nbsp here ideographic　space nel\u0085char",
    "file\u001cgroup\u001drecord\u001eunit\u001fseparators",
    "$100 @user #tag 50% a+b=c {json: [1, 2]} <html></html>",
    "http://example.com/path?q=1&r=2",
    "'s 't 're 've 'm 'll 'd 'S 'RE",
    "aé́ combining accent after NFC",
    "",
    " ",
    "\n",
    "x",
]


def fuzz_cases(n, seed=0):
    rnd = random.Random(seed)
    pools = ["abcXYZ", "0123456789", " \t\n\r", ".,;:!?'\"()-_", "éüñç", "你好世界", "😀🚀", " 　", "<|>"]
    out = []
    for _ in range(n):
        s = "".join(rnd.choice(rnd.choice(pools)) for _ in range(rnd.randint(1, 60)))
        out.append(s)
    return out


def quanta_tokenize(model, text):
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".txt") as f:
        f.write(text.encode("utf-8"))
        path = f.name
    try:
        r = subprocess.run([str(QUANTA), "tokenize", "-m", model, "-f", path], capture_output=True, text=True)
    finally:
        Path(path).unlink()
    if r.returncode != 0:
        sys.exit(f"quanta failed: {r.stderr}")
    line = r.stdout.strip()
    return [int(x) for x in line.split(",")] if line else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-0.5b-f32.qnt"))
    ap.add_argument("--fuzz", type=int, default=40)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "qwen2.5-0.5b-instruct"))
    big = [
        (Path(sys.base_prefix) / "LICENSE.txt").read_text(encoding="utf-8")[:20000],
        (Path(sys.base_prefix) / "Lib" / "json" / "decoder.py").read_text(encoding="utf-8"),
        (ROOT / "docs" / "RESEARCH.md").read_text(encoding="utf-8"),
    ]
    cases = [unicodedata.normalize("NFC", c) for c in CASES + big + fuzz_cases(args.fuzz)]

    failures = 0
    for i, text in enumerate(cases):
        want = tok.encode(text)
        got = quanta_tokenize(args.model, text)
        if got != want:
            failures += 1
            k = next((j for j, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            print(f"FAIL case {i}: {text[:60]!r}")
            print(f"   first diff at token {k}: quanta {got[k:k+5]} vs hf {want[k:k+5]}")
            print(f"   hf pieces around: {[tok.decode([t]) for t in want[max(0,k-2):k+3]]}")
    total_tokens = sum(len(tok.encode(c)) for c in cases)
    print(f"{len(cases) - failures}/{len(cases)} cases identical ({total_tokens} tokens total)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
