"""Compare Quanta against Hugging Face reference outputs (from scripts/reference.py).

  python tests/test_parity.py [--model models/qwen2.5-0.5b-f32.qnt] [--tol 1e-3] [--gen 64]

Checks, per prompt:
  * logits for every prompt position (max abs diff, top-1 agreement)
  * greedy continuation token ids match exactly (first --gen tokens)
For quantized models pass a looser --tol; top-1 / greedy agreement is reported either way.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUANTA = ROOT / "build" / ("quanta.exe" if sys.platform == "win32" else "quanta")


def run(args):
    r = subprocess.run([str(QUANTA), *args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"quanta failed: {r.stderr}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-0.5b-f32.qnt"))
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--kv-f16", action="store_true", help="test the default 16-bit KV cache (not exact)")
    args = ap.parse_args()
    args.extra = [] if args.kv_f16 else ["--kv-f32"]

    cases = json.loads((ROOT / "tests" / "data" / "ref.json").read_text(encoding="utf-8"))
    ok = True
    for i, case in enumerate(cases):
        ids = ",".join(map(str, case["input_ids"]))
        ref = np.load(ROOT / "tests" / "data" / f"ref_logits_{i}.npy")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "logits.bin"
            run(["logits", "-m", args.model, "--ids", ids, "-o", str(out), *args.extra])
            got = np.fromfile(out, dtype=np.float32).reshape(ref.shape)

        diff = np.abs(got - ref).max()
        top1 = (got.argmax(-1) == ref.argmax(-1)).mean()

        r = run(["gen", "-m", args.model, "--ids", ids, "-n", str(args.gen), *args.extra])
        gen = [int(x) for x in r.stdout.strip().split(",")]
        want = case["greedy_ids"][: args.gen]
        n_match = next((k for k, (a, b) in enumerate(zip(gen, want)) if a != b), len(want))
        speed = r.stderr.strip().splitlines()[-1]

        passed = diff < args.tol and n_match == len(want)
        ok &= passed
        print(f"[{i}] {'PASS' if passed else 'FAIL'}  max|dlogit|={diff:.2e}  top1={top1:.0%}  "
              f"greedy match {n_match}/{len(want)}  ({speed})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
