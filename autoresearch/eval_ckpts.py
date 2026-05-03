#!/usr/bin/env python
"""Eval all H100 checkpoints + the original gen_continued.pt for comparison."""
import sys, os
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import evaluate, gpu_cleanup
from train import Generator, load_data_full

CKPT_DIR = Path("autoresearch/colab_run/h100_ckpts")
ORIGINAL = Path("autoresearch/colab_run/gen_continued.pt")


def main():
    device = torch.device("cuda")
    data = load_data_full(device)

    paths = [ORIGINAL] + sorted(CKPT_DIR.glob("*.pt*"))
    results = []
    for p in paths:
        if not p.exists():
            continue
        print(f"\n=== {p.name} ===", flush=True)
        gen = Generator().to(device)
        sd = torch.load(p, map_location=device, weights_only=True)
        gen.load_state_dict(sd, strict=False)
        result = evaluate(gen, data, device)
        print(f"  score={result['score']:.6f} seg={result['seg_term']:.4f} "
              f"pose={result['pose_term']:.4f} rate={result['rate_term']:.4f}")
        results.append((p.name, result['score'], result['seg_term'],
                         result['pose_term'], result['rate_term']))
        del gen; gpu_cleanup()

    print("\n\n=== SUMMARY (sorted by score) ===")
    results.sort(key=lambda r: r[1])
    print(f"{'name':<35} {'score':>8} {'seg':>8} {'pose':>8} {'rate':>8}")
    for name, s, seg, pose, rate in results:
        print(f"{name:<35} {s:>8.4f} {seg:>8.4f} {pose:>8.4f} {rate:>8.4f}")


if __name__ == "__main__":
    main()
