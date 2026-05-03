#!/usr/bin/env python
"""
A5: Differential coordinate encoding (delta-coded positions).

Currently each patch stores absolute (x, y) as 2×u16 = 4 bytes.
If we sort patches by 1D index = y*1164+x within each pair, then encode the GAP
to the next position via exp-Golomb / signed-Golomb-Rice / variable-length, we
might save bytes since gaps are smaller than absolute positions.

This is the AV1 motion-vector-difference style encoding.

We test by packing differently and measuring bz2 savings.
"""
import sys, os, pickle, struct, bz2
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def encode_diff_coords(rgb_patches):
    """Per-pair, sort patches by y*W+x, then encode each as gap from previous.
    Use a u16 for gap (most fit in 16 bits since max gap < 1164*874 = ~1M but still u32).
    For brevity store gap as varint (1 byte if <128, 2 bytes if <16384, else 3+).
    """
    parts = [struct.pack("<H", len(rgb_patches))]
    W = 1164
    for pi in sorted(rgb_patches.keys()):
        ps = rgb_patches[pi]
        # Sort by 1D index
        sorted_ps = sorted(ps, key=lambda p: p[1] * W + p[0])
        parts.append(struct.pack("<HH", pi, len(sorted_ps)))
        prev_idx = 0
        for (x, y, c, d) in sorted_ps:
            cur_idx = y * W + x
            gap = cur_idx - prev_idx
            # varint encoding (LEB128-like for unsigned)
            while gap >= 128:
                parts.append(struct.pack("<B", (gap & 0x7F) | 0x80))
                gap >>= 7
            parts.append(struct.pack("<B", gap))
            parts.append(struct.pack("<Bb", c, d))
            prev_idx = cur_idx
    return b''.join(parts)


def main():
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    rgb_patches = bp['rgb_patches']
    score_bl = bp['score']

    print(f"=== A5: differential coordinate encoding ===")
    raw = encode_diff_coords(rgb_patches)
    compressed = bz2.compress(raw, compresslevel=9)
    print(f"Raw diff-coded: {len(raw)}B")
    print(f"After bz2: {len(compressed)}B")
    print(f"Original bz2 (absolute coords): {bp['sb_rgb_bz2']}B")
    print(f"Diff-coded saving vs bz2-absolute: {bp['sb_rgb_bz2'] - len(compressed)}B "
          f"({100 * (1 - len(compressed)/bp['sb_rgb_bz2']):+.1f}%)")

    from prepare import MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE
    import math
    sb_total = bp['sb_mask_bz2'] + len(compressed)
    rate = (MASK_BYTES + POSE_BYTES + bp['model_bytes'] + sb_total) / UNCOMPRESSED_SIZE
    score = 100*bp['seg_dist'] + math.sqrt(10*bp['pose_dist']) + 25*rate
    print(f"\nNew score: {score:.4f} (delta {score-score_bl:+.4f})")

    import csv
    with open(OUTPUT_DIR / "a5_diff_coords_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["scheme", "sb_total", "score", "delta"])
        w.writerow(["baseline_bz2", bp['sb_total_bz2'], score_bl, 0])
        w.writerow(["diff_coded_bz2", sb_total, score, score-score_bl])


if __name__ == "__main__":
    main()
