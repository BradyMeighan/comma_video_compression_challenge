#!/usr/bin/env python
"""E3: Zopfli (better DEFLATE) vs bz2."""
import sys, os, pickle, struct, bz2
from pathlib import Path
import zopfli.zlib as zopfli_zlib
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def main():
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        data = pickle.load(f)
    from explore_e2_zstd import serialize_channel_only, serialize_mask
    raw_mask = serialize_mask(data['mask_patches'])
    raw_rgb = serialize_channel_only(data['rgb_patches'])

    schemes = []
    for name, fn in [
        ("bz2_l9", lambda b: bz2.compress(b, 9)),
        ("zlib_l9", lambda b: zlib.compress(b, 9)),
        ("zopfli", lambda b: zopfli_zlib.compress(b, numiterations=15)),
    ]:
        cm = fn(raw_mask); cr = fn(raw_rgb)
        schemes.append((name, len(cm), len(cr), len(cm)+len(cr)))

    from prepare import MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE
    import math
    score_bz2 = data['score']
    sb_bz2 = data['sb_total_bz2']
    print(f"=== E3: Zopfli vs bz2 vs zlib ===")
    print(f"BZ2 baseline: {sb_bz2}B / score {score_bz2:.4f}\n")
    for name, cm, cr, total in schemes:
        rate = (MASK_BYTES + POSE_BYTES + data['model_bytes'] + total) / UNCOMPRESSED_SIZE
        score = 100*data['seg_dist'] + math.sqrt(10*data['pose_dist']) + 25*rate
        print(f"  {name}: mask={cm}B rgb={cr}B total={total}B score={score:.4f} delta={score-score_bz2:+.4f}")

    import csv
    with open(OUTPUT_DIR / "e3_zopfli_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["scheme", "sb_mask", "sb_rgb", "sb_total", "score", "delta_vs_bz2"])
        for name, cm, cr, total in schemes:
            rate = (MASK_BYTES + POSE_BYTES + data['model_bytes'] + total) / UNCOMPRESSED_SIZE
            score = 100*data['seg_dist'] + math.sqrt(10*data['pose_dist']) + 25*rate
            w.writerow([name, cm, cr, total, score, score-score_bz2])


if __name__ == "__main__":
    main()
