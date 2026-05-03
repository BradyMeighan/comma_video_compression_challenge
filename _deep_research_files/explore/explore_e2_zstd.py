#!/usr/bin/env python
"""E2: zstd with dictionary trained on patch payloads."""
import sys, os, pickle, struct
from pathlib import Path
import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def serialize_channel_only(rgb_patches):
    parts = [struct.pack("<H", len(rgb_patches))]
    for pi in sorted(rgb_patches.keys()):
        ps = rgb_patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    return b''.join(parts)


def serialize_mask(mask_patches):
    parts = [struct.pack("<H", len(mask_patches))]
    for pi in sorted(mask_patches.keys()):
        ps = mask_patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c) in ps:
            parts.append(struct.pack("<HHB", x, y, c))
    return b''.join(parts)


def main():
    pkl_path = OUTPUT_DIR / "baseline_patches.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    mask_patches = data['mask_patches']
    rgb_patches = data['rgb_patches']
    sb_total_bz2 = data['sb_total_bz2']
    score_bz2 = data['score']
    model_bytes = data['model_bytes']

    raw_mask = serialize_mask(mask_patches)
    raw_rgb = serialize_channel_only(rgb_patches)

    print(f"=== E2: zstd vs bz2 ===")
    print(f"Raw bytes: mask={len(raw_mask)}B rgb={len(raw_rgb)}B")

    # Test multiple zstd configs
    results = []
    for level in [3, 9, 15, 22]:
        cctx = zstd.ZstdCompressor(level=level)
        cm = cctx.compress(raw_mask)
        cr = cctx.compress(raw_rgb)
        total = len(cm) + len(cr)
        results.append((f"zstd_l{level}", len(cm), len(cr), total))
        print(f"zstd level={level}: mask={len(cm)}B rgb={len(cr)}B total={total}B")

    # Try training a dictionary on the rgb stream by chunking
    # Split rgb into chunks per pair, train dict
    chunks = []
    for pi in sorted(rgb_patches.keys()):
        ps = rgb_patches[pi]
        chunk = struct.pack("<HH", pi, len(ps))
        for (x, y, c, d) in ps:
            chunk += struct.pack("<HHBb", x, y, c, d)
        chunks.append(chunk)
    if len(chunks) >= 7:
        try:
            dict_data = zstd.train_dictionary(2048, chunks)
            cctx_d = zstd.ZstdCompressor(dict_data=dict_data, level=22)
            cm_d = cctx_d.compress(raw_mask)
            cr_d = cctx_d.compress(raw_rgb)
            dict_size = len(dict_data.as_bytes())
            total_d = len(cm_d) + len(cr_d) + dict_size
            print(f"zstd_dict (size={dict_size}B): mask={len(cm_d)}B rgb={len(cr_d)}B "
                  f"+dict={dict_size}B = total={total_d}B")
            results.append((f"zstd_l22_dict", len(cm_d), len(cr_d), total_d))
        except Exception as e:
            print(f"dict training failed: {e}")

    # Score
    from prepare import MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE
    import math
    print(f"\n--- vs bz2 baseline {sb_total_bz2}B / score {score_bz2:.4f} ---")
    for name, cm, cr, total in results:
        rate = (MASK_BYTES + POSE_BYTES + model_bytes + total) / UNCOMPRESSED_SIZE
        score = 100*data['seg_dist'] + math.sqrt(10*data['pose_dist']) + 25*rate
        print(f"  {name}: {total}B score={score:.4f} delta={score-score_bz2:+.4f}")

    # Save
    import csv
    with open(OUTPUT_DIR / "e2_zstd_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["scheme", "sb_mask", "sb_rgb", "sb_total", "score", "delta_vs_bz2"])
        w.writerow(["bz2_baseline", data['sb_mask_bz2'], data['sb_rgb_bz2'], sb_total_bz2, score_bz2, 0])
        for name, cm, cr, total in results:
            rate = (MASK_BYTES + POSE_BYTES + model_bytes + total) / UNCOMPRESSED_SIZE
            score = 100*data['seg_dist'] + math.sqrt(10*data['pose_dist']) + 25*rate
            w.writerow([name, cm, cr, total, score, score-score_bz2])


if __name__ == "__main__":
    main()
