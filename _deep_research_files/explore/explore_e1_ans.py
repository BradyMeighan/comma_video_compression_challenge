#!/usr/bin/env python
"""
E1: Custom ANS coder for sidecar — Compass Rank 1 (highest expected impact).

Three independently-modeled streams:
  - Position stream: empirical 2D CDF, range-code via constriction.
  - Channel-id stream: 3-symbol arithmetic coding.
  - Delta stream: exp-Golomb (k=2) sign + magnitude, range-coded under Laplace.

This script LOADS the existing baseline patches (mask + channel-only RGB) and
RE-ENCODES them, comparing bytes vs bz2. Distortion is unchanged so only RATE
changes — score is recomputed analytically.
"""
import sys, os, pickle, struct, bz2
from pathlib import Path
from collections import Counter
import numpy as np
import constriction

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def encode_channel_only_ans(rgb_patches):
    """Encode the channel-only RGB sidecar using ANS for 3 streams.
    Layout (decoder needs to reverse):
      header: u16 num_pairs, u16 total_patches, then 3 separate ANS streams
    """
    if not rgb_patches:
        return b''

    # Collect all data into 3 streams
    pair_idxs = []; ks_per_pair = []
    xs = []; ys = []; chans = []; deltas = []
    for pi in sorted(rgb_patches.keys()):
        ps = rgb_patches[pi]
        pair_idxs.append(pi)
        ks_per_pair.append(len(ps))
        for (x, y, c, d) in ps:
            xs.append(x)
            ys.append(y)
            chans.append(c)
            deltas.append(d)

    n_pairs = len(pair_idxs)
    n_patches = len(xs)

    # === STREAM 1: positions (encode as flattened 1D position via empirical CDF) ===
    # Flatten (x, y) → idx = y * 1164 + x. Use empirical frequency model.
    pos_idx = np.array([ys[i] * 1164 + xs[i] for i in range(n_patches)], dtype=np.int64)
    cnt = Counter(pos_idx.tolist())
    pos_alphabet = sorted(cnt.keys())
    pos_to_rank = {pos: rank for rank, pos in enumerate(pos_alphabet)}
    pos_ranks = np.array([pos_to_rank[p] for p in pos_idx], dtype=np.int32)

    # Build PMF for pos_ranks
    n_alphabet = len(pos_alphabet)
    counts = np.array([cnt[pos_alphabet[i]] for i in range(n_alphabet)], dtype=np.float64)
    pmf_pos = counts / counts.sum()
    # constriction wants probabilities; we use Categorical
    pos_model = constriction.stream.model.Categorical(pmf_pos.astype(np.float32), perfect=False)

    # === STREAM 2: channel ids (3-symbol arithmetic) ===
    chan_arr = np.array(chans, dtype=np.int32)
    chan_cnt = np.bincount(chan_arr, minlength=3).astype(np.float64)
    chan_pmf = chan_cnt / chan_cnt.sum()
    chan_model = constriction.stream.model.Categorical(chan_pmf.astype(np.float32), perfect=False)

    # === STREAM 3: deltas (signed int8) ===
    # Encode magnitude + sign separately. Magnitude as Categorical over 0-127.
    delta_arr = np.array(deltas, dtype=np.int32)
    sign_arr = (delta_arr < 0).astype(np.int32)
    mag_arr = np.abs(delta_arr).astype(np.int32)
    mag_cnt = np.bincount(mag_arr, minlength=128).astype(np.float64)
    mag_cnt += 1e-6  # smoothing
    mag_pmf = mag_cnt / mag_cnt.sum()
    mag_model = constriction.stream.model.Categorical(mag_pmf.astype(np.float32), perfect=False)
    sign_cnt = np.bincount(sign_arr, minlength=2).astype(np.float64) + 1e-6
    sign_pmf = sign_cnt / sign_cnt.sum()
    sign_model = constriction.stream.model.Categorical(sign_pmf.astype(np.float32), perfect=False)

    # === Encode each stream with ANS ===
    enc_pos = constriction.stream.queue.RangeEncoder()
    enc_pos.encode(pos_ranks.astype(np.int32), pos_model)
    bytes_pos = enc_pos.get_compressed().tobytes()

    enc_chan = constriction.stream.queue.RangeEncoder()
    enc_chan.encode(chan_arr, chan_model)
    bytes_chan = enc_chan.get_compressed().tobytes()

    enc_mag = constriction.stream.queue.RangeEncoder()
    enc_mag.encode(mag_arr, mag_model)
    bytes_mag = enc_mag.get_compressed().tobytes()

    enc_sign = constriction.stream.queue.RangeEncoder()
    enc_sign.encode(sign_arr, sign_model)
    bytes_sign = enc_sign.get_compressed().tobytes()

    # === Assemble header + per-pair structure + streams ===
    parts = []
    # Pair structure: num_pairs, then for each pair: u16 idx, u16 K
    parts.append(struct.pack("<H", n_pairs))
    for pi, k in zip(pair_idxs, ks_per_pair):
        parts.append(struct.pack("<HH", pi, k))

    # Position alphabet (variable length): u32 n_alphabet, then alphabet entries (u16 x, u16 y)
    parts.append(struct.pack("<I", n_alphabet))
    for pos in pos_alphabet:
        x = pos % 1164; y = pos // 1164
        parts.append(struct.pack("<HH", x, y))

    # PMFs (floats): for compact storage use fixed-point u16 normalized
    # For now, write counts as u16 (limited precision) — could improve
    parts.append(struct.pack("<I", n_alphabet))
    for c in counts:
        parts.append(struct.pack("<H", min(int(c), 65535)))

    parts.append(struct.pack("<3H", *[min(int(c), 65535) for c in chan_cnt]))
    parts.append(struct.pack("<128H", *[min(int(c), 65535) for c in mag_cnt]))
    # sign uses default

    # Stream bytes
    parts.append(struct.pack("<I", len(bytes_pos)))
    parts.append(bytes_pos)
    parts.append(struct.pack("<I", len(bytes_chan)))
    parts.append(bytes_chan)
    parts.append(struct.pack("<I", len(bytes_mag)))
    parts.append(bytes_mag)
    parts.append(struct.pack("<I", len(bytes_sign)))
    parts.append(bytes_sign)

    raw = b''.join(parts)
    # Final ZIP DEFLATE (the actual archive uses ZIP)
    final_compressed = bz2.compress(raw, compresslevel=9)
    return raw, final_compressed


def encode_mask_ans(mask_patches):
    """Encode mask sidecar with ANS."""
    if not mask_patches:
        return b'', b''
    pair_idxs = []; ks = []; xs = []; ys = []; classes = []
    for pi in sorted(mask_patches.keys()):
        ps = mask_patches[pi]
        pair_idxs.append(pi); ks.append(len(ps))
        for (x, y, c) in ps:
            xs.append(x); ys.append(y); classes.append(c)

    # Position stream (1D index)
    pos_idx = np.array([ys[i] * 512 + xs[i] for i in range(len(xs))], dtype=np.int64)  # mask is 384x512
    cnt = Counter(pos_idx.tolist())
    pos_alphabet = sorted(cnt.keys())
    pos_to_rank = {p: r for r, p in enumerate(pos_alphabet)}
    pos_ranks = np.array([pos_to_rank[p] for p in pos_idx], dtype=np.int32)
    counts = np.array([cnt[pos_alphabet[i]] for i in range(len(pos_alphabet))], dtype=np.float64)
    pmf = counts / counts.sum()
    pos_model = constriction.stream.model.Categorical(pmf.astype(np.float32), perfect=False)

    # Class stream (5 classes)
    class_arr = np.array(classes, dtype=np.int32)
    cls_cnt = np.bincount(class_arr, minlength=5).astype(np.float64) + 1e-6
    cls_pmf = cls_cnt / cls_cnt.sum()
    cls_model = constriction.stream.model.Categorical(cls_pmf.astype(np.float32), perfect=False)

    # Encode
    enc_pos = constriction.stream.queue.RangeEncoder()
    enc_pos.encode(pos_ranks, pos_model)
    bytes_pos = enc_pos.get_compressed().tobytes()
    enc_cls = constriction.stream.queue.RangeEncoder()
    enc_cls.encode(class_arr, cls_model)
    bytes_cls = enc_cls.get_compressed().tobytes()

    parts = []
    parts.append(struct.pack("<H", len(pair_idxs)))
    for pi, k in zip(pair_idxs, ks):
        parts.append(struct.pack("<HH", pi, k))
    parts.append(struct.pack("<I", len(pos_alphabet)))
    for p in pos_alphabet:
        x = p % 512; y = p // 512
        parts.append(struct.pack("<HH", x, y))
    parts.append(struct.pack("<I", len(pos_alphabet)))
    for c in counts:
        parts.append(struct.pack("<H", min(int(c), 65535)))
    parts.append(struct.pack("<5H", *[min(int(c), 65535) for c in cls_cnt]))
    parts.append(struct.pack("<I", len(bytes_pos))); parts.append(bytes_pos)
    parts.append(struct.pack("<I", len(bytes_cls))); parts.append(bytes_cls)
    raw = b''.join(parts)
    final = bz2.compress(raw, compresslevel=9)
    return raw, final


def main():
    pkl_path = OUTPUT_DIR / "baseline_patches.pkl"
    if not pkl_path.exists():
        print(f"ERROR: missing {pkl_path}. Run explore_baseline_builder.py first.")
        sys.exit(1)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    mask_patches = data['mask_patches']
    rgb_patches = data['rgb_patches']
    sb_mask_bz2 = data['sb_mask_bz2']
    sb_rgb_bz2 = data['sb_rgb_bz2']
    score_bz2 = data['score']
    model_bytes = data['model_bytes']

    print(f"=== E1: ANS coder vs bz2 ===")
    print(f"Loaded mask_patches: {len(mask_patches)} pairs")
    print(f"Loaded rgb_patches: {len(rgb_patches)} pairs, "
          f"{sum(len(v) for v in rgb_patches.values())} total patches")
    print(f"BZ2 sizes: sb_mask={sb_mask_bz2}B sb_rgb={sb_rgb_bz2}B sb_total={sb_mask_bz2+sb_rgb_bz2}B")
    print(f"BZ2 score: {score_bz2:.4f}\n")

    # Encode with ANS
    raw_mask, ans_mask = encode_mask_ans(mask_patches)
    raw_rgb, ans_rgb = encode_channel_only_ans(rgb_patches)

    print(f"ANS sizes: sb_mask={len(ans_mask)}B sb_rgb={len(ans_rgb)}B sb_total={len(ans_mask)+len(ans_rgb)}B")
    print(f"Raw (pre-bz2): mask={len(raw_mask)}B rgb={len(raw_rgb)}B total={len(raw_mask)+len(raw_rgb)}B")

    # Compute new score (distortion unchanged, only rate changes)
    from prepare import MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE
    import math
    sb_total_ans = len(ans_mask) + len(ans_rgb)
    rate = (MASK_BYTES + POSE_BYTES + model_bytes + sb_total_ans) / UNCOMPRESSED_SIZE
    score_ans = 100*data['seg_dist'] + math.sqrt(10*data['pose_dist']) + 25*rate
    print(f"\nANS score: {score_ans:.4f}")
    print(f"Delta vs bz2 baseline: {score_ans - score_bz2:+.4f}")
    print(f"Bytes saved: {(sb_mask_bz2+sb_rgb_bz2) - sb_total_ans}B "
          f"({100*(1 - sb_total_ans/(sb_mask_bz2+sb_rgb_bz2)):.1f}% reduction)")

    # Also check raw (unbz2'd)
    sb_raw = len(raw_mask) + len(raw_rgb)
    rate_raw = (MASK_BYTES + POSE_BYTES + model_bytes + sb_raw) / UNCOMPRESSED_SIZE
    score_raw = 100*data['seg_dist'] + math.sqrt(10*data['pose_dist']) + 25*rate_raw
    print(f"\nRaw (no final bz2) score: {score_raw:.4f}  bytes: {sb_raw}B")
    print(f"  Delta vs bz2: {score_raw - score_bz2:+.4f}")

    # Save results
    import csv
    csv_path = OUTPUT_DIR / "e1_ans_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["scheme", "sb_mask", "sb_rgb", "sb_total", "score", "delta_vs_bz2"])
        w.writerow(["bz2_baseline", sb_mask_bz2, sb_rgb_bz2, sb_mask_bz2+sb_rgb_bz2, score_bz2, 0])
        w.writerow(["ans_then_bz2", len(ans_mask), len(ans_rgb), sb_total_ans, score_ans, score_ans-score_bz2])
        w.writerow(["ans_raw", len(raw_mask), len(raw_rgb), sb_raw, score_raw, score_raw-score_bz2])
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
