"""Analyze seg map compression opportunities: temporal redundancy,
spatial downscale impact, context prediction, and pose vector contribution."""
import struct, bz2, lzma, zlib, time
import numpy as np
from pathlib import Path
from itertools import permutations

# ── Load current seg maps ────────────────────────────────────────────────
seg_path = Path("submissions/phase2/archive/0/seg.bin")
pose_path = Path("submissions/phase2/archive/0/pose.bin")

with open(seg_path, 'rb') as f:
    n, H, W, flags = struct.unpack('<IIII', f.read(16))
    has_perm = bool(flags & (1 << 4))
    is_ri = bool(flags & (1 << 3))
    compress_method = flags & 0x3

    perm_lut = None
    fwd_perm = None
    if has_perm:
        fwd_perm = list(f.read(5))
        inv_lut = np.zeros(5, dtype=np.uint8)
        for orig, enc in enumerate(fwd_perm):
            inv_lut[enc] = orig
        perm_lut = inv_lut

    raw_compressed = f.read()
    seg_compressed_size = 16 + (5 if has_perm else 0) + len(raw_compressed)
    raw = bz2.decompress(raw_compressed)
    if is_ri:
        seg_maps = np.frombuffer(raw, dtype=np.uint8).reshape(H, n, W)
        seg_maps = np.ascontiguousarray(seg_maps.transpose(1, 0, 2))
    else:
        seg_maps = np.frombuffer(raw, dtype=np.uint8).reshape(n, H, W)
    if perm_lut is not None:
        seg_maps = perm_lut[seg_maps]

pose_size = pose_path.stat().st_size

print(f"Seg maps: {seg_maps.shape} ({n} frames, {H}x{W})")
print(f"Current seg.bin: {seg_compressed_size:,} bytes ({seg_compressed_size/1024:.1f} KB)")
print(f"Current pose.bin: {pose_size:,} bytes ({pose_size/1024:.1f} KB)")
print(f"Current total: {(seg_compressed_size + pose_size + 60)/1024:.1f} KB")
print(f"Class distribution: {np.bincount(seg_maps.flatten(), minlength=5)}")
print()

# ── 1. TEMPORAL REDUNDANCY ANALYSIS ──────────────────────────────────────
print("=" * 60)
print("1. TEMPORAL REDUNDANCY ANALYSIS")
print("=" * 60)

identical_frames = 0
total_changed_pixels = 0
change_rates = []
for i in range(1, n):
    diff = seg_maps[i] != seg_maps[i - 1]
    changed = diff.sum()
    rate = changed / (H * W)
    change_rates.append(rate)
    if changed == 0:
        identical_frames += 1
    total_changed_pixels += changed

change_rates = np.array(change_rates)
print(f"Identical consecutive frames: {identical_frames}/{n-1} ({100*identical_frames/(n-1):.1f}%)")
print(f"Mean change rate: {change_rates.mean()*100:.3f}%")
print(f"Median change rate: {np.median(change_rates)*100:.3f}%")
print(f"Max change rate: {change_rates.max()*100:.3f}%")
print(f"Frames with <0.1% change: {(change_rates < 0.001).sum()}")
print(f"Frames with <1% change: {(change_rates < 0.01).sum()}")
print(f"Frames with <5% change: {(change_rates < 0.05).sum()}")

# Test: skip identical frames, store bitmask + unique frames only
unique_indices = [0]  # always store first frame
for i in range(1, n):
    if not np.array_equal(seg_maps[i], seg_maps[i - 1]):
        unique_indices.append(i)

print(f"\nUnique frames needed: {len(unique_indices)}/{n} ({100*len(unique_indices)/n:.1f}%)")

# Build: bitmask (1 bit per frame: 0=same as prev, 1=new) + unique frames
bitmask = np.zeros(n, dtype=np.uint8)
for idx in unique_indices:
    bitmask[idx] = 1
bitmask_bytes = np.packbits(bitmask).tobytes()

unique_data = seg_maps[unique_indices]
# Row-interleave the unique frames
n_unique = len(unique_indices)
ri_unique = np.ascontiguousarray(unique_data.transpose(1, 0, 2).reshape(-1, W))

# Apply best permutation
best_perm = fwd_perm if fwd_perm else list(range(5))
perm_arr = np.array(best_perm, dtype=np.uint8)
ri_perm = perm_arr[ri_unique]

unique_compressed = bz2.compress(ri_perm.tobytes(), 9)
total_skip = len(bitmask_bytes) + len(unique_compressed) + 20  # header
print(f"Skip-identical approach: {total_skip:,} bytes ({total_skip/1024:.1f} KB)")
print(f"  Bitmask: {len(bitmask_bytes)} bytes, Unique data: {len(unique_compressed):,} bytes")
print(f"  Savings vs current: {seg_compressed_size - total_skip:,} bytes ({100*(seg_compressed_size - total_skip)/seg_compressed_size:.1f}%)")

# Test: store XOR deltas for changed frames only (skip identical)
print("\n--- Delta encoding for changed frames ---")
delta_frames = [seg_maps[0]]
for i in range(1, n):
    if np.array_equal(seg_maps[i], seg_maps[i-1]):
        continue
    delta = seg_maps[i] ^ seg_maps[i-1]
    delta_frames.append(delta)

delta_arr = np.array(delta_frames, dtype=np.uint8)
ri_delta = np.ascontiguousarray(delta_arr.transpose(1, 0, 2).reshape(-1, W))
# Don't apply perm to XOR deltas (values go beyond 0-4)
delta_compressed = bz2.compress(ri_delta.tobytes(), 9)
total_delta = len(bitmask_bytes) + len(delta_compressed) + 20
print(f"Delta (XOR) + skip-identical: {total_delta:,} bytes ({total_delta/1024:.1f} KB)")

print()

# ── 2. CONTEXT PREDICTION ────────────────────────────────────────────────
print("=" * 60)
print("2. CONTEXT PREDICTION (predict from left + above + prev frame)")
print("=" * 60)

def context_predict(maps):
    """Predict each pixel from (left, above, previous-frame). Return residuals."""
    n, h, w = maps.shape
    residuals = np.zeros_like(maps)

    for i in range(n):
        for y in range(h):
            for x in range(w):
                # Simple predictor: use left neighbor, fall back to above, fall back to prev frame
                if x > 0:
                    pred = maps[i, y, x - 1]
                elif y > 0:
                    pred = maps[i, y - 1, x]
                elif i > 0:
                    pred = maps[i - 1, y, x]
                else:
                    pred = 0
                residuals[i, y, x] = maps[i, y, x] if maps[i, y, x] != pred else 0
    return residuals

# Fast vectorized version of context prediction
def context_predict_fast(maps):
    """Vectorized context prediction: use previous frame as predictor."""
    n, h, w = maps.shape
    residuals = np.zeros_like(maps)
    residuals[0] = maps[0]
    for i in range(1, n):
        match = maps[i] == maps[i-1]
        residuals[i] = np.where(match, 0, maps[i])
    return residuals

def left_predict_fast(maps):
    """Predict from left neighbor."""
    n, h, w = maps.shape
    residuals = np.zeros_like(maps)
    residuals[:, :, 0] = maps[:, :, 0]  # first column: no left neighbor
    match = maps[:, :, 1:] == maps[:, :, :-1]
    residuals[:, :, 1:] = np.where(match, 0, maps[:, :, 1:])
    return residuals

print("Testing temporal prediction (prev frame)...")
t0 = time.time()
resid_temporal = context_predict_fast(seg_maps)
zeros_temporal = (resid_temporal == 0).sum() / resid_temporal.size
print(f"  Zero rate: {zeros_temporal*100:.2f}% (took {time.time()-t0:.1f}s)")

# Compress the residuals
ri_resid = np.ascontiguousarray(resid_temporal.transpose(1, 0, 2).reshape(-1, W))
resid_compressed = bz2.compress(ri_resid.tobytes(), 9)
print(f"  Temporal residuals + bz2: {len(resid_compressed):,} bytes ({len(resid_compressed)/1024:.1f} KB)")

print("\nTesting left prediction...")
resid_left = left_predict_fast(seg_maps)
zeros_left = (resid_left == 0).sum() / resid_left.size
print(f"  Zero rate: {zeros_left*100:.2f}%")
ri_resid_left = np.ascontiguousarray(resid_left.transpose(1, 0, 2).reshape(-1, W))
resid_left_compressed = bz2.compress(ri_resid_left.tobytes(), 9)
print(f"  Left residuals + bz2: {len(resid_left_compressed):,} bytes ({len(resid_left_compressed)/1024:.1f} KB)")

# Combined: temporal then left
print("\nTesting temporal + left prediction...")
resid_combined = left_predict_fast(context_predict_fast(seg_maps))
zeros_combined = (resid_combined == 0).sum() / resid_combined.size
print(f"  Zero rate: {zeros_combined*100:.2f}%")
ri_resid_comb = np.ascontiguousarray(resid_combined.transpose(1, 0, 2).reshape(-1, W))
resid_comb_compressed = bz2.compress(ri_resid_comb.tobytes(), 9)
print(f"  Combined residuals + bz2: {len(resid_comb_compressed):,} bytes ({len(resid_comb_compressed)/1024:.1f} KB)")

# XOR-based prediction (different from value-based)
print("\nTesting XOR temporal prediction...")
xor_maps = np.zeros_like(seg_maps)
xor_maps[0] = seg_maps[0]
for i in range(1, n):
    xor_maps[i] = seg_maps[i] ^ seg_maps[i-1]
ri_xor = np.ascontiguousarray(xor_maps.transpose(1, 0, 2).reshape(-1, W))
xor_compressed = bz2.compress(ri_xor.tobytes(), 9)
print(f"  XOR temporal + bz2: {len(xor_compressed):,} bytes ({len(xor_compressed)/1024:.1f} KB)")

print()

# ── 3. SPATIAL DOWNSCALE ─────────────────────────────────────────────────
print("=" * 60)
print("3. SPATIAL DOWNSCALE")
print("=" * 60)

for scale_name, sH, sW in [("3/4", 288, 384), ("2/3", 256, 342), ("1/2", 192, 256), ("3/8", 144, 192)]:
    # Nearest-neighbor downscale
    step_h = H / sH
    step_w = W / sW
    indices_h = (np.arange(sH) * step_h + step_h/2).astype(int).clip(0, H-1)
    indices_w = (np.arange(sW) * step_w + step_w/2).astype(int).clip(0, W-1)
    downscaled = seg_maps[:, indices_h][:, :, indices_w]

    # Upscale back to original
    up_indices_h = (np.arange(H) * sH / H).astype(int).clip(0, sH-1)
    up_indices_w = (np.arange(W) * sW / W).astype(int).clip(0, sW-1)
    upscaled = downscaled[:, up_indices_h][:, :, up_indices_w]

    # Measure error
    pixel_error = (upscaled != seg_maps).mean()
    segnet_penalty = 100 * pixel_error  # approximate SegNet score penalty

    # Compress downscaled
    ri_down = np.ascontiguousarray(downscaled.transpose(1, 0, 2).reshape(-1, sW))
    perm_down = perm_arr[ri_down]
    down_compressed = bz2.compress(perm_down.tobytes(), 9)

    rate_savings = (seg_compressed_size - len(down_compressed) - 20) / 37_545_489
    rate_score_savings = 25 * rate_savings

    print(f"Scale {scale_name} ({sW}x{sH}):")
    print(f"  Compressed: {len(down_compressed):,} bytes ({len(down_compressed)/1024:.1f} KB)")
    print(f"  Pixel error: {pixel_error*100:.3f}% -> SegNet penalty: +{segnet_penalty:.3f}")
    print(f"  Rate savings: {rate_score_savings:.4f} score points")
    print(f"  Net score change: {segnet_penalty - rate_score_savings:+.4f} {'(WORSE)' if segnet_penalty > rate_score_savings else '(BETTER)'}")

print()

# ── 4. POSE VECTOR ANALYSIS ─────────────────────────────────────────────
print("=" * 60)
print("4. POSE VECTOR ANALYSIS")
print("=" * 60)

with open(pose_path, 'rb') as f:
    header = f.read(12)
    n_p, d, pose_flags = struct.unpack('<III', header)
    pose_method = pose_flags & 0x3
    is_f16 = bool(pose_flags & (1 << 2))
    raw_pose = bz2.decompress(f.read())
    if is_f16:
        pose_vectors = np.frombuffer(raw_pose, dtype=np.float16).astype(np.float32).reshape(n_p, d)
    else:
        pose_vectors = np.frombuffer(raw_pose, dtype=np.float32).reshape(n_p, d)

print(f"Pose vectors: {pose_vectors.shape}")
print(f"Pose.bin size: {pose_size} bytes")
print(f"Pose value range: [{pose_vectors.min():.4f}, {pose_vectors.max():.4f}]")
print(f"Pose value mean: {pose_vectors.mean():.6f}, std: {pose_vectors.std():.6f}")
print(f"Rate contribution of pose.bin: {25 * pose_size / 37_545_489:.6f}")
print(f"If dropped: saves {25 * pose_size / 37_545_489:.4f} score from rate")
print()

# ── 5. ALTERNATIVE COMPRESSORS ON BEST ENCODING ─────────────────────────
print("=" * 60)
print("5. COMPRESSOR COMPARISON (on current row-interleaved + perm data)")
print("=" * 60)

ri_data = np.ascontiguousarray(seg_maps.transpose(1, 0, 2).reshape(-1, W))
best_data = perm_arr[ri_data].tobytes()

compressors = {
    'bz2-9': lambda d: bz2.compress(d, 9),
    'zlib-9': lambda d: zlib.compress(d, 9),
    'lzma-9': lambda d: lzma.compress(d, preset=9),
    'lzma-extreme': lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME),
}

# Try lzma with different filters
try:
    lzma_filters_delta = [
        {"id": lzma.FILTER_DELTA, "dist": W},  # delta filter with row width
        {"id": lzma.FILTER_LZMA2, "preset": 9}
    ]
    compressors['lzma-delta-row'] = lambda d: lzma.compress(d, format=lzma.FORMAT_RAW, filters=lzma_filters_delta)
except:
    pass

try:
    lzma_filters_delta1 = [
        {"id": lzma.FILTER_DELTA, "dist": 1},
        {"id": lzma.FILTER_LZMA2, "preset": 9}
    ]
    compressors['lzma-delta-1'] = lambda d: lzma.compress(d, format=lzma.FORMAT_RAW, filters=lzma_filters_delta1)
except:
    pass

for name, compress_fn in compressors.items():
    t0 = time.time()
    try:
        result = compress_fn(best_data)
        elapsed = time.time() - t0
        print(f"  {name:20s}: {len(result):>8,} bytes ({len(result)/1024:.1f} KB) [{elapsed:.1f}s]")
    except Exception as e:
        print(f"  {name:20s}: FAILED ({e})")

print()

# ── 6. LOSSY: SMOOTH SMALL REGIONS ──────────────────────────────────────
print("=" * 60)
print("6. LOSSY: MORPHOLOGICAL SMOOTHING (remove tiny isolated regions)")
print("=" * 60)

try:
    from scipy import ndimage

    for min_size in [2, 4, 8, 16, 32]:
        smoothed = seg_maps.copy()
        total_changed = 0
        for i in range(n):
            frame = seg_maps[i]
            for cls in range(5):
                mask = frame == cls
                labeled, num_features = ndimage.label(mask)
                for region_id in range(1, num_features + 1):
                    region = labeled == region_id
                    if region.sum() < min_size:
                        # Replace with most common neighbor class
                        dilated = ndimage.binary_dilation(region, iterations=1)
                        border = dilated & ~region
                        if border.sum() > 0:
                            neighbor_vals = frame[border]
                            replacement = np.bincount(neighbor_vals, minlength=5).argmax()
                            smoothed[i][region] = replacement
                            total_changed += region.sum()

        pixel_error = (smoothed != seg_maps).mean()
        segnet_penalty = 100 * pixel_error

        # Compress smoothed
        ri_smooth = np.ascontiguousarray(smoothed.transpose(1, 0, 2).reshape(-1, W))
        smooth_perm = perm_arr[ri_smooth]
        smooth_compressed = bz2.compress(smooth_perm.tobytes(), 9)

        rate_savings = (seg_compressed_size - len(smooth_compressed) - 20) / 37_545_489
        rate_score_savings = 25 * rate_savings

        print(f"Min region size {min_size:>3d}:")
        print(f"  Changed pixels: {total_changed:,} ({pixel_error*100:.4f}%)")
        print(f"  Compressed: {len(smooth_compressed):,} bytes ({len(smooth_compressed)/1024:.1f} KB)")
        print(f"  SegNet penalty: +{segnet_penalty:.4f}, Rate savings: -{rate_score_savings:.4f}")
        print(f"  Net: {segnet_penalty - rate_score_savings:+.4f} {'(WORSE)' if segnet_penalty > rate_score_savings else '(BETTER)'}")

except ImportError:
    print("scipy not available, skipping morphological analysis")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Current archive: {(seg_compressed_size + pose_size + 60)/1024:.1f} KB")
print(f"Current rate score: {25 * (seg_compressed_size + pose_size + 60) / 37_545_489:.4f}")
