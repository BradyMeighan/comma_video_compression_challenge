"""Corrected context prediction analysis.
The original had a bug: value 0 was ambiguous (class 0 vs "matches prediction").
Fix: store (value + 1) for mismatches, 0 for matches."""
import struct, bz2, time
import numpy as np
from pathlib import Path

# ── Load seg maps ────────────────────────────────────────────────────────
seg_path = Path("submissions/phase2/archive/0/seg.bin")
with open(seg_path, 'rb') as f:
    n, H, W, flags = struct.unpack('<IIII', f.read(16))
    has_perm = bool(flags & (1 << 4))
    is_ri = bool(flags & (1 << 3))
    fwd_perm = list(f.read(5)) if has_perm else list(range(5))
    inv_lut = np.zeros(5, dtype=np.uint8)
    for orig, enc in enumerate(fwd_perm):
        inv_lut[enc] = orig
    raw = bz2.decompress(f.read())
    if is_ri:
        seg_maps = np.frombuffer(raw, dtype=np.uint8).reshape(H, n, W)
        seg_maps = np.ascontiguousarray(seg_maps.transpose(1, 0, 2))
    else:
        seg_maps = np.frombuffer(raw, dtype=np.uint8).reshape(n, H, W)
    if has_perm:
        seg_maps = inv_lut[seg_maps]

seg_size_current = seg_path.stat().st_size
print(f"Loaded: {seg_maps.shape}, current seg.bin: {seg_size_current:,} bytes ({seg_size_current/1024:.1f} KB)")
print(f"Current raw row-interleaved+perm bz2:")
ri_raw = np.ascontiguousarray(seg_maps.transpose(1, 0, 2).reshape(-1, W))
perm_arr = np.array(fwd_perm, dtype=np.uint8)
baseline = bz2.compress(perm_arr[ri_raw].tobytes(), 9)
print(f"  {len(baseline):,} bytes ({len(baseline)/1024:.1f} KB)\n")

# ── Correct context prediction ──────────────────────────────────────────
# Encode: 0 = matches prediction, (actual_class + 1) = mismatch
# Decode: if val == 0 use prediction, else val - 1 is actual class

def temporal_predict_correct(maps):
    """Predict from previous frame. 0=match, val+1=mismatch."""
    n, h, w = maps.shape
    out = np.zeros_like(maps)
    out[0] = maps[0] + 1  # first frame: no prediction, store raw+1
    for i in range(1, n):
        match = maps[i] == maps[i - 1]
        out[i] = np.where(match, 0, maps[i] + 1)
    return out

def left_predict_correct(maps):
    """Predict from left neighbor. 0=match, val+1=mismatch."""
    n, h, w = maps.shape
    out = np.zeros_like(maps)
    out[:, :, 0] = maps[:, :, 0] + 1  # first col: no prediction
    match = maps[:, :, 1:] == maps[:, :, :-1]
    out[:, :, 1:] = np.where(match, 0, maps[:, :, 1:] + 1)
    return out

def above_predict_correct(maps):
    """Predict from above neighbor. 0=match, val+1=mismatch."""
    n, h, w = maps.shape
    out = np.zeros_like(maps)
    out[:, 0, :] = maps[:, 0, :] + 1  # first row: no prediction
    match = maps[:, 1:, :] == maps[:, :-1, :]
    out[:, 1:, :] = np.where(match, 0, maps[:, 1:, :] + 1)
    return out

def verify_roundtrip(maps, encode_fn, decode_fn, name):
    """Verify encode->decode is lossless."""
    encoded = encode_fn(maps)
    decoded = decode_fn(encoded, maps.shape)
    if np.array_equal(decoded, maps):
        print(f"  OK: {name} roundtrip verified lossless")
        return True
    else:
        diff = (decoded != maps).sum()
        print(f"  FAIL: {name} roundtrip LOSSY: {diff} pixel errors")
        return False

def decode_temporal(encoded, shape):
    n, h, w = shape
    out = np.zeros(shape, dtype=np.uint8)
    out[0] = encoded[0] - 1
    for i in range(1, n):
        match = encoded[i] == 0
        out[i] = np.where(match, out[i-1], encoded[i] - 1)
    return out

def decode_left(encoded, shape):
    n, h, w = shape
    out = np.zeros(shape, dtype=np.uint8)
    out[:, :, 0] = encoded[:, :, 0] - 1
    for x in range(1, w):
        match = encoded[:, :, x] == 0
        out[:, :, x] = np.where(match, out[:, :, x-1], encoded[:, :, x] - 1)
    return out

def decode_above(encoded, shape):
    n, h, w = shape
    out = np.zeros(shape, dtype=np.uint8)
    out[:, 0, :] = encoded[:, 0, :] - 1
    for y in range(1, h):
        match = encoded[:, y, :] == 0
        out[:, y, :] = np.where(match, out[:, y-1, :], encoded[:, y, :] - 1)
    return out

def test_encoding(maps, encode_fn, decode_fn, name, apply_perm=False):
    """Test an encoding scheme: encode, verify, compress, report."""
    print(f"\n--- {name} ---")
    encoded = encode_fn(maps)

    # Verify lossless
    verify_roundtrip(maps, encode_fn, decode_fn, name)

    zero_rate = (encoded == 0).sum() / encoded.size
    print(f"  Zero rate: {zero_rate*100:.2f}%")
    print(f"  Value distribution: {np.bincount(encoded.flatten(), minlength=6)}")

    # Try different layouts
    results = {}

    # Row-interleaved
    ri = np.ascontiguousarray(encoded.transpose(1, 0, 2).reshape(-1, W))
    c = bz2.compress(ri.tobytes(), 9)
    results['ri-bz2'] = len(c)
    print(f"  Row-interleaved bz2: {len(c):,} bytes ({len(c)/1024:.1f} KB)")

    # Flat
    flat = encoded.tobytes()
    c2 = bz2.compress(flat, 9)
    results['flat-bz2'] = len(c2)
    print(f"  Flat bz2: {len(c2):,} bytes ({len(c2)/1024:.1f} KB)")

    # With permutation on encoded (remap 0-5 values)
    from itertools import permutations
    best_size = min(results.values())
    # Quick perm search on the 6 values (720 perms)
    print(f"  Searching 720 permutations of 6 values...")
    best_perm_size = best_size
    best_p = None
    for p in permutations(range(6)):
        lut = np.array(p, dtype=np.uint8)
        remapped = lut[ri]
        sz = len(bz2.compress(remapped.tobytes(), 9))
        if sz < best_perm_size:
            best_perm_size = sz
            best_p = p
    if best_p:
        print(f"  Best perm {best_p}: {best_perm_size:,} bytes ({best_perm_size/1024:.1f} KB)")
    else:
        print(f"  No perm improvement found")

    best_final = min(best_perm_size, best_size)
    savings = seg_size_current - best_final - 25  # header overhead
    rate_improvement = 25 * savings / 37_545_489
    print(f"  Best: {best_final:,} bytes ({best_final/1024:.1f} KB)")
    print(f"  vs current: {savings:+,} bytes ({rate_improvement:+.4f} rate score)")

    return best_final

# ── Run all tests ────────────────────────────────────────────────────────
print("=" * 60)
print("CORRECTED CONTEXT PREDICTION ANALYSIS")
print("=" * 60)

test_encoding(seg_maps, temporal_predict_correct, decode_temporal, "Temporal (prev frame)")
test_encoding(seg_maps, left_predict_correct, decode_left, "Left neighbor")
test_encoding(seg_maps, above_predict_correct, decode_above, "Above neighbor")

# ── Combined: temporal then left ────────────────────────────────────────
print("\n--- Combined: temporal then left ---")
# First apply temporal, then on the residuals apply left prediction
temp_encoded = temporal_predict_correct(seg_maps)
# For combined, left-predict the temporal residuals (values 0-5)
# This is trickier - we're predicting in the residual domain
# Simple approach: just predict from left neighbor in the encoded domain
comb = np.zeros_like(temp_encoded)
comb[:, :, 0] = temp_encoded[:, :, 0]  # first col: store as-is
for x in range(1, W):
    match = temp_encoded[:, :, x] == temp_encoded[:, :, x-1]
    # Use 6 as "matches left" sentinel (values are 0-5, so 6 is safe)
    comb[:, :, x] = np.where(match, 6, temp_encoded[:, :, x])

zero_rate = (comb == 0).sum() / comb.size
six_rate = (comb == 6).sum() / comb.size
print(f"  Zero rate (temporal match): {zero_rate*100:.2f}%")
print(f"  Six rate (left match): {six_rate*100:.2f}%")
print(f"  Total predicted: {(zero_rate + six_rate)*100:.2f}%")

ri_comb = np.ascontiguousarray(comb.transpose(1, 0, 2).reshape(-1, W))
c_comb = bz2.compress(ri_comb.tobytes(), 9)
print(f"  Combined ri-bz2: {len(c_comb):,} bytes ({len(c_comb)/1024:.1f} KB)")

# Verify roundtrip
decoded_temp = decode_temporal(temp_encoded, seg_maps.shape)
assert np.array_equal(decoded_temp, seg_maps), "Temporal decode failed"

# Decode combined
decoded_comb_temp = np.zeros_like(comb)
decoded_comb_temp[:, :, 0] = comb[:, :, 0]
for x in range(1, W):
    is_left_match = comb[:, :, x] == 6
    decoded_comb_temp[:, :, x] = np.where(is_left_match, decoded_comb_temp[:, :, x-1], comb[:, :, x])
decoded_comb = decode_temporal(decoded_comb_temp, seg_maps.shape)
if np.array_equal(decoded_comb, seg_maps):
    print(f"  OK: Combined roundtrip verified lossless")
else:
    print(f"  FAIL: Combined roundtrip LOSSY: {(decoded_comb != seg_maps).sum()} errors")

savings = seg_size_current - len(c_comb) - 25
rate_improvement = 25 * savings / 37_545_489
print(f"  vs current: {savings:+,} bytes ({rate_improvement:+.4f} rate score)")

# ── Bitplane separation on prediction residuals ─────────────────────────
print("\n--- Bitplane separation on temporal residuals ---")
temp_enc = temporal_predict_correct(seg_maps)  # values 0-5, need 3 bits
bit0 = ((temp_enc >> 0) & 1).astype(np.uint8)
bit1 = ((temp_enc >> 1) & 1).astype(np.uint8)
bit2 = ((temp_enc >> 2) & 1).astype(np.uint8)

# Pack each bitplane and compress separately
planes_total = 0
for i, plane in enumerate([bit0, bit1, bit2]):
    packed = np.packbits(plane.flatten())
    compressed = bz2.compress(packed.tobytes(), 9)
    planes_total += len(compressed)
    print(f"  Bit {i}: {len(compressed):,} bytes ({plane.mean()*100:.1f}% ones)")

print(f"  Bitplane total: {planes_total:,} bytes ({planes_total/1024:.1f} KB)")

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: Best approaches vs current")
print("=" * 60)
print(f"Current seg.bin: {seg_size_current:,} bytes ({seg_size_current/1024:.1f} KB)")
print(f"Current rate score: {25 * seg_size_current / 37_545_489:.4f}")
