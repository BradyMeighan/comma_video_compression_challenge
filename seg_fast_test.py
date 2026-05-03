"""Fast seg map compression comparison. No perm search, just raw results."""
import struct, bz2
import numpy as np
from pathlib import Path

seg_path = Path("submissions/phase2/archive/0/seg.bin")
with open(seg_path, 'rb') as f:
    n, H, W, flags = struct.unpack('<IIII', f.read(16))
    fwd_perm = list(f.read(5)) if (flags & (1<<4)) else list(range(5))
    inv_lut = np.zeros(5, dtype=np.uint8)
    for o, e in enumerate(fwd_perm):
        inv_lut[e] = o
    raw = bz2.decompress(f.read())
    sm = np.frombuffer(raw, dtype=np.uint8).reshape(H, n, W) if (flags & (1<<3)) else np.frombuffer(raw, dtype=np.uint8).reshape(n, H, W)
    if flags & (1<<3): sm = np.ascontiguousarray(sm.transpose(1, 0, 2))
    if flags & (1<<4): sm = inv_lut[sm]

SZ = seg_path.stat().st_size
U = 37_545_489
print(f"Current: {SZ:,} B ({SZ/1024:.1f} KB) | rate score: {25*SZ/U:.4f}")

def compress_ri(data):
    """Row-interleave and bz2 compress."""
    ri = np.ascontiguousarray(data.transpose(1, 0, 2).reshape(-1, data.shape[2]))
    return len(bz2.compress(ri.tobytes(), 9))

def report(name, sz, header=25):
    total = sz + header
    delta = SZ - total
    print(f"  {name:40s}: {total:>8,} B ({total/1024:>6.1f} KB) | delta: {delta:>+7,} B | rate: {25*total/U:.4f}")

# Baseline: current approach
report("Current (raw ri+perm bz2)", compress_ri(np.array(fwd_perm, dtype=np.uint8)[sm]), 0)

# 1. Temporal prediction: 0=match prev, val+1=mismatch
tp = np.zeros_like(sm)
tp[0] = sm[0] + 1
for i in range(1, n):
    m = sm[i] == sm[i-1]
    tp[i] = np.where(m, 0, sm[i] + 1)
report("Temporal pred (0=match, v+1=miss)", compress_ri(tp))

# 2. Left prediction
lp = np.zeros_like(sm)
lp[:,:,0] = sm[:,:,0] + 1
m = sm[:,:,1:] == sm[:,:,:-1]
lp[:,:,1:] = np.where(m, 0, sm[:,:,1:] + 1)
report("Left pred", compress_ri(lp))

# 3. Above prediction
ap = np.zeros_like(sm)
ap[:,0,:] = sm[:,0,:] + 1
m = sm[:,1:,:] == sm[:,:-1,:]
ap[:,1:,:] = np.where(m, 0, sm[:,1:,:] + 1)
report("Above pred", compress_ri(ap))

# 4. Temporal then left (two-stage)
# temporal first
tp2 = np.zeros_like(sm)
tp2[0] = sm[0] + 1
for i in range(1, n):
    m = sm[i] == sm[i-1]
    tp2[i] = np.where(m, 0, sm[i] + 1)
# then left on temporal residuals (vals 0-5, use 6 as left-match sentinel)
tl = np.zeros((n, H, W), dtype=np.uint8)
tl[:,:,0] = tp2[:,:,0]
m = tp2[:,:,1:] == tp2[:,:,:-1]
tl[:,:,1:] = np.where(m, 6, tp2[:,:,1:])
report("Temporal+Left pred", compress_ri(tl))

# 5. Left then temporal
lp2 = np.zeros_like(sm)
lp2[:,:,0] = sm[:,:,0] + 1
m = sm[:,:,1:] == sm[:,:,:-1]
lp2[:,:,1:] = np.where(m, 0, sm[:,:,1:] + 1)
lt = np.zeros_like(lp2)
lt[0] = lp2[0]
m = lp2[1:] == lp2[:-1]
lt[1:] = np.where(m, 6, lp2[1:])
report("Left+Temporal pred", compress_ri(lt))

# 6. Median-edge predictor (JPEG-LS style)
# Predict from left, above, and left+above-corner
me = np.zeros_like(sm)
me[:,0,:] = sm[:,0,:] + 1  # first row
me[:,:,0] = sm[:,:,0] + 1  # first col (overwrite corner, ok)
for y in range(1, H):
    for x in range(1, W):
        pass  # too slow for full video, skip
# Skip median-edge, too slow without vectorization

# 7. Different data layouts with temporal pred
print("\n  --- Layout variations on temporal pred ---")
tp_flat = tp.tobytes()
report("  Temporal pred FLAT bz2", len(bz2.compress(tp_flat, 9)), 0)

# Frame-major (all pixels of frame 0, then frame 1, etc)
tp_frame = tp.tobytes()
report("  Temporal pred frame-major bz2", len(bz2.compress(tp_frame, 9)), 0)

# Column-interleaved
ci = np.ascontiguousarray(tp.transpose(2, 0, 1).reshape(-1, H))
report("  Temporal pred col-interleaved bz2", len(bz2.compress(ci.tobytes(), 9)), 0)

# 8. Bitplane on temporal pred (values 0-5, 3 bits)
print("\n  --- Bitplane separation on temporal pred ---")
b0 = np.packbits(((tp >> 0) & 1).flatten().astype(np.uint8))
b1 = np.packbits(((tp >> 1) & 1).flatten().astype(np.uint8))
b2 = np.packbits(((tp >> 2) & 1).flatten().astype(np.uint8))
bp_total = sum(len(bz2.compress(b.tobytes(), 9)) for b in [b0, b1, b2])
report("  Bitplane (3 planes) bz2", bp_total, 0)

# 9. Binary mask approach: store "changed" bitmask + values for changed pixels only
print("\n  --- Binary mask + changed values ---")
changed_mask = np.zeros((n, H, W), dtype=np.uint8)
changed_vals = []
changed_mask[0] = 1  # first frame always "changed"
changed_vals.append(sm[0].flatten())
for i in range(1, n):
    diff = sm[i] != sm[i-1]
    changed_mask[i] = diff.astype(np.uint8)
    changed_vals.append(sm[i][diff])

mask_packed = np.packbits(changed_mask.flatten())
mask_compressed = len(bz2.compress(mask_packed.tobytes(), 9))
vals_arr = np.concatenate(changed_vals).astype(np.uint8)
vals_compressed = len(bz2.compress(vals_arr.tobytes(), 9))
report("  Binary mask + changed vals", mask_compressed + vals_compressed, 0)
print(f"    (mask: {mask_compressed:,} B, vals: {vals_compressed:,} B, "
      f"changed pixels: {len(vals_arr):,}/{n*H*W:,} = {100*len(vals_arr)/(n*H*W):.2f}%)")

# 10. Run-length on temporal pred
print("\n  --- RLE approaches ---")
tp_flat_arr = tp.flatten()
# Simple RLE: (value, count) pairs
runs = []
cur_val = tp_flat_arr[0]
cur_count = 1
for v in tp_flat_arr[1:]:
    if v == cur_val and cur_count < 255:
        cur_count += 1
    else:
        runs.append((cur_val, cur_count))
        cur_val = v
        cur_count = 1
runs.append((cur_val, cur_count))
rle_data = np.array(runs, dtype=np.uint8).tobytes()
rle_compressed = len(bz2.compress(rle_data, 9))
report("  RLE on temporal pred + bz2", rle_compressed, 0)
print(f"    ({len(runs):,} runs from {len(tp_flat_arr):,} pixels)")

print(f"\nDone.")
