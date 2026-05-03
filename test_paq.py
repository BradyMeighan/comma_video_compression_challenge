"""Test paq (paq9a) vs bz2 on seg map data."""
import struct, bz2, time
import numpy as np
from pathlib import Path
import paq

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

perm_arr = np.array(fwd_perm, dtype=np.uint8)
ri = np.ascontiguousarray(sm.transpose(1, 0, 2).reshape(-1, W))
ri_perm = perm_arr[ri]
data = ri_perm.tobytes()

print(f"Raw data: {len(data):,} bytes ({len(data)/1024/1024:.1f} MB)")

# bz2 baseline
t0 = time.time()
bz2_result = bz2.compress(data, 9)
bz2_time = time.time() - t0
print(f"bz2-9:  {len(bz2_result):>10,} bytes ({len(bz2_result)/1024:.1f} KB) [{bz2_time:.1f}s compress]")

# paq - test on chunks since full data might be very slow
# First test a small chunk to estimate speed
chunk_size = len(data) // 100  # 1% of data
chunk = data[:chunk_size]

print(f"\nTesting paq on 1% chunk ({chunk_size:,} bytes)...")
t0 = time.time()
paq_chunk = paq.compress(chunk)
paq_chunk_time = time.time() - t0
print(f"  paq chunk: {len(paq_chunk):,} bytes [{paq_chunk_time:.1f}s]")

bz2_chunk = bz2.compress(chunk, 9)
print(f"  bz2 chunk: {len(bz2_chunk):,} bytes")
print(f"  paq/bz2 ratio: {len(paq_chunk)/len(bz2_chunk):.3f}")

# Verify roundtrip
t0 = time.time()
rt = paq.decompress(paq_chunk)
decomp_time = time.time() - t0
assert rt == chunk, "Roundtrip failed!"
print(f"  Decompress: {decomp_time:.1f}s")

# Estimate full compression time
est_full = paq_chunk_time * 100
print(f"\nEstimated full compression time: {est_full:.0f}s ({est_full/60:.1f} min)")
est_full_size = len(paq_chunk) * 100
print(f"Estimated full compressed size: {est_full_size:,} bytes ({est_full_size/1024:.1f} KB)")

# If chunk looks promising and won't take forever, try 10%
if paq_chunk_time < 5:
    chunk10 = data[:len(data)//10]
    print(f"\nTesting paq on 10% ({len(chunk10):,} bytes)...")
    t0 = time.time()
    paq10 = paq.compress(chunk10)
    t10 = time.time() - t0
    bz2_10 = bz2.compress(chunk10, 9)
    print(f"  paq: {len(paq10):,} bytes [{t10:.1f}s]")
    print(f"  bz2: {len(bz2_10):,} bytes")
    print(f"  paq/bz2 ratio: {len(paq10)/len(bz2_10):.3f}")

    # Decompress timing
    t0 = time.time()
    rt10 = paq.decompress(paq10)
    d10 = time.time() - t0
    assert rt10 == chunk10
    print(f"  Decompress: {d10:.1f}s")

    est = t10 * 10
    print(f"  Estimated full: {est:.0f}s ({est/60:.1f} min)")

# If still reasonable, try full
if paq_chunk_time < 2:
    print(f"\nTesting paq on FULL data ({len(data):,} bytes)...")
    t0 = time.time()
    paq_full = paq.compress(data)
    t_full = time.time() - t0
    print(f"  paq FULL: {len(paq_full):,} bytes ({len(paq_full)/1024:.1f} KB) [{t_full:.1f}s compress]")
    print(f"  bz2 FULL: {len(bz2_result):,} bytes ({len(bz2_result)/1024:.1f} KB)")
    print(f"  paq/bz2 ratio: {len(paq_full)/len(bz2_result):.3f}")
    if len(paq_full) < len(bz2_result):
        saved = len(bz2_result) - len(paq_full)
        score_saved = 25 * saved / 37_545_489
        print(f"  SAVINGS: {saved:,} bytes -> {score_saved:.4f} score improvement")
    else:
        extra = len(paq_full) - len(bz2_result)
        print(f"  WORSE by {extra:,} bytes")

    # Decompress full
    t0 = time.time()
    rt_full = paq.decompress(paq_full)
    d_full = time.time() - t0
    assert rt_full == data
    print(f"  Decompress FULL: {d_full:.1f}s")
