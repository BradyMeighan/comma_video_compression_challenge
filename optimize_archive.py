#!/usr/bin/env python
"""
Try different compression strategies for seg maps to minimize archive size.
"""
import struct, zlib, bz2, lzma, time, itertools
import numpy as np

from pathlib import Path

ROOT = Path(__file__).parent
seg_bin = ROOT / 'submissions' / 'adversarial_decode' / 'archive' / '0' / 'seg.bin'

with open(seg_bin, 'rb') as f:
    n, H, W, flags = struct.unpack('<IIII', f.read(16))
    has_perm = bool(flags & (1 << 4))
    if has_perm:
        fwd_perm = list(f.read(5))
        inv_lut = np.zeros(5, dtype=np.uint8)
        for orig_class, encoded_val in enumerate(fwd_perm):
            inv_lut[encoded_val] = orig_class
    
    raw_bytes = bz2.decompress(f.read())

ri = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(H, n, W)
ri = np.ascontiguousarray(ri.transpose(1, 0, 2))  # n, H, W

if has_perm:
    seg_maps = inv_lut[ri]
else:
    seg_maps = ri

print(f"Seg maps shape: {seg_maps.shape}, classes: {np.unique(seg_maps)}")
print(f"Raw size: {seg_maps.nbytes:,} bytes")
print(f"Current seg.bin: {seg_bin.stat().st_size:,} bytes")

# Test all compression combinations
results = []

# Permutations of 5 classes
from itertools import permutations
all_perms = list(permutations(range(5)))

for perm in all_perms[:10]:  # test first 10 + best known
    perm_lut = np.array(perm, dtype=np.uint8)
    encoded = perm_lut[seg_maps]
    
    # Row-interleaved
    ri_data = np.ascontiguousarray(encoded.transpose(1, 0, 2)).tobytes()
    
    # Normal order
    flat_data = encoded.tobytes()
    
    # Left-residual
    lr_data = encoded.copy()
    for c in range(1, W):
        lr_data[:, :, c] = (lr_data[:, :, c].astype(np.int16) - lr_data[:, :, c-1].astype(np.int16)) % 5
    lr_ri = np.ascontiguousarray(lr_data.transpose(1, 0, 2)).tobytes()
    lr_flat = lr_data.tobytes()
    
    # 4-bit packing (row-interleaved)
    arr = np.frombuffer(ri_data, dtype=np.uint8)
    if len(arr) % 2 == 1:
        arr = np.append(arr, np.uint8(0))
    packed = (arr[0::2] & 0x0F) | ((arr[1::2] & 0x0F) << 4)
    packed_data = packed.tobytes()
    
    # 4-bit left-residual
    arr_lr = np.frombuffer(lr_ri, dtype=np.uint8)
    if len(arr_lr) % 2 == 1:
        arr_lr = np.append(arr_lr, np.uint8(0))
    packed_lr = (arr_lr[0::2] & 0x0F) | ((arr_lr[1::2] & 0x0F) << 4)
    packed_lr_data = packed_lr.tobytes()
    
    for name, data in [
        ("ri", ri_data),
        ("flat", flat_data),
        ("lr_ri", lr_ri),
        ("lr_flat", lr_flat),
        ("4bit_ri", packed_data),
        ("4bit_lr_ri", packed_lr_data),
    ]:
        for comp_name, comp_fn in [
            ("bz2_1", lambda d: bz2.compress(d, 1)),
            ("bz2_2", lambda d: bz2.compress(d, 2)),
            ("bz2_9", lambda d: bz2.compress(d, 9)),
            ("zlib_9", lambda d: zlib.compress(d, 9)),
            ("lzma", lambda d: lzma.compress(d)),
        ]:
            compressed = comp_fn(data)
            header_size = 16 + 5  # base header + perm
            total = header_size + len(compressed)
            results.append((total, perm, name, comp_name, len(compressed)))

# Also test BEST_PERM from encode.py
BEST_PERM = (2, 4, 1, 3, 0)
perm_lut = np.array(BEST_PERM, dtype=np.uint8)
encoded = perm_lut[seg_maps]
ri_data = np.ascontiguousarray(encoded.transpose(1, 0, 2)).tobytes()
for comp_name, comp_fn in [
    ("bz2_1", lambda d: bz2.compress(d, 1)),
    ("bz2_2", lambda d: bz2.compress(d, 2)),
    ("bz2_9", lambda d: bz2.compress(d, 9)),
    ("zlib_9", lambda d: zlib.compress(d, 9)),
    ("lzma", lambda d: lzma.compress(d)),
]:
    compressed = comp_fn(ri_data)
    header_size = 16 + 5
    total = header_size + len(compressed)
    results.append((total, BEST_PERM, "ri_BEST", comp_name, len(compressed)))

# Left-residual with BEST_PERM
lr_data = encoded.copy()
for c in range(1, W):
    lr_data[:, :, c] = (lr_data[:, :, c].astype(np.int16) - lr_data[:, :, c-1].astype(np.int16)) % 5
lr_ri = np.ascontiguousarray(lr_data.transpose(1, 0, 2)).tobytes()
for comp_name, comp_fn in [
    ("bz2_1", lambda d: bz2.compress(d, 1)),
    ("bz2_2", lambda d: bz2.compress(d, 2)),
    ("bz2_9", lambda d: bz2.compress(d, 9)),
    ("zlib_9", lambda d: zlib.compress(d, 9)),
    ("lzma", lambda d: lzma.compress(d)),
]:
    compressed = comp_fn(lr_ri)
    header_size = 16 + 5
    total = header_size + len(compressed)
    results.append((total, BEST_PERM, "lr_ri_BEST", comp_name, len(compressed)))

# 4-bit with BEST_PERM
arr = np.frombuffer(ri_data, dtype=np.uint8)
if len(arr) % 2 == 1:
    arr = np.append(arr, np.uint8(0))
packed = (arr[0::2] & 0x0F) | ((arr[1::2] & 0x0F) << 4)
for comp_name, comp_fn in [
    ("bz2_1", lambda d: bz2.compress(d, 1)),
    ("bz2_9", lambda d: bz2.compress(d, 9)),
    ("lzma", lambda d: lzma.compress(d)),
]:
    compressed = comp_fn(packed.tobytes())
    header_size = 16 + 5
    total = header_size + len(compressed)
    results.append((total, BEST_PERM, "4bit_ri_BEST", comp_name, len(compressed)))

results.sort()
print(f"\nTop 20 results:")
for total, perm, layout, comp, data_size in results[:20]:
    print(f"  {total:>8,} bytes = {total/1024:.1f} KB  "
          f"perm={perm} layout={layout:15s} comp={comp:8s} "
          f"data={data_size:,}")

print(f"\nCurrent: 324,148 bytes = 316.6 KB")
best_total = results[0][0]
savings = 324148 - best_total
print(f"Best:    {best_total:,} bytes = {best_total/1024:.1f} KB")
print(f"Savings: {savings:,} bytes = {savings/1024:.1f} KB")
rate_savings = 25 * savings / 37_545_489
print(f"Score improvement from rate: {rate_savings:.4f}")
