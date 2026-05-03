"""Find optimal permutation for paq compressor."""
import struct, bz2, time
import numpy as np
from pathlib import Path
from itertools import permutations
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

ri = np.ascontiguousarray(sm.transpose(1, 0, 2).reshape(-1, W))

print(f"Current bz2 perm: {tuple(fwd_perm)}")
print(f"Searching 120 perms for paq...")

best_perm = None
best_sz = float('inf')
for i, p in enumerate(permutations(range(5))):
    lut = np.array(p, dtype=np.uint8)
    data = lut[ri].tobytes()
    sz = len(paq.compress(data))
    if sz < best_sz:
        best_sz = sz
        best_perm = p
        print(f"  perm={p} -> {sz:,} ({sz/1024:.1f} KB) [NEW BEST]")
    if (i+1) % 20 == 0:
        print(f"  ...tested {i+1}/120")

print(f"\nBest paq perm: {best_perm} -> {best_sz:,} bytes ({best_sz/1024:.1f} KB)")

# Compare
bz2_best_data = np.array(fwd_perm, dtype=np.uint8)[ri].tobytes()
bz2_sz = len(bz2.compress(bz2_best_data, 9))
paq_bz2perm = len(paq.compress(bz2_best_data))

print(f"\nComparison:")
print(f"  bz2 (bz2-optimal perm {tuple(fwd_perm)}): {bz2_sz:,} bytes ({bz2_sz/1024:.1f} KB)")
print(f"  paq (bz2-optimal perm):                    {paq_bz2perm:,} bytes ({paq_bz2perm/1024:.1f} KB)")
print(f"  paq (paq-optimal perm {best_perm}):        {best_sz:,} bytes ({best_sz/1024:.1f} KB)")
print(f"  Savings paq-optimal vs bz2: {bz2_sz - best_sz:,} bytes -> {25*(bz2_sz - best_sz)/37_545_489:.4f} score")
