#!/usr/bin/env python
"""
Pack multiple frame1_ab_q*.bin sidecars into one chain file:
  frame1_ab_chain_q.bin

Format (before bz2):
  <uint32 K><uint32 N><int8[K,N,2]>
where K = number of stages and N = number of pairs.
"""
import bz2
import struct
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "submissions" / "evenframe_meta_v1" / "archive"


def stage_sort_key(path: Path):
    n = path.name
    # Initial coarse odd-frame stage.
    if n == "frame1_ab_q.bin":
        return (0, 0, n)

    # Main residual chain: frame1_ab_q_t300w55(.bin) then _r2.._rN
    m_main = re.match(r"^frame1_ab_q_t300w55(?:_r(\d+))?\.bin$", n)
    if m_main:
        r = int(m_main.group(1)) if m_main.group(1) else 1
        return (1, r, n)

    # Final full-coverage residuals should come after the main chain.
    m_full = re.match(r"^frame1_ab_q_full600_r(\d+)\.bin$", n)
    if m_full:
        return (2, int(m_full.group(1)), n)

    # Fallback deterministic order.
    return (9, 9999, n)


def load_stage(path: Path) -> np.ndarray:
    q = np.frombuffer(bz2.decompress(path.read_bytes()), dtype=np.int8)
    q = q.reshape(-1, 2)  # (N,2)
    return q


def main():
    paths = sorted(ARCHIVE.glob("frame1_ab_q*.bin"), key=stage_sort_key)
    if not paths:
        raise FileNotFoundError("No frame1_ab_q*.bin files found.")

    stages = [load_stage(p) for p in paths]
    n = stages[0].shape[0]
    for i, st in enumerate(stages):
        if st.shape != (n, 2):
            raise ValueError(f"Stage {i} shape mismatch: {st.shape} != {(n, 2)}")

    arr = np.stack(stages, axis=0).astype(np.int8)  # (K,N,2)
    k = arr.shape[0]

    payload = struct.pack("<II", k, n) + arr.tobytes()
    packed = bz2.compress(payload, compresslevel=9)
    out = ARCHIVE / "frame1_ab_chain_q.bin"
    out.write_bytes(packed)

    raw_total = sum(p.stat().st_size for p in paths)
    print(f"Packed {k} stages, N={n}")
    print(f"input total bytes: {raw_total}")
    print(f"output bytes: {len(packed)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

