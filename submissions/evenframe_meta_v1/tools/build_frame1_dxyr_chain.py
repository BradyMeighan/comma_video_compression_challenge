#!/usr/bin/env python
"""
Pack multiple frame1_dxyr_q*.bin sidecars into one chain file:
  frame1_dxyr_chain_q.bin

Format (before bz2):
  <uint32 K><uint32 N><int8[K,N,3]>
where int8 values are:
  dx_q, dy_q in 0.1 model-px; th_q in 0.1 degrees.
"""
import bz2
import struct
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "submissions" / "evenframe_meta_v1" / "archive"


def natural_key(path: Path):
    n = path.name
    m = re.search(r"_r(\d+)\.bin$", n)
    if m:
        return (0, int(m.group(1)), n)
    return (1, 0, n)


def load_stage(path: Path) -> np.ndarray:
    q = np.frombuffer(bz2.decompress(path.read_bytes()), dtype=np.int8)
    q = q.reshape(-1, 3)  # (N,3)
    return q


def main():
    paths = sorted(ARCHIVE.glob("frame1_dxyr_q*.bin"), key=natural_key)
    if not paths:
        raise FileNotFoundError("No frame1_dxyr_q*.bin files found.")

    stages = [load_stage(p) for p in paths]
    n = stages[0].shape[0]
    for i, st in enumerate(stages):
        if st.shape != (n, 3):
            raise ValueError(f"Stage {i} shape mismatch: {st.shape} != {(n, 3)}")

    arr = np.stack(stages, axis=0).astype(np.int8)  # (K,N,3)
    k = arr.shape[0]

    payload = struct.pack("<II", k, n) + arr.tobytes()
    packed = bz2.compress(payload, compresslevel=9)
    out = ARCHIVE / "frame1_dxyr_chain_q.bin"
    out.write_bytes(packed)

    raw_total = sum(p.stat().st_size for p in paths)
    print(f"Packed {k} stages, N={n}")
    print(f"input total bytes: {raw_total}")
    print(f"output bytes: {len(packed)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

