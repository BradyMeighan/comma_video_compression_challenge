#!/usr/bin/env python
"""
Blend sweep between baseline odd frames and REN-corrected odd frames.

alpha=0.0 -> baseline
alpha=1.0 -> full REN odd-frame replacement
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "submissions" / "evenframe_meta_v1"
RAW_BASE = SUB / "inflated" / "0.raw"
RAW_REN = SUB / "inflated" / "0_ren.raw"
UNCOMPRESSED_BYTES = 37_545_489
W_CAM, H_CAM = 1164, 874
N_FRAMES = 1200


def build_blend(alpha: float, out_path: Path):
    shape = (N_FRAMES, H_CAM, W_CAM, 3)
    base = np.memmap(RAW_BASE, dtype=np.uint8, mode="r", shape=shape)
    ren = np.memmap(RAW_REN, dtype=np.uint8, mode="r", shape=shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for i in range(N_FRAMES):
            if i % 2 == 0:
                fr = base[i]
            else:
                if alpha <= 0.0:
                    fr = base[i]
                elif alpha >= 1.0:
                    fr = ren[i]
                else:
                    fr = np.clip((1.0 - alpha) * base[i].astype(np.float32) + alpha * ren[i].astype(np.float32), 0, 255).astype(np.uint8)
            f.write(fr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--model-bytes", type=int, required=True)
    args = ap.parse_args()

    if not RAW_BASE.exists():
        raise FileNotFoundError(f"Missing baseline raw: {RAW_BASE}")
    if not RAW_REN.exists():
        raise FileNotFoundError(f"Missing REN raw: {RAW_REN}")

    archive_bytes = (SUB / "archive.zip").stat().st_size
    total_bytes = archive_bytes + int(args.model_bytes)
    print(
        f"Archive bytes={archive_bytes}, model bytes={args.model_bytes}, total={total_bytes}, 25*rate={25*total_bytes/UNCOMPRESSED_BYTES:.4f}",
        flush=True,
    )

    py = ROOT / ".venv" / "Scripts" / "python.exe"
    for a in args.alphas:
        out_raw = SUB / "inflated" / f"0_ren_blend_{a:.2f}.raw"
        print(f"\n=== alpha={a:.2f} ===", flush=True)
        build_blend(a, out_raw)
        cmd = [
            str(py),
            "-m",
            "submissions.evenframe_meta_v1.fast_eval",
            str(out_raw),
            str(total_bytes),
        ]
        subprocess.run(cmd, check=False, cwd=str(ROOT))
        try:
            out_raw.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()

