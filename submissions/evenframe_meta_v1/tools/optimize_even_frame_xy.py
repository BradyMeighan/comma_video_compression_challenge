#!/usr/bin/env python
"""
Optimize per-pair (dx, dy) shifts on even frames only (model-space).
"""
import bz2
import math
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "submissions" / "av1_repro" / "inflated" / "0.raw"
GT_CACHE = ROOT / "submissions" / "av1_repro" / "_cache" / "gt.pt"
ARCHIVE_ZIP = ROOT / "submissions" / "av1_repro" / "archive.zip"

UNCOMPRESSED_BYTES = 37_545_489
W_CAM, H_CAM = 1164, 874
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_posenet():
    import sys

    sys.path.insert(0, str(ROOT))
    from modules import PoseNet, posenet_sd_path

    m = PoseNet().eval().to(DEVICE)
    m.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def eval_pose_grid(posenet, f0: np.ndarray, f1: np.ndarray, gt_pose: torch.Tensor, dxy_grid):
    cands = []
    for dx, dy in dxy_grid:
        M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        s = cv2.warpAffine(
            f0,
            M,
            dsize=(f0.shape[1], f0.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        cands.append(s)
    cands = np.stack(cands, axis=0)
    f1_rep = np.repeat(f1[None, ...], cands.shape[0], axis=0)

    x0 = torch.from_numpy(cands.copy()).to(DEVICE).float()
    x1 = torch.from_numpy(f1_rep.copy()).to(DEVICE).float()
    x = torch.stack([x0, x1], dim=1)
    x = einops.rearrange(x, "b t h w c -> b t c h w")
    with torch.inference_mode():
        pose = posenet(posenet.preprocess_input(x))["pose"][:, :6]
        mse = (pose - gt_pose.to(DEVICE).unsqueeze(0)).pow(2).mean(dim=1)
    return mse.detach().cpu().numpy()


def main():
    print(f"Device: {DEVICE}", flush=True)
    posenet = load_posenet()
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_pose = gt["pose"]
    n_pairs = gt_pose.shape[0]

    raw = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    # Pre-resize to model-space once.
    even_m = np.zeros((n_pairs, 384, 512, 3), dtype=np.uint8)
    odd_m = np.zeros((n_pairs, 384, 512, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even_m[i] = cv2.resize(raw[2 * i], (512, 384), interpolation=cv2.INTER_LINEAR)
        odd_m[i] = cv2.resize(raw[2 * i + 1], (512, 384), interpolation=cv2.INTER_LINEAR)

    dx_vals = np.arange(-2.5, 2.51, 0.5, dtype=np.float32)
    dy_vals = np.arange(-1.0, 1.01, 0.5, dtype=np.float32)
    grid = [(float(dx), float(dy)) for dy in dy_vals for dx in dx_vals]

    base_mse = np.zeros(n_pairs, dtype=np.float32)
    best_mse = np.zeros(n_pairs, dtype=np.float32)
    best_dx = np.zeros(n_pairs, dtype=np.float32)
    best_dy = np.zeros(n_pairs, dtype=np.float32)

    # fine refinement around coarse best
    fine_dx = np.arange(-0.4, 0.41, 0.1, dtype=np.float32)
    fine_dy = np.arange(-0.2, 0.21, 0.1, dtype=np.float32)

    for i in range(n_pairs):
        f0 = even_m[i]
        f1 = odd_m[i]
        g = gt_pose[i]

        b = eval_pose_grid(posenet, f0, f1, g, [(0.0, 0.0)])[0]
        base_mse[i] = b

        coarse_mse = eval_pose_grid(posenet, f0, f1, g, grid)
        cidx = int(np.argmin(coarse_mse))
        cdx, cdy = grid[cidx]

        fine_grid = [
            (float(cdx + ddx), float(cdy + ddy))
            for ddy in fine_dy
            for ddx in fine_dx
        ]
        fine_mse = eval_pose_grid(posenet, f0, f1, g, fine_grid)
        fidx = int(np.argmin(fine_mse))
        fdx, fdy = fine_grid[fidx]
        fmse = float(fine_mse[fidx])

        best_dx[i] = fdx
        best_dy[i] = fdy
        best_mse[i] = fmse

        if i % 50 == 0:
            print(
                f"pair {i:3d}: base={b:.6f} best={fmse:.6f} d=({fdx:+.2f},{fdy:+.2f})",
                flush=True,
            )

    base_pose = float(base_mse.mean())
    opt_pose = float(best_mse.mean())
    baseline_seg = 0.0054975
    archive_bytes = ARCHIVE_ZIP.stat().st_size

    # Quantize to compact metadata (int8, 0.1 px units, model-space)
    dx_q = np.clip(np.round(best_dx * 10), -127, 127).astype(np.int8)
    dy_q = np.clip(np.round(best_dy * 10), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([dx_q, dy_q], axis=1).tobytes(), compresslevel=9)
    meta_bytes = len(packed)

    base_score = 100 * baseline_seg + math.sqrt(10 * base_pose) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = 100 * baseline_seg + math.sqrt(10 * opt_pose) + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)

    print("\n=== XY optimization (projected) ===", flush=True)
    print(f"baseline pose mse: {base_pose:.8f} sqrt={math.sqrt(10*base_pose):.4f}", flush=True)
    print(f"optimized pose mse: {opt_pose:.8f} sqrt={math.sqrt(10*opt_pose):.4f}", flush=True)
    print(f"meta bytes (bz2): {meta_bytes}", flush=True)
    print(f"baseline score: {base_score:.4f}", flush=True)
    print(f"optimized score: {opt_score:.4f}", flush=True)
    print(f"improvement: {base_score - opt_score:+.4f}", flush=True)

    out = ROOT / "submissions" / "av1_repro" / "archive" / "frame0_dxy_q.bin"
    out.write_bytes(packed)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

