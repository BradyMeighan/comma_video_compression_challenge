#!/usr/bin/env python
"""
Optimize per-pair horizontal shifts on EVEN frames only.

Rationale:
- SegNet uses only the last frame in each pair (odd-indexed frame).
- PoseNet uses both frames.
- Therefore, we can modify even frames to reduce PoseNet error with near-zero
  impact on SegNet.

This script:
1) Loads current codec output raw frames.
2) For each pair, searches dx in a small grid for frame0 shift that minimizes
   PoseNet MSE to ground-truth pose.
3) Reports the projected final score including tiny metadata overhead for dx.
"""
import bz2
import math
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).parent
RAW_PATH = ROOT / "submissions" / "av1_repro" / "inflated" / "0.raw"
ARCHIVE_ZIP = ROOT / "submissions" / "av1_repro" / "archive.zip"
GT_CACHE = ROOT / "submissions" / "av1_repro" / "_cache" / "gt.pt"

UNCOMPRESSED_BYTES = 37_545_489
W_CAM, H_CAM = 1164, 874
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    import sys

    sys.path.insert(0, str(ROOT))
    from modules import PoseNet, posenet_sd_path

    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    for p in posenet.parameters():
        p.requires_grad_(False)
    return posenet


def resize_to_model(frame: np.ndarray, out_size=(512, 384)) -> np.ndarray:
    # PoseNet preprocess uses bilinear resize to 512x384.
    return cv2.resize(frame, out_size, interpolation=cv2.INTER_LINEAR)


def eval_pose_mse_for_candidates(
    posenet, f0_candidates: np.ndarray, f1_model: np.ndarray, gt_pose: torch.Tensor
) -> np.ndarray:
    """
    f0_candidates: (C, Hm, Wm, 3), uint8
    f1_model:      (Hm, Wm, 3), uint8
    gt_pose:       (6,) tensor on CPU
    Returns:       (C,) pose mse
    """
    C = f0_candidates.shape[0]
    f1_rep = np.repeat(f1_model[None, ...], C, axis=0)

    x0 = torch.from_numpy(f0_candidates.copy()).to(DEVICE).float()
    x1 = torch.from_numpy(f1_rep.copy()).to(DEVICE).float()
    x = torch.stack([x0, x1], dim=1)  # (C, 2, H, W, 3)
    x = einops.rearrange(x, "b t h w c -> b t c h w")

    with torch.inference_mode():
        pose = posenet(posenet.preprocess_input(x))["pose"][:, :6]
        mse = (pose - gt_pose.to(DEVICE).unsqueeze(0)).pow(2).mean(dim=1)
    return mse.detach().cpu().numpy()


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")
    if not ARCHIVE_ZIP.exists():
        raise FileNotFoundError(f"Missing archive.zip: {ARCHIVE_ZIP}")

    print(f"Device: {DEVICE}", flush=True)
    posenet = load_models()
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_pose = gt["pose"]  # (600, 6)
    gt_seg = gt["seg"]    # (600, Hm, Wm)
    n_pairs = gt_pose.shape[0]

    raw = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    archive_bytes = ARCHIVE_ZIP.stat().st_size

    # Baseline components from known best run.
    # Seg is unchanged by even-frame-only transforms.
    baseline_seg = float((gt_seg != gt_seg).float().mean().item())  # exactly 0, placeholder
    # Use measured baseline seg from pipeline for score projection.
    # Keeping explicit value avoids needing a full segnet pass.
    baseline_seg = 0.0054975

    # Coarse + fine candidate grids in model-resolution pixels.
    coarse = np.arange(-2.5, 2.51, 0.5, dtype=np.float32)
    fine = np.arange(-0.40, 0.401, 0.10, dtype=np.float32)

    best_dx_model = np.zeros(n_pairs, dtype=np.float32)
    best_pose_mse = np.zeros(n_pairs, dtype=np.float32)
    baseline_pose_mse = np.zeros(n_pairs, dtype=np.float32)

    # Pre-resize all odd frames once (used repeatedly).
    odd_model_frames = np.zeros((n_pairs, 384, 512, 3), dtype=np.uint8)
    even_model_frames = np.zeros((n_pairs, 384, 512, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even_model_frames[i] = resize_to_model(raw[2 * i], (512, 384))
        odd_model_frames[i] = resize_to_model(raw[2 * i + 1], (512, 384))

    for i in range(n_pairs):
        f0m = even_model_frames[i]
        f1m = odd_model_frames[i]
        gt_i = gt_pose[i]

        # Baseline pose MSE at dx=0
        base_mse = eval_pose_mse_for_candidates(
            posenet, f0m[None, ...], f1m, gt_i
        )[0]
        baseline_pose_mse[i] = base_mse

        # Coarse search
        coarse_frames = []
        for dx in coarse:
            M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0]], dtype=np.float32)
            shifted = cv2.warpAffine(
                f0m,
                M,
                dsize=(f0m.shape[1], f0m.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            coarse_frames.append(shifted)
        coarse_frames = np.stack(coarse_frames, axis=0)
        coarse_mse = eval_pose_mse_for_candidates(posenet, coarse_frames, f1m, gt_i)
        cidx = int(np.argmin(coarse_mse))
        cdx = float(coarse[cidx])

        # Fine search around best coarse
        fine_grid = cdx + fine
        fine_frames = []
        for dx in fine_grid:
            M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, 0.0]], dtype=np.float32)
            shifted = cv2.warpAffine(
                f0m,
                M,
                dsize=(f0m.shape[1], f0m.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            fine_frames.append(shifted)
        fine_frames = np.stack(fine_frames, axis=0)
        fine_mse = eval_pose_mse_for_candidates(posenet, fine_frames, f1m, gt_i)
        fidx = int(np.argmin(fine_mse))
        fdx = float(fine_grid[fidx])
        fmse = float(fine_mse[fidx])

        best_dx_model[i] = fdx
        best_pose_mse[i] = fmse

        if i % 50 == 0:
            print(
                f"pair {i:3d}/{n_pairs}: base={base_mse:.6f} best={fmse:.6f} dx={fdx:+.2f}",
                flush=True,
            )

    base_pose = float(baseline_pose_mse.mean())
    opt_pose = float(best_pose_mse.mean())

    # Quantize model-space dx to int8 in 0.1 px units for storage.
    dx_q = np.clip(np.round(best_dx_model * 10.0), -127, 127).astype(np.int8)
    packed = bz2.compress(dx_q.tobytes(), compresslevel=9)
    meta_bytes = len(packed)

    base_score = 100 * baseline_seg + math.sqrt(10 * base_pose) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = (
        100 * baseline_seg
        + math.sqrt(10 * opt_pose)
        + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)
    )

    print("\n=== Even-frame dx optimization result ===", flush=True)
    print(f"archive bytes: {archive_bytes} ({archive_bytes/1024:.1f} KB)", flush=True)
    print(f"dx metadata bytes (bz2): {meta_bytes} ({meta_bytes/1024:.2f} KB)", flush=True)
    print(f"baseline pose mse: {base_pose:.8f}  sqrt(10*p)={math.sqrt(10*base_pose):.4f}", flush=True)
    print(f"optimized pose mse: {opt_pose:.8f} sqrt(10*p)={math.sqrt(10*opt_pose):.4f}", flush=True)
    print(f"baseline projected score: {base_score:.4f}", flush=True)
    print(f"optimized projected score: {opt_score:.4f}", flush=True)
    print(f"projected improvement: {base_score - opt_score:+.4f}", flush=True)

    # Save metadata for potential decode integration.
    out_meta = ROOT / "submissions" / "av1_repro" / "archive" / "frame0_dx_q.bin"
    out_meta.write_bytes(packed)
    print(f"wrote metadata: {out_meta}", flush=True)


if __name__ == "__main__":
    main()

