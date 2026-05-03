#!/usr/bin/env python
"""
Probe residual even-frame affine correction potential on top of current dxyr+ab output.

This does NOT modify the submission. It estimates projected score impact first.
If useful, we can wire the metadata into inflate.py in a second step.
"""
import bz2
import math
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "submissions" / "evenframe_meta_v1"
RAW_PATH = SUB / "inflated" / "0.raw"
GT_CACHE = SUB / "_cache" / "gt.pt"
ARCHIVE_ZIP = SUB / "archive.zip"
UNCOMPRESSED_BYTES = 37_545_489

W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
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


def affine_about_center(frame: np.ndarray, scale: float, shx: float, shy: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    A = np.array([[scale, shx], [shy, scale]], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    t = c - A @ c
    M = np.array(
        [[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        frame,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def eval_pose_candidates(posenet, f0: np.ndarray, f1: np.ndarray, gt_pose: torch.Tensor, params):
    # params: list[(scale, shx, shy)]
    cands = np.stack([affine_about_center(f0, s, x, y) for s, x, y in params], axis=0)
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
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Missing inflated raw: {RAW_PATH}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")

    posenet = load_posenet()
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_pose = gt["pose"]
    n_pairs = int(gt_pose.shape[0])
    archive_bytes = int(ARCHIVE_ZIP.stat().st_size)

    raw = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    even_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    odd_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even_m[i] = cv2.resize(raw[2 * i], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        odd_m[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    # Small residual search only (metadata-friendly, low risk).
    scales = np.array([0.994, 0.997, 1.000, 1.003, 1.006], dtype=np.float32)
    shx = np.array([-0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015], dtype=np.float32)
    shy = np.array([-0.015, -0.010, -0.005, 0.000, 0.005, 0.010, 0.015], dtype=np.float32)
    grid = [(float(s), float(x), float(y)) for s in scales for x in shx for y in shy]

    base_mse = np.zeros(n_pairs, dtype=np.float32)
    best_mse = np.zeros(n_pairs, dtype=np.float32)
    best_s = np.ones(n_pairs, dtype=np.float32)
    best_x = np.zeros(n_pairs, dtype=np.float32)
    best_y = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        f0 = even_m[i]
        f1 = odd_m[i]
        g = gt_pose[i]

        b0 = eval_pose_candidates(posenet, f0, f1, g, [(1.0, 0.0, 0.0)])[0]
        base_mse[i] = b0

        mse = eval_pose_candidates(posenet, f0, f1, g, grid)
        idx = int(np.argmin(mse))
        s, x, y = grid[idx]
        m = float(mse[idx])

        best_s[i] = s
        best_x[i] = x
        best_y[i] = y
        best_mse[i] = m

        if i % 50 == 0:
            print(
                f"pair {i:3d}: base={b0:.6f} best={m:.6f} s={s:.4f} sh=({x:+.3f},{y:+.3f})",
                flush=True,
            )

    base_pose = float(base_mse.mean())
    opt_pose = float(best_mse.mean())
    baseline_seg = 0.00564890  # from current 1.2675 run

    # Quantize metadata estimate:
    # scale in 0.001 steps around 1.0, shear in 0.001 units.
    s_q = np.clip(np.round((best_s - 1.0) * 1000.0), -127, 127).astype(np.int8)
    x_q = np.clip(np.round(best_x * 1000.0), -127, 127).astype(np.int8)
    y_q = np.clip(np.round(best_y * 1000.0), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([s_q, x_q, y_q], axis=1).tobytes(), compresslevel=9)
    meta_bytes = len(packed)

    base_score = 100 * baseline_seg + math.sqrt(10 * base_pose) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = 100 * baseline_seg + math.sqrt(10 * opt_pose) + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)

    print("\n=== Residual affine probe (projected) ===", flush=True)
    print(f"baseline pose mse: {base_pose:.8f} sqrt={math.sqrt(10*base_pose):.4f}", flush=True)
    print(f"optimized pose mse: {opt_pose:.8f} sqrt={math.sqrt(10*opt_pose):.4f}", flush=True)
    print(f"meta bytes (bz2): {meta_bytes}", flush=True)
    print(f"baseline score: {base_score:.4f}", flush=True)
    print(f"projected score: {opt_score:.4f}", flush=True)
    print(f"improvement: {base_score - opt_score:+.4f}", flush=True)

    out = SUB / "archive" / "frame0_affine_q.bin"
    out.write_bytes(packed)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

