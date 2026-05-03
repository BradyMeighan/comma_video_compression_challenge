#!/usr/bin/env python
"""
Optimize residual (dx,dy,theta) on even frames on top of current corrected output.

Writes: frame0_dxyr2_q.bin
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
RAW_PATH = ROOT / "submissions" / "av1_repro" / "inflated" / "0.raw"  # current corrected output
GT_CACHE = ROOT / "submissions" / "av1_repro" / "_cache" / "gt.pt"
ARCHIVE_ZIP = ROOT / "submissions" / "av1_repro" / "archive.zip"
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


def warp_model(frame: np.ndarray, dx: float, dy: float, th: float) -> np.ndarray:
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), th, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(frame, M, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def eval_pose(posenet, f0: np.ndarray, f1: np.ndarray, gt_pose: torch.Tensor, params):
    cands = np.stack([warp_model(f0, dx, dy, th) for dx, dy, th in params], axis=0)
    f1r = np.repeat(f1[None, ...], cands.shape[0], axis=0)
    x0 = torch.from_numpy(cands.copy()).to(DEVICE).float()
    x1 = torch.from_numpy(f1r.copy()).to(DEVICE).float()
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
    n_pairs = gt["pose"].shape[0]
    raw = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    archive_bytes = ARCHIVE_ZIP.stat().st_size

    even = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    odd = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even[i] = cv2.resize(raw[2 * i], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        odd[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    dx_c = np.array([-0.6, -0.3, 0.0, 0.3, 0.6], dtype=np.float32)
    dy_c = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float32)
    th_c = np.array([-0.4, -0.2, 0.0, 0.2, 0.4], dtype=np.float32)
    coarse = [(float(dx), float(dy), float(th)) for th in th_c for dy in dy_c for dx in dx_c]

    dx_f = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    dy_f = np.array([-0.1, 0.0, 0.1], dtype=np.float32)
    th_f = np.array([-0.1, 0.0, 0.1], dtype=np.float32)

    base = np.zeros(n_pairs, dtype=np.float32)
    best = np.zeros(n_pairs, dtype=np.float32)
    bdx = np.zeros(n_pairs, dtype=np.float32)
    bdy = np.zeros(n_pairs, dtype=np.float32)
    bth = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        f0, f1, g = even[i], odd[i], gt["pose"][i]
        b0 = eval_pose(posenet, f0, f1, g, [(0.0, 0.0, 0.0)])[0]
        base[i] = b0
        cm = eval_pose(posenet, f0, f1, g, coarse)
        cidx = int(np.argmin(cm))
        cdx, cdy, cth = coarse[cidx]
        fine = [(float(cdx + ddx), float(cdy + ddy), float(cth + dth)) for dth in th_f for ddy in dy_f for ddx in dx_f]
        fm = eval_pose(posenet, f0, f1, g, fine)
        fidx = int(np.argmin(fm))
        dx, dy, th = fine[fidx]
        bdx[i], bdy[i], bth[i] = dx, dy, th
        best[i] = float(fm[fidx])
        if i % 50 == 0:
            print(f"pair {i:3d}: base={b0:.6f} best={best[i]:.6f} d=({dx:+.2f},{dy:+.2f}) th={th:+.2f}", flush=True)

    base_pose = float(base.mean())
    opt_pose = float(best.mean())
    baseline_seg = 0.00564963

    qx = np.clip(np.round(bdx * 10), -127, 127).astype(np.int8)
    qy = np.clip(np.round(bdy * 10), -127, 127).astype(np.int8)
    qt = np.clip(np.round(bth * 10), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([qx, qy, qt], axis=1).tobytes(), compresslevel=9)
    meta_bytes = len(packed)

    base_score = 100 * baseline_seg + math.sqrt(10 * base_pose) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = 100 * baseline_seg + math.sqrt(10 * opt_pose) + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)
    print("\n=== Residual XYTheta (projected) ===", flush=True)
    print(f"baseline pose mse: {base_pose:.8f} sqrt={math.sqrt(10*base_pose):.4f}", flush=True)
    print(f"optimized pose mse: {opt_pose:.8f} sqrt={math.sqrt(10*opt_pose):.4f}", flush=True)
    print(f"meta bytes: {meta_bytes}", flush=True)
    print(f"baseline score: {base_score:.4f}", flush=True)
    print(f"optimized score: {opt_score:.4f}", flush=True)
    print(f"improvement: {base_score - opt_score:+.4f}", flush=True)

    out = ROOT / "submissions" / "av1_repro" / "archive" / "frame0_dxyr2_q.bin"
    out.write_bytes(packed)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

