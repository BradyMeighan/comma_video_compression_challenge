#!/usr/bin/env python
"""
Probe odd-frame per-pair (dx, dy, theta) sidecar with joint seg+pose objective.

This tests whether residual error is better explained by geometric mismatch
than by sparse boundary pixel edits.
"""
import argparse
import bz2
import math
import subprocess
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
RAW_IN = SUB / "inflated" / "0.raw"
GT_CACHE = SUB / "_cache" / "gt.pt"
ARCHIVE_ZIP = SUB / "archive.zip"
FAST_EVAL_MODULE = f"submissions.{SUB.name}.fast_eval"
UNCOMPRESSED_BYTES = 37_545_489
W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_WEIGHT = 100.0


def load_models():
    import sys

    sys.path.insert(0, str(ROOT))
    from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path

    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)
    return segnet, posenet


def warp_model(frame: np.ndarray, dx: float, dy: float, th: float) -> np.ndarray:
    h, w = frame.shape[:2]
    center = (w * 0.5, h * 0.5)
    M = cv2.getRotationMatrix2D(center, th, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(
        frame,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def eval_candidates(
    segnet,
    posenet,
    even_m: np.ndarray,
    odd_m: np.ndarray,
    gt_seg: torch.Tensor,
    gt_pose: torch.Tensor,
    params,
    pose_weight: float,
):
    cands = np.stack([warp_model(odd_m, dx, dy, th) for dx, dy, th in params], axis=0)  # (C,H,W,3)
    c = cands.shape[0]

    # Seg objective on odd frame
    x_seg = torch.from_numpy(cands.copy()).to(DEVICE).float()
    x_seg = einops.rearrange(x_seg, "c h w ch -> c ch h w")
    with torch.inference_mode():
        seg_pred = segnet(x_seg).argmax(1)
        seg_dist = (seg_pred != gt_seg.to(DEVICE).unsqueeze(0)).float().mean(dim=(1, 2))

    # Pose objective (even fixed, odd candidate)
    even_rep = np.repeat(even_m[None, ...], c, axis=0)
    x0 = torch.from_numpy(even_rep.copy()).to(DEVICE).float()
    x1 = torch.from_numpy(cands.copy()).to(DEVICE).float()
    x = torch.stack([x0, x1], dim=1)
    x = einops.rearrange(x, "b t h w ch -> b t ch h w")
    with torch.inference_mode():
        pose = posenet(posenet.preprocess_input(x))["pose"][:, :6]
        pose_mse = (pose - gt_pose.to(DEVICE).unsqueeze(0)).pow(2).mean(dim=1)

    obj = SEG_WEIGHT * seg_dist + pose_weight * pose_mse
    return seg_dist.detach().cpu().numpy(), pose_mse.detach().cpu().numpy(), obj.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-in", type=str, default=str(RAW_IN))
    ap.add_argument("--archive-bytes", type=int, default=0)
    ap.add_argument("--pose-weight", type=float, default=55.0)
    ap.add_argument("--tag", type=str, default="oddxyth")
    args = ap.parse_args()

    raw_in = Path(args.raw_in)
    if not raw_in.exists():
        raise FileNotFoundError(f"Missing raw input: {raw_in}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")
    archive_bytes = int(args.archive_bytes) if int(args.archive_bytes) > 0 else int(ARCHIVE_ZIP.stat().st_size)

    out_raw = SUB / "inflated" / f"0_{args.tag}.raw"
    out_meta = SUB / "archive" / f"frame1_dxyr_q_{args.tag}.bin"

    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_seg = gt["seg"]   # (N,H,W), odd frame target
    gt_pose = gt["pose"] # (N,6)
    n_pairs = int(gt_seg.shape[0])

    raw = np.fromfile(raw_in, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    even_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    odd_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even_m[i] = cv2.resize(raw[2 * i], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        odd_m[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    # Fast coarse+fine search
    dx_coarse = np.array([-0.8, -0.4, 0.0, 0.4, 0.8], dtype=np.float32)
    dy_coarse = np.array([-0.4, 0.0, 0.4], dtype=np.float32)
    th_coarse = np.array([-0.6, 0.0, 0.6], dtype=np.float32)
    coarse = [(float(dx), float(dy), float(th)) for th in th_coarse for dy in dy_coarse for dx in dx_coarse]
    dxf = np.array([-0.2, 0.0, 0.2], dtype=np.float32)
    dyf = np.array([-0.2, 0.0, 0.2], dtype=np.float32)
    thf = np.array([-0.2, 0.0, 0.2], dtype=np.float32)

    base_seg = np.zeros(n_pairs, dtype=np.float32)
    base_pose = np.zeros(n_pairs, dtype=np.float32)
    best_seg = np.zeros(n_pairs, dtype=np.float32)
    best_pose = np.zeros(n_pairs, dtype=np.float32)
    best_dx = np.zeros(n_pairs, dtype=np.float32)
    best_dy = np.zeros(n_pairs, dtype=np.float32)
    best_th = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        s0, p0, _ = eval_candidates(
            segnet, posenet, even_m[i], odd_m[i], gt_seg[i], gt_pose[i], [(0.0, 0.0, 0.0)], args.pose_weight
        )
        base_seg[i], base_pose[i] = float(s0[0]), float(p0[0])

        sc, pc, oc = eval_candidates(
            segnet, posenet, even_m[i], odd_m[i], gt_seg[i], gt_pose[i], coarse, args.pose_weight
        )
        cidx = int(np.argmin(oc))
        cdx, cdy, cth = coarse[cidx]
        fine = [
            (float(cdx + ddx), float(cdy + ddy), float(cth + dth))
            for dth in thf
            for ddy in dyf
            for ddx in dxf
        ]
        sf, pf, of = eval_candidates(
            segnet, posenet, even_m[i], odd_m[i], gt_seg[i], gt_pose[i], fine, args.pose_weight
        )
        fidx = int(np.argmin(of))
        fdx, fdy, fth = fine[fidx]
        best_dx[i], best_dy[i], best_th[i] = fdx, fdy, fth
        best_seg[i], best_pose[i] = float(sf[fidx]), float(pf[fidx])

        if i % 50 == 0:
            print(
                f"pair {i:3d}: seg {base_seg[i]:.6f}->{best_seg[i]:.6f} "
                f"pose {base_pose[i]:.6f}->{best_pose[i]:.6f} "
                f"d=({fdx:+.2f},{fdy:+.2f}) th={fth:+.2f}",
                flush=True,
            )

    # Save metadata
    dx_q = np.clip(np.round(best_dx * 10.0), -127, 127).astype(np.int8)
    dy_q = np.clip(np.round(best_dy * 10.0), -127, 127).astype(np.int8)
    th_q = np.clip(np.round(best_th * 10.0), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([dx_q, dy_q, th_q], axis=1).tobytes(), compresslevel=9)
    out_meta.write_bytes(packed)
    meta_bytes = len(packed)
    print(f"wrote {out_meta} ({meta_bytes} bytes)", flush=True)

    # Apply to full-res odd frames
    raw_new = raw.copy()
    sx = W_CAM / MODEL_W
    sy = H_CAM / MODEL_H
    for i in range(n_pairs):
        dx = float(best_dx[i] * sx)
        dy = float(best_dy[i] * sy)
        th = float(best_th[i])
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6 and abs(th) <= 1e-6:
            continue
        arr = raw_new[2 * i + 1]
        center = (W_CAM * 0.5, H_CAM * 0.5)
        M = cv2.getRotationMatrix2D(center, th, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        raw_new[2 * i + 1] = cv2.warpAffine(
            arr,
            M,
            dsize=(W_CAM, H_CAM),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    raw_new.tofile(out_raw)
    print(f"wrote {out_raw}", flush=True)

    base_score = 100 * float(base_seg.mean()) + math.sqrt(10 * float(base_pose.mean())) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = 100 * float(best_seg.mean()) + math.sqrt(10 * float(best_pose.mean())) + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)
    print("\n=== projected ===", flush=True)
    print(f"seg {base_seg.mean():.8f} -> {best_seg.mean():.8f}", flush=True)
    print(f"pose {base_pose.mean():.8f} -> {best_pose.mean():.8f}", flush=True)
    print(f"score {base_score:.4f} -> {opt_score:.4f}", flush=True)

    total_bytes = archive_bytes + meta_bytes
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    cmd = [
        str(py),
        "-m",
        FAST_EVAL_MODULE,
        str(out_raw),
        str(total_bytes),
    ]
    print("\nRunning fast_eval...", flush=True)
    subprocess.run(cmd, check=False, cwd=str(ROOT))


if __name__ == "__main__":
    main()

