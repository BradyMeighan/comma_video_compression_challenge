#!/usr/bin/env python
"""
Optimize per-pair even-frame (dx,dy,theta) on an input raw sequence, then apply.

Useful for testing whether stronger compression can be recovered by sidecars.
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
GT_CACHE = SUB / "_cache" / "gt.pt"
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


def warp_model_frame(frame: np.ndarray, dx: float, dy: float, theta_deg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    center = (w * 0.5, h * 0.5)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(
        frame,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def eval_pose_candidates(posenet, f0: np.ndarray, f1: np.ndarray, gt_pose: torch.Tensor, params):
    cands = np.stack([warp_model_frame(f0, dx, dy, th) for dx, dy, th in params], axis=0)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-in", type=str, required=True)
    ap.add_argument("--raw-out", type=str, required=True)
    ap.add_argument("--meta-out", type=str, default=str(SUB / "archive" / "frame0_dxyr_q.bin"))
    ap.add_argument("--archive-bytes", type=int, default=0)
    ap.add_argument("--eval-total-bytes", type=int, default=0)
    ap.add_argument("--run-fast-eval", action="store_true")
    args = ap.parse_args()

    raw_in = Path(args.raw_in)
    raw_out = Path(args.raw_out)
    meta_out = Path(args.meta_out)
    if not raw_in.exists():
        raise FileNotFoundError(f"Missing raw input: {raw_in}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")

    print(f"Device: {DEVICE}", flush=True)
    posenet = load_posenet()
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_pose = gt["pose"]
    n_pairs = int(gt_pose.shape[0])

    raw = np.fromfile(raw_in, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    even_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    odd_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even_m[i] = cv2.resize(raw[2 * i], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        odd_m[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    dx_coarse = np.arange(-2.0, 2.01, 0.5, dtype=np.float32)
    dy_coarse = np.arange(-0.8, 0.81, 0.4, dtype=np.float32)
    th_coarse = np.arange(-1.0, 1.01, 0.5, dtype=np.float32)
    coarse = [(float(dx), float(dy), float(th)) for th in th_coarse for dy in dy_coarse for dx in dx_coarse]
    dx_fine = np.arange(-0.3, 0.31, 0.1, dtype=np.float32)
    dy_fine = np.arange(-0.2, 0.21, 0.1, dtype=np.float32)
    th_fine = np.arange(-0.4, 0.41, 0.2, dtype=np.float32)

    base_mse = np.zeros(n_pairs, dtype=np.float32)
    best_mse = np.zeros(n_pairs, dtype=np.float32)
    best_dx = np.zeros(n_pairs, dtype=np.float32)
    best_dy = np.zeros(n_pairs, dtype=np.float32)
    best_th = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        f0 = even_m[i]
        f1 = odd_m[i]
        g = gt_pose[i]

        b = eval_pose_candidates(posenet, f0, f1, g, [(0.0, 0.0, 0.0)])[0]
        base_mse[i] = b
        cm = eval_pose_candidates(posenet, f0, f1, g, coarse)
        cidx = int(np.argmin(cm))
        cdx, cdy, cth = coarse[cidx]

        fine = [
            (float(cdx + ddx), float(cdy + ddy), float(cth + dth))
            for dth in th_fine
            for ddy in dy_fine
            for ddx in dx_fine
        ]
        fm = eval_pose_candidates(posenet, f0, f1, g, fine)
        fidx = int(np.argmin(fm))
        fdx, fdy, fth = fine[fidx]
        best_dx[i], best_dy[i], best_th[i] = fdx, fdy, fth
        best_mse[i] = float(fm[fidx])

        if i % 50 == 0:
            print(
                f"pair {i:3d}: base={b:.6f} best={best_mse[i]:.6f} "
                f"d=({fdx:+.2f},{fdy:+.2f}) th={fth:+.2f}",
                flush=True,
            )

    # quantize metadata
    dx_q = np.clip(np.round(best_dx * 10), -127, 127).astype(np.int8)
    dy_q = np.clip(np.round(best_dy * 10), -127, 127).astype(np.int8)
    th_q = np.clip(np.round(best_th * 10), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([dx_q, dy_q, th_q], axis=1).tobytes(), compresslevel=9)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_bytes(packed)
    meta_bytes = len(packed)
    print(f"wrote {meta_out} ({meta_bytes} bytes)", flush=True)

    # apply on full-res even frames
    raw_new = raw.copy()
    sx = W_CAM / MODEL_W
    sy = H_CAM / MODEL_H
    for i in range(n_pairs):
        dx = float(best_dx[i] * sx)
        dy = float(best_dy[i] * sy)
        th = float(best_th[i])
        if abs(dx) <= 1e-6 and abs(dy) <= 1e-6 and abs(th) <= 1e-6:
            continue
        arr = raw_new[2 * i]
        center = (W_CAM * 0.5, H_CAM * 0.5)
        M = cv2.getRotationMatrix2D(center, th, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        raw_new[2 * i] = cv2.warpAffine(
            arr,
            M,
            dsize=(W_CAM, H_CAM),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_new.tofile(raw_out)
    print(f"wrote {raw_out}", flush=True)

    base_pose = float(base_mse.mean())
    opt_pose = float(best_mse.mean())
    print(f"pose mse: {base_pose:.8f} -> {opt_pose:.8f}", flush=True)
    print(f"sqrt term: {math.sqrt(10*base_pose):.4f} -> {math.sqrt(10*opt_pose):.4f}", flush=True)
    if args.archive_bytes > 0:
        print(
            f"if bytes include metadata additively: 25*rate delta ~= {25*meta_bytes/37_545_489:.4f}",
            flush=True,
        )

    if args.run_fast_eval:
        total = int(args.eval_total_bytes) if int(args.eval_total_bytes) > 0 else int(args.archive_bytes)
        if total <= 0:
            raise ValueError("Need --eval-total-bytes or --archive-bytes when --run-fast-eval is set")
        cmd = [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "submissions.evenframe_meta_v2_rateprobe.fast_eval",
            str(raw_out),
            str(total),
        ]
        print("Running fast_eval...", flush=True)
        subprocess.run(cmd, check=False, cwd=str(ROOT))


if __name__ == "__main__":
    main()

