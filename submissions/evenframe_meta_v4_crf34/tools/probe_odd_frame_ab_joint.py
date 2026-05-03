#!/usr/bin/env python
"""
Probe odd-frame (frame1) per-pair gain/bias sidecar with joint seg+pose objective.

This is a metadata-only, rule-safe alternative to REN:
- optimize only odd frames (SegNet frame) for each pair
- include pose penalty to avoid breaking already-good pose
- save tiny sidecar candidate: frame1_ab_q.bin
"""
import argparse
import bz2
import math
from pathlib import Path

import einops
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
RAW_IN = SUB / "inflated" / "0.raw"
RAW_OUT = SUB / "inflated" / "0_frame1ab.raw"
GT_CACHE = SUB / "_cache" / "gt.pt"
ARCHIVE_ZIP = SUB / "archive.zip"
META_OUT = SUB / "archive" / "frame1_ab_q.bin"
FAST_EVAL_MODULE = f"submissions.{SUB.name}.fast_eval"
UNCOMPRESSED_BYTES = 37_545_489

W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEG_WEIGHT = 100.0


def parse_floats(csv: str):
    return np.array([float(x.strip()) for x in csv.split(",") if x.strip()], dtype=np.float32)


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


def apply_ab(frame: np.ndarray, a: float, b: float) -> np.ndarray:
    x = frame.astype(np.float32) * a + b
    return np.clip(x, 0, 255).astype(np.uint8)


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
    # params: list[(a,b)]
    cands = np.stack([apply_ab(odd_m, a, b) for a, b in params], axis=0)  # (C,H,W,3)
    c = cands.shape[0]

    # Seg objective
    x_seg = torch.from_numpy(cands.copy()).to(DEVICE).float()
    x_seg = einops.rearrange(x_seg, "c h w ch -> c ch h w")
    with torch.inference_mode():
        seg_pred = segnet(x_seg).argmax(1)  # (C,H,W)
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

    # Weighted local objective
    obj = SEG_WEIGHT * seg_dist + pose_weight * pose_mse
    return seg_dist.detach().cpu().numpy(), pose_mse.detach().cpu().numpy(), obj.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=220)
    ap.add_argument("--pose-weight", type=float, default=49.0)
    ap.add_argument("--a-grid", type=str, default="0.92,0.95,0.98,1.00,1.02,1.05,1.08")
    ap.add_argument("--b-grid", type=str, default="-10,-6,-3,0,3,6,10")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--raw-in", type=str, default=str(RAW_IN))
    ap.add_argument("--archive-bytes", type=int, default=0, help="Override base archive bytes; 0 means read archive.zip")
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    raw_in = Path(args.raw_in)
    if not raw_in.exists():
        raise FileNotFoundError(f"Missing raw input: {raw_in}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")

    tag = f"_{args.tag}" if args.tag else ""
    raw_out = SUB / "inflated" / f"0_frame1ab{tag}.raw"
    meta_out = SUB / "archive" / f"frame1_ab_q{tag}.bin"

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

    # Baseline pairwise metrics and worst-seg mining
    base_seg = np.zeros(n_pairs, dtype=np.float32)
    base_pose = np.zeros(n_pairs, dtype=np.float32)
    for i in range(n_pairs):
        s, p, _ = eval_candidates(
            segnet,
            posenet,
            even_m[i],
            odd_m[i],
            gt_seg[i],
            gt_pose[i],
            [(1.0, 0.0)],
            args.pose_weight,
        )
        base_seg[i] = float(s[0])
        base_pose[i] = float(p[0])
        if i % 100 == 0:
            print(f"baseline pair {i:3d}: seg={base_seg[i]:.6f} pose={base_pose[i]:.6f}", flush=True)

    top_k = min(max(1, int(args.top_k)), n_pairs)
    opt_idx = np.argsort(-base_seg)[:top_k]
    opt_mask = np.zeros(n_pairs, dtype=bool)
    opt_mask[opt_idx] = True
    print(f"Optimizing top-{top_k} pairs by seg disagreement", flush=True)
    print(f"  top-k mean seg={base_seg[opt_idx].mean():.6f}, all mean seg={base_seg.mean():.6f}", flush=True)

    # Candidate grid
    a_vals = parse_floats(args.a_grid)
    b_vals = parse_floats(args.b_grid)
    grid = [(float(a), float(b)) for a in a_vals for b in b_vals]

    best_a = np.ones(n_pairs, dtype=np.float32)
    best_b = np.zeros(n_pairs, dtype=np.float32)
    best_seg = base_seg.copy()
    best_pose = base_pose.copy()

    for rank, i in enumerate(opt_idx):
        s, p, obj = eval_candidates(
            segnet,
            posenet,
            even_m[i],
            odd_m[i],
            gt_seg[i],
            gt_pose[i],
            grid,
            args.pose_weight,
        )
        j = int(np.argmin(obj))
        best_a[i], best_b[i] = grid[j]
        best_seg[i], best_pose[i] = float(s[j]), float(p[j])
        if rank % 40 == 0:
            print(
                f"opt {rank:3d}/{top_k}: pair={i:3d} "
                f"seg {base_seg[i]:.6f}->{best_seg[i]:.6f} "
                f"pose {base_pose[i]:.6f}->{best_pose[i]:.6f} "
                f"a={best_a[i]:.3f} b={best_b[i]:+.1f}",
                flush=True,
            )

    # Save sidecar
    a_q = np.clip(np.round((best_a - 1.0) * 100.0), -127, 127).astype(np.int8)
    b_q = np.clip(np.round(best_b), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([a_q, b_q], axis=1).tobytes(), compresslevel=9)
    meta_out.write_bytes(packed)
    meta_bytes = len(packed)
    print(f"wrote {meta_out} ({meta_bytes} bytes)", flush=True)

    # Apply transform on odd full-res frames
    raw_new = raw.copy()
    for i in range(n_pairs):
        if abs(float(best_a[i]) - 1.0) > 1e-6 or abs(float(best_b[i])) > 1e-6:
            odd = raw_new[2 * i + 1].astype(np.float32)
            odd = np.clip(odd * float(best_a[i]) + float(best_b[i]), 0, 255).astype(np.uint8)
            raw_new[2 * i + 1] = odd
    raw_new.tofile(raw_out)
    print(f"wrote {raw_out}", flush=True)

    # Projected score from model-space metrics
    base_seg_mean = float(base_seg.mean())
    base_pose_mean = float(base_pose.mean())
    opt_seg_mean = float(best_seg.mean())
    opt_pose_mean = float(best_pose.mean())
    archive_bytes = int(args.archive_bytes) if int(args.archive_bytes) > 0 else int(ARCHIVE_ZIP.stat().st_size)
    base_score = 100 * base_seg_mean + math.sqrt(10 * base_pose_mean) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = 100 * opt_seg_mean + math.sqrt(10 * opt_pose_mean) + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)
    print("\n=== projected ===", flush=True)
    print(f"seg {base_seg_mean:.8f} -> {opt_seg_mean:.8f}", flush=True)
    print(f"pose {base_pose_mean:.8f} -> {opt_pose_mean:.8f}", flush=True)
    print(f"score {base_score:.4f} -> {opt_score:.4f}", flush=True)

    # Real fast_eval
    total_bytes = archive_bytes + meta_bytes
    py = ROOT / ".venv" / "Scripts" / "python.exe"
    cmd = [
        str(py),
        "-m",
        FAST_EVAL_MODULE,
        str(raw_out),
        str(total_bytes),
    ]
    print("\nRunning fast_eval...", flush=True)
    import subprocess
    subprocess.run(cmd, check=False, cwd=str(ROOT))


if __name__ == "__main__":
    import cv2
    main()

