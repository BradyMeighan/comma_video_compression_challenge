#!/usr/bin/env python
"""
Analyze residual seg/pose error sources for a decoded raw file.
"""
import argparse
import math
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
UNCOMPRESSED_BYTES = 37_545_489
W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["road", "lane_marking", "undrivable", "movable", "car_vehicle"]


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


def boundary_mask(seg: np.ndarray) -> np.ndarray:
    """
    seg: (N,H,W) uint8/int64
    Return 1-px boundary mask via morphological gradient per frame.
    """
    n, h, w = seg.shape
    out = np.zeros((n, h, w), dtype=bool)
    k = np.ones((3, 3), np.uint8)
    for i in range(n):
        s = seg[i].astype(np.uint8)
        grad = cv2.morphologyEx(s, cv2.MORPH_GRADIENT, k)
        out[i] = grad > 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-in", type=str, default=str(RAW_IN))
    ap.add_argument("--archive-bytes", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    raw_in = Path(args.raw_in)
    if not raw_in.exists():
        raise FileNotFoundError(raw_in)
    if not GT_CACHE.exists():
        raise FileNotFoundError(GT_CACHE)
    archive_bytes = int(args.archive_bytes) if int(args.archive_bytes) > 0 else int((SUB / "archive.zip").stat().st_size)

    segnet, posenet = load_models()
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_seg = gt["seg"].cpu().numpy().astype(np.int64)   # (N,H,W)
    gt_pose = gt["pose"].cpu()                          # (N,6)
    n_pairs = int(gt_seg.shape[0])

    raw = np.fromfile(raw_in, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    f0 = raw[0::2].copy()
    f1 = raw[1::2].copy()

    pred_seg_chunks = []
    pred_pose_chunks = []
    bs = int(args.batch_size)
    for s in range(0, n_pairs, bs):
        e = min(n_pairs, s + bs)
        x0 = torch.from_numpy(f0[s:e].copy()).to(DEVICE).float()
        x1 = torch.from_numpy(f1[s:e].copy()).to(DEVICE).float()
        x = torch.stack([x0, x1], dim=1)
        x = einops.rearrange(x, "b t h w c -> b t c h w")
        with torch.inference_mode():
            seg = segnet(segnet.preprocess_input(x)).argmax(1).cpu().numpy().astype(np.int64)
            pose = posenet(posenet.preprocess_input(x))["pose"][:, :6].cpu()
        pred_seg_chunks.append(seg)
        pred_pose_chunks.append(pose)
    pred_seg = np.concatenate(pred_seg_chunks, axis=0)
    pred_pose = torch.cat(pred_pose_chunks, dim=0)

    err = (pred_seg != gt_seg)
    per_pair_seg = err.reshape(n_pairs, -1).mean(axis=1)
    per_pair_pose = (pred_pose - gt_pose).pow(2).mean(dim=1).numpy()
    mean_seg = float(per_pair_seg.mean())
    mean_pose = float(per_pair_pose.mean())
    score = 100 * mean_seg + math.sqrt(10 * mean_pose) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)

    print(f"pairs={n_pairs} device={DEVICE}")
    print(f"seg={mean_seg:.8f} pose={mean_pose:.8f} archive={archive_bytes} score={score:.4f}")
    print(
        f"contrib: seg={100*mean_seg:.4f} pose={math.sqrt(10*mean_pose):.4f} "
        f"rate={25*(archive_bytes/UNCOMPRESSED_BYTES):.4f}"
    )

    # Concentration
    for k in [10, 20, 50, 100]:
        top_seg = np.sort(per_pair_seg)[-k:].sum() / np.maximum(per_pair_seg.sum(), 1e-12)
        top_pose = np.sort(per_pair_pose)[-k:].sum() / np.maximum(per_pair_pose.sum(), 1e-12)
        print(f"top{k:3d} share: seg={top_seg*100:5.1f}% pose={top_pose*100:5.1f}%")

    # Confusion matrix
    cm = np.zeros((5, 5), dtype=np.int64)
    gt_flat = gt_seg.reshape(-1)
    pr_flat = pred_seg.reshape(-1)
    valid = (gt_flat >= 0) & (gt_flat < 5) & (pr_flat >= 0) & (pr_flat < 5)
    idx = gt_flat[valid] * 5 + pr_flat[valid]
    binc = np.bincount(idx, minlength=25)
    cm = binc.reshape(5, 5)

    print("\nConfusions (gt -> pred) top off-diagonal:")
    rows = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            rows.append((int(cm[i, j]), i, j))
    rows.sort(reverse=True)
    for c, i, j in rows[:10]:
        if c <= 0:
            break
        print(f"  {CLASS_NAMES[i]:>12} -> {CLASS_NAMES[j]:<12} : {c}")

    # Boundary vs interior
    bmask = boundary_mask(gt_seg)
    b_err = err[bmask].mean() if bmask.any() else 0.0
    i_err = err[~bmask].mean() if (~bmask).any() else 0.0
    b_share = err[bmask].sum() / np.maximum(err.sum(), 1)
    b_ratio = (b_err / np.maximum(i_err, 1e-12)) if i_err > 0 else 0.0
    print("\nBoundary analysis:")
    print(f"  boundary_err={b_err:.6f} interior_err={i_err:.6f}")
    print(f"  boundary_share_of_all_errors={100*b_share:.2f}%")
    print(f"  boundary_vs_interior_error_ratio={b_ratio:.2f}x")

    # Spatial map (coarse)
    gh, gw = 12, 16
    eh = MODEL_H // gh
    ew = MODEL_W // gw
    heat = np.zeros((gh, gw), dtype=np.float64)
    for iy in range(gh):
        for ix in range(gw):
            ys, ye = iy * eh, (iy + 1) * eh
            xs, xe = ix * ew, (ix + 1) * ew
            heat[iy, ix] = err[:, ys:ye, xs:xe].mean()
    flat = [(float(heat[iy, ix]), iy, ix) for iy in range(gh) for ix in range(gw)]
    flat.sort(reverse=True)
    print("\nWorst spatial blocks (12x16 grid, model space):")
    for v, iy, ix in flat[:8]:
        print(f"  block(y={iy:02d}, x={ix:02d}) err={v:.6f}")

    # Worst pairs
    print("\nWorst seg pairs:")
    idx_seg = np.argsort(per_pair_seg)[-10:][::-1]
    for i in idx_seg:
        print(f"  pair {i:3d}: seg={per_pair_seg[i]:.6f} pose={per_pair_pose[i]:.6f}")

    print("\nWorst pose pairs:")
    idx_pose = np.argsort(per_pair_pose)[-10:][::-1]
    for i in idx_pose:
        print(f"  pair {i:3d}: pose={per_pair_pose[i]:.6f} seg={per_pair_seg[i]:.6f}")


if __name__ == "__main__":
    main()

