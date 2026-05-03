#!/usr/bin/env python
"""
Sweep compact odd-frame sparse boundary correction sidecars.

Idea:
- detect seg mistakes at GT boundaries on odd frames
- keep only top-K points per pair
- store per-point tiny RGB deltas
- apply local patches at decode-time (odd frames only)
- measure seg/pose/rate tradeoff
"""
import argparse
import bz2
import math
import struct
from pathlib import Path

import av
import cv2
import einops
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "submissions" / "evenframe_meta_v1" / "inflated" / "0.raw"
GT_CACHE = SUB / "_cache" / "gt.pt"
GT_VIDEO = ROOT / "videos" / "0.mkv"
UNCOMPRESSED = 37_545_489
W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_ints(csv: str):
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def parse_floats(csv: str):
    return [float(x.strip()) for x in csv.split(",") if x.strip()]


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


def boundary_mask(seg_gt: np.ndarray) -> np.ndarray:
    # seg_gt: (N,H,W)
    c = seg_gt
    up = np.roll(c, 1, axis=1)
    dn = np.roll(c, -1, axis=1)
    lf = np.roll(c, 1, axis=2)
    rt = np.roll(c, -1, axis=2)
    return (c != up) | (c != dn) | (c != lf) | (c != rt)


def decode_gt_odd_model() -> np.ndarray:
    from frame_utils import yuv420_to_rgb

    container = av.open(str(GT_VIDEO))
    stream = container.streams.video[0]
    out = []
    n = 0
    for fr in container.decode(stream):
        if n % 2 == 1:
            rgb = yuv420_to_rgb(fr).numpy()
            out.append(cv2.resize(rgb, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR))
        n += 1
    container.close()
    return np.stack(out, axis=0).astype(np.uint8)


def predict_seg(segnet, raw_mm: np.memmap, n_pairs: int) -> np.ndarray:
    preds = []
    bs = 16
    with torch.inference_mode():
        for i in range(0, n_pairs, bs):
            e = min(i + bs, n_pairs)
            f0 = torch.from_numpy(raw_mm[2 * i : 2 * e : 2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw_mm[2 * i + 1 : 2 * e : 2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, "b t h w c -> b t c h w")
            seg = segnet(segnet.preprocess_input(x)).argmax(1).cpu().numpy()
            preds.append(seg)
    return np.concatenate(preds, axis=0).astype(np.uint8)


def eval_with_sparse(
    segnet,
    posenet,
    gt_seg_t: torch.Tensor,
    gt_pose_t: torch.Tensor,
    raw_mm: np.memmap,
    pts_xy: np.ndarray,   # (N,K,2) uint16 full-res coords
    pts_d: np.ndarray,    # (N,K,3) int8 deltas
    radius: int,
    gain: float,
    total_bytes: int,
) -> tuple[float, float, float]:
    n_pairs, k = pts_xy.shape[0], pts_xy.shape[1]
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    kernel /= kernel.sum()

    seg_dists, pose_dists = [], []
    bs = 12
    with torch.inference_mode():
        for i in range(0, n_pairs, bs):
            e = min(i + bs, n_pairs)
            bsz = e - i
            f0 = raw_mm[2 * i : 2 * e : 2].copy()  # uint8
            f1 = raw_mm[2 * i + 1 : 2 * e : 2].copy().astype(np.float32)

            for b in range(bsz):
                pair = i + b
                arr = f1[b]
                for j in range(k):
                    d = pts_d[pair, j].astype(np.float32) * gain
                    if d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0:
                        continue
                    x = int(pts_xy[pair, j, 0])
                    y = int(pts_xy[pair, j, 1])
                    x1 = max(0, x - radius)
                    x2 = min(W_CAM, x + radius + 1)
                    y1 = max(0, y - radius)
                    y2 = min(H_CAM, y + radius + 1)
                    kx1 = x1 - (x - radius)
                    kx2 = kx1 + (x2 - x1)
                    ky1 = y1 - (y - radius)
                    ky2 = ky1 + (y2 - y1)
                    w = kernel[ky1:ky2, kx1:kx2, None]
                    arr[y1:y2, x1:x2] += w * d[None, None, :]
                f1[b] = np.clip(arr, 0, 255)

            f0_t = torch.from_numpy(f0).to(DEVICE).float()
            f1_t = torch.from_numpy(f1.astype(np.uint8)).to(DEVICE).float()
            x = torch.stack([f0_t, f1_t], dim=1)
            x = einops.rearrange(x, "b t h w c -> b t c h w")

            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            gt_seg = gt_seg_t[i:e].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1, 2)).cpu().tolist())

            pose = posenet(posenet.preprocess_input(x))["pose"][:, :6]
            gt_pose = gt_pose_t[i:e].to(DEVICE)
            pose_dists.extend((pose - gt_pose).pow(2).mean(1).cpu().tolist())

    seg_d = float(np.mean(seg_dists))
    pose_d = float(np.mean(pose_dists))
    rate = total_bytes / UNCOMPRESSED
    score = 100 * seg_d + math.sqrt(10 * pose_d) + 25 * rate
    return seg_d, pose_d, score


def build_points_for_k(
    k: int,
    raw_mm: np.memmap,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    gt_odd_model: np.ndarray,
    dclip: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = gt_seg.shape[0]
    # model-space decoded odd for color-diff ranking / deltas
    dec_odd_model = np.zeros((n, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n):
        dec_odd_model[i] = cv2.resize(raw_mm[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    bmask = boundary_mask(gt_seg)
    mismatch = pred_seg != gt_seg
    cand = mismatch & bmask
    color_err = np.abs(gt_odd_model.astype(np.int16) - dec_odd_model.astype(np.int16)).sum(axis=3).astype(np.int32)

    pts_xy = np.zeros((n, k, 2), dtype=np.uint16)
    pts_d = np.zeros((n, k, 3), dtype=np.int8)

    sx = W_CAM / MODEL_W
    sy = H_CAM / MODEL_H
    for i in range(n):
        cm = cand[i]
        score = color_err[i].copy()
        score[~cm] = -1
        flat = score.reshape(-1)
        # fallback if not enough candidate points
        if np.count_nonzero(cm) < k:
            flat = color_err[i].reshape(-1)
        if k >= flat.size:
            idx = np.argsort(flat)[::-1][:k]
        else:
            idx = np.argpartition(flat, -k)[-k:]
            idx = idx[np.argsort(flat[idx])[::-1]]

        ys = (idx // MODEL_W).astype(np.int32)
        xs = (idx % MODEL_W).astype(np.int32)
        xf = np.clip(np.round((xs + 0.5) * sx - 0.5), 0, W_CAM - 1).astype(np.uint16)
        yf = np.clip(np.round((ys + 0.5) * sy - 0.5), 0, H_CAM - 1).astype(np.uint16)
        pts_xy[i, :, 0] = xf
        pts_xy[i, :, 1] = yf

        d = gt_odd_model[i, ys, xs].astype(np.int16) - dec_odd_model[i, ys, xs].astype(np.int16)
        d = np.clip(d, -dclip, dclip).astype(np.int8)
        pts_d[i] = d
    return pts_xy, pts_d


def sidecar_size_bytes(pts_xy: np.ndarray, pts_d: np.ndarray, radius: int, gain_q: int) -> int:
    n, k = pts_xy.shape[0], pts_xy.shape[1]
    payload = (
        struct.pack("<IIII", n, k, radius, gain_q)
        + pts_xy.astype(np.uint16).tobytes()
        + pts_d.astype(np.int8).tobytes()
    )
    return len(bz2.compress(payload, compresslevel=9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-in", type=str, default=str(RAW_DEFAULT))
    ap.add_argument("--archive-bytes", type=int, default=903995)
    ap.add_argument("--k-list", type=str, default="8,16,32,48")
    ap.add_argument("--gain-list", type=str, default="0.8,1.0")
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--dclip", type=int, default=48)
    args = ap.parse_args()

    raw_path = Path(args.raw_in)
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw input: {raw_path}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")
    if not GT_VIDEO.exists():
        raise FileNotFoundError(f"Missing GT video: {GT_VIDEO}")

    print(f"Device: {DEVICE}", flush=True)
    gt = torch.load(GT_CACHE, weights_only=True)
    gt_seg = gt["seg"].numpy().astype(np.uint8)   # (600,384,512)
    gt_pose_t = gt["pose"].float()
    n_pairs = gt_seg.shape[0]

    raw_mm = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(n_pairs * 2, H_CAM, W_CAM, 3))
    segnet, posenet = load_models()

    print("Decoding GT odd frames (model-size)...", flush=True)
    gt_odd_model = decode_gt_odd_model()
    print("Predicting baseline seg maps...", flush=True)
    pred_seg = predict_seg(segnet, raw_mm, n_pairs)

    # baseline score
    base_seg, base_pose, base_score = eval_with_sparse(
        segnet=segnet,
        posenet=posenet,
        gt_seg_t=torch.from_numpy(gt_seg),
        gt_pose_t=gt_pose_t,
        raw_mm=raw_mm,
        pts_xy=np.zeros((n_pairs, 0, 2), dtype=np.uint16),
        pts_d=np.zeros((n_pairs, 0, 3), dtype=np.int8),
        radius=args.radius,
        gain=1.0,
        total_bytes=args.archive_bytes,
    )
    print(
        f"baseline: seg={base_seg:.8f} pose={base_pose:.8f} "
        f"25r={25*args.archive_bytes/UNCOMPRESSED:.4f} score={base_score:.4f}",
        flush=True,
    )

    k_list = parse_ints(args.k_list)
    gain_list = parse_floats(args.gain_list)
    best = None
    for k in k_list:
        print(f"\nBuilding points for K={k}...", flush=True)
        pts_xy, pts_d = build_points_for_k(
            k=k,
            raw_mm=raw_mm,
            gt_seg=gt_seg,
            pred_seg=pred_seg,
            gt_odd_model=gt_odd_model,
            dclip=args.dclip,
        )
        for gain in gain_list:
            side_bytes = sidecar_size_bytes(pts_xy, pts_d, radius=args.radius, gain_q=int(round(gain * 1000)))
            total = args.archive_bytes + side_bytes
            seg_d, pose_d, score = eval_with_sparse(
                segnet=segnet,
                posenet=posenet,
                gt_seg_t=torch.from_numpy(gt_seg),
                gt_pose_t=gt_pose_t,
                raw_mm=raw_mm,
                pts_xy=pts_xy,
                pts_d=pts_d,
                radius=args.radius,
                gain=gain,
                total_bytes=total,
            )
            print(
                f"K={k:2d} gain={gain:.2f} sidecar={side_bytes:6d}B "
                f"seg={seg_d:.8f} pose={pose_d:.8f} score={score:.4f}",
                flush=True,
            )
            if best is None or score < best["score"]:
                best = {
                    "k": k,
                    "gain": gain,
                    "side_bytes": side_bytes,
                    "seg": seg_d,
                    "pose": pose_d,
                    "score": score,
                }

    print("\n=== Best sparse boundary config ===", flush=True)
    print(best, flush=True)


if __name__ == "__main__":
    main()

