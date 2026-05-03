#!/usr/bin/env python
"""
Validate even-frame (dx, dy) correction with full SegNet+PoseNet evaluation.
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
DXY_META_PATH = ROOT / "submissions" / "av1_repro" / "archive" / "frame0_dxy_q.bin"
ARCHIVE_ZIP = ROOT / "submissions" / "av1_repro" / "archive.zip"
GT_CACHE = ROOT / "submissions" / "av1_repro" / "_cache" / "gt.pt"

UNCOMPRESSED_BYTES = 37_545_489
W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models():
    import sys

    sys.path.insert(0, str(ROOT))
    from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path

    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def main():
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    gt = torch.load(GT_CACHE, weights_only=True)
    n_pairs = gt["seg"].shape[0]
    archive_bytes = ARCHIVE_ZIP.stat().st_size

    raw = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)

    q = np.frombuffer(bz2.decompress(DXY_META_PATH.read_bytes()), dtype=np.int8).reshape(-1, 2)
    if q.shape[0] != n_pairs:
        raise ValueError(f"metadata length mismatch: {q.shape[0]} vs {n_pairs}")
    dxy_model = q.astype(np.float32) / 10.0
    sx = W_CAM / MODEL_W
    sy = H_CAM / MODEL_H
    dx_full = dxy_model[:, 0] * sx
    dy_full = dxy_model[:, 1] * sy

    shifted = raw.copy()
    for i in range(n_pairs):
        M = np.array(
            [[1.0, 0.0, float(dx_full[i])], [0.0, 1.0, float(dy_full[i])]],
            dtype=np.float32,
        )
        shifted[2 * i] = cv2.warpAffine(
            shifted[2 * i],
            M,
            dsize=(W_CAM, H_CAM),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    seg_dists, pose_dists = [], []
    bs = 16
    with torch.inference_mode():
        for i in range(0, n_pairs, bs):
            end = min(i + bs, n_pairs)
            f0 = torch.from_numpy(shifted[2 * i : 2 * end : 2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(shifted[2 * i + 1 : 2 * end : 2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, "b t h w c -> b t c h w")

            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            gt_seg = gt["seg"][i:end].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1, 2)).cpu().tolist())

            pose = posenet(posenet.preprocess_input(x))["pose"][:, :6]
            gt_pose = gt["pose"][i:end].to(DEVICE)
            pose_dists.extend((pose - gt_pose).pow(2).mean(1).cpu().tolist())

    seg = float(np.mean(seg_dists))
    pose = float(np.mean(pose_dists))
    meta_bytes = len(DXY_META_PATH.read_bytes())
    rate = (archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES
    score = 100 * seg + math.sqrt(10 * pose) + 25 * rate

    print("\n=== Full validation (even-frame dxy) ===", flush=True)
    print(f"archive bytes: {archive_bytes} ({archive_bytes/1024:.1f} KB)", flush=True)
    print(f"metadata bytes: {meta_bytes} ({meta_bytes/1024:.2f} KB)", flush=True)
    print(
        f"seg={seg:.8f} pose={pose:.8f} "
        f"100s={100*seg:.4f} sqrtp={math.sqrt(10*pose):.4f} 25r={25*rate:.4f}",
        flush=True,
    )
    print(f"score={score:.4f}", flush=True)


if __name__ == "__main__":
    main()

