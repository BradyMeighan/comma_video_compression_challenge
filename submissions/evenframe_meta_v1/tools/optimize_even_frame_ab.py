#!/usr/bin/env python
"""
Optimize per-pair affine photometric correction on EVEN frames only.
frame0' = clip(a * frame0 + b)

Run on top of current corrected raw output to improve residual pose error.
"""
import bz2
import math
from pathlib import Path

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


def apply_ab(frame: np.ndarray, a: float, b: float) -> np.ndarray:
    x = frame.astype(np.float32) * a + b
    return np.clip(x, 0, 255).astype(np.uint8)


def eval_pose_candidates(posenet, f0: np.ndarray, f1: np.ndarray, gt_pose: torch.Tensor, params):
    cands = np.stack([apply_ab(f0, a, b) for a, b in params], axis=0)
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
    archive_bytes = ARCHIVE_ZIP.stat().st_size

    raw = np.fromfile(RAW_PATH, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)

    even_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    odd_m = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        even_m[i] = cv2.resize(raw[2 * i], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        odd_m[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    a_coarse = np.array([0.92, 0.96, 1.00, 1.04, 1.08], dtype=np.float32)
    b_coarse = np.array([-10.0, -6.0, -2.0, 0.0, 2.0, 6.0, 10.0], dtype=np.float32)
    coarse = [(float(a), float(b)) for a in a_coarse for b in b_coarse]

    a_fine = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    b_fine = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    base_mse = np.zeros(n_pairs, dtype=np.float32)
    best_mse = np.zeros(n_pairs, dtype=np.float32)
    best_a = np.zeros(n_pairs, dtype=np.float32)
    best_b = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        f0 = even_m[i]
        f1 = odd_m[i]
        g = gt_pose[i]

        b0 = eval_pose_candidates(posenet, f0, f1, g, [(1.0, 0.0)])[0]
        base_mse[i] = b0

        coarse_mse = eval_pose_candidates(posenet, f0, f1, g, coarse)
        cidx = int(np.argmin(coarse_mse))
        ca, cb = coarse[cidx]

        fine = [(float(ca + da), float(cb + db)) for da in a_fine for db in b_fine]
        fine_mse = eval_pose_candidates(posenet, f0, f1, g, fine)
        fidx = int(np.argmin(fine_mse))
        fa, fb = fine[fidx]
        fm = float(fine_mse[fidx])

        best_a[i] = fa
        best_b[i] = fb
        best_mse[i] = fm

        if i % 50 == 0:
            print(f"pair {i:3d}: base={b0:.6f} best={fm:.6f} a={fa:.3f} b={fb:+.1f}", flush=True)

    base_pose = float(base_mse.mean())
    opt_pose = float(best_mse.mean())
    baseline_seg = 0.00564963  # current seg from official eval with dxyr

    # quantize metadata
    # a_q in 0.01 around 1.0, int8: a = 1 + a_q/100
    a_q = np.clip(np.round((best_a - 1.0) * 100.0), -127, 127).astype(np.int8)
    # b_q in 1.0 levels
    b_q = np.clip(np.round(best_b), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([a_q, b_q], axis=1).tobytes(), compresslevel=9)
    meta_bytes = len(packed)

    base_score = 100 * baseline_seg + math.sqrt(10 * base_pose) + 25 * (archive_bytes / UNCOMPRESSED_BYTES)
    opt_score = 100 * baseline_seg + math.sqrt(10 * opt_pose) + 25 * ((archive_bytes + meta_bytes) / UNCOMPRESSED_BYTES)

    print("\n=== AB optimization (projected) ===", flush=True)
    print(f"baseline pose mse: {base_pose:.8f} sqrt={math.sqrt(10*base_pose):.4f}", flush=True)
    print(f"optimized pose mse: {opt_pose:.8f} sqrt={math.sqrt(10*opt_pose):.4f}", flush=True)
    print(f"meta bytes (bz2): {meta_bytes}", flush=True)
    print(f"baseline score: {base_score:.4f}", flush=True)
    print(f"optimized score: {opt_score:.4f}", flush=True)
    print(f"improvement: {base_score - opt_score:+.4f}", flush=True)

    out = ROOT / "submissions" / "av1_repro" / "archive" / "frame0_ab_q.bin"
    out.write_bytes(packed)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    import cv2
    main()

