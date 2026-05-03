#!/usr/bin/env python
"""
Optimize per-pair even-frame photometric (a,b) on an input raw sequence, then apply.
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
FAST_EVAL_MODULE = f"submissions.{SUB.name}.fast_eval"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-in", type=str, required=True)
    ap.add_argument("--raw-out", type=str, required=True)
    ap.add_argument("--meta-out", type=str, default=str(SUB / "archive" / "frame0_ab_q.bin"))
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

    a_coarse = np.array([0.92, 0.95, 0.98, 1.00, 1.02, 1.05, 1.08], dtype=np.float32)
    b_coarse = np.array([-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0], dtype=np.float32)
    coarse = [(float(a), float(b)) for a in a_coarse for b in b_coarse]
    a_fine = np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03], dtype=np.float32)
    b_fine = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    base_mse = np.zeros(n_pairs, dtype=np.float32)
    best_mse = np.zeros(n_pairs, dtype=np.float32)
    best_a = np.ones(n_pairs, dtype=np.float32)
    best_b = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        f0 = even_m[i]
        f1 = odd_m[i]
        g = gt_pose[i]
        b0 = eval_pose_candidates(posenet, f0, f1, g, [(1.0, 0.0)])[0]
        base_mse[i] = b0

        cm = eval_pose_candidates(posenet, f0, f1, g, coarse)
        cidx = int(np.argmin(cm))
        ca, cb = coarse[cidx]

        fine = [(float(ca + da), float(cb + db)) for da in a_fine for db in b_fine]
        fm = eval_pose_candidates(posenet, f0, f1, g, fine)
        fidx = int(np.argmin(fm))
        fa, fb = fine[fidx]
        best_a[i], best_b[i] = fa, fb
        best_mse[i] = float(fm[fidx])

        if i % 50 == 0:
            print(
                f"pair {i:3d}: base={b0:.6f} best={best_mse[i]:.6f} a={fa:.3f} b={fb:+.1f}",
                flush=True,
            )

    a_q = np.clip(np.round((best_a - 1.0) * 100.0), -127, 127).astype(np.int8)
    b_q = np.clip(np.round(best_b), -127, 127).astype(np.int8)
    packed = bz2.compress(np.stack([a_q, b_q], axis=1).tobytes(), compresslevel=9)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_bytes(packed)
    meta_bytes = len(packed)
    print(f"wrote {meta_out} ({meta_bytes} bytes)", flush=True)

    raw_new = raw.copy()
    for i in range(n_pairs):
        a = float(best_a[i])
        b = float(best_b[i])
        if abs(a - 1.0) <= 1e-6 and abs(b) <= 1e-6:
            continue
        raw_new[2 * i] = apply_ab(raw_new[2 * i], a, b)
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_new.tofile(raw_out)
    print(f"wrote {raw_out}", flush=True)

    base_pose = float(base_mse.mean())
    opt_pose = float(best_mse.mean())
    print(f"pose mse: {base_pose:.8f} -> {opt_pose:.8f}", flush=True)
    print(f"sqrt term: {math.sqrt(10*base_pose):.4f} -> {math.sqrt(10*opt_pose):.4f}", flush=True)
    if args.archive_bytes > 0:
        print(f"if bytes include metadata additively: 25*rate delta ~= {25*meta_bytes/37_545_489:.4f}", flush=True)

    if args.run_fast_eval:
        total = int(args.eval_total_bytes) if int(args.eval_total_bytes) > 0 else int(args.archive_bytes)
        if total <= 0:
            raise ValueError("Need --eval-total-bytes or --archive-bytes when --run-fast-eval is set")
        cmd = [
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            "-m",
            FAST_EVAL_MODULE,
            str(raw_out),
            str(total),
        ]
        print("Running fast_eval...", flush=True)
        subprocess.run(cmd, check=False, cwd=str(ROOT))


if __name__ == "__main__":
    main()

