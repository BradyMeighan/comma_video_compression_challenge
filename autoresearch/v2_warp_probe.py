#!/usr/bin/env python
"""Probe tiny per-pair frame-1 warp sidecars for the v2 mask-generator stack.

The decoder would not need SegNet/PoseNet for this idea: the encoder searches
small model-space translations offline, stores quantized dx/dy, and the decoder
applies the warp to generated frame 1. SegNet ignores frame 1, so this should
mostly affect PoseNet while spending only a few hundred bytes.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import lzma
import math
import os
import pickle
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FULL_DATA", "1")
os.environ.setdefault("CONFIG", "B")
# sidecar_stack imports an older helper that insists MODEL_PATH exists even
# though this probe only needs its cached evaluators.
os.environ.setdefault("MODEL_PATH", "autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt")

from prepare import (  # noqa: E402
    MODEL_H,
    MODEL_W,
    OUT_H,
    OUT_W,
    UNCOMPRESSED_SIZE,
    estimate_model_bytes,
    get_pose6,
    load_posenet,
    pack_pair_yuv6,
)
from sidecar_channel_only import apply_channel_patches  # noqa: E402
from sidecar_stack import fast_eval, fast_compose, per_pair_pose_mse  # noqa: E402
from train import Generator, load_data_full  # noqa: E402


OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
CACHE_DIR = OUTPUT_DIR / "v2_cache"


def _base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = (xx + 0.5) * (2.0 / width) - 1.0
    y = (yy + 0.5) * (2.0 / height) - 1.0
    return torch.stack([x, y], dim=-1)


def warp_chw_batch(
    frame_chw: torch.Tensor,
    params: list[tuple[float, float]],
    base_grid: torch.Tensor,
) -> torch.Tensor:
    """Return frame shifted by dx/dy pixels in output space for every param."""
    n = len(params)
    height, width = frame_chw.shape[-2:]
    offsets = torch.tensor(params, device=frame_chw.device, dtype=torch.float32)
    grid = base_grid.unsqueeze(0).expand(n, -1, -1, -1).clone()
    # output(x, y) samples input(x - dx, y - dy)
    grid[..., 0] -= offsets[:, 0].view(n, 1, 1) * (2.0 / width)
    grid[..., 1] -= offsets[:, 1].view(n, 1, 1) * (2.0 / height)
    inp = frame_chw.unsqueeze(0).expand(n, -1, -1, -1)
    return F.grid_sample(
        inp,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


def pose_mse_candidates(
    posenet: torch.nn.Module,
    f1_model_chw: torch.Tensor,
    f2_model_chw: torch.Tensor,
    gt_pose: torch.Tensor,
    params: list[tuple[float, float]],
    base_grid: torch.Tensor,
    batch_cands: int,
) -> np.ndarray:
    vals: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(params), batch_cands):
            chunk = params[start:start + batch_cands]
            f1w = warp_chw_batch(f1_model_chw, chunk, base_grid)
            f2r = f2_model_chw.unsqueeze(0).expand(f1w.shape[0], -1, -1, -1)
            pred = get_pose6(posenet, pack_pair_yuv6(f1w, f2r)).float()
            mse = (pred - gt_pose.view(1, 6)).pow(2).mean(dim=1)
            vals.append(mse.detach().cpu().numpy())
    return np.concatenate(vals)


def quantize_params(dx: float, dy: float, qscale: float) -> tuple[int, int, float, float]:
    qx = int(np.clip(np.round(dx * qscale), -127, 127))
    qy = int(np.clip(np.round(dy * qscale), -127, 127))
    return qx, qy, qx / qscale, qy / qscale


def find_xy_warps(
    f1_all: torch.Tensor,
    f2_all: torch.Tensor,
    poses: torch.Tensor,
    pair_indices: list[int],
    posenet: torch.nn.Module,
    device: torch.device,
    coarse: list[tuple[float, float]],
    fine_dx: np.ndarray,
    fine_dy: np.ndarray,
    qscale: float,
    batch_cands: int,
) -> dict[int, dict[str, float | int]]:
    base_grid = _base_grid(MODEL_H, MODEL_W, device)
    out: dict[int, dict[str, float | int]] = {}
    t0 = time.time()

    for rank_i, pi in enumerate(pair_indices, 1):
        f1 = f1_all[pi:pi + 1].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[pi:pi + 1].to(device).float().permute(0, 3, 1, 2)
        f1m = F.interpolate(f1, (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)[0]
        f2m = F.interpolate(f2, (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)[0]
        gt = poses[pi].to(device).float()

        coarse_mse = pose_mse_candidates(
            posenet, f1m, f2m, gt, coarse, base_grid, batch_cands)
        base_idx = coarse.index((0.0, 0.0))
        base_mse = float(coarse_mse[base_idx])
        cdx, cdy = coarse[int(np.argmin(coarse_mse))]

        fine = [
            (float(cdx + ddx), float(cdy + ddy))
            for ddy in fine_dy
            for ddx in fine_dx
        ]
        fine_mse = pose_mse_candidates(
            posenet, f1m, f2m, gt, fine, base_grid, batch_cands)
        best_i = int(np.argmin(fine_mse))
        best_dx, best_dy = fine[best_i]
        qx, qy, qdx, qdy = quantize_params(best_dx, best_dy, qscale)

        # Re-score the exact quantized value that would be decoded.
        q_mse = pose_mse_candidates(
            posenet, f1m, f2m, gt, [(qdx, qdy)], base_grid, batch_cands)[0]
        out[int(pi)] = {
            "base_mse": base_mse,
            "best_mse": float(q_mse),
            "improve": float(base_mse - q_mse),
            "dx": float(qdx),
            "dy": float(qdy),
            "qx": int(qx),
            "qy": int(qy),
        }

        if rank_i == 1 or rank_i % 25 == 0:
            accepted = sum(1 for r in out.values() if r["improve"] > 0 and (r["qx"] or r["qy"]))
            print(
                f"  searched {rank_i:>3}/{len(pair_indices)} "
                f"accepted={accepted:>3} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    return out


def serialize_sparse_warps(rows: dict[int, dict[str, float | int]]) -> bytes:
    active = [
        (pi, int(r["qx"]), int(r["qy"]))
        for pi, r in sorted(rows.items())
        if float(r["improve"]) > 0.0 and (int(r["qx"]) != 0 or int(r["qy"]) != 0)
    ]
    parts = [b"W2S\0", struct.pack("<H", len(active))]
    for pi, qx, qy in active:
        parts.append(struct.pack("<Hbb", int(pi), qx, qy))
    return b"".join(parts)


def serialize_dense_warps(rows: dict[int, dict[str, float | int]], n_pairs: int) -> bytes:
    q = np.zeros((n_pairs, 2), dtype=np.int8)
    for pi, r in rows.items():
        if float(r["improve"]) > 0.0 and (int(r["qx"]) != 0 or int(r["qy"]) != 0):
            q[int(pi), 0] = int(r["qx"])
            q[int(pi), 1] = int(r["qy"])
    return b"W2D\0" + q.tobytes()


def compressed_sizes(raw: bytes) -> dict[str, int]:
    return {
        "raw": len(raw),
        "bz2": len(bz2.compress(raw, compresslevel=9)),
        "lzma": len(lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)),
    }


def choose_rows_by_predicted_net(
    rows: dict[int, dict[str, float | int]],
    n_pairs: int,
    base_pose_dist: float,
) -> tuple[dict[int, dict[str, float | int]], dict[str, float | int]]:
    """Prefix sweep by per-pair predicted improvement, using dense-LZMA bytes."""
    active = [
        (pi, r)
        for pi, r in rows.items()
        if float(r["improve"]) > 0.0 and (int(r["qx"]) != 0 or int(r["qy"]) != 0)
    ]
    active.sort(key=lambda item: float(item[1]["improve"]), reverse=True)
    best_rows: dict[int, dict[str, float | int]] = {}
    best_info: dict[str, float | int] = {
        "k": 0,
        "pred_pose_dist": base_pose_dist,
        "pred_d_pose_term": 0.0,
        "bytes_lzma": compressed_sizes(serialize_dense_warps({}, n_pairs))["lzma"],
        "pred_net": 25 * compressed_sizes(serialize_dense_warps({}, n_pairs))["lzma"] / UNCOMPRESSED_SIZE,
    }
    accum = 0.0
    cur: dict[int, dict[str, float | int]] = {}
    for k, (pi, r) in enumerate(active, 1):
        cur[pi] = r
        accum += float(r["improve"])
        pred_pose_dist = max(0.0, base_pose_dist - accum / n_pairs)
        d_pose = math.sqrt(10.0 * pred_pose_dist) - math.sqrt(10.0 * base_pose_dist)
        raw = serialize_dense_warps(cur, n_pairs)
        cb = compressed_sizes(raw)["lzma"]
        d_bytes = 25 * cb / UNCOMPRESSED_SIZE
        net = d_pose + d_bytes
        if net < float(best_info["pred_net"]):
            best_rows = dict(cur)
            best_info = {
                "k": k,
                "pred_pose_dist": pred_pose_dist,
                "pred_d_pose_term": d_pose,
                "bytes_lzma": cb,
                "pred_net": net,
            }
    return best_rows, best_info


def apply_warps_hwc(
    f1_all: torch.Tensor,
    rows: dict[int, dict[str, float | int]],
    qscale: float,
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    active = [
        (int(pi), int(r["qx"]), int(r["qy"]))
        for pi, r in sorted(rows.items())
        if float(r["improve"]) > 0.0 and (int(r["qx"]) != 0 or int(r["qy"]) != 0)
    ]
    out = f1_all.clone()
    grid = _base_grid(OUT_H, OUT_W, device)
    sx = OUT_W / MODEL_W
    sy = OUT_H / MODEL_H
    with torch.inference_mode():
        for start in range(0, len(active), batch_size):
            chunk = active[start:start + batch_size]
            pair_ids = [x[0] for x in chunk]
            params = [(x[1] / qscale * sx, x[2] / qscale * sy) for x in chunk]
            batch = out[pair_ids].to(device).float().permute(0, 3, 1, 2)
            b = batch.shape[0]
            offsets = torch.tensor(params, device=device, dtype=torch.float32)
            g = grid.unsqueeze(0).expand(b, -1, -1, -1).clone()
            g[..., 0] -= offsets[:, 0].view(b, 1, 1) * (2.0 / OUT_W)
            g[..., 1] -= offsets[:, 1].view(b, 1, 1) * (2.0 / OUT_H)
            warped = F.grid_sample(
                batch,
                g,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
            out[pair_ids] = warped.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--base", choices=("x5", "x5rgb"), default="x5rgb")
    ap.add_argument("--batch-cands", type=int, default=64)
    ap.add_argument("--qscale", type=float, default=10.0)
    ap.add_argument("--eval-final", action="store_true", default=True)
    ap.add_argument("--no-eval-final", action="store_false", dest="eval_final")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"Device: {device}", flush=True)
    print(f"Base: {args.base}, n_pairs={args.n_pairs}", flush=True)

    if not (CACHE_DIR / "frames_x5.pt").exists():
        raise FileNotFoundError(f"Missing {CACHE_DIR / 'frames_x5.pt'}")

    print("Loading full data + PoseNet + cached frames...", flush=True)
    data = load_data_full(device)
    posenet = load_posenet(device)
    for p in posenet.parameters():
        p.requires_grad_(False)
    gen_shell = Generator()
    model_bytes = estimate_model_bytes(gen_shell)

    frames = torch.load(CACHE_DIR / "frames_x5.pt", weights_only=False)
    f1_base = frames["f1"]
    f2_base = frames["f2"]
    if args.base == "x5rgb":
        with open(CACHE_DIR / "rgb_patches_x5.pkl", "rb") as f:
            rgb_patches = pickle.load(f)
        f1_work = apply_channel_patches(f1_base, rgb_patches)
    else:
        f1_work = f1_base

    print("Evaluating current base and ranking pose errors...", flush=True)
    seg_base, pose_base = fast_eval(f1_work, f2_base, data["val_rgb"], device)
    base_score = fast_compose(seg_base, pose_base, model_bytes, sidecar_bytes=0)
    pose_per_pair = per_pair_pose_mse(f1_work, f2_base, data["val_poses"], posenet, device)
    rank = np.argsort(pose_per_pair)[::-1]
    pairs = [int(x) for x in rank[:min(args.n_pairs, len(rank))]]
    print(
        f"  base seg={seg_base:.9f} pose={pose_base:.9f} "
        f"pose_term={base_score['pose_term']:.6f}",
        flush=True,
    )

    dx_vals = np.arange(-2.5, 2.51, 0.5, dtype=np.float32)
    dy_vals = np.arange(-1.0, 1.01, 0.5, dtype=np.float32)
    coarse = [(float(dx), float(dy)) for dy in dy_vals for dx in dx_vals]
    fine_dx = np.arange(-0.4, 0.41, 0.1, dtype=np.float32)
    fine_dy = np.arange(-0.2, 0.21, 0.1, dtype=np.float32)

    print("Searching per-pair dx/dy in model space...", flush=True)
    t0 = time.time()
    rows = find_xy_warps(
        f1_work,
        f2_base,
        data["val_poses"],
        pairs,
        posenet,
        device,
        coarse,
        fine_dx,
        fine_dy,
        args.qscale,
        args.batch_cands,
    )
    elapsed = time.time() - t0

    all_active = {
        pi: r for pi, r in rows.items()
        if float(r["improve"]) > 0.0 and (int(r["qx"]) != 0 or int(r["qy"]) != 0)
    }
    best_rows, best_info = choose_rows_by_predicted_net(rows, f1_work.shape[0], pose_base)

    raw_sparse = serialize_sparse_warps(all_active)
    raw_dense = serialize_dense_warps(all_active, f1_work.shape[0])
    sizes_sparse = compressed_sizes(raw_sparse)
    sizes_dense = compressed_sizes(raw_dense)

    print("\n=== warp probe prediction ===", flush=True)
    print(f"searched={len(rows)} active_positive={len(all_active)} elapsed={elapsed:.0f}s", flush=True)
    print(f"all-active sparse bytes: {sizes_sparse}", flush=True)
    print(f"all-active dense  bytes: {sizes_dense}", flush=True)
    print(
        f"best dense-prefix k={best_info['k']} bytes_lzma={best_info['bytes_lzma']} "
        f"pred_d_pose={best_info['pred_d_pose_term']:+.6f} "
        f"pred_net={best_info['pred_net']:+.6f}",
        flush=True,
    )

    final = {}
    if args.eval_final and best_rows:
        print("\nApplying best prefix at camera resolution and running full eval...", flush=True)
        f1_warped = apply_warps_hwc(f1_work, best_rows, args.qscale, device)
        seg_new, pose_new = fast_eval(f1_warped, f2_base, data["val_rgb"], device)
        raw_best_dense = serialize_dense_warps(best_rows, f1_work.shape[0])
        raw_best_sparse = serialize_sparse_warps(best_rows)
        best_dense_sizes = compressed_sizes(raw_best_dense)
        best_sparse_sizes = compressed_sizes(raw_best_sparse)
        best_bytes = min(best_dense_sizes["lzma"], best_sparse_sizes["lzma"])
        new_score = fast_compose(seg_new, pose_new, model_bytes, sidecar_bytes=best_bytes)
        delta = (
            (new_score["seg_term"] - base_score["seg_term"])
            + (new_score["pose_term"] - base_score["pose_term"])
            + new_score["rate_term"] - fast_compose(seg_base, pose_base, model_bytes, 0)["rate_term"]
        )
        final = {
            "seg": seg_new,
            "pose": pose_new,
            "pose_term": new_score["pose_term"],
            "best_bytes_lzma": best_bytes,
            "dense_lzma": best_dense_sizes["lzma"],
            "sparse_lzma": best_sparse_sizes["lzma"],
            "score_delta_additive": delta,
        }
        print(
            f"final seg={seg_new:.9f} pose={pose_new:.9f} "
            f"pose_term={new_score['pose_term']:.6f}",
            flush=True,
        )
        print(
            f"best warp bytes lzma={best_bytes} "
            f"(dense={best_dense_sizes['lzma']}, sparse={best_sparse_sizes['lzma']}) "
            f"additive score delta={delta:+.6f}",
            flush=True,
        )

    out_csv = OUTPUT_DIR / f"v2_warp_probe_{args.base}_n{args.n_pairs}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pair",
            "base_mse",
            "best_mse",
            "improve",
            "dx",
            "dy",
            "qx",
            "qy",
        ])
        for pi, r in sorted(rows.items(), key=lambda item: float(item[1]["improve"]), reverse=True):
            w.writerow([
                pi,
                r["base_mse"],
                r["best_mse"],
                r["improve"],
                r["dx"],
                r["dy"],
                r["qx"],
                r["qy"],
            ])
    print(f"wrote {out_csv}", flush=True)

    summary_csv = OUTPUT_DIR / "v2_warp_probe_summary.csv"
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "base",
                "n_pairs",
                "active",
                "best_k",
                "pred_net",
                "pred_d_pose",
                "pred_bytes_lzma",
                "final_score_delta",
                "final_pose_term",
                "final_bytes_lzma",
                "elapsed_s",
            ])
        w.writerow([
            args.base,
            args.n_pairs,
            len(all_active),
            best_info["k"],
            best_info["pred_net"],
            best_info["pred_d_pose_term"],
            best_info["bytes_lzma"],
            final.get("score_delta_additive", ""),
            final.get("pose_term", ""),
            final.get("best_bytes_lzma", ""),
            elapsed,
        ])
    print(f"summary appended: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
