#!/usr/bin/env python
"""Select v2 warp candidates using actual camera-resolution PoseNet gains.

v2_warp_probe searches in 384x512 model space. This script takes its candidate
CSV, applies all nonzero candidates at camera resolution, measures the actual
per-pair pose MSE changes, and chooses the byte-aware subset from those actual
gains.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("FULL_DATA", "1")
os.environ.setdefault("CONFIG", "B")
os.environ.setdefault("MODEL_PATH", "autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt")

from prepare import UNCOMPRESSED_SIZE, estimate_model_bytes, load_posenet  # noqa: E402
from sidecar_channel_only import apply_channel_patches  # noqa: E402
from sidecar_stack import fast_compose, fast_eval, per_pair_pose_mse  # noqa: E402
from train import Generator, load_data_full  # noqa: E402
from v2_warp_probe import (  # noqa: E402
    CACHE_DIR,
    OUTPUT_DIR,
    apply_warps_hwc,
    compressed_sizes,
    serialize_dense_warps,
    serialize_sparse_warps,
)


def read_candidates(path: Path) -> dict[int, dict[str, float | int]]:
    rows: dict[int, dict[str, float | int]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pi = int(row["pair"])
            qx = int(row["qx"])
            qy = int(row["qy"])
            if qx == 0 and qy == 0:
                continue
            rows[pi] = {
                "qx": qx,
                "qy": qy,
                "dx": float(row["dx"]),
                "dy": float(row["dy"]),
                "improve": float(row["improve"]),
                "base_mse": float(row["base_mse"]),
                "best_mse": float(row["best_mse"]),
            }
    return rows


def choose_actual_subset(
    rows: dict[int, dict[str, float | int]],
    actual_improve: dict[int, float],
    n_pairs: int,
    base_pose_dist: float,
) -> tuple[dict[int, dict[str, float | int]], dict[str, float | int]]:
    active: list[tuple[int, dict[str, float | int]]] = []
    for pi, r in rows.items():
        imp = float(actual_improve.get(pi, 0.0))
        if imp <= 0.0:
            continue
        nr = dict(r)
        nr["improve"] = imp
        active.append((pi, nr))
    active.sort(key=lambda item: float(item[1]["improve"]), reverse=True)

    best_rows: dict[int, dict[str, float | int]] = {}
    best_info: dict[str, float | int] = {
        "k": 0,
        "pred_net": 0.0,
        "pred_d_pose": 0.0,
        "bytes_lzma": 0,
    }
    cur: dict[int, dict[str, float | int]] = {}
    accum = 0.0
    for k, (pi, r) in enumerate(active, 1):
        cur[pi] = r
        accum += float(r["improve"])
        pose_dist = max(0.0, base_pose_dist - accum / n_pairs)
        d_pose = math.sqrt(10.0 * pose_dist) - math.sqrt(10.0 * base_pose_dist)
        dense = compressed_sizes(serialize_dense_warps(cur, n_pairs))["lzma"]
        sparse = compressed_sizes(serialize_sparse_warps(cur))["lzma"]
        byte_cost = 25.0 * min(dense, sparse) / UNCOMPRESSED_SIZE
        net = d_pose + byte_cost
        if net < float(best_info["pred_net"]):
            best_rows = dict(cur)
            best_info = {
                "k": k,
                "pred_net": net,
                "pred_d_pose": d_pose,
                "bytes_lzma": min(dense, sparse),
                "dense_lzma": dense,
                "sparse_lzma": sparse,
            }
    return best_rows, best_info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=("x5", "x5rgb"), required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--qscale", type=float, default=10.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Base: {args.base}", flush=True)
    print(f"Candidates: {args.candidates}", flush=True)

    rows = read_candidates(args.candidates)
    if not rows:
        raise ValueError("No nonzero candidates found.")
    print(f"loaded nonzero candidates: {len(rows)}", flush=True)

    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(Generator())

    frames = torch.load(CACHE_DIR / "frames_x5.pt", weights_only=False)
    f1_base = frames["f1"]
    f2_base = frames["f2"]
    if args.base == "x5rgb":
        with open(CACHE_DIR / "rgb_patches_x5.pkl", "rb") as f:
            rgb_patches = pickle.load(f)
        f1_work = apply_channel_patches(f1_base, rgb_patches)
    else:
        f1_work = f1_base

    print("Evaluating base + all candidate warps at camera resolution...", flush=True)
    seg_base, pose_base = fast_eval(f1_work, f2_base, data["val_rgb"], device)
    pair_base = per_pair_pose_mse(f1_work, f2_base, data["val_poses"], posenet, device)
    f1_all = apply_warps_hwc(f1_work, rows, args.qscale, device)
    pair_all = per_pair_pose_mse(f1_all, f2_base, data["val_poses"], posenet, device)
    actual_improve = {pi: float(pair_base[pi] - pair_all[pi]) for pi in rows}
    n_pos = sum(1 for v in actual_improve.values() if v > 0)
    print(f"actual-positive candidates: {n_pos}/{len(rows)}", flush=True)

    best_rows, info = choose_actual_subset(rows, actual_improve, f1_work.shape[0], pose_base)
    print(
        f"best actual subset: k={info['k']} bytes={info['bytes_lzma']} "
        f"d_pose={info['pred_d_pose']:+.6f} net={info['pred_net']:+.6f}",
        flush=True,
    )

    if best_rows:
        print("Applying selected subset and running full distortion eval...", flush=True)
        f1_best = apply_warps_hwc(f1_work, best_rows, args.qscale, device)
        seg_new, pose_new = fast_eval(f1_best, f2_base, data["val_rgb"], device)
        dense = compressed_sizes(serialize_dense_warps(best_rows, f1_work.shape[0]))["lzma"]
        sparse = compressed_sizes(serialize_sparse_warps(best_rows))["lzma"]
        best_bytes = min(dense, sparse)
        base_score = fast_compose(seg_base, pose_base, model_bytes, 0)
        new_score = fast_compose(seg_new, pose_new, model_bytes, best_bytes)
        delta = new_score["score"] - base_score["score"]
        print(
            f"final seg={seg_new:.9f} pose={pose_new:.9f} "
            f"pose_term={new_score['pose_term']:.6f}",
            flush=True,
        )
        print(
            f"bytes lzma={best_bytes} (dense={dense}, sparse={sparse}) "
            f"score_delta={delta:+.6f}",
            flush=True,
        )

    out_csv = OUTPUT_DIR / f"v2_warp_actual_select_{args.base}_{args.candidates.stem}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "qx", "qy", "model_improve", "actual_improve"])
        for pi, r in sorted(rows.items(), key=lambda item: actual_improve[item[0]], reverse=True):
            w.writerow([pi, r["qx"], r["qy"], r["improve"], actual_improve[pi]])
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
