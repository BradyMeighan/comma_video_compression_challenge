#!/usr/bin/env python
"""Apply actual-selected X5 warps, then refit RGB sidecars on the warped frames."""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
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

from prepare import estimate_model_bytes, load_posenet  # noqa: E402
from sidecar_channel_only import (  # noqa: E402
    apply_channel_patches,
    channel_sidecar_size,
    find_channel_only_patches,
)
from sidecar_stack import fast_compose, fast_eval, per_pair_pose_mse  # noqa: E402
from train import Generator, load_data_full  # noqa: E402
from v2_tournament_v2 import find_channel_iterative  # noqa: E402
from v2_warp_actual_select import choose_actual_subset, read_candidates  # noqa: E402
from v2_warp_probe import (  # noqa: E402
    CACHE_DIR,
    OUTPUT_DIR,
    apply_warps_hwc,
    compressed_sizes,
    serialize_dense_warps,
    serialize_sparse_warps,
)


def fit_rgb_config(
    name: str,
    tiers: list[tuple[int, int | tuple[str, int, int]]],
    f1_start: torch.Tensor,
    f2_all: torch.Tensor,
    poses: torch.Tensor,
    posenet: torch.nn.Module,
    rank: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, list[tuple[int, int, int, int]]], int, float]:
    cur_f1 = f1_start.clone()
    all_patches: dict[int, list[tuple[int, int, int, int]]] = {}
    offset = 0
    t0 = time.time()
    for n_pairs, spec in tiers:
        pair_ids = [int(x) for x in rank[offset:offset + n_pairs]]
        offset += n_pairs
        if isinstance(spec, tuple):
            _, k_each, n_passes = spec
            patches = find_channel_iterative(
                cur_f1, f2_all, poses, posenet, pair_ids,
                K_each=k_each, n_passes=n_passes, n_iter=80, device=device)
        else:
            patches = find_channel_only_patches(
                cur_f1, f2_all, poses, posenet, pair_ids,
                K=int(spec), n_iter=80, device=device)
        cur_f1 = apply_channel_patches(cur_f1, patches)
        for pi, ps in patches.items():
            all_patches.setdefault(int(pi), []).extend(ps)
    return cur_f1, all_patches, channel_sidecar_size(all_patches), time.time() - t0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, default=OUTPUT_DIR / "v2_warp_probe_x5_n600.csv")
    ap.add_argument("--qscale", type=float, default=10.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Candidates: {args.candidates}", flush=True)

    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(Generator())
    frames = torch.load(CACHE_DIR / "frames_x5.pt", weights_only=False)
    f1_x5 = frames["f1"]
    f2_x5 = frames["f2"]

    print("Selecting actual camera-res warp subset on X5...", flush=True)
    rows = read_candidates(args.candidates)
    seg_x5, pose_x5 = fast_eval(f1_x5, f2_x5, data["val_rgb"], device)
    pair_base = per_pair_pose_mse(f1_x5, f2_x5, data["val_poses"], posenet, device)
    f1_all_warp = apply_warps_hwc(f1_x5, rows, args.qscale, device)
    pair_all = per_pair_pose_mse(f1_all_warp, f2_x5, data["val_poses"], posenet, device)
    actual_improve = {pi: float(pair_base[pi] - pair_all[pi]) for pi in rows}
    warp_rows, warp_info = choose_actual_subset(rows, actual_improve, f1_x5.shape[0], pose_x5)
    warp_dense = compressed_sizes(serialize_dense_warps(warp_rows, f1_x5.shape[0]))["lzma"]
    warp_sparse = compressed_sizes(serialize_sparse_warps(warp_rows))["lzma"]
    warp_bytes = min(warp_dense, warp_sparse)
    f1_warp = apply_warps_hwc(f1_x5, warp_rows, args.qscale, device)
    seg_warp, pose_warp = fast_eval(f1_warp, f2_x5, data["val_rgb"], device)
    print(
        f"warp: k={warp_info['k']} bytes={warp_bytes} "
        f"pose_term={fast_compose(seg_warp, pose_warp, model_bytes, warp_bytes)['pose_term']:.6f}",
        flush=True,
    )

    pose_rank_warp = np.argsort(
        per_pair_pose_mse(f1_warp, f2_x5, data["val_poses"], posenet, device)
    )[::-1]

    configs = [
        ("warp_rgb_top100_K3+400_K1", [(100, 3), (400, 1)]),
        ("warp_rgb_top500_K1x2iter", [(500, ("iter", 1, 2))]),
        ("warp_rgb_top250_K5+250_K2", [(250, 5), (250, 2)]),
    ]

    out_csv = OUTPUT_DIR / "v2_warp_then_rgb_probe.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cfg",
            "warp_bytes",
            "rgb_bytes",
            "total_bytes",
            "seg_term",
            "pose_term",
            "score",
            "delta_vs_x5",
            "elapsed_s",
        ])

        base_score = fast_compose(seg_x5, pose_x5, model_bytes, 0)
        warp_score = fast_compose(seg_warp, pose_warp, model_bytes, warp_bytes)
        w.writerow([
            "warp_only",
            warp_bytes,
            0,
            warp_bytes,
            warp_score["seg_term"],
            warp_score["pose_term"],
            warp_score["score"],
            warp_score["score"] - base_score["score"],
            0,
        ])

        # Existing cached RGB reference after X5, not refit after warp.
        with open(CACHE_DIR / "rgb_patches_x5.pkl", "rb") as pf:
            cached_rgb = pickle.load(pf)
        f1_cached = apply_channel_patches(f1_x5, cached_rgb)
        seg_cached, pose_cached = fast_eval(f1_cached, f2_x5, data["val_rgb"], device)
        cached_bytes = channel_sidecar_size(cached_rgb)
        cached_score = fast_compose(seg_cached, pose_cached, model_bytes, cached_bytes)
        w.writerow([
            "x5_cached_rgb_reference",
            0,
            cached_bytes,
            cached_bytes,
            cached_score["seg_term"],
            cached_score["pose_term"],
            cached_score["score"],
            cached_score["score"] - base_score["score"],
            0,
        ])

        for name, tiers in configs:
            print(f"\nFitting {name}...", flush=True)
            f1_rgb, rgb_patches, rgb_bytes, elapsed = fit_rgb_config(
                name, tiers, f1_warp, f2_x5, data["val_poses"],
                posenet, pose_rank_warp, device)
            seg, pose = fast_eval(f1_rgb, f2_x5, data["val_rgb"], device)
            total_bytes = warp_bytes + rgb_bytes
            score = fast_compose(seg, pose, model_bytes, total_bytes)
            print(
                f"  {name}: rgb={rgb_bytes}B total={total_bytes}B "
                f"score={score['score']:.6f} pose={score['pose_term']:.6f} "
                f"delta={score['score'] - base_score['score']:+.6f} ({elapsed:.0f}s)",
                flush=True,
            )
            w.writerow([
                name,
                warp_bytes,
                rgb_bytes,
                total_bytes,
                score["seg_term"],
                score["pose_term"],
                score["score"],
                score["score"] - base_score["score"],
                elapsed,
            ])

    print(f"\nwrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
