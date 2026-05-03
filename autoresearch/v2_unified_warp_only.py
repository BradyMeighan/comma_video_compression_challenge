#!/usr/bin/env python
"""Unified non-RGB stack plus exact-selected frame-1 warp sidecar.

This answers the important ordering question: if RGB patches are byte-negative,
can X2+CMAES+S2+C3 plus a tiny geometric sidecar beat the current unified+RGB
result?
"""
from __future__ import annotations

import csv
import lzma
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

from prepare import MODEL_H, MODEL_W  # noqa: E402
from sidecar_mask_verified import regenerate_frames_from_masks  # noqa: E402
from sidecar_stack import fast_eval, per_pair_pose_mse  # noqa: E402
from v2_c3_pose_vector import (  # noqa: E402
    apply_pose_deltas_and_regen_full,
    find_pose_deltas_gridsearch,
)
from v2_per_pair_select import collect_pattern_patches, per_pair_select  # noqa: E402
from v2_s2_strip_cmaes import PATTERN_SIZES  # noqa: E402
from v2_shared import State, compose_score  # noqa: E402
from v2_unified_pipeline import build_combined_raw_v8  # noqa: E402
from v2_warp_actual_select import choose_actual_subset  # noqa: E402
from v2_warp_probe import (  # noqa: E402
    apply_warps_hwc,
    find_xy_warps,
    serialize_dense_warps,
    serialize_sparse_warps,
)


CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def lzma_len(raw: bytes) -> int:
    return len(lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6))


def main() -> None:
    s = State()
    target_dims = (1, 2, 5)
    wider_range = (-5, -3, -1, 0, 1, 3, 5)
    print("\n=== Unified non-RGB + warp ===", flush=True)

    if not (CACHE_DIR / "masks_x5.pt").exists():
        raise FileNotFoundError("Run v2_cache_builder.py first.")
    masks_after_x2 = torch.load(CACHE_DIR / "masks_after_x2.pt", weights_only=False)
    masks_x5 = torch.load(CACHE_DIR / "masks_x5.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", "rb") as f:
        x2_patches = pickle.load(f)
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", "rb") as f:
        cmaes_patches = pickle.load(f)
    print(f"cache: x2={len(x2_patches)} cmaes={len(cmaes_patches)}", flush=True)

    print("\nCollecting S2 patterns...", flush=True)
    pattern_patches = collect_pattern_patches(
        s, masks_after_x2, [int(x) for x in s.rank[:100]])

    print("\nCollecting C3 wider pose deltas...", flush=True)
    pose_deltas = find_pose_deltas_gridsearch(
        s.gen, masks_x5, s.poses, s.posenet,
        [int(x) for x in s.rank[:200]], s.device,
        target_dims=target_dims, delta_range=wider_range)

    print("\nPer-pair selecting non-RGB methods...", flush=True)
    top_pairs = [int(x) for x in s.rank[:600]]
    selected = per_pair_select(
        s, s.masks_cpu, x2_patches, cmaes_patches,
        pattern_patches, pose_deltas, top_pairs, s.poses)
    sel_x2 = {pi: x2_patches[pi] for pi, c in selected.items() if c[0] and pi in x2_patches}
    sel_cmaes = {pi: cmaes_patches[pi] for pi, c in selected.items() if c[1] and pi in cmaes_patches}
    sel_pattern = {pi: pattern_patches[pi] for pi, c in selected.items() if c[2] and pi in pattern_patches}
    sel_pose = {pi: pose_deltas[pi] for pi, c in selected.items() if c[3] and pi in pose_deltas}
    print(
        f"selected: x2={len(sel_x2)} cmaes={len(sel_cmaes)} "
        f"pattern={len(sel_pattern)} pose={len(sel_pose)}",
        flush=True,
    )

    final_masks = s.masks_cpu.clone()
    for pi, ps in sel_x2.items():
        for x, y, c in ps:
            final_masks[pi, y:y + 2, x:x + 2] = c
    for pi, ps in sel_pattern.items():
        for x, y, p_id, c in ps:
            ph, pw = PATTERN_SIZES[int(p_id)]
            final_masks[pi, y:min(y + ph, MODEL_H), x:min(x + pw, MODEL_W)] = c
    for pi, ps in sel_cmaes.items():
        for x, y, c in ps:
            final_masks[pi, y, x] = c

    print("\nRegenerating non-RGB frames...", flush=True)
    scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=s.device)
    if sel_pose:
        f1_new, f2_new = apply_pose_deltas_and_regen_full(
            s.gen, final_masks, s.poses, sel_pose, s.device, scale, target_dims)
    else:
        f1_new, f2_new = regenerate_frames_from_masks(s.gen, final_masks, s.poses, s.device)

    raw_base = build_combined_raw_v8(
        sel_x2, sel_cmaes, sel_pattern, sel_pose, {}, target_dims=target_dims)
    sb_base = lzma_len(raw_base)
    seg_base, pose_base = fast_eval(f1_new, f2_new, s.data["val_rgb"], s.device)
    score_base = compose_score(seg_base, pose_base, s.model_bytes, sb_base)
    print(
        f"nonRGB: sb_lzma={sb_base} score={score_base['score']:.6f} "
        f"seg={score_base['seg_term']:.6f} pose={score_base['pose_term']:.6f}",
        flush=True,
    )

    print("\nSearching model-space warp candidates on non-RGB unified frames...", flush=True)
    pose_rank = np.argsort(per_pair_pose_mse(f1_new, f2_new, s.poses, s.posenet, s.device))[::-1]
    dx_vals = np.arange(-2.5, 2.51, 0.5, dtype=np.float32)
    dy_vals = np.arange(-1.0, 1.01, 0.5, dtype=np.float32)
    coarse = [(float(dx), float(dy)) for dy in dy_vals for dx in dx_vals]
    fine_dx = np.arange(-0.4, 0.41, 0.1, dtype=np.float32)
    fine_dy = np.arange(-0.2, 0.21, 0.1, dtype=np.float32)
    t0 = time.time()
    rows = find_xy_warps(
        f1_new, f2_new, s.poses, [int(x) for x in pose_rank[:600]], s.posenet, s.device,
        coarse, fine_dx, fine_dy, qscale=10.0, batch_cands=64)
    print(f"warp search done in {time.time() - t0:.0f}s", flush=True)

    print("\nActual-selecting warps at camera resolution...", flush=True)
    pair_base = per_pair_pose_mse(f1_new, f2_new, s.poses, s.posenet, s.device)
    f1_all_warp = apply_warps_hwc(f1_new, rows, qscale=10.0, device=s.device)
    pair_all = per_pair_pose_mse(f1_all_warp, f2_new, s.poses, s.posenet, s.device)
    actual_improve = {pi: float(pair_base[pi] - pair_all[pi]) for pi in rows}
    warp_rows, warp_info = choose_actual_subset(rows, actual_improve, f1_new.shape[0], pose_base)
    print(
        f"warp selected: k={warp_info['k']} pred_net={warp_info['pred_net']:+.6f} "
        f"pred_d_pose={warp_info['pred_d_pose']:+.6f}",
        flush=True,
    )

    f1_best = apply_warps_hwc(f1_new, warp_rows, qscale=10.0, device=s.device)
    seg_warp, pose_warp = fast_eval(f1_best, f2_new, s.data["val_rgb"], s.device)
    raw_dense = raw_base + b"WARP_DENSE\0" + serialize_dense_warps(warp_rows, f1_new.shape[0])
    raw_sparse = raw_base + b"WARP_SPARSE\0" + serialize_sparse_warps(warp_rows)
    sb_dense = lzma_len(raw_dense)
    sb_sparse = lzma_len(raw_sparse)
    sb_warp = min(sb_dense, sb_sparse)
    score_warp = compose_score(seg_warp, pose_warp, s.model_bytes, sb_warp)
    print("\n=== Unified warp-only result ===", flush=True)
    print(
        f"nonRGB score={score_base['score']:.6f} sb={sb_base} "
        f"seg={score_base['seg_term']:.6f} pose={score_base['pose_term']:.6f}",
        flush=True,
    )
    print(
        f"warp score={score_warp['score']:.6f} sb={sb_warp} "
        f"(dense={sb_dense}, sparse={sb_sparse}) "
        f"seg={score_warp['seg_term']:.6f} pose={score_warp['pose_term']:.6f}",
        flush=True,
    )
    print(f"delta={score_warp['score'] - score_base['score']:+.6f}", flush=True)

    out_csv = OUTPUT_DIR / "v2_unified_warp_only_results.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cfg", "sb_lzma", "score", "seg_term", "pose_term", "n_warp", "dense_lzma", "sparse_lzma"])
        w.writerow(["nonRGB", sb_base, score_base["score"], score_base["seg_term"], score_base["pose_term"], 0, "", ""])
        w.writerow([
            "nonRGB_warp",
            sb_warp,
            score_warp["score"],
            score_warp["seg_term"],
            score_warp["pose_term"],
            warp_info["k"],
            sb_dense,
            sb_sparse,
        ])
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
